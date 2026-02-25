"""Worker — pulls jobs from the queue and dispatches to module runners.

Supports multi-threaded processing with graceful shutdown on SIGINT (Ctrl+C).
GPU-bound modules (local_ai, embedding) are serialized to avoid VRAM contention.
"""
from __future__ import annotations

import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any

import sqlite3
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID

from imganalyzer.db.queue import JobQueue
from imganalyzer.db.repository import Repository
from imganalyzer.pipeline.modules import ModuleRunner, write_xmp_from_db

console = Console()

# Modules that use GPU — must run single-threaded
GPU_MODULES = {"local_ai", "embedding"}
# Modules that are I/O-bound — can run in parallel
IO_MODULES = {"metadata", "technical", "cloud_ai", "aesthetic"}


class Worker:
    """Process queue jobs with optional parallelism and graceful shutdown."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        workers: int = 1,
        force: bool = False,
        cloud_provider: str = "openai",
        detection_prompt: str | None = None,
        detection_threshold: float | None = None,
        face_match_threshold: float | None = None,
        verbose: bool = False,
        stale_timeout_minutes: int = 10,
        write_xmp: bool = True,
    ) -> None:
        self.conn = conn
        self.repo = Repository(conn)
        self.queue = JobQueue(conn)
        self.runner = ModuleRunner(
            conn=conn,
            repo=self.repo,
            force=force,
            cloud_provider=cloud_provider,
            detection_prompt=detection_prompt,
            detection_threshold=detection_threshold,
            face_match_threshold=face_match_threshold,
            verbose=verbose,
        )
        self.workers = max(1, workers)
        self.verbose = verbose
        self.stale_timeout = stale_timeout_minutes
        self.write_xmp = write_xmp
        self._shutdown = threading.Event()
        # Track images that had a job complete this run (for XMP generation)
        self._xmp_candidates: set[int] = set()

    def run(self, batch_size: int = 10) -> dict[str, int]:
        """Main processing loop.  Blocks until queue is empty or Ctrl+C.

        Returns summary: {done: N, failed: N, skipped: N}.
        """
        # Install signal handler for graceful shutdown
        original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_sigint)

        try:
            return self._run_loop(batch_size)
        finally:
            signal.signal(signal.SIGINT, original_handler)

    def _run_loop(self, batch_size: int) -> dict[str, int]:
        stats = {"done": 0, "failed": 0, "skipped": 0}

        # Recover stale jobs from previous crashes
        recovered = self.queue.recover_stale(self.stale_timeout)
        if recovered:
            console.print(f"[yellow]Recovered {recovered} stale job(s) from previous run[/yellow]")

        # Retry previously failed jobs
        retried = self.queue.retry_failed()
        if retried:
            console.print(f"[yellow]Retrying {retried} previously failed job(s)[/yellow]")

        total_pending = self.queue.pending_count()
        if total_pending == 0:
            console.print("[green]No pending jobs in queue.[/green]")
            return stats

        console.print(f"[cyan]Processing {total_pending} pending job(s)...[/cyan]")
        console.print("[dim]Press Ctrl+C to pause (current batch will finish).[/dim]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            console=console,
        ) as progress:
            task = progress.add_task("Processing", total=total_pending)

            while not self._shutdown.is_set():
                jobs = self.queue.claim(batch_size=batch_size)
                if not jobs:
                    break

                # Split into GPU and IO jobs
                gpu_jobs = [j for j in jobs if j["module"] in GPU_MODULES]
                io_jobs = [j for j in jobs if j["module"] in IO_MODULES]

                # Process GPU jobs serially
                for job in gpu_jobs:
                    if self._shutdown.is_set():
                        break
                    result_status = self._process_job(job)
                    stats[result_status] += 1
                    progress.advance(task)

                # Process IO jobs in parallel
                if io_jobs and not self._shutdown.is_set():
                    with ThreadPoolExecutor(max_workers=self.workers) as pool:
                        futures: dict[Future, dict[str, Any]] = {}
                        for job in io_jobs:
                            if self._shutdown.is_set():
                                break
                            fut = pool.submit(self._process_job, job)
                            futures[fut] = job

                        for fut in futures:
                            if self._shutdown.is_set():
                                break
                            try:
                                result_status = fut.result(timeout=300)
                                stats[result_status] = stats.get(result_status, 0) + 1
                            except Exception as exc:
                                job = futures[fut]
                                self.queue.mark_failed(job["id"], str(exc))
                                stats["failed"] += 1
                            progress.advance(task)

        if self._shutdown.is_set():
            console.print("\n[yellow]Paused.[/yellow] Run `imganalyzer run` to resume.")
        else:
            console.print(f"\n[green]Complete.[/green]")

        # Write XMP sidecars for images that had jobs complete
        if self.write_xmp and self._xmp_candidates:
            xmp_written = self._write_pending_xmps()
            if xmp_written:
                console.print(f"  XMP sidecars written: {xmp_written}")

        console.print(
            f"  Done: {stats['done']}  Failed: {stats['failed']}  "
            f"Skipped: {stats['skipped']}"
        )
        return stats

    def _process_job(self, job: dict[str, Any]) -> str:
        """Process a single job.  Returns 'done', 'failed', or 'skipped'."""
        image_id = job["image_id"]
        module = job["module"]
        job_id = job["id"]

        try:
            # Cache check
            if not self.runner.should_run(image_id, module):
                self.queue.mark_skipped(job_id, "already_analyzed")
                return "skipped"

            # People guard for cloud/aesthetic
            if module in ("cloud_ai", "aesthetic"):
                local_data = self.repo.get_analysis(image_id, "local_ai")
                if local_data and local_data.get("has_people"):
                    self.queue.mark_skipped(job_id, "has_people")
                    return "skipped"

            self.runner.run(image_id, module)
            self.queue.mark_done(job_id)

            # Track image for XMP generation after all jobs finish
            if self.write_xmp:
                self._xmp_candidates.add(image_id)

            return "done"

        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            self.queue.mark_failed(job_id, error_msg)
            if self.verbose:
                console.print(f"  [red]Failed: image={image_id} module={module}: {error_msg}[/red]")
            return "failed"

    def _write_pending_xmps(self) -> int:
        """Write XMP sidecars for images that had at least one job complete."""
        count = 0
        for image_id in self._xmp_candidates:
            try:
                xmp_path = write_xmp_from_db(self.repo, image_id)
                if xmp_path and self.verbose:
                    console.print(f"  [dim]XMP written: {xmp_path}[/dim]")
                if xmp_path:
                    count += 1
            except Exception as exc:
                if self.verbose:
                    console.print(
                        f"  [red]XMP write failed for image {image_id}: {exc}[/red]"
                    )
        self._xmp_candidates.clear()
        return count

    def _handle_sigint(self, signum: int, frame: Any) -> None:
        console.print("\n[yellow]Ctrl+C received — finishing current batch...[/yellow]")
        self._shutdown.set()
