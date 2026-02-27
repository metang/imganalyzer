"""Worker — pulls jobs from the queue and dispatches to module runners.

Supports multi-threaded processing with graceful shutdown on SIGINT (Ctrl+C).
GPU-bound modules (local_ai, embedding) are serialized to avoid VRAM contention.

Structured output
-----------------
Each completed job emits a ``[RESULT]`` line to stdout so that the Electron
GUI can parse per-job progress without depending on Rich formatting:

    [RESULT] {"path": "/a/b.jpg", "module": "technical", "status": "done", "ms": 45}
    [RESULT] {"path": "/a/b.jpg", "module": "cloud_ai",  "status": "skipped", "ms": 0, "error": "prerequisite not met: local_ai"}
    [RESULT] {"path": "/a/b.jpg", "module": "local_ai",  "status": "failed",  "ms": 812, "error": "CUDA OOM"}

These lines are always emitted (regardless of --verbose) and are JSON-safe
(the ``error`` field has inner quotes escaped).
"""
from __future__ import annotations

import json
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any

import sqlite3
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID

from imganalyzer.db.connection import get_db_path
from imganalyzer.db.queue import JobQueue
from imganalyzer.db.repository import Repository
from imganalyzer.pipeline.modules import ModuleRunner, write_xmp_from_db

console = Console()

# Modules that use GPU — must run single-threaded
GPU_MODULES = {"local_ai", "embedding"}
# Local I/O-bound modules — parallel, governed by `workers`
LOCAL_IO_MODULES = {"metadata", "technical"}
# Cloud API modules — parallel, governed by `cloud_workers`
CLOUD_MODULES = {"cloud_ai", "aesthetic"}
# Combined set (kept for backwards-compat references)
IO_MODULES = LOCAL_IO_MODULES | CLOUD_MODULES

# Prerequisite map: a module may only run after its prerequisite has
# completed successfully for the same image (checked via is_analyzed).
# If the prerequisite has not run, the job is skipped with a log message.
#
# NOTE: "aesthetic" is intentionally NOT listed here.  The people-privacy
# guard below (lines ~283-289) already skips aesthetic when local_ai has
# been run and detected people (has_people=1).  When local_ai has NOT run,
# get_analysis returns None → has_people is falsy → aesthetic proceeds.
# Requiring local_ai as a hard prerequisite would force users to run the
# full local_ai suite just to get aesthetic scores on images with no people.
_PREREQUISITES: dict[str, str] = {
    "cloud_ai": "local_ai",
}


def _emit_result(path: str, module: str, status: str, ms: int, error: str = "") -> None:
    """Print a machine-readable [RESULT] line to stdout."""
    payload: dict[str, Any] = {
        "path":   path,
        "module": module,
        "status": status,
        "ms":     ms,
    }
    if error:
        payload["error"] = error
    # Use json.dumps so all string values are correctly escaped
    print(f"[RESULT] {json.dumps(payload)}", flush=True)


class Worker:
    """Process queue jobs with optional parallelism and graceful shutdown."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        workers: int = 1,
        cloud_workers: int = 4,
        force: bool = False,
        cloud_provider: str = "openai",
        detection_prompt: str | None = None,
        detection_threshold: float | None = None,
        face_match_threshold: float | None = None,
        verbose: bool = False,
        stale_timeout_minutes: int = 10,
        write_xmp: bool = True,
    ) -> None:
        # Main-thread connection — used only for queue management (claim, recover, etc.)
        self.conn = conn
        self.repo = Repository(conn)
        self.queue = JobQueue(conn)

        # Per-worker-thread config (no conn stored here — opened fresh per thread)
        self.force = force
        self.cloud_provider = cloud_provider
        self.detection_prompt = detection_prompt
        self.detection_threshold = detection_threshold
        self.face_match_threshold = face_match_threshold

        self.workers = max(1, workers)
        self.cloud_workers = max(1, cloud_workers)
        self.verbose = verbose
        self.stale_timeout = stale_timeout_minutes
        self.write_xmp = write_xmp
        self._shutdown = threading.Event()
        # Track images that had a job complete this run (for XMP generation)
        self._xmp_candidates: set[int] = set()
        self._xmp_lock = threading.Lock()
        # Thread-local storage for per-thread DB objects
        self._local = threading.local()

    def _get_thread_db(self) -> tuple[sqlite3.Connection, Repository, JobQueue, "ModuleRunner"]:
        """Return (conn, repo, queue, runner) local to the current thread.

        Opens a fresh SQLite connection the first time a thread calls this,
        ensuring we never share a single connection across threads.
        """
        local = self._local
        if not hasattr(local, "conn") or local.conn is None:
            db_path = get_db_path()
            conn = sqlite3.connect(
                str(db_path),
                timeout=30,
                isolation_level=None,
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=5000")
            repo = Repository(conn)
            queue = JobQueue(conn)
            runner = ModuleRunner(
                conn=conn,
                repo=repo,
                force=self.force,
                cloud_provider=self.cloud_provider,
                detection_prompt=self.detection_prompt,
                detection_threshold=self.detection_threshold,
                face_match_threshold=self.face_match_threshold,
                verbose=self.verbose,
            )
            local.conn = conn
            local.repo = repo
            local.queue = queue
            local.runner = runner
        return local.conn, local.repo, local.queue, local.runner

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

                # Split into GPU, local-IO, and cloud jobs
                gpu_jobs       = [j for j in jobs if j["module"] in GPU_MODULES]
                local_io_jobs  = [j for j in jobs if j["module"] in LOCAL_IO_MODULES]
                cloud_jobs     = [j for j in jobs if j["module"] in CLOUD_MODULES]

                # Process GPU jobs serially
                for job in gpu_jobs:
                    if self._shutdown.is_set():
                        break
                    result_status = self._process_job(job)
                    stats[result_status] += 1
                    progress.advance(task)

                # Process local-IO and cloud jobs concurrently in separate pools
                if (local_io_jobs or cloud_jobs) and not self._shutdown.is_set():
                    futures: dict[Future, dict[str, Any]] = {}
                    with (
                        ThreadPoolExecutor(max_workers=self.workers)       as local_pool,
                        ThreadPoolExecutor(max_workers=self.cloud_workers) as cloud_pool,
                    ):
                        for job in local_io_jobs:
                            if self._shutdown.is_set():
                                break
                            fut = local_pool.submit(self._process_job, job)
                            futures[fut] = job
                        for job in cloud_jobs:
                            if self._shutdown.is_set():
                                break
                            fut = cloud_pool.submit(self._process_job, job)
                            futures[fut] = job

                    for fut, job in futures.items():
                        if self._shutdown.is_set():
                            break
                        try:
                            result_status = fut.result(timeout=300)
                            stats[result_status] = stats.get(result_status, 0) + 1
                        except Exception as exc:
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
        """Process a single job.  Returns 'done', 'failed', or 'skipped'.

        Uses a per-thread SQLite connection (via _get_thread_db) so that
        ThreadPoolExecutor workers never share a single connection.
        """
        image_id = job["image_id"]
        module = job["module"]
        job_id = job["id"]

        # Obtain thread-local DB objects (opens a fresh connection if needed)
        _, repo, queue, runner = self._get_thread_db()

        # Resolve image path for [RESULT] lines (best-effort)
        image_row = repo.get_image(image_id)
        path = image_row["file_path"] if image_row else f"id={image_id}"

        start_ms = int(time.time() * 1000)

        try:
            # ── Cache check ──────────────────────────────────────────────────
            if not runner.should_run(image_id, module):
                queue.mark_skipped(job_id, "already_analyzed")
                _emit_result(path, module, "skipped", 0, "already_analyzed")
                return "skipped"

            # ── Prerequisite check (DB-driven) ───────────────────────────────
            prereq = _PREREQUISITES.get(module)
            if prereq and not repo.is_analyzed(image_id, prereq):
                reason = f"prerequisite not met: {prereq} has not run for this image"
                queue.mark_skipped(job_id, f"prerequisite_not_met:{prereq}")
                _emit_result(path, module, "skipped", 0, reason)
                if self.verbose:
                    console.print(
                        f"  [yellow]Skipped[/yellow] {path} "
                        f"[dim]module={module} — {reason}[/dim]"
                    )
                return "skipped"

            # ── People guard for cloud/aesthetic (privacy) ───────────────────
            if module in ("cloud_ai", "aesthetic"):
                local_data = repo.get_analysis(image_id, "local_ai")
                if local_data and local_data.get("has_people"):
                    queue.mark_skipped(job_id, "has_people")
                    _emit_result(path, module, "skipped", 0, "has_people")
                    return "skipped"

            # ── Run the module ───────────────────────────────────────────────
            runner.run(image_id, module)
            elapsed = int(time.time() * 1000) - start_ms
            queue.mark_done(job_id)
            _emit_result(path, module, "done", elapsed)

            # Track image for XMP generation after all jobs finish
            if self.write_xmp:
                with self._xmp_lock:
                    self._xmp_candidates.add(image_id)

            return "done"

        except Exception as exc:
            elapsed = int(time.time() * 1000) - start_ms
            error_msg = f"{type(exc).__name__}: {exc}"
            queue.mark_failed(job_id, error_msg)
            _emit_result(path, module, "failed", elapsed, error_msg)
            if self.verbose:
                console.print(f"  [red]Failed:[/red] {path} module={module}: {error_msg}")
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
