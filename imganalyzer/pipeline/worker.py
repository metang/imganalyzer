"""Worker — pulls jobs from the queue and dispatches to module runners.

Supports multi-threaded processing with graceful shutdown on SIGINT (Ctrl+C).
GPU-bound modules are serialized to avoid VRAM contention; small GPU models
can run concurrently when they fit within the VRAM budget.

Processing strategy
-------------------
The worker uses a VRAM-budget-aware scheduler with three GPU phases:

Phase 0 — ``objects`` pass (GPU, serial):
  Drain ALL pending ``objects`` jobs first.  Once ``objects`` is done for an
  image, ``has_person`` is known → ``cloud_ai``, ``aesthetic``, ``ocr``, and
  ``faces`` jobs are unblocked.  ``metadata`` / ``technical`` run concurrently
  in a thread pool throughout all phases.

Phase 1 — ``blip2`` (GPU, exclusive):
  BLIP-2 requires ~6 GB VRAM and runs alone (exclusive mode).

Phase 2 — ``faces`` + ``ocr`` + ``embedding`` (GPU, co-resident):
  These small models (~2.75 GB total) run concurrently with separate CUDA
  streams, saving two model load/unload cycles compared to serial execution.

Cloud / IO work runs in parallel thread pools throughout all phases, with
the cloud pool boosted to 2× when GPU is idle.

Structured output
-----------------
Each completed job emits a ``[RESULT]`` line to stdout so that the Electron
GUI can parse per-job progress without depending on Rich formatting:

    [RESULT] {"path": "/a/b.jpg", "module": "technical", "status": "done", "ms": 45}
    [RESULT] {"path": "/a/b.jpg", "module": "cloud_ai",  "status": "skipped", "ms": 0, "error": "prerequisite not met: objects"}
    [RESULT] {"path": "/a/b.jpg", "module": "objects",   "status": "failed",  "ms": 812, "error": "CUDA OOM"}

These lines are always emitted (regardless of --verbose) and are JSON-safe
(the ``error`` field has inner quotes escaped).
"""
from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Any

import sqlite3
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from imganalyzer.db.connection import get_db_path
from imganalyzer.db.queue import JobQueue
from imganalyzer.db.repository import Repository
from imganalyzer.pipeline.modules import (
    ModuleRunner, write_xmp_from_db, unload_gpu_model, _read_image, _pre_resize,
)
from imganalyzer.pipeline.vram_budget import VRAMBudget
from imganalyzer.pipeline.scheduler import ResourceScheduler

console = Console()

# Modules that use GPU — must run single-threaded on the main thread
GPU_MODULES = {"local_ai", "embedding", "blip2", "objects", "ocr", "faces"}
# Local I/O-bound modules — parallel, governed by `workers`
LOCAL_IO_MODULES = {"metadata", "technical"}
# Cloud API modules — parallel, governed by `cloud_workers`
CLOUD_MODULES = {"cloud_ai", "aesthetic"}
# Combined set (kept for backwards-compat references)
IO_MODULES = LOCAL_IO_MODULES | CLOUD_MODULES

# Modules whose output contributes to the FTS5 search index.
# The search index is rebuilt *once* per image after all its jobs complete,
# rather than after every individual module write (saves ~3M FTS5 cycles
# at 500K images with 6 text-producing modules each).
_FTS_MODULES = {"metadata", "local_ai", "blip2", "faces", "cloud_ai"}

# The ``objects`` pass must complete for an image before cloud/aesthetic
# may run (it provides ``has_person`` for the privacy gate).
# ``ocr`` and ``faces`` also depend on ``objects`` for their gate flags.
_PREREQUISITES: dict[str, str] = {
    "cloud_ai": "objects",
    "aesthetic": "objects",
    "ocr": "objects",
    "faces": "objects",
}


# ── Result notification callback ──────────────────────────────────────────────
# When running under the JSON-RPC server, this is set to a function that
# sends a ``run/result`` notification directly (bypassing print).
# When running from the CLI, it stays None and _emit_result falls back to
# printing a [RESULT] line to stdout.
_result_notify: Any = None   # Callable[[dict], None] | None


def _emit_result(
    path: str,
    module: str,
    status: str,
    ms: int,
    error: str = "",
    keywords: list[str] | None = None,
) -> None:
    """Emit a machine-readable result — via callback or [RESULT] stdout line."""
    payload: dict[str, Any] = {
        "path":   path,
        "module": module,
        "status": status,
        "ms":     ms,
    }
    if error:
        payload["error"] = error
    if keywords:
        payload["keywords"] = keywords

    if _result_notify is not None:
        # Server mode: send JSON-RPC notification directly
        try:
            _result_notify(payload)
        except Exception:
            pass  # callback failure is non-fatal
    else:
        # CLI mode: print [RESULT] line to stdout
        print(f"[RESULT] {json.dumps(payload)}", flush=True)


class Worker:
    """Process queue jobs with optional parallelism and graceful shutdown."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        workers: int = min(os.cpu_count() or 1, 8),
        cloud_workers: int = 4,
        force: bool = False,
        cloud_provider: str = "openai",
        detection_prompt: str | None = None,
        detection_threshold: float | None = None,
        face_match_threshold: float | None = None,
        verbose: bool = False,
        stale_timeout_minutes: int = 10,
        write_xmp: bool = True,
        retry_failed: bool = False,
        profile: bool = False,
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
        self.retry_failed_on_start = retry_failed
        self._shutdown = threading.Event()

        # Profiler
        from imganalyzer.pipeline.profiler import ProfileCollector, NullProfiler
        self.profiler: Any = ProfileCollector(conn) if profile else NullProfiler()
        # Track images that had a job complete this run (for XMP generation)
        self._xmp_candidates: set[int] = set()
        self._xmp_lock = threading.Lock()
        # Track images needing FTS5 search index rebuild (deferred).
        # Instead of rebuilding after every module write (5-7 SELECTs + FTS5
        # DELETE+INSERT per call, ~3M calls at 500K images), we mark images
        # as dirty and flush periodically (every 60s) and at end-of-run.
        self._fts_dirty: set[int] = set()
        self._fts_lock = threading.Lock()
        # Periodic flush: ensure FTS5 + XMP are flushed at least every 60s
        # so that a crash loses at most 1 minute of search index / XMP work.
        # (Analysis data + queue status are committed per-job and survive crashes.)
        self._flush_interval_s = 60
        self._last_flush_time = time.time()
        # Thread-local storage for per-thread DB objects
        self._local = threading.local()
        # Prefetch cache: maps image_id → pre-read image_data for IO/GPU overlap.
        # Populated by the scheduler's prefetch producer threads, consumed by
        # _process_job on the GPU thread.  Thread-safe via dict atomicity.
        self._prefetch_cache: dict[int, dict[str, Any]] = {}

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
            conn.execute("PRAGMA busy_timeout=30000")
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
                profiler=self.profiler,
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
        # Install signal handler for graceful shutdown.
        # signal.signal() only works from the main thread — when running
        # under the JSON-RPC server the worker is spawned in a daemon
        # thread, so we skip signal handling and rely on the server's
        # cancel_run mechanism (sets _shutdown via _run_cancel event).
        original_handler = None
        is_main = threading.current_thread() is threading.main_thread()
        if is_main:
            original_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._handle_sigint)

        try:
            return self._run_loop(batch_size)
        finally:
            if is_main and original_handler is not None:
                signal.signal(signal.SIGINT, original_handler)

    def _run_loop(self, batch_size: int) -> dict[str, int]:
        stats = {"done": 0, "failed": 0, "skipped": 0}

        # Cap PyTorch CUDA memory usage to 70% of physical VRAM to leave
        # headroom for other applications and avoid virtual memory spilling.
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(0.70)
        except Exception:
            pass

        # Initialize VRAM budget and scheduler
        vram = VRAMBudget()  # auto-detects GPU VRAM, applies 70% cap
        scheduler = ResourceScheduler(
            vram_budget=vram,
            gpu_batch_sizes=self._GPU_BATCH_SIZES,
            default_batch_size=batch_size,
            cpu_workers=self.workers,
            cloud_workers=self.cloud_workers,
            shutdown_event=self._shutdown,
        )

        # Recover stale jobs from previous crashes
        recovered = self.queue.recover_stale(self.stale_timeout)
        sys.stderr.write(f"[worker.run] stale_timeout={self.stale_timeout}, recovered={recovered}\n")
        sys.stderr.flush()
        if recovered:
            console.print(f"[yellow]Recovered {recovered} stale job(s) from previous run[/yellow]")

        # Retry previously failed jobs (only when explicitly requested;
        # the GUI has a dedicated "Retry failed" button that calls the
        # rebuild RPC to re-enqueue failed jobs before starting a run).
        if self.retry_failed_on_start:
            retried = self.queue.retry_failed()
            if retried:
                console.print(f"[yellow]Retrying {retried} previously failed job(s)[/yellow]")

        total_pending = self.queue.pending_count()
        sys.stderr.write(f"[worker.run] total_pending={total_pending}\n")
        sys.stderr.flush()
        if total_pending == 0:
            console.print("[green]No pending jobs in queue.[/green]")
            return stats

        # Start profiler run if enabled
        self.profiler.start_run(total_images=total_pending)

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

            # ── Helper: submit IO jobs to their pools ─────────────────────────
            def _submit_io_jobs(
                local_pool: ThreadPoolExecutor,
                cloud_pool: ThreadPoolExecutor,
            ) -> dict[Future, dict[str, Any]]:
                """Claim and submit a batch of local-IO and cloud jobs."""
                futures: dict[Future, dict[str, Any]] = {}
                for module_set, pool in (
                    (LOCAL_IO_MODULES, local_pool),
                    (CLOUD_MODULES, cloud_pool),
                ):
                    for mod in module_set:
                        jobs = self.queue.claim(batch_size=batch_size, module=mod)
                        for i, job in enumerate(jobs):
                            if self._shutdown.is_set():
                                for j in range(i, len(jobs)):
                                    self.queue.mark_pending(jobs[j]["id"])
                                return futures
                            try:
                                fut = pool.submit(self._process_job, job)
                                futures[fut] = job
                            except RuntimeError:
                                for j in range(i, len(jobs)):
                                    self.queue.mark_pending(jobs[j]["id"])
                                return futures
                return futures

            # ── Helper: collect results from futures ──────────────────────────
            def _collect_futures(futures: dict[Future, dict[str, Any]]) -> None:
                for fut, job in futures.items():
                    try:
                        result_status = fut.result()
                        stats[result_status] = stats.get(result_status, 0) + 1
                    except Exception as exc:
                        error_msg = f"{type(exc).__name__}: {exc}"
                        self.queue.mark_failed(job["id"], error_msg)
                        stats["failed"] += 1
                        image_row = self.repo.get_image(job["image_id"])
                        path = image_row["file_path"] if image_row else f"id={job['image_id']}"
                        _emit_result(path, job["module"], "failed", 0, error_msg)
                    progress.advance(task)

            # ── Helper: claim jobs from queue ─────────────────────────────────
            def _claim_fn(batch_sz: int, module: str) -> list[dict[str, Any]]:
                _, _, tl_queue, _ = self._get_thread_db()
                return tl_queue.claim(batch_size=batch_sz, module=module)

            # ── Helper: advance progress bar ──────────────────────────────────
            def _advance_fn(count: int) -> None:
                progress.advance(task, advance=count)

            # ── Helper: prefetch image for IO/GPU overlap ─────────────────────
            def _prefetch_image(job: dict[str, Any]) -> dict[str, Any] | None:
                """Read + decode + resize an image on a background IO thread.

                Stores result in ``_prefetch_cache[image_id]`` so that
                ``_process_job`` → ``_cached_read_image`` can skip disk IO.
                """
                image_id = job["image_id"]
                _, repo, _, _ = self._get_thread_db()
                image_row = repo.get_image(image_id)
                if image_row is None:
                    return None
                path = Path(image_row["file_path"])
                if not path.exists():
                    return None
                try:
                    file_size = path.stat().st_size
                    fmt = path.suffix.lower()
                    with self.profiler.span("prefetch", image_id=image_id,
                                            image_file_size=file_size, image_format=fmt):
                        data = _read_image(path)
                        data = _pre_resize(data)
                    self._prefetch_cache[image_id] = data
                    return data
                except Exception:
                    return None

            # ════════════════════════════════════════════════════════════════
            # GPU Phases (scheduler-driven)
            # Phase 0: objects (exclusive, unlocks dependencies)
            # Phase 1: blip2 (exclusive, large model)
            # Phase 2: faces + ocr + embedding (co-resident, ~2.75 GB)
            # ════════════════════════════════════════════════════════════════
            phase_labels = [
                "Phase 0 — object detection (people flag)",
                "Phase 1 — BLIP-2 captioning (exclusive GPU)",
                "Phase 2 — faces + OCR + embeddings (co-resident GPU)",
            ]

            for phase_idx in range(len(scheduler.gpu_phases)):
                if self._shutdown.is_set():
                    break

                phase_modules = scheduler.modules_for_phase(phase_idx)
                has_pending = any(
                    self.queue.pending_count(module=mod) > 0
                    for mod in phase_modules
                )
                if not has_pending:
                    continue

                console.print(f"[dim]{phase_labels[phase_idx]}[/dim]")

                with self.profiler.span("gpu_phase", phase=phase_idx), (
                    ThreadPoolExecutor(max_workers=self.workers)       as local_pool,
                    ThreadPoolExecutor(max_workers=self.cloud_workers) as cloud_pool,
                ):
                    scheduler.run_gpu_phase(
                        phase_idx,
                        claim_fn=_claim_fn,
                        process_batch_fn=self._process_job_batch,
                        process_single_fn=self._process_job,
                        submit_io_fn=_submit_io_jobs,
                        collect_fn=_collect_futures,
                        advance_fn=_advance_fn,
                        flush_fn=self._maybe_periodic_flush,
                        local_pool=local_pool,
                        cloud_pool=cloud_pool,
                        stats=stats,
                        unload_fn=unload_gpu_model,
                        prefetch_fn=_prefetch_image,
                    )

            # ════════════════════════════════════════════════════════════════
            # IO drain: remaining cloud/IO jobs with boosted cloud pool
            # ════════════════════════════════════════════════════════════════
            if not self._shutdown.is_set():
                boosted_cloud = scheduler.boosted_cloud_workers()
                with self.profiler.span("io_drain"), (
                    ThreadPoolExecutor(max_workers=self.workers)       as local_pool,
                    ThreadPoolExecutor(max_workers=boosted_cloud)      as cloud_pool,
                ):
                    console.print(
                        f"[dim]IO drain — cloud threads boosted to {boosted_cloud}[/dim]"
                    )
                    scheduler.run_io_drain(
                        submit_io_fn=_submit_io_jobs,
                        collect_fn=_collect_futures,
                        flush_fn=self._maybe_periodic_flush,
                        local_pool=local_pool,
                        cloud_pool=cloud_pool,
                    )

        if self._shutdown.is_set():
            console.print("\n[yellow]Paused.[/yellow] Run `imganalyzer run` to resume.")
        else:
            console.print(f"\n[green]Complete.[/green]")

        # Final flush: pick up any FTS5 / XMP work accumulated since the
        # last periodic flush (at most ~60s worth).
        if self._fts_dirty:
            fts_count = self._flush_fts_dirty()
            if fts_count and self.verbose:
                console.print(f"  FTS5 search index rebuilt for {fts_count} image(s)")

        if self.write_xmp and self._xmp_candidates:
            xmp_written = self._write_pending_xmps()
            if xmp_written:
                console.print(f"  XMP sidecars written: {xmp_written}")

        console.print(
            f"  Done: {stats['done']}  Failed: {stats['failed']}  "
            f"Skipped: {stats['skipped']}"
        )

        # End profiler run
        self.profiler.end_run()

        return stats

    # ── GPU batch sizes per module ──────────────────────────────────────
    # Tuned to stay within ~70% of GPU VRAM (e.g. ~11 GB on a 16 GB card)
    # to leave headroom for other applications and CUDA allocator overhead.
    _GPU_BATCH_SIZES: dict[str, int] = {
        "objects":   4,   # GroundingDINO mixed fp16/fp32, ~1.1 GB model
        "blip2":     1,   # BLIP-2 fp16, ~4.7 GB model + beam search
        "embedding": 16,  # CLIP ViT-L/14 fp16, ~0.95 GB model
        "faces":     8,   # InsightFace ONNX — claim granularity for prefetch
        "ocr":       4,   # TrOCR — claim granularity for prefetch
    }

    def _process_job_batch(
        self,
        jobs: list[dict[str, Any]],
        module: str,
    ) -> dict[str, int]:
        """Process a batch of GPU jobs using a single batched forward pass.

        Returns a stats dict: ``{done: N, failed: N, skipped: N}``.

        For modules with batch support (objects, blip2, embedding) this
        method reads all images, runs a single batched GPU call, and
        writes all results atomically.  If batching fails (OOM etc.) the
        batch method's internal fallback handles per-image sequential
        processing.  Modules without batch support fall through to
        serial ``_process_job`` calls.
        """
        batch_stats: dict[str, int] = {"done": 0, "failed": 0, "skipped": 0}

        if not jobs:
            return batch_stats

        # Modules that support batched GPU inference
        BATCH_MODULES = {"objects", "blip2", "embedding"}
        if module not in BATCH_MODULES:
            # Fallback: process each job individually
            for job in jobs:
                if self._shutdown.is_set():
                    break
                status = self._process_job(job)
                batch_stats[status] += 1
            return batch_stats

        _, repo, queue, runner = self._get_thread_db()

        # ── Pre-flight: check cache, prereqs, read images ────────────
        valid_jobs: list[dict[str, Any]] = []
        valid_image_data: list[dict[str, Any]] = []
        valid_image_ids: list[int] = []

        from imganalyzer.pipeline.modules import _read_image, _pre_resize

        for job in jobs:
            image_id = job["image_id"]
            job_id = job["id"]

            image_row = repo.get_image(image_id)
            path_str = image_row["file_path"] if image_row else f"id={image_id}"

            # Cache check
            if not runner.should_run(image_id, module):
                queue.mark_skipped(job_id, "already_analyzed")
                _emit_result(path_str, module, "skipped", 0, "already_analyzed")
                batch_stats["skipped"] += 1
                continue

            # Prerequisite check — defer (not skip) so job retries after prereq completes
            prereq = _PREREQUISITES.get(module)
            if prereq and not repo.is_analyzed(image_id, prereq):
                reason = f"prerequisite not met: {prereq}"
                queue.mark_pending(job_id)
                batch_stats["skipped"] += 1
                continue

            # For embedding, image_data reading is handled inside
            # run_embedding_batch, so we just need to validate the job
            if module == "embedding":
                valid_jobs.append(job)
                valid_image_ids.append(image_id)
                continue

            # Read image for objects/blip2
            from pathlib import Path
            path = Path(path_str)
            if not path.exists():
                error_msg = f"FileNotFoundError: Image file not found: {path}"
                queue.mark_failed(job_id, error_msg)
                _emit_result(path_str, module, "failed", 0, error_msg)
                batch_stats["failed"] += 1
                continue

            try:
                image_data = _read_image(path)
                image_data = _pre_resize(image_data)
            except Exception as exc:
                error_msg = f"{type(exc).__name__}: {exc}"
                queue.mark_failed(job_id, error_msg)
                _emit_result(path_str, module, "failed", 0, error_msg)
                batch_stats["failed"] += 1
                continue

            valid_jobs.append(job)
            valid_image_data.append(image_data)
            valid_image_ids.append(image_id)

        if not valid_jobs:
            return batch_stats

        # ── Batched GPU forward pass + DB write ──────────────────────
        start_ms = int(time.time() * 1000)

        try:
            if module == "objects":
                from imganalyzer.pipeline.passes.objects import run_objects_batch
                run_objects_batch(
                    valid_image_data, repo, valid_image_ids, runner.conn,
                    prompt=self.detection_prompt,
                    threshold=self.detection_threshold,
                )
            elif module == "blip2":
                from imganalyzer.pipeline.passes.blip2 import run_blip2_batch
                run_blip2_batch(
                    valid_image_data, repo, valid_image_ids, runner.conn,
                )
            elif module == "embedding":
                runner.run_embedding_batch(valid_jobs)

            elapsed = int(time.time() * 1000) - start_ms
            per_image_ms = elapsed // max(len(valid_jobs), 1)

            # Mark all as done
            for job in valid_jobs:
                image_id = job["image_id"]
                image_row = repo.get_image(image_id)
                path_str = image_row["file_path"] if image_row else f"id={image_id}"
                queue.mark_done(job["id"])
                _emit_result(path_str, module, "done", per_image_ms)
                batch_stats["done"] += 1

                if self.write_xmp:
                    with self._xmp_lock:
                        self._xmp_candidates.add(image_id)
                if module in _FTS_MODULES:
                    with self._fts_lock:
                        self._fts_dirty.add(image_id)

        except Exception as exc:
            # Batch failed entirely — fall back to per-image processing
            elapsed = int(time.time() * 1000) - start_ms
            if self.verbose:
                console.print(
                    f"  [yellow]Batch {module} failed ({type(exc).__name__}), "
                    f"falling back to per-image processing[/yellow]"
                )
            for job in valid_jobs:
                if self._shutdown.is_set():
                    break
                status = self._process_job(job)
                batch_stats[status] += 1

        return batch_stats

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
            # Defer back to pending so the job retries after prereq completes
            prereq = _PREREQUISITES.get(module)
            if prereq and not repo.is_analyzed(image_id, prereq):
                queue.mark_pending(job_id)
                return "skipped"

            # ── People guard for cloud/aesthetic (privacy) ───────────────────
            # Primary source: analysis_objects.has_person (set by the objects pass).
            # Fallback: analysis_local_ai.has_people (set by the legacy local_ai
            # pass) for databases populated before the split-pass refactor.
            if module in ("cloud_ai", "aesthetic"):
                has_people = False
                objects_data = repo.get_analysis(image_id, "objects")
                if objects_data is not None:
                    has_people = bool(objects_data.get("has_person"))
                else:
                    local_data = repo.get_analysis(image_id, "local_ai")
                    if local_data:
                        has_people = bool(local_data.get("has_people"))
                if has_people:
                    queue.mark_skipped(job_id, "has_people")
                    _emit_result(path, module, "skipped", 0, "has_people")
                    return "skipped"

            # ── Prime image cache from prefetch (IO/GPU overlap) ────────────
            prefetched = self._prefetch_cache.pop(image_id, None)
            if prefetched is not None:
                runner.prime_image_cache(Path(path), prefetched)

            # ── Run the module ───────────────────────────────────────────────
            result = runner.run(image_id, module)
            elapsed = int(time.time() * 1000) - start_ms
            queue.mark_done(job_id)
            # For cloud_ai, include keywords in the result notification
            kw = result.get("keywords") if module == "cloud_ai" and result else None
            _emit_result(path, module, "done", elapsed, keywords=kw)

            # Track image for XMP generation after all jobs finish
            if self.write_xmp:
                with self._xmp_lock:
                    self._xmp_candidates.add(image_id)

            # Mark image as needing FTS5 search index rebuild (deferred).
            # Only modules that produce searchable text need this.
            if module in _FTS_MODULES:
                with self._fts_lock:
                    self._fts_dirty.add(image_id)

            return "done"

        except Exception as exc:
            elapsed = int(time.time() * 1000) - start_ms
            error_msg = f"{type(exc).__name__}: {exc}"
            queue.mark_failed(job_id, error_msg)
            _emit_result(path, module, "failed", elapsed, error_msg)
            if self.verbose:
                console.print(f"  [red]Failed:[/red] {path} module={module}: {error_msg}")
            return "failed"

    def _maybe_periodic_flush(self) -> None:
        """Flush FTS5 + XMP if at least ``_flush_interval_s`` seconds elapsed.

        Called from the main-thread GPU loop between batches.  This
        ensures that a crash loses at most ~60s of search-index and XMP
        work.  Analysis data and queue status are always committed
        per-job and are not affected.
        """
        now = time.time()
        if now - self._last_flush_time < self._flush_interval_s:
            return
        self._last_flush_time = now

        # Piggyback profiler flush
        self.profiler.maybe_flush()

        # Snapshot and clear dirty sets under their locks
        with self._fts_lock:
            fts_snapshot = list(self._fts_dirty)
            self._fts_dirty.clear()
        with self._xmp_lock:
            xmp_snapshot = set(self._xmp_candidates)
            self._xmp_candidates.clear()

        # Flush FTS5 — use small batches (50 rows) with short transactions
        # so we don't hold the write lock long enough to cause
        # ``OperationalError: database is locked`` in concurrent cloud
        # worker threads (which have a 30s busy_timeout).
        if fts_snapshot:
            failed_ids: list[int] = []
            BATCH = 50
            for start in range(0, len(fts_snapshot), BATCH):
                chunk = fts_snapshot[start:start + BATCH]
                self.conn.execute("BEGIN IMMEDIATE")
                try:
                    for image_id in chunk:
                        try:
                            self.repo.update_search_index(image_id)
                        except Exception:
                            failed_ids.append(image_id)
                    self.conn.commit()
                except Exception:
                    self.conn.rollback()
                    failed_ids.extend(chunk)
            # Re-add failed IDs so they'll be retried on next flush
            if failed_ids:
                with self._fts_lock:
                    self._fts_dirty.update(failed_ids)
            if self.verbose:
                count = len(fts_snapshot) - len(failed_ids)
                if count > 0:
                    console.print(
                        f"  [dim]Periodic flush: FTS5 rebuilt for "
                        f"{count} image(s)[/dim]"
                    )

        # Flush XMP
        if self.write_xmp and xmp_snapshot:
            count = 0
            failed_xmp: set[int] = set()
            for image_id in xmp_snapshot:
                try:
                    xmp_path = write_xmp_from_db(self.repo, image_id)
                    if xmp_path:
                        count += 1
                except Exception:
                    failed_xmp.add(image_id)
            if failed_xmp:
                with self._xmp_lock:
                    self._xmp_candidates.update(failed_xmp)
            if count and self.verbose:
                console.print(
                    f"  [dim]Periodic flush: {count} XMP sidecar(s) written[/dim]"
                )

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

    def _flush_fts_dirty(self) -> int:
        """Rebuild FTS5 search index for all images marked dirty.

        Processes in batches of 50 inside a single transaction to keep
        write-lock hold time short (consistent with _maybe_periodic_flush).
        Uses the main-thread connection (``self.conn``) since this runs
        after all worker threads have joined.
        """
        dirty = list(self._fts_dirty)
        self._fts_dirty.clear()
        if not dirty:
            return 0

        BATCH = 50
        rebuilt = 0
        for start in range(0, len(dirty), BATCH):
            chunk = dirty[start:start + BATCH]
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                for image_id in chunk:
                    try:
                        self.repo.update_search_index(image_id)
                        rebuilt += 1
                    except Exception:
                        pass  # best-effort; don't crash the batch
                self.conn.commit()
            except Exception:
                self.conn.rollback()
        return rebuilt

    def _handle_sigint(self, signum: int, frame: Any) -> None:
        console.print("\n[yellow]Ctrl+C received — finishing current batch...[/yellow]")
        self._shutdown.set()
