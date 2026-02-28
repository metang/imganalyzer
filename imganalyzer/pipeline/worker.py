"""Worker — pulls jobs from the queue and dispatches to module runners.

Supports multi-threaded processing with graceful shutdown on SIGINT (Ctrl+C).
GPU-bound modules are serialized to avoid VRAM contention.

Processing strategy
-------------------
The worker uses a two-phase loop designed to unlock cloud parallelism as
early as possible:

Phase 1 — ``objects`` pass (GPU, serial):
  Drain ALL pending ``objects`` jobs first, one image at a time.  Once
  ``objects`` is done for an image, ``has_person`` is known → ``cloud_ai``
  and ``aesthetic`` jobs for that image are unblocked.

  ``metadata`` / ``technical`` jobs run concurrently in a thread pool
  throughout both phases.

Phase 2 — remaining passes (GPU serial + cloud parallel):
  The remaining GPU passes (``blip2``, ``ocr``, ``faces``, ``embedding``,
  ``local_ai``) continue running serially on the main thread.
  Meanwhile, ``cloud_ai`` and ``aesthetic`` jobs run in the cloud thread
  pool, overlapping freely with the GPU work.

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
from typing import Any

import sqlite3
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from imganalyzer.db.connection import get_db_path
from imganalyzer.db.queue import JobQueue
from imganalyzer.db.repository import Repository
from imganalyzer.pipeline.modules import ModuleRunner, write_xmp_from_db, unload_gpu_model

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


def _emit_result(path: str, module: str, status: str, ms: int, error: str = "") -> None:
    """Emit a machine-readable result — via callback or [RESULT] stdout line."""
    payload: dict[str, Any] = {
        "path":   path,
        "module": module,
        "status": status,
        "ms":     ms,
    }
    if error:
        payload["error"] = error

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

        # Recover stale jobs from previous crashes
        recovered = self.queue.recover_stale(self.stale_timeout)
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

            # ── Helper: drain a specific GPU module serially ──────────────────
            def _drain_gpu_module(module_name: str) -> None:
                """Process all pending jobs for one GPU module, one at a time."""
                while not self._shutdown.is_set():
                    jobs = self.queue.claim(batch_size=batch_size, module=module_name)
                    if not jobs:
                        break
                    for job in jobs:
                        if self._shutdown.is_set():
                            break
                        result_status = self._process_job(job)
                        stats[result_status] += 1
                        progress.advance(task)

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
                                # Release this and all remaining claimed jobs
                                # back to pending for the next run.
                                for j in range(i, len(jobs)):
                                    self.queue.mark_pending(jobs[j]["id"])
                                return futures
                            try:
                                fut = pool.submit(self._process_job, job)
                                futures[fut] = job
                            except RuntimeError:
                                # Pool already shut down (app exit race) —
                                # release the claimed job back to pending so
                                # it can be retried on the next run.
                                self.queue.mark_pending(job["id"])
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
                        # Emit a [RESULT] line so the UI shows the failure.
                        # (_process_job normally does this itself, but if the
                        # future raises we need a fallback.)
                        image_row = self.repo.get_image(job["image_id"])
                        path = image_row["file_path"] if image_row else f"id={job['image_id']}"
                        _emit_result(path, job["module"], "failed", 0, error_msg)
                    progress.advance(task)

            # ════════════════════════════════════════════════════════════════
            # Phase 1: Drain ALL ``objects`` jobs (GPU, serial).
            #
            # ``objects`` runs GroundingDINO across every image first so that
            # ``has_person`` is known for the whole batch.  That unblocks
            # ``cloud_ai`` / ``aesthetic`` which need this flag before they
            # can decide whether to run.
            #
            # ``metadata`` / ``technical`` run concurrently in a thread pool
            # throughout this phase.
            # ════════════════════════════════════════════════════════════════
            if self.queue.pending_count(module="objects") > 0 and not self._shutdown.is_set():
                console.print("[dim]Phase 1 — object detection (people flag)[/dim]")

            objects_batch_size = self._GPU_BATCH_SIZES["objects"]

            with (
                ThreadPoolExecutor(max_workers=self.workers)       as local_pool,
                ThreadPoolExecutor(max_workers=self.cloud_workers) as cloud_pool,
            ):
                pending_futures: dict[Future, dict[str, Any]] = {}

                while not self._shutdown.is_set():
                    # Batch: claim up to objects_batch_size jobs at once
                    obj_jobs = self.queue.claim(batch_size=objects_batch_size, module="objects")
                    if not obj_jobs:
                        # No more objects jobs — collect any remaining IO futures
                        # then break out of phase 1
                        _collect_futures(pending_futures)
                        pending_futures = {}
                        # Unload GroundingDINO to free ~0.9 GB VRAM before
                        # Phase 2 loads different models.
                        unload_gpu_model("objects")
                        break

                    # Process the batch on the main thread (single GPU forward pass)
                    batch_result = self._process_job_batch(obj_jobs, "objects")
                    for k in ("done", "failed", "skipped"):
                        stats[k] += batch_result[k]
                    progress.advance(task, advance=len(obj_jobs))

                    # After each batch, fire a fresh batch of IO jobs
                    # (metadata/technical/cloud_ai/aesthetic)
                    new_io = _submit_io_jobs(local_pool, cloud_pool)
                    pending_futures.update(new_io)

                    # Reap any IO futures that are already done to keep the
                    # dict from growing unbounded
                    done_futs = [f for f in pending_futures if f.done()]
                    for fut in done_futs:
                        job = pending_futures.pop(fut)
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

                    # Periodic flush: commit FTS5 + XMP every ~60s so a crash
                    # loses at most 1 minute of search-index / XMP work.
                    self._maybe_periodic_flush()

            # ════════════════════════════════════════════════════════════════
            # Phase 2: Remaining GPU passes (batched where supported) +
            #          cloud passes (parallel)
            #
            # blip2 / embedding use batched GPU forward passes via
            # ``_process_job_batch()``.  ocr / faces / local_ai remain
            # serial (ONNX can't batch; local_ai is legacy).  cloud_ai /
            # aesthetic run concurrently in the cloud thread pool.
            # ════════════════════════════════════════════════════════════════
            if not self._shutdown.is_set():
                # GPU pass order: blip2 first (captioning, no prereqs),
                # then ocr/faces (need objects — already done), then
                # local_ai (legacy full pipeline), then embedding last
                # (needs descriptions to exist).
                remaining_gpu = ["blip2", "ocr", "faces", "local_ai", "embedding"]

                with (
                    ThreadPoolExecutor(max_workers=self.workers)       as local_pool,
                    ThreadPoolExecutor(max_workers=self.cloud_workers) as cloud_pool,
                ):
                    # Kick off all available cloud + local-IO work immediately
                    pending_futures = _submit_io_jobs(local_pool, cloud_pool)

                    for gpu_module in remaining_gpu:
                        if self._shutdown.is_set():
                            break

                        # Use the tuned batch size if this module supports
                        # batching, otherwise fall back to the queue claim
                        # batch_size (which _process_job_batch will handle
                        # with serial _process_job calls).
                        gpu_batch_sz = self._GPU_BATCH_SIZES.get(gpu_module, batch_size)

                        while not self._shutdown.is_set():
                            gpu_jobs = self.queue.claim(batch_size=gpu_batch_sz, module=gpu_module)
                            if not gpu_jobs:
                                break

                            # Dispatch: batch-capable modules use a single
                            # GPU forward pass; others fall through to serial
                            batch_result = self._process_job_batch(gpu_jobs, gpu_module)
                            for k in ("done", "failed", "skipped"):
                                stats[k] += batch_result[k]
                            progress.advance(task, advance=len(gpu_jobs))

                            # After each GPU batch, sweep for newly available
                            # cloud/IO work and submit it
                            new_io = _submit_io_jobs(local_pool, cloud_pool)
                            pending_futures.update(new_io)

                            # Reap completed futures
                            done_futs = [f for f in pending_futures if f.done()]
                            for fut in done_futs:
                                job2 = pending_futures.pop(fut)
                                try:
                                    result_status = fut.result()
                                    stats[result_status] = stats.get(result_status, 0) + 1
                                except Exception as exc:
                                    error_msg = f"{type(exc).__name__}: {exc}"
                                    self.queue.mark_failed(job2["id"], error_msg)
                                    stats["failed"] += 1
                                    image_row = self.repo.get_image(job2["image_id"])
                                    path = image_row["file_path"] if image_row else f"id={job2['image_id']}"
                                    _emit_result(path, job2["module"], "failed", 0, error_msg)
                                progress.advance(task)

                            # Periodic flush: commit FTS5 + XMP every ~60s
                            self._maybe_periodic_flush()

                        # All jobs for this GPU module are done — unload its
                        # model to free VRAM before the next module loads.
                        # Peak VRAM drops from ~9.5 GB (all co-resident) to
                        # ~4.7 GB (single largest model = BLIP-2).
                        unload_gpu_model(gpu_module)

                    # All GPU passes done — collect remaining IO futures
                    _collect_futures(pending_futures)

                    # ════════════════════════════════════════════════════════════
                    # Phase 3: Drain any remaining IO/cloud jobs.
                    #
                    # When GPU work finishes before all IO/cloud jobs have been
                    # submitted (common on resume when only cloud_ai / aesthetic
                    # remain), keep claiming and processing IO batches until
                    # the queue is empty.
                    # ════════════════════════════════════════════════════════════
                    while not self._shutdown.is_set():
                        io_futures = _submit_io_jobs(local_pool, cloud_pool)
                        if not io_futures:
                            break
                        _collect_futures(io_futures)
                        self._maybe_periodic_flush()

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
        return stats

    # ── GPU batch sizes per module ──────────────────────────────────────
    # Tuned to stay within the 14 GB VRAM ceiling with model unloading.
    _GPU_BATCH_SIZES: dict[str, int] = {
        "objects":   8,   # GroundingDINO mixed fp16/fp32, ~1.1 GB model
        "blip2":     2,   # BLIP-2 fp16, beam search is memory-hungry
        "embedding": 32,  # CLIP ViT-L/14 fp16, ~0.95 GB model
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

            # Prerequisite check
            prereq = _PREREQUISITES.get(module)
            if prereq and not repo.is_analyzed(image_id, prereq):
                reason = f"prerequisite not met: {prereq}"
                queue.mark_skipped(job_id, f"prerequisite_not_met:{prereq}")
                _emit_result(path_str, module, "skipped", 0, reason)
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

            # ── Run the module ───────────────────────────────────────────────
            runner.run(image_id, module)
            elapsed = int(time.time() * 1000) - start_ms
            queue.mark_done(job_id)
            _emit_result(path, module, "done", elapsed)

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
            BATCH = 50
            for start in range(0, len(fts_snapshot), BATCH):
                chunk = fts_snapshot[start:start + BATCH]
                self.conn.execute("BEGIN IMMEDIATE")
                try:
                    for image_id in chunk:
                        try:
                            self.repo.update_search_index(image_id)
                        except Exception:
                            pass
                    self.conn.commit()
                except Exception:
                    self.conn.rollback()
            if self.verbose:
                console.print(
                    f"  [dim]Periodic flush: FTS5 rebuilt for "
                    f"{len(fts_snapshot)} image(s)[/dim]"
                )

        # Flush XMP
        if self.write_xmp and xmp_snapshot:
            count = 0
            for image_id in xmp_snapshot:
                try:
                    xmp_path = write_xmp_from_db(self.repo, image_id)
                    if xmp_path:
                        count += 1
                except Exception:
                    pass
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
