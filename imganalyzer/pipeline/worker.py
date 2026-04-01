"""Worker — pulls jobs from the queue and dispatches to module runners.

Supports multi-threaded processing with graceful shutdown on SIGINT (Ctrl+C).
GPU-bound modules are serialized to avoid VRAM contention; small GPU models
can run concurrently when they fit within the VRAM budget.

Processing strategy
-------------------
The worker uses a VRAM-budget-aware scheduler with GPU phases:

Phase 0 — ``caption`` pass (GPU, serial):
  qwen3.5 via Ollama generates descriptions, keywords, scene classification,
  and aesthetic scoring.  The Ollama model is unloaded after this phase.

Phase 1 — ``objects`` pass (GPU, serial):
  Drain ALL pending ``objects`` jobs first.  Once ``objects`` is done for an
  image, ``faces`` jobs are unblocked.  ``metadata`` / ``technical`` run
  concurrently in a thread pool throughout all phases.

Phase 2 — ``faces`` + ``embedding`` (GPU, co-resident):
  These models (~1.95 GB total) run concurrently with separate CUDA streams.

Phase 3 — ``perception`` (GPU, exclusive):
  UniPercept (~13.8 GB effective, 4-bit) runs as the last phase in each mini-batch.
  While CUDA processes perception, macOS workers continue caption jobs.

Mini-batch interleaving
-----------------------
When chunk_size > 100, each chunk is split into mini-batches of ~100 images.
Instead of running ALL captions before ANY objects, phases are interleaved
per mini-batch: [caption ×100 → objects ×100 → faces+embed ×100 → repeat].
This gives first fully-analyzed results in ~40 min instead of ~3.5 hours.
Model switch overhead is ~18s per mini-batch boundary (<1% of batch time).

Structured output
-----------------
Each completed job emits a ``[RESULT]`` line to stdout so that the Electron
GUI can parse per-job progress without depending on Rich formatting:

    [RESULT] {"path": "/a/b.jpg", "module": "technical", "status": "done", "ms": 45}
    [RESULT] {"path": "/a/b.jpg", "module": "objects",   "status": "failed",  "ms": 812, "error": "CUDA OOM"}

These lines are always emitted (regardless of --verbose) and are JSON-safe
(the ``error`` field has inner quotes escaped).
"""
from __future__ import annotations

import json
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future, CancelledError
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
GPU_MODULES = {"caption", "embedding", "objects", "faces", "perception"}
# Local I/O-bound modules — parallel, governed by `workers`
LOCAL_IO_MODULES = {"metadata", "technical"}
# Combined IO set
IO_MODULES = LOCAL_IO_MODULES

# Modules whose output contributes to the FTS5 search index.
# The search index is rebuilt *once* per image after all its jobs complete,
# rather than after every individual module write (saves ~3M FTS5 cycles
# at 500K images with 6 text-producing modules each).
_FTS_MODULES = {"metadata", "caption", "faces"}

# The ``objects`` pass must complete for an image before faces/embedding
# may run (it provides detection-derived gating/context).
_PREREQUISITES: dict[str, str] = {
    "faces": "objects",
    "embedding": "objects",
}
_DEPENDENTS: dict[str, list[str]] = {}
for _mod, _prereq in _PREREQUISITES.items():
    _DEPENDENTS.setdefault(_prereq, []).append(_mod)

_LOCK_RETRY_ATTEMPTS = 4
_LOCK_RETRY_INITIAL_DELAY_S = 0.15


def _is_transient_db_lock_error(exc: Exception) -> bool:
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    text = str(exc).lower()
    return (
        "database is locked" in text
        or "database table is locked" in text
        or "database schema is locked" in text
    )


# ── Result notification callback ──────────────────────────────────────────────
# When running under the JSON-RPC server, this is set to a function that
# sends a ``run/result`` notification directly (bypassing print).
# When running from the CLI, it stays None and _emit_result falls back to
# printing a [RESULT] line to stdout.
_result_notify: Any = None   # Callable[[dict], None] | None
_chunk_notify: Any = None    # Callable[[dict], None] | None


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
        force: bool = False,
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
        self.detection_prompt = detection_prompt
        self.detection_threshold = detection_threshold
        self.face_match_threshold = face_match_threshold

        self.workers = max(1, workers)
        self.verbose = verbose
        self.stale_timeout = stale_timeout_minutes
        self.write_xmp = write_xmp
        self.retry_failed_on_start = retry_failed
        self._shutdown = threading.Event()

        # Current chunk image IDs — exposed so the coordinator's job claim
        # handler can prefer these images for distributed workers, keeping
        # workers focused on the same chunk as the coordinator.
        self.current_chunk_ids: set[int] | None = None
        self.current_chunk_index: int = 0
        self.total_chunks: int = 0

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
        # Cooldown window after a perception VRAM timeout so we keep processing
        # other modules instead of repeatedly stalling on the same phase.
        self._perception_vram_cooldown_until = 0.0

    @staticmethod
    def _mini_batch_size(chunk_size: int) -> int:
        """Compute optimal mini-batch size for GPU phase interleaving.

        Mini-batches allow images to become fully analyzed sooner by
        interleaving caption (Ollama, ~25s/img) with fast modules
        (objects/faces/embedding, ~2s/img) instead of waiting for ALL
        captions to finish before starting fast modules.

        Model switch overhead is ~18s per mini-batch boundary.  We want
        this to be < 2% of mini-batch time, so minimum batch is 50
        (18s / 50×25s = 1.4%).  Default is 100 (0.7% overhead).

        Returns chunk_size itself if interleaving isn't worthwhile.
        """
        if chunk_size <= 100:
            return chunk_size  # single mini-batch, no interleaving
        return min(100, max(50, chunk_size // 5))

    def _get_thread_db(self) -> tuple[sqlite3.Connection, Repository, JobQueue, "ModuleRunner"]:
        """Return (conn, repo, queue, runner) local to the current thread.

        Opens a fresh SQLite connection the first time a thread calls this,
        ensuring we never share a single connection across threads.
        """
        local = self._local
        if not hasattr(local, "conn") or local.conn is None:
            from imganalyzer.db.connection import create_connection
            conn = create_connection(busy_timeout_ms=30000)
            repo = Repository(conn)
            queue = JobQueue(conn)
            runner = ModuleRunner(
                conn=conn,
                repo=repo,
                force=self.force,
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

    def _pending_image_ids_with_running_wait(
        self,
        poll_interval_s: float = 1.0,
        reclaim_interval_s: float = 15.0,
    ) -> list[int]:
        """Return pending image_ids, waiting while leased jobs are still running.

        In distributed runs, workers can temporarily lease most pending jobs.
        During that window ``pending=0`` does not mean the queue is finished —
        it only means work is currently in ``running`` state on workers.
        """
        idle_polls = 0
        while not self._shutdown.is_set():
            pending_image_ids = self.queue.get_pending_image_ids()
            if pending_image_ids:
                return pending_image_ids

            running_jobs = self.queue.running_count()
            if running_jobs <= 0:
                return []

            idle_polls += 1
            reclaim_every = max(
                1,
                int(max(reclaim_interval_s, 0.0) / max(poll_interval_s, 0.1)),
            )
            released = 0
            # Reclaim expired worker leases while waiting so stale
            # distributed jobs are eventually returned to pending.
            if idle_polls % reclaim_every == 0:
                try:
                    released = self.queue.release_expired_leases()
                except sqlite3.OperationalError as exc:
                    if self.verbose:
                        console.print(
                            "[dim]Deferred lease reclaim (database busy): "
                            f"{exc}[/dim]"
                        )
            if released > 0:
                idle_polls = 0
                continue

            log_every = max(1, int(30 / max(poll_interval_s, 0.1)))
            if idle_polls % log_every == 0:
                console.print(
                    f"[dim]Waiting for {running_jobs} running distributed job(s) "
                    "to free pending work...[/dim]"
                )
            self._shutdown.wait(poll_interval_s)

        return []

    def _is_perception_vram_timeout(
        self,
        phase_modules: list[str],
        exc: RuntimeError,
    ) -> bool:
        if phase_modules != ["perception"]:
            return False
        msg = str(exc)
        return (
            "Timed out waiting for VRAM for perception" in msg
            or "Cannot load perception" in msg
        )

    def _defer_perception_phase(self, cooldown_s: float = 30.0) -> None:
        self._perception_vram_cooldown_until = time.monotonic() + max(1.0, cooldown_s)
        if self.verbose:
            console.print(
                "[yellow]Perception phase deferred: insufficient free VRAM. "
                f"Retrying in ~{int(cooldown_s)}s while other modules continue.[/yellow]"
            )

    def _perception_on_cooldown(self, phase_modules: list[str]) -> bool:
        return (
            phase_modules == ["perception"]
            and time.monotonic() < self._perception_vram_cooldown_until
        )

    def run(self, batch_size: int = 10, chunk_size: int = 0) -> dict[str, int]:
        """Main processing loop.  Blocks until queue is empty or Ctrl+C.

        If *chunk_size* > 0, images are processed in chunks: each chunk runs
        through all phases before moving to the next, so images become fully
        analyzed incrementally.  ``chunk_size=0`` processes all images at once
        (original behaviour).

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
            return self._run_loop(batch_size, chunk_size=chunk_size)
        finally:
            if is_main and original_handler is not None:
                signal.signal(signal.SIGINT, original_handler)

    def _run_loop(self, batch_size: int, chunk_size: int = 0) -> dict[str, int]:
        stats = {"done": 0, "failed": 0, "skipped": 0}

        # Initialize VRAM budget and scheduler
        vram = VRAMBudget()  # auto-detects GPU VRAM, applies 70% cap
        # Cap PyTorch CUDA memory usage conservatively.
        # (MPS uses unified memory — no per-process fraction API.)
        try:
            import torch
            if torch.cuda.is_available():
                fraction = 0.70
                torch.cuda.set_per_process_memory_fraction(fraction)
        except Exception:
            pass

        scheduler = ResourceScheduler(
            vram_budget=vram,
            gpu_batch_sizes=self._GPU_BATCH_SIZES,
            default_batch_size=batch_size,
            cpu_workers=self.workers,
            shutdown_event=self._shutdown,
        )

        # Backward compatibility: remap legacy queue modules to their
        # current replacements (blip2/cloud_ai/local_ai → caption, aesthetic → perception).
        remapped = self.queue.remap_pending_modules({
            "blip2": "caption",
            "cloud_ai": "caption",
            "local_ai": "caption",
            "aesthetic": "perception",
        })
        if remapped["updated"] or remapped["deleted"]:
            console.print(
                "[yellow]Migrated legacy queue modules:[/yellow] "
                f"updated={remapped['updated']} deleted_duplicates={remapped['deleted']}"
            )

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

            # ── Helper: submit IO jobs to their pool ──────────────────────────

            def _submit_io_jobs(
                local_pool: ThreadPoolExecutor,
                cloud_pool: ThreadPoolExecutor,
            ) -> dict[Future, dict[str, Any]]:
                """Claim and submit a batch of local-IO jobs."""
                futures: dict[Future, dict[str, Any]] = {}
                for mod in LOCAL_IO_MODULES:
                    jobs = self.queue.claim(
                        batch_size=batch_size, module=mod
                    )
                    for i, job in enumerate(jobs):
                        if self._shutdown.is_set():
                            for j in range(i, len(jobs)):
                                self.queue.mark_pending(jobs[j]["id"])
                            return futures
                        try:
                            fut = local_pool.submit(self._process_job, job)
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
                    except CancelledError:
                        self.queue.mark_pending(job["id"])
                    except Exception as exc:
                        error_msg = f"{type(exc).__name__}: {exc}"
                        self.queue.mark_failed(job["id"], error_msg)
                        stats["failed"] += 1
                        image_row = self.repo.get_image(job["image_id"])
                        path = image_row["file_path"] if image_row else f"id={job['image_id']}"
                        _emit_result(path, job["module"], "failed", 0, error_msg)
                    progress.advance(task)

            def _cancel_futures(futures: dict[Future, dict[str, Any]]) -> None:
                for fut, job in list(futures.items()):
                    if fut.done():
                        continue
                    if fut.cancel():
                        self.queue.mark_pending(job["id"])
                        futures.pop(fut, None)

            # ── Helper: claim jobs from queue ─────────────────────────────────
            # Maximum seconds the master waits for distributed workers to
            # release work before giving up on a phase.
            _DISTRIBUTED_WAIT_S = 5.0
            _DISTRIBUTED_POLL_S = 1.0

            def _claim_fn(batch_sz: int, module: str) -> list[dict[str, Any]]:
                _, _, tl_queue, _ = self._get_thread_db()
                jobs = tl_queue.claim(batch_size=batch_sz, module=module)
                if jobs or self._shutdown.is_set():
                    return jobs
                # No pending jobs — check if distributed workers hold leased
                # jobs for this module.  If so, wait briefly for them to
                # complete (releasing dependent work) or expire.
                leased = tl_queue.leased_running_count(module=module)
                if leased <= 0:
                    return []
                waited = 0.0
                while waited < _DISTRIBUTED_WAIT_S and not self._shutdown.is_set():
                    self._shutdown.wait(_DISTRIBUTED_POLL_S)
                    waited += _DISTRIBUTED_POLL_S
                    try:
                        tl_queue.release_expired_leases()
                    except Exception:
                        pass
                    jobs = tl_queue.claim(batch_size=batch_sz, module=module)
                    if jobs:
                        return jobs
                return []

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
            # Sweep loop — after processing all chunks, re-check the queue
            # for remaining pending work (e.g. new images added, distributed
            # workers completing prerequisites).  Only exits when there is
            # truly nothing left or the user pauses.
            # ════════════════════════════════════════════════════════════════
            sweep = 0
            chunk_offset = 0          # cumulative chunk count across sweeps
            overall_total_chunks = 0  # total chunks across all sweeps
            _consecutive_empty_sweeps = 0
            _MAX_EMPTY_SWEEPS = 3
            while not self._shutdown.is_set():
                all_image_ids = self._pending_image_ids_with_running_wait(poll_interval_s=1.0)
                if not all_image_ids:
                    # Safety: re-check globally before exiting.  The previous
                    # call may have missed newly pending work due to timing.
                    _consecutive_empty_sweeps += 1
                    if _consecutive_empty_sweeps < _MAX_EMPTY_SWEEPS:
                        global_pending = self.queue.pending_count()
                        if global_pending > 0:
                            console.print(
                                f"[yellow]Sweep returned empty but {global_pending} "
                                f"pending globally — retrying "
                                f"({_consecutive_empty_sweeps}/{_MAX_EMPTY_SWEEPS})[/yellow]"
                            )
                            self._shutdown.wait(2.0)
                            continue
                    console.print(
                        "[dim]Sweep loop: no more pending/running work, exiting.[/dim]"
                    )
                    break  # truly nothing left
                _consecutive_empty_sweeps = 0

                sweep += 1
                if sweep > 1:
                    console.print(
                        f"\n[cyan]Sweep {sweep}: {len(all_image_ids)} images "
                        f"still pending, continuing...[/cyan]"
                    )
                    progress.update(task, total=stats["done"] + stats["failed"] + len(all_image_ids))

                if chunk_size > 0 and len(all_image_ids) > chunk_size:
                    chunks = [
                        set(all_image_ids[i : i + chunk_size])
                        for i in range(0, len(all_image_ids), chunk_size)
                    ]
                else:
                    chunks = [None]  # type: ignore[list-item]

                sweep_chunks = len(chunks)
                if sweep == 1:
                    overall_total_chunks = sweep_chunks
                else:
                    overall_total_chunks = chunk_offset + sweep_chunks
                total_chunks = overall_total_chunks
                if sweep_chunks > 1 or (sweep == 1 and sweep_chunks > 1):
                    console.print(
                        f"[cyan]Chunked processing: {sweep_chunks} chunks "
                        f"of {chunk_size} images each[/cyan]"
                    )

                for chunk_idx, chunk_ids in enumerate(chunks):
                    if self._shutdown.is_set():
                        break

                    global_chunk_idx = chunk_offset + chunk_idx

                    # Expose current chunk to coordinator's job claim handler
                    # so distributed workers are directed to the same chunk.
                    self.current_chunk_ids = chunk_ids
                    self.current_chunk_index = global_chunk_idx
                    self.total_chunks = total_chunks
                    chunk_start_ms = time.perf_counter() * 1000
                    chunk_stats_before = {k: stats[k] for k in ("done", "failed", "skipped")}

                    if total_chunks > 1:
                        console.print(
                            f"\n[bold cyan]━━ Chunk {global_chunk_idx + 1}/{total_chunks} "
                            f"({len(chunk_ids)} images) ━━[/bold cyan]"  # type: ignore[arg-type]
                        )

                    # Chunk-scoped IO submit — same as _submit_io_jobs but uses chunk_ids filter
                    def _chunk_submit_io_jobs(
                        local_pool: ThreadPoolExecutor,
                        cloud_pool: ThreadPoolExecutor,
                    ) -> dict[Future, dict[str, Any]]:
                        futures: dict[Future, dict[str, Any]] = {}
                        for mod in LOCAL_IO_MODULES:
                            jobs = self.queue.claim(
                                batch_size=batch_size, module=mod,
                                image_ids=chunk_ids,
                            )
                            for i, job in enumerate(jobs):
                                if self._shutdown.is_set():
                                    for j in range(i, len(jobs)):
                                        self.queue.mark_pending(jobs[j]["id"])
                                    return futures
                                try:
                                    fut = local_pool.submit(self._process_job, job)
                                    futures[fut] = job
                                except RuntimeError:
                                    for j in range(i, len(jobs)):
                                        self.queue.mark_pending(jobs[j]["id"])
                                    return futures
                        return futures

                    # Use chunk-scoped functions when chunking, original otherwise
                    active_submit_io = _chunk_submit_io_jobs if chunk_ids is not None else _submit_io_jobs

                    # ════════════════════════════════════════════════════════════
                    # Mini-batch interleaving
                    #
                    # Instead of processing ALL captions (~3.5h for 500 images)
                    # before ANY objects/faces/embedding, interleave GPU phases
                    # in mini-batches of ~100 images:
                    #
                    #   [caption ×100] → [objects ×100] → [faces+embed ×100]
                    #   [caption ×100] → [objects ×100] → [faces+embed ×100]
                    #   ...
                    #
                    # Benefits:
                    # - First 100 images fully analyzed in ~40 min (not 3.5h)
                    # - Distributed workers help with caption during fast-module
                    #   windows (chunk affinity keeps them on the same chunk)
                    # - Model switch overhead: ~18s per mini-batch boundary
                    #   (< 1% of mini-batch time)
                    # ════════════════════════════════════════════════════════════
                    phase_labels = [
                        f"Phase {i} — {', '.join(scheduler.modules_for_phase(i))}"
                        for i in range(len(scheduler.gpu_phases))
                    ]

                    # Split chunk into mini-batches for interleaving
                    if chunk_ids is not None:
                        mini_size = self._mini_batch_size(len(chunk_ids))
                        chunk_list = list(chunk_ids)
                        mini_batches: list[set[int]] = [
                            set(chunk_list[i : i + mini_size])
                            for i in range(0, len(chunk_list), mini_size)
                        ]
                    else:
                        # No chunking — single mini-batch of everything
                        mini_batches = [None]  # type: ignore[list-item]

                    total_minis = len(mini_batches)
                    if total_minis > 1:
                        console.print(
                            f"[dim]  Mini-batch interleaving: {total_minis} batches "
                            f"of ~{mini_size} images[/dim]"
                        )

                    for mini_idx, mini_ids in enumerate(mini_batches):
                        if self._shutdown.is_set():
                            break

                        # Build mini-batch-scoped claim function.
                        # Falls back to chunk scope (or global) if mini_ids is None.
                        if mini_ids is not None:
                            def _mini_claim_fn(
                                batch_sz: int, module: str,
                                _ids: set[int] = mini_ids,
                            ) -> list[dict[str, Any]]:
                                _, _, tl_queue, _ = self._get_thread_db()
                                jobs = tl_queue.claim(
                                    batch_size=batch_sz, module=module, image_ids=_ids,
                                )
                                if jobs or self._shutdown.is_set():
                                    return jobs
                                leased = tl_queue.leased_running_count(
                                    module=module, image_ids=_ids,
                                )
                                if leased <= 0:
                                    return []
                                waited = 0.0
                                while waited < _DISTRIBUTED_WAIT_S and not self._shutdown.is_set():
                                    self._shutdown.wait(_DISTRIBUTED_POLL_S)
                                    waited += _DISTRIBUTED_POLL_S
                                    try:
                                        tl_queue.release_expired_leases()
                                    except Exception:
                                        pass
                                    jobs = tl_queue.claim(
                                        batch_size=batch_sz, module=module, image_ids=_ids,
                                    )
                                    if jobs:
                                        return jobs
                                return []
                            active_claim_fn = _mini_claim_fn
                        else:
                            active_claim_fn = _claim_fn

                        if total_minis > 1:
                            console.print(
                                f"[dim]  ── Mini-batch {mini_idx + 1}/{total_minis} "
                                f"({len(mini_ids)} images) ──[/dim]"  # type: ignore[arg-type]
                            )

                        for phase_idx in range(len(scheduler.gpu_phases)):
                            if self._shutdown.is_set():
                                break

                            phase_modules = scheduler.modules_for_phase(phase_idx)
                            if self._perception_on_cooldown(phase_modules):
                                continue
                            has_pending = any(
                                self.queue.pending_count(
                                    module=mod,
                                    image_ids=mini_ids if mini_ids is not None else chunk_ids,
                                ) > 0
                                for mod in phase_modules
                            )
                            if not has_pending:
                                continue

                            if total_minis <= 1:
                                console.print(f"[dim]{phase_labels[phase_idx]}[/dim]")

                            with (
                                self.profiler.span(
                                    "gpu_phase", phase=phase_idx, mini_batch=mini_idx,
                                ),
                                ThreadPoolExecutor(max_workers=self.workers) as local_pool,
                                ThreadPoolExecutor(max_workers=1)            as cloud_pool,
                            ):
                                try:
                                    scheduler.run_gpu_phase(
                                        phase_idx,
                                        claim_fn=active_claim_fn,
                                        process_batch_fn=self._process_job_batch,
                                        process_single_fn=self._process_job,
                                        submit_io_fn=active_submit_io,
                                        collect_fn=_collect_futures,
                                        advance_fn=_advance_fn,
                                        flush_fn=self._maybe_periodic_flush,
                                        local_pool=local_pool,
                                        cloud_pool=cloud_pool,
                                        stats=stats,
                                        unload_fn=unload_gpu_model,
                                        prefetch_fn=_prefetch_image,
                                        cancel_futures_fn=_cancel_futures,
                                    )
                                except RuntimeError as exc:
                                    if self._is_perception_vram_timeout(phase_modules, exc):
                                        self._defer_perception_phase(cooldown_s=30.0)
                                        continue
                                    raise

                    # ════════════════════════════════════════════════════════════
                    # IO drain (per-chunk)
                    # Independent GPU modules (e.g. perception) are deferred to
                    # after all chunks so they don't block chunk progression.
                    # ════════════════════════════════════════════════════════════
                    if not self._shutdown.is_set():
                        with (
                            self.profiler.span("io_drain"),
                            ThreadPoolExecutor(max_workers=self.workers)  as local_pool,
                            ThreadPoolExecutor(max_workers=1)             as cloud_pool,
                        ):
                            console.print("[dim]IO drain[/dim]")

                            scheduler.run_io_drain(
                                submit_io_fn=active_submit_io,
                                collect_fn=_collect_futures,
                                flush_fn=self._maybe_periodic_flush,
                                local_pool=local_pool,
                                cloud_pool=cloud_pool,
                                cancel_futures_fn=_cancel_futures,
                            )

                    # ════════════════════════════════════════════════════════════
                    # Chunk retry: pick up remaining pending jobs in this chunk.
                    # Jobs may still be pending because distributed workers
                    # finished prerequisites (e.g. objects → caption unlocked)
                    # or because slow modules weren't fully drained.
                    # ════════════════════════════════════════════════════════════
                    chunk_retry = 0
                    last_stale_recovery = time.perf_counter()
                    stale_recovery_interval_s = 60.0  # recover stuck jobs every 60s
                    while not self._shutdown.is_set() and chunk_ids is not None:
                        # ── Periodic stale + lease recovery ──────────────────
                        # During long chunk processing, jobs claimed by
                        # distributed workers can get stuck in "running" state
                        # indefinitely.  Periodically reclaim them.
                        now_pc = time.perf_counter()
                        if now_pc - last_stale_recovery >= stale_recovery_interval_s:
                            try:
                                recovered = self.queue.recover_stale(
                                    self.stale_timeout,
                                )
                                released = self.queue.release_expired_leases()
                                if recovered or released:
                                    console.print(
                                        f"[yellow]  Recovered {recovered} stale + "
                                        f"{released} expired lease(s)[/yellow]"
                                    )
                            except sqlite3.OperationalError:
                                pass  # database busy — try again next interval
                            last_stale_recovery = now_pc

                        remaining = self.queue.pending_count(image_ids=chunk_ids)
                        running = self.queue.running_count(image_ids=chunk_ids)
                        if remaining == 0 and running == 0:
                            break

                        if remaining == 0 and running > 0:
                            # All remaining work is in-flight on distributed
                            # workers.  Wait for completions / lease expiry.
                            console.print(
                                f"[dim]  Waiting for {running} running job(s) "
                                "in chunk (will reclaim stale)...[/dim]"
                            )
                            self._shutdown.wait(min(5.0, stale_recovery_interval_s))
                            continue

                        chunk_retry += 1
                        console.print(
                            f"[dim]  Chunk pass {chunk_retry + 1}: "
                            f"{remaining} pending, {running} running[/dim]"
                        )

                        # Chunk-level claim for retries (no mini-batch split)
                        def _retry_claim_fn(
                            batch_sz: int, module: str,
                            _ids: set[int] = chunk_ids,
                        ) -> list[dict[str, Any]]:
                            _, _, tl_queue, _ = self._get_thread_db()
                            jobs = tl_queue.claim(
                                batch_size=batch_sz, module=module, image_ids=_ids,
                            )
                            if jobs or self._shutdown.is_set():
                                return jobs
                            leased = tl_queue.leased_running_count(
                                module=module, image_ids=_ids,
                            )
                            if leased <= 0:
                                return []
                            waited = 0.0
                            while waited < _DISTRIBUTED_WAIT_S and not self._shutdown.is_set():
                                self._shutdown.wait(_DISTRIBUTED_POLL_S)
                                waited += _DISTRIBUTED_POLL_S
                                try:
                                    tl_queue.release_expired_leases()
                                except Exception:
                                    pass
                                jobs = tl_queue.claim(
                                    batch_size=batch_sz, module=module, image_ids=_ids,
                                )
                                if jobs:
                                    return jobs
                            return []

                        # ── IO drain first (metadata / technical) ────────────
                        # IO modules are not part of any GPU phase, so they
                        # must be drained explicitly here.
                        if not self._shutdown.is_set():
                            io_pending = any(
                                self.queue.pending_count(
                                    module=mod, image_ids=chunk_ids,
                                ) > 0
                                for mod in LOCAL_IO_MODULES
                            )
                            if io_pending:
                                with (
                                    ThreadPoolExecutor(max_workers=self.workers) as local_pool,
                                    ThreadPoolExecutor(max_workers=1)            as cloud_pool,
                                ):
                                    scheduler.run_io_drain(
                                        submit_io_fn=active_submit_io,
                                        collect_fn=_collect_futures,
                                        flush_fn=self._maybe_periodic_flush,
                                        local_pool=local_pool,
                                        cloud_pool=cloud_pool,
                                        cancel_futures_fn=_cancel_futures,
                                    )

                        # ── GPU phases ────────────────────────────────────────
                        processed_any = False
                        for phase_idx in range(len(scheduler.gpu_phases)):
                            if self._shutdown.is_set():
                                break
                            phase_modules = scheduler.modules_for_phase(phase_idx)
                            if self._perception_on_cooldown(phase_modules):
                                continue
                            has_pending = any(
                                self.queue.pending_count(
                                    module=mod, image_ids=chunk_ids,
                                ) > 0
                                for mod in phase_modules
                            )
                            if not has_pending:
                                continue
                            processed_any = True
                            with (
                                self.profiler.span(
                                    "gpu_phase_retry",
                                    phase=phase_idx, retry=chunk_retry,
                                ),
                                ThreadPoolExecutor(max_workers=self.workers) as local_pool,
                                ThreadPoolExecutor(max_workers=1)            as cloud_pool,
                            ):
                                try:
                                    scheduler.run_gpu_phase(
                                        phase_idx,
                                        claim_fn=_retry_claim_fn,
                                        process_batch_fn=self._process_job_batch,
                                        process_single_fn=self._process_job,
                                        submit_io_fn=active_submit_io,
                                        collect_fn=_collect_futures,
                                        advance_fn=_advance_fn,
                                        flush_fn=self._maybe_periodic_flush,
                                        local_pool=local_pool,
                                        cloud_pool=cloud_pool,
                                        stats=stats,
                                        unload_fn=unload_gpu_model,
                                        prefetch_fn=_prefetch_image,
                                        cancel_futures_fn=_cancel_futures,
                                    )
                                except RuntimeError as exc:
                                    if self._is_perception_vram_timeout(phase_modules, exc):
                                        self._defer_perception_phase(cooldown_s=30.0)
                                        continue
                                    raise

                        if not processed_any:
                            # No GPU phase had work — remaining jobs are
                            # IO-only or running on workers.  The IO drain
                            # above already handled IO jobs; loop back to
                            # check running count and wait if needed.
                            continue

                        # IO drain for retried jobs
                        if not self._shutdown.is_set():
                            with (
                                ThreadPoolExecutor(max_workers=self.workers) as local_pool,
                                ThreadPoolExecutor(max_workers=1)            as cloud_pool,
                            ):
                                scheduler.run_io_drain(
                                    submit_io_fn=active_submit_io,
                                    collect_fn=_collect_futures,
                                    flush_fn=self._maybe_periodic_flush,
                                    local_pool=local_pool,
                                    cloud_pool=cloud_pool,
                                    cancel_futures_fn=_cancel_futures,
                                )

                    if not self._shutdown.is_set():
                        chunk_elapsed_ms = time.perf_counter() * 1000 - chunk_start_ms
                        chunk_passes = sum(
                            stats[k] - chunk_stats_before[k]
                            for k in ("done", "failed", "skipped")
                        )
                        if _chunk_notify is not None:
                            try:
                                _chunk_notify({
                                    "chunkIndex": global_chunk_idx,
                                    "totalChunks": total_chunks,
                                    "chunkSize": len(chunk_ids) if chunk_ids else 0,
                                    "durationMs": round(chunk_elapsed_ms),
                                    "passesCompleted": chunk_passes,
                                })
                            except Exception:
                                pass

                    if total_chunks > 1 and not self._shutdown.is_set():
                        console.print(
                            f"[green]Chunk {global_chunk_idx + 1}/{total_chunks} complete ✓[/green]"
                        )

                chunk_offset += len(chunks)

                # ════════════════════════════════════════════════════════════════
                # Independent GPU modules — currently empty (perception moved into
                # the phase pipeline).  Kept as a fallback for any future modules
                # that need to run outside the sequential pipeline.
                # ════════════════════════════════════════════════════════════════
                if not self._shutdown.is_set():
                    indie_gpu = [
                        mod for mod in scheduler.independent_gpu_modules()
                        if self.queue.pending_count(module=mod) > 0
                    ]
                    if indie_gpu:
                        total_indie = sum(
                            self.queue.pending_count(module=mod) for mod in indie_gpu
                        )
                        console.print(
                            f"[cyan]Independent GPU pass: "
                            f"{', '.join(indie_gpu)} ({total_indie} jobs)[/cyan]"
                        )

                        try:
                            unload_gpu_model("caption")
                        except Exception:
                            pass

                        def _drain_indie_gpu(module: str) -> None:
                            while not self._shutdown.is_set():
                                _, _, tl_queue, _ = self._get_thread_db()
                                jobs = tl_queue.claim(batch_size=1, module=module)
                                if not jobs:
                                    break
                                for job in jobs:
                                    self._process_job(job)

                        with (
                            self.profiler.span("indie_gpu"),
                            ThreadPoolExecutor(max_workers=self.workers)  as local_pool,
                            ThreadPoolExecutor(max_workers=1)             as cloud_pool,
                        ):
                            gpu_threads: list[threading.Thread] = []
                            for mod in indie_gpu:
                                t = threading.Thread(
                                    target=_drain_indie_gpu,
                                    args=(mod,),
                                    daemon=True,
                                    name=f"gpu-indie-{mod}",
                                )
                                gpu_threads.append(t)
                                t.start()

                            for t in gpu_threads:
                                while t.is_alive():
                                    new = _submit_io_jobs(local_pool, cloud_pool)
                                    if new:
                                        _collect_futures(new)
                                    t.join(timeout=1.0)

        # Clear chunk affinity — no longer directing workers to a specific chunk.
        self.current_chunk_ids = None

        if self._shutdown.is_set():
            console.print("\n[yellow]Paused.[/yellow] Run `imganalyzer run` to resume.")
            stats["paused"] = True
        else:
            console.print("\n[green]Complete.[/green]")

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
        "embedding": 16,  # CLIP ViT-L/14 fp16, ~0.95 GB model
        "faces":     8,   # InsightFace ONNX — claim granularity for prefetch
    }

    def _process_job_batch(
        self,
        jobs: list[dict[str, Any]],
        module: str,
    ) -> dict[str, int]:
        """Process a batch of GPU jobs using a single batched forward pass.

        Returns a stats dict: ``{done: N, failed: N, skipped: N}``.

        For modules with batch support (objects, embedding) this
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
        BATCH_MODULES = {"objects", "embedding"}
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
                prereq_status = queue.get_image_module_job_status(image_id, prereq)
                if prereq_status in ("failed", "skipped"):
                    queue.mark_skipped(job_id, f"prerequisite_{prereq}_{prereq_status}")
                else:
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
            elif module == "embedding":
                runner.run_embedding_batch(valid_jobs)

            elapsed = int(time.time() * 1000) - start_ms
            per_image_ms = elapsed // max(len(valid_jobs), 1)

            # Mark all as done
            for job in valid_jobs:
                image_id = job["image_id"]
                image_row = repo.get_image(image_id)
                path_str = image_row["file_path"] if image_row else f"id={image_id}"
                queue.mark_done(job["id"], processing_ms=per_image_ms)
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

        attempts = _LOCK_RETRY_ATTEMPTS
        delay_s = _LOCK_RETRY_INITIAL_DELAY_S
        for attempt in range(1, attempts + 1):
            try:
            # ── Cache check ──────────────────────────────────────────────────
                if not runner.should_run(image_id, module):
                    queue.mark_skipped(job_id, "already_analyzed")
                    _emit_result(path, module, "skipped", 0, "already_analyzed")
                    return "skipped"

            # ── Prerequisite check (DB-driven) ───────────────────────────────
            # Defer back to pending so the job retries after prereq completes.
            # Use queue.defer() (not mark_pending) to bump queued_at — this
            # prevents the same ineligible job from starving eligible ones
            # that are further back in the queue.
                prereq = _PREREQUISITES.get(module)
                if prereq and not repo.is_analyzed(image_id, prereq):
                    prereq_status = queue.get_image_module_job_status(image_id, prereq)
                    if prereq_status in ("failed", "skipped"):
                        queue.mark_skipped(job_id, f"prerequisite_{prereq}_{prereq_status}")
                    else:
                        queue.defer(job_id)
                    return "skipped"

            # ── Prime image cache from prefetch (IO/GPU overlap) ────────────
                prefetched = self._prefetch_cache.pop(image_id, None)
                if prefetched is not None:
                    runner.prime_image_cache(Path(path), prefetched)

            # ── Run the module ───────────────────────────────────────────────
                result = runner.run(image_id, module)
                elapsed = int(time.time() * 1000) - start_ms
                queue.mark_done(job_id, processing_ms=elapsed)
                kw = result.get("keywords") if module == "caption" and result else None
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

            except ImportError as exc:
                elapsed = int(time.time() * 1000) - start_ms
                queue.mark_skipped(job_id, "missing_dependency")
                dependent_modules = _DEPENDENTS.get(module, [])
                if dependent_modules:
                    queue.mark_image_pending_modules_skipped(
                        image_id,
                        dependent_modules,
                        f"prerequisite_{module}_missing_dependency",
                    )
                _emit_result(path, module, "skipped", elapsed, f"missing dependency: {exc}")
                if self.verbose:
                    console.print(
                        f"  [yellow]Skipped:[/yellow] {path} module={module}: "
                        f"missing dependency ({exc})"
                    )
                return "skipped"

            except ValueError as exc:
                err_lower = str(exc).lower()
                if (
                    "libraw cannot decode" in err_lower
                    or "libraw postprocess failed" in err_lower
                    or "pillow cannot decode" in err_lower
                ):
                    elapsed = int(time.time() * 1000) - start_ms
                    queue.mark_skipped(job_id, "corrupt_file")
                    queue.mark_image_pending_jobs_skipped(image_id, "corrupt_file")
                    _emit_result(path, module, "skipped", elapsed, f"corrupt file: {exc}")
                    # Persist corrupt file path for later handling
                    conn, _, _, _ = self._get_thread_db()
                    conn.execute(
                        "INSERT OR IGNORE INTO corrupt_files (image_id, file_path, error_msg)"
                        " VALUES (?, ?, ?)",
                        [image_id, path, str(exc)],
                    )
                    conn.commit()
                    if self.verbose:
                        console.print(
                            f"  [yellow]Skipped:[/yellow] {path} module={module}: corrupt file"
                        )
                    return "skipped"
                raise

            except Exception as exc:
                if _is_transient_db_lock_error(exc) and attempt < attempts:
                    if self.verbose:
                        console.print(
                            "[yellow]Transient DB lock while processing "
                            f"{path} module={module}; retrying ({attempt}/{attempts - 1}) "
                            f"in {delay_s:.2f}s[/yellow]"
                        )
                    self._shutdown.wait(delay_s)
                    delay_s = min(delay_s * 2, 1.0)
                    continue
                elapsed = int(time.time() * 1000) - start_ms
                error_msg = f"{type(exc).__name__}: {exc}"
                try:
                    queue.mark_failed(job_id, error_msg)
                except Exception:
                    pass
                _emit_result(path, module, "failed", elapsed, error_msg)
                if self.verbose:
                    console.print(f"  [red]Failed:[/red] {path} module={module}: {error_msg}")
                return "failed"

        elapsed = int(time.time() * 1000) - start_ms
        error_msg = "OperationalError: database is locked"
        try:
            queue.mark_failed(job_id, error_msg)
        except Exception:
            pass
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

        # Flush FTS5 — per-image micro-transactions to minimize lock hold time.
        # Each image update holds the write lock for ~5-50ms instead of
        # 250ms-2.5s for a batch of 50.
        if fts_snapshot:
            failed_ids: list[int] = []
            for image_id in fts_snapshot:
                try:
                    from imganalyzer.db.connection import begin_immediate
                    begin_immediate(self.conn)
                    try:
                        self.repo.update_search_artifacts(image_id)
                        self.conn.commit()
                    except Exception:
                        self.conn.rollback()
                        raise
                except Exception as exc:
                    failed_ids.append(image_id)
                    if self.verbose:
                        console.print(
                            f"  [yellow]FTS update failed for image {image_id}: {exc}[/yellow]"
                        )
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
        candidates = set(self._xmp_candidates)
        self._xmp_candidates.clear()
        failed_xmp: set[int] = set()
        for image_id in candidates:
            try:
                xmp_path = write_xmp_from_db(self.repo, image_id)
                if xmp_path and self.verbose:
                    console.print(f"  [dim]XMP written: {xmp_path}[/dim]")
                if xmp_path:
                    count += 1
            except Exception as exc:
                failed_xmp.add(image_id)
                if self.verbose:
                    console.print(
                        f"  [red]XMP write failed for image {image_id}: {exc}[/red]"
                    )
        if failed_xmp:
            self._xmp_candidates.update(failed_xmp)
        return count

    def _flush_fts_dirty(self) -> int:
        """Rebuild FTS5 search index for all images marked dirty.

        Uses per-image micro-transactions (~5-50ms lock each) consistent
        with _maybe_periodic_flush.  Runs on the main-thread connection
        (``self.conn``) after all worker threads have joined.
        """
        dirty = list(self._fts_dirty)
        self._fts_dirty.clear()
        if not dirty:
            return 0

        rebuilt = 0
        failed_ids: list[int] = []
        for image_id in dirty:
            try:
                from imganalyzer.db.connection import begin_immediate
                begin_immediate(self.conn)
                try:
                    self.repo.update_search_artifacts(image_id)
                    self.conn.commit()
                    rebuilt += 1
                except Exception:
                    self.conn.rollback()
                    raise
            except Exception as exc:
                failed_ids.append(image_id)
                if self.verbose:
                    console.print(
                        f"  [yellow]FTS update failed for image {image_id}: {exc}[/yellow]"
                    )
        if failed_ids:
            self._fts_dirty.update(failed_ids)
        return rebuilt

    def _handle_sigint(self, signum: int, frame: Any) -> None:
        console.print("\n[yellow]Ctrl+C received — finishing current batch...[/yellow]")
        self._shutdown.set()
