"""Resource-aware scheduler for the batch processing pipeline.

VRAM-budget-aware scheduler that can run multiple small GPU models
concurrently while keeping large models (UniPercept) exclusive.

Key improvements over the old phase-based approach:
  - Small GPU models (faces 1.0 GB + embedding 0.95 GB)
    can run concurrently when they fit within the VRAM budget.
  - Fewer model load/unload cycles.
"""

from __future__ import annotations

import gc
import queue as _queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Optional

from imganalyzer.pipeline.vram_budget import VRAMBudget

# ── Module classifications ────────────────────────────────────────────────────

GPU_MODULES: frozenset[str] = frozenset({
    "caption", "objects", "faces", "embedding", "perception",
})
LOCAL_IO_MODULES: frozenset[str] = frozenset({"metadata", "technical"})
IO_MODULES: frozenset[str] = LOCAL_IO_MODULES

# Dependency graph: module -> prerequisite that must complete first.
_PREREQUISITES: dict[str, str] = {
    "faces":     "objects",
}

# GPU modules that support batched forward passes.
_BATCH_CAPABLE: frozenset[str] = frozenset({"objects", "embedding"})

# Ordered phases for GPU execution.  Within a phase, all modules can
# be loaded simultaneously (VRAM permitting).  Between phases, all
# models from the previous phase are unloaded.
#
# Phase 0: caption      (qwen3.5 via Ollama, ~8.7 GB — must unload before objects)
# Phase 1: objects       (GroundingDINO, ~2.4 GB — unlocks faces dep)
# Phase 2: faces, embedding (co-resident — total ~1.95 GB)
# Phase 3: perception    (UniPercept, ~13.8 GB effective, CUDA-only)
#
# Perception is interleaved per mini-batch so the CUDA machine doesn't
# spend hours on perception at the end while macOS workers sit idle.
# During perception, macOS workers continue processing caption jobs.
_GPU_PHASES: list[list[str]] = [
    ["caption"],
    ["objects"],
    ["faces", "embedding"],
    ["perception"],
]

# GPU modules that run independently alongside the IO drain rather than
# in the sequential phase pipeline.  Currently empty — perception was
# moved into the phase pipeline for better scheduling with distributed
# workers (macOS workers can't do perception, so deferring it to the end
# wastes CUDA time while macOS workers idle).
INDEPENDENT_GPU_MODULES: frozenset[str] = frozenset()


class ResourceScheduler:
    """Coordinate GPU and CPU work with VRAM-aware scheduling.

    Parameters
    ----------
    vram_budget:
        VRAMBudget instance for tracking GPU memory.
    gpu_batch_sizes:
        Per-module batch sizes for GPU modules (e.g. {"objects": 4}).
    default_batch_size:
        Fallback batch size for modules not in *gpu_batch_sizes*.
    cpu_workers:
        Thread count for local I/O (metadata, technical).
    shutdown_event:
        External shutdown signal (e.g. Worker._shutdown).
    """

    def __init__(
        self,
        vram_budget: VRAMBudget,
        gpu_batch_sizes: dict[str, int],
        default_batch_size: int = 10,
        cpu_workers: int = 4,
        shutdown_event: Optional[threading.Event] = None,
    ) -> None:
        self.vram = vram_budget
        self.gpu_batch_sizes = gpu_batch_sizes
        self.default_batch_size = default_batch_size
        self.cpu_workers = cpu_workers
        self._shutdown = shutdown_event or threading.Event()

    # ── Phase planning ────────────────────────────────────────────────────

    @property
    def gpu_phases(self) -> list[list[str]]:
        """Return the GPU execution phases."""
        return _GPU_PHASES

    def modules_for_phase(self, phase_idx: int) -> list[str]:
        """Return GPU modules scheduled for *phase_idx*."""
        if 0 <= phase_idx < len(_GPU_PHASES):
            return list(_GPU_PHASES[phase_idx])
        return []

    def is_co_resident_phase(self, phase_idx: int) -> bool:
        """Return True if *phase_idx* has multiple co-resident GPU modules."""
        return 0 <= phase_idx < len(_GPU_PHASES) and len(_GPU_PHASES[phase_idx]) > 1

    # ── Batch size helpers ────────────────────────────────────────────────

    def batch_size_for(self, module: str) -> int:
        """Return the appropriate batch size for *module*."""
        return self.gpu_batch_sizes.get(module, self.default_batch_size)

    def is_batch_capable(self, module: str) -> bool:
        """Return True if *module* supports batched GPU forward passes."""
        return module in _BATCH_CAPABLE



    # ── Module classification ─────────────────────────────────────────────

    @staticmethod
    def is_gpu(module: str) -> bool:
        return module in GPU_MODULES

    @staticmethod
    def is_io(module: str) -> bool:
        return module in IO_MODULES

    @staticmethod
    def is_local_io(module: str) -> bool:
        return module in LOCAL_IO_MODULES

    @staticmethod
    def prerequisite_for(module: str) -> Optional[str]:
        """Return the prerequisite module name, or None."""
        return _PREREQUISITES.get(module)

    @property
    def is_shutdown(self) -> bool:
        return self._shutdown.is_set()

    # ── CUDA memory readiness ───────────────────────────────────────────────

    @staticmethod
    def _cuda_free_gb() -> float | None:
        """Return currently free GPU memory in GB, or None when unavailable."""
        try:
            import torch
            if torch.cuda.is_available():
                free_bytes, _total_bytes = torch.cuda.mem_get_info()
                return free_bytes / (1024 ** 3)
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # MPS has no free-memory query; return None to skip
                # memory-based scheduling decisions.
                return None
        except Exception:
            pass
        return None

    @staticmethod
    def _force_cuda_cleanup() -> None:
        """Best-effort allocator cleanup before the next GPU phase."""
        gc.collect()
        try:
            from imganalyzer.device import empty_cache
            empty_cache()
            import torch
            if torch.cuda.is_available() and hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def _ready_free_vram_gb(self, module: str, needed_gb: float) -> float:
        """Return required free VRAM threshold before starting *module*.

        Exclusive modules may require almost all VRAM on paper. On desktop GPUs,
        a portion is often permanently reserved by the display driver/compositor,
        so waiting for full model size can stall forever even when inference works.
        """
        if needed_gb <= 0:
            return 0.0
        if not self.vram.is_exclusive(module):
            return needed_gb
        desktop_reserve_gb = max(2.0, self.vram.total_gb * 0.15)
        return max(0.0, min(needed_gb, self.vram.total_gb - desktop_reserve_gb))

    def _wait_for_vram_ready(self, module: str) -> None:
        """Wait until enough CUDA free memory is visible for *module*.

        This avoids racing into the next phase before memory from the previous
        model unload is returned to the allocator/driver.
        """
        needed_gb = self.vram.vram_for(module)
        ready_needed_gb = self._ready_free_vram_gb(module, needed_gb)
        if needed_gb <= 0:
            return
        # If the model cannot fit physically on this GPU, don't spin forever.
        if needed_gb > self.vram.total_gb:
            return
        if ready_needed_gb <= 0:
            return
        # No CUDA visibility (CPU mode / unsupported backend): nothing to wait for.
        if self._cuda_free_gb() is None:
            return

        timeout_s = 120.0
        deadline = time.monotonic() + timeout_s
        while not self.is_shutdown:
            free_gb = self._cuda_free_gb()
            if free_gb is None or free_gb >= ready_needed_gb:
                return
            self._force_cuda_cleanup()
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Timed out waiting for VRAM for {module}: "
                    f"need {needed_gb:.2f} GB "
                    f"(ready threshold {ready_needed_gb:.2f} GB), "
                    f"free {free_gb:.2f} GB"
                )
            self._shutdown.wait(timeout=0.25)

    # ── Execution helpers ─────────────────────────────────────────────────

    def run_gpu_phase(
        self,
        phase_idx: int,
        *,
        claim_fn: Callable[[int, str], list[dict[str, Any]]],
        process_batch_fn: Callable[[list[dict[str, Any]], str], dict[str, int]],
        process_single_fn: Callable[[dict[str, Any]], str],
        submit_io_fn: Callable[
            [ThreadPoolExecutor, ThreadPoolExecutor],
            dict[Future, dict[str, Any]],
        ],
        collect_fn: Callable[[dict[Future, dict[str, Any]]], None],
        advance_fn: Callable[[int], None],
        flush_fn: Callable[[], None],
        local_pool: ThreadPoolExecutor,
        cloud_pool: ThreadPoolExecutor,
        stats: dict[str, int],
        unload_fn: Callable[[str], None],
        prefetch_fn: Optional[Callable[[dict[str, Any]], Optional[dict[str, Any]]]] = None,
        cancel_futures_fn: Optional[Callable[[dict[Future, dict[str, Any]]], None]] = None,
    ) -> None:
        """Execute one GPU phase, potentially with co-resident models.

        For single-module phases (objects), this drains all jobs
        for that module sequentially on the calling thread.

        For multi-module phases (faces + embedding), each module
        gets its own thread so they can process concurrently with separate
        CUDA streams.
        """
        modules = self.modules_for_phase(phase_idx)
        if not modules:
            return

        reserved_modules: list[str] = []
        pending_futures: dict[Future, dict[str, Any]] = {}
        try:
            # Reserve VRAM for all modules in this phase
            for mod in modules:
                if self.vram.vram_for(mod) > 0:
                    self._wait_for_vram_ready(mod)
                    if self.is_shutdown:
                        break
                    self.vram.reserve(mod)
                    reserved_modules.append(mod)

            if self.is_shutdown:
                return

            if len(modules) == 1:
                # Single-module phase: run on current thread
                mod = modules[0]
                batch_sz = self.batch_size_for(mod)

                while not self.is_shutdown:
                    jobs = claim_fn(batch_sz, mod)
                    if not jobs:
                        break

                    batch_result = process_batch_fn(jobs, mod)
                    for k in ("done", "failed", "skipped"):
                        stats[k] += batch_result[k]
                    advance_fn(len(jobs))

                    # Sweep for IO/cloud work
                    new_io = submit_io_fn(local_pool, cloud_pool)
                    pending_futures.update(new_io)

                    # Reap completed IO futures
                    done_futs = {f: pending_futures.pop(f) for f in list(pending_futures) if f.done()}
                    if done_futs:
                        collect_fn(done_futs)

                    flush_fn()
            else:
                # Multi-module phase: each GPU module gets its own thread.
                # When a prefetch_fn is provided, IO threads read+decode images
                # ahead of GPU consumption so the GPU never waits for disk IO.
                gpu_threads: list[threading.Thread] = []
                lock = threading.Lock()

                _PREFETCH_DEPTH = 4   # max images buffered ahead of GPU

                def _drain_module(mod: str) -> None:
                    batch_sz = self.batch_size_for(mod)
                    use_prefetch = prefetch_fn is not None and not self.is_batch_capable(mod)

                    if not use_prefetch:
                        # Original path for batch-capable modules
                        while not self.is_shutdown:
                            jobs = claim_fn(batch_sz, mod)
                            if not jobs:
                                break
                            batch_result = process_batch_fn(jobs, mod)
                            with lock:
                                for k in ("done", "failed", "skipped"):
                                    stats[k] += batch_result[k]
                            advance_fn(len(jobs))
                        return

                    # ── Prefetched pipeline for non-batch modules ────────────
                    # Producer threads read images; GPU thread consumes.
                    prefetch_q: _queue.Queue[
                        tuple[dict[str, Any], dict[str, Any] | None] | None
                    ] = _queue.Queue(maxsize=_PREFETCH_DEPTH)
                    producer_done = threading.Event()

                    def _producer() -> None:
                        """Claim jobs and prefetch images into the queue."""
                        try:
                            while not self.is_shutdown:
                                jobs = claim_fn(batch_sz, mod)
                                if not jobs:
                                    break
                                for job in jobs:
                                    if self.is_shutdown:
                                        return
                                    if prefetch_fn is None:
                                        img_data = None
                                    else:
                                        try:
                                            img_data = prefetch_fn(job)
                                        except Exception:
                                            img_data = None
                                    prefetch_q.put((job, img_data))
                        finally:
                            producer_done.set()
                            # Sentinel: tell consumer no more items
                            prefetch_q.put(None)

                    # Start producer thread
                    prod_t = threading.Thread(
                        target=_producer, daemon=True,
                        name=f"prefetch-{mod}",
                    )
                    prod_t.start()

                    # Consumer: GPU inference on prefetched images
                    while True:
                        try:
                            item = prefetch_q.get(timeout=120)
                        except _queue.Empty:
                            # Producer may have died without sentinel
                            if producer_done.is_set() or self.is_shutdown:
                                break
                            continue
                        if item is None:
                            break
                        job, img_data = item
                        # process_single_fn will use the cached image if
                        # prefetch_fn populated the thread-local cache
                        status = process_single_fn(job)
                        with lock:
                            stats[status] = stats.get(status, 0) + 1
                        advance_fn(1)

                    prod_t.join(timeout=5)

                for mod in modules:
                    t = threading.Thread(
                        target=_drain_module,
                        args=(mod,),
                        daemon=True,
                        name=f"gpu-{mod}",
                    )
                    gpu_threads.append(t)
                    t.start()

                # While GPU threads work, keep submitting and reaping IO jobs.
                # Phase timeout prevents indefinite hang if a GPU thread stalls.
                phase_deadline = time.monotonic() + 1800  # 30 min max per phase
                alive = True
                while alive and not self.is_shutdown:
                    if time.monotonic() > phase_deadline:
                        import logging
                        logging.getLogger(__name__).warning(
                            "GPU phase timeout (30 min) — forcing phase exit"
                        )
                        break

                    new_io = submit_io_fn(local_pool, cloud_pool)
                    pending_futures.update(new_io)

                    done_futs = {f: pending_futures.pop(f) for f in list(pending_futures) if f.done()}
                    if done_futs:
                        collect_fn(done_futs)

                    flush_fn()

                    alive = any(t.is_alive() for t in gpu_threads)
                    if alive:
                        # Brief sleep to avoid busy-waiting
                        self._shutdown.wait(timeout=0.1)

                # Join all GPU threads
                for t in gpu_threads:
                    t.join(timeout=5)

        finally:
            # Unload all models reserved in this phase.
            for mod in reserved_modules:
                try:
                    unload_fn(mod)
                finally:
                    self.vram.release(mod)
            # Force allocator cleanup now so large model memory is released
            # before we wait on outstanding IO/cloud futures.
            self._force_cuda_cleanup()
            # Cancel uncompleted futures on shutdown so the pool can exit.
            if self.is_shutdown and pending_futures and cancel_futures_fn is not None:
                cancel_futures_fn(pending_futures)
            # Cancel uncompleted cloud futures at phase boundary so the
            # ThreadPoolExecutor can shut down promptly and the next GPU
            # phase isn't blocked waiting for a slow cloud backlog.
            elif pending_futures and cancel_futures_fn is not None:
                cancel_futures_fn(pending_futures)
            if pending_futures:
                collect_fn(pending_futures)

    @staticmethod
    def independent_gpu_modules() -> frozenset[str]:
        """GPU modules that run outside the sequential phase pipeline."""
        return INDEPENDENT_GPU_MODULES

    def run_io_drain(
        self,
        *,
        submit_io_fn: Callable[
            [ThreadPoolExecutor, ThreadPoolExecutor],
            dict[Future, dict[str, Any]],
        ],
        collect_fn: Callable[[dict[Future, dict[str, Any]]], None],
        flush_fn: Callable[[], None],
        local_pool: ThreadPoolExecutor,
        cloud_pool: ThreadPoolExecutor,
        cancel_futures_fn: Optional[Callable[[dict[Future, dict[str, Any]]], None]] = None,
    ) -> None:
        """Drain remaining IO/cloud jobs after GPU work finishes."""
        pending_futures: dict[Future, dict[str, Any]] = {}

        while True:
            if self.is_shutdown and pending_futures and cancel_futures_fn is not None:
                cancel_futures_fn(pending_futures)

            done_futs = {f: pending_futures.pop(f) for f in list(pending_futures) if f.done()}
            if done_futs:
                collect_fn(done_futs)
                flush_fn()
                continue

            if not pending_futures:
                if self.is_shutdown:
                    break
                new_futures = submit_io_fn(local_pool, cloud_pool)
                if not new_futures:
                    break
                pending_futures.update(new_futures)
                continue

            self._shutdown.wait(timeout=0.1)

        if self.is_shutdown and pending_futures and cancel_futures_fn is not None:
            cancel_futures_fn(pending_futures)
        if pending_futures:
            collect_fn(pending_futures)
            flush_fn()
