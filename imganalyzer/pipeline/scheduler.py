"""Resource-aware scheduler for the batch processing pipeline.

Replaces the rigid two-phase GPU pipeline with a VRAM-budget-aware
scheduler that can run multiple small GPU models concurrently while
keeping large models (BLIP-2) exclusive.

Key improvements over the old phase-based approach:
  - Small GPU models (faces 0.5 GB + ocr 1.3 GB + embedding 0.95 GB)
    can run concurrently when they fit within the VRAM budget.
  - Cloud thread pool is boosted when GPU is idle.
  - Fewer model load/unload cycles (3 vs 5).
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
    "objects", "blip2", "ocr", "faces", "embedding",
})
LOCAL_IO_MODULES: frozenset[str] = frozenset({"metadata", "technical"})
CLOUD_MODULES: frozenset[str] = frozenset({"cloud_ai", "aesthetic"})
IO_MODULES: frozenset[str] = LOCAL_IO_MODULES | CLOUD_MODULES

# Dependency graph: module -> prerequisite that must complete first.
_PREREQUISITES: dict[str, str] = {
    "cloud_ai":  "objects",
    "aesthetic": "objects",
    "ocr":       "objects",
    "faces":     "objects",
}

# GPU modules that support batched forward passes.
_BATCH_CAPABLE: frozenset[str] = frozenset({"objects", "blip2", "embedding"})

# Ordered phases for GPU execution.  Within a phase, all modules can
# be loaded simultaneously (VRAM permitting).  Between phases, all
# models from the previous phase are unloaded.
#
# Phase 0: objects   (must run first — unlocks cloud/aesthetic/ocr/faces)
# Phase 1: blip2     (exclusive — too large to share)
# Phase 2: faces, ocr, embedding (co-resident — total ~2.75 GB)
_GPU_PHASES: list[list[str]] = [
    ["objects"],
    ["blip2"],
    ["faces", "ocr", "embedding"],
]


class ResourceScheduler:
    """Coordinate GPU, CPU, and cloud work with VRAM-aware scheduling.

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
    cloud_workers:
        Thread count for cloud API calls.
    cloud_boost_factor:
        Multiplier for cloud threads when GPU is idle (default 2).
    shutdown_event:
        External shutdown signal (e.g. Worker._shutdown).
    """

    def __init__(
        self,
        vram_budget: VRAMBudget,
        gpu_batch_sizes: dict[str, int],
        default_batch_size: int = 10,
        cpu_workers: int = 4,
        cloud_workers: int = 4,
        cloud_boost_factor: int = 2,
        shutdown_event: Optional[threading.Event] = None,
    ) -> None:
        self.vram = vram_budget
        self.gpu_batch_sizes = gpu_batch_sizes
        self.default_batch_size = default_batch_size
        self.cpu_workers = cpu_workers
        self.cloud_workers = cloud_workers
        self.cloud_boost_factor = cloud_boost_factor
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

    # ── Cloud pool sizing ─────────────────────────────────────────────────

    def boosted_cloud_workers(self) -> int:
        """Return cloud worker count for when GPU is idle."""
        return self.cloud_workers * self.cloud_boost_factor

    # ── Module classification ─────────────────────────────────────────────

    @staticmethod
    def is_gpu(module: str) -> bool:
        return module in GPU_MODULES

    @staticmethod
    def is_io(module: str) -> bool:
        return module in IO_MODULES

    @staticmethod
    def is_cloud(module: str) -> bool:
        return module in CLOUD_MODULES

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
        """Return currently free CUDA memory in GB, or None when unavailable."""
        try:
            import torch
            if not torch.cuda.is_available():
                return None
            free_bytes, _total_bytes = torch.cuda.mem_get_info()
            return free_bytes / (1024 ** 3)
        except Exception:
            return None

    @staticmethod
    def _force_cuda_cleanup() -> None:
        """Best-effort allocator cleanup before the next GPU phase."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # ipc_collect helps release cached blocks held for IPC sharing.
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception:
            pass

    def _wait_for_vram_ready(self, module: str) -> None:
        """Wait until enough CUDA free memory is visible for *module*.

        This avoids racing into the next phase before memory from the previous
        model unload is returned to the allocator/driver.
        """
        needed_gb = self.vram.vram_for(module)
        if needed_gb <= 0:
            return
        # If the model cannot fit physically on this GPU, don't spin forever.
        if needed_gb > self.vram.total_gb:
            return
        # No CUDA visibility (CPU mode / unsupported backend): nothing to wait for.
        if self._cuda_free_gb() is None:
            return

        timeout_s = 120.0
        deadline = time.monotonic() + timeout_s
        while not self.is_shutdown:
            free_gb = self._cuda_free_gb()
            if free_gb is None or free_gb >= needed_gb:
                return
            self._force_cuda_cleanup()
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Timed out waiting for VRAM for {module}: "
                    f"need {needed_gb:.2f} GB, free {free_gb:.2f} GB"
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
    ) -> None:
        """Execute one GPU phase, potentially with co-resident models.

        For single-module phases (objects, blip2), this drains all jobs
        for that module sequentially on the calling thread.

        For multi-module phases (faces + ocr + embedding), each module
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
                        item = prefetch_q.get()
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

                # While GPU threads work, keep submitting and reaping IO jobs
                alive = True
                while alive and not self.is_shutdown:
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
            # Collect remaining IO futures after unload so cloud/IO drain does
            # not keep GPU-heavy models resident longer than needed.
            if pending_futures:
                collect_fn(pending_futures)

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
    ) -> None:
        """Drain remaining IO/cloud jobs after GPU work finishes."""
        while not self.is_shutdown:
            io_futures = submit_io_fn(local_pool, cloud_pool)
            if not io_futures:
                break
            collect_fn(io_futures)
            flush_fn()
