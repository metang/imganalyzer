"""Batch processing profiler — fine-grained performance instrumentation.

Collects timed spans (IO reads, GPU inference, DB writes, model load/unload,
prefetch events) in a thread-safe in-memory buffer and periodically flushes
to SQLite for post-run analysis.

Usage::

    profiler = ProfileCollector(conn)
    profiler.start_run(total_images=1000)

    with profiler.span("io_read", image_id=42, module="objects",
                        image_file_size=5_000_000, image_format=".jpg",
                        image_width=4000, image_height=3000):
        data = read_image(path)

    profiler.end_run()

When profiling is disabled, use ``NullProfiler`` which has the same API
but does nothing (zero overhead).
"""
from __future__ import annotations

import os
import platform
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator, Optional


@dataclass
class _Event:
    """A single profiled time span."""
    run_id: int
    image_id: Optional[int]
    module: Optional[str]
    phase: Optional[int]
    event_type: str
    start_ts: float          # seconds since run start (perf_counter based)
    duration_ms: float
    thread_name: str
    batch_size: int = 1
    image_file_size: Optional[int] = None
    image_format: Optional[str] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None


@dataclass
class _Snapshot:
    """A periodic system resource sample."""
    run_id: int
    ts: float                # seconds since run start
    gpu_util_pct: Optional[float] = None
    gpu_mem_used_mb: Optional[float] = None
    gpu_mem_total_mb: Optional[float] = None
    cpu_pct: Optional[float] = None
    ram_used_mb: Optional[float] = None
    prefetch_queue_depth: Optional[int] = None


class ProfileCollector:
    """Thread-safe profiler that collects events and flushes to SQLite."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self._lock = threading.Lock()
        self._events: list[_Event] = []
        self._snapshots: list[_Snapshot] = []
        self._run_id: Optional[int] = None
        self._run_start: float = 0.0       # perf_counter at run start
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_stop = threading.Event()
        self._flush_interval = 60          # seconds between auto-flushes
        self._last_flush = 0.0
        # Prefetch queue depth — updated externally by the scheduler
        self._prefetch_depth: int = 0

    @property
    def run_id(self) -> Optional[int]:
        return self._run_id

    @property
    def enabled(self) -> bool:
        return self._run_id is not None

    # ── Run lifecycle ─────────────────────────────────────────────────────

    def start_run(self, total_images: int = 0) -> int:
        """Begin a new profiling run. Returns the run_id."""
        gpu_name, gpu_vram_gb = self._detect_gpu()
        cpu_count = os.cpu_count() or 1
        ram_gb = self._detect_ram()

        cur = self.conn.execute(
            """INSERT INTO profiler_runs
               (started_at, total_images, gpu_name, gpu_vram_gb, cpu_count, ram_gb)
               VALUES (datetime('now'), ?, ?, ?, ?, ?)""",
            [total_images, gpu_name, gpu_vram_gb, cpu_count, ram_gb],
        )
        self.conn.commit()
        self._run_id = cur.lastrowid
        self._run_start = time.perf_counter()
        self._last_flush = time.perf_counter()

        # Start system monitor thread
        self._monitor_stop.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="profiler-monitor",
        )
        self._monitor_thread.start()

        return self._run_id  # type: ignore[return-value]

    def end_run(self) -> None:
        """Finalize the current profiling run."""
        if self._run_id is None:
            return

        # Stop monitor thread
        self._monitor_stop.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
            self._monitor_thread = None

        # Final flush
        self.flush()

        # Update run end time
        self.conn.execute(
            "UPDATE profiler_runs SET ended_at = datetime('now') WHERE id = ?",
            [self._run_id],
        )
        self.conn.commit()
        self._run_id = None

    # ── Event recording ───────────────────────────────────────────────────

    @contextmanager
    def span(
        self,
        event_type: str,
        *,
        image_id: Optional[int] = None,
        module: Optional[str] = None,
        phase: Optional[int] = None,
        batch_size: int = 1,
        image_file_size: Optional[int] = None,
        image_format: Optional[str] = None,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ) -> Generator[None, None, None]:
        """Context manager that records a timed span."""
        if self._run_id is None:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            event = _Event(
                run_id=self._run_id,
                image_id=image_id,
                module=module,
                phase=phase,
                event_type=event_type,
                start_ts=start - self._run_start,
                duration_ms=elapsed_ms,
                thread_name=threading.current_thread().name,
                batch_size=batch_size,
                image_file_size=image_file_size,
                image_format=image_format,
                image_width=image_width,
                image_height=image_height,
            )
            with self._lock:
                self._events.append(event)

    def record_event(
        self,
        event_type: str,
        duration_ms: float,
        *,
        image_id: Optional[int] = None,
        module: Optional[str] = None,
        phase: Optional[int] = None,
        batch_size: int = 1,
        image_file_size: Optional[int] = None,
        image_format: Optional[str] = None,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ) -> None:
        """Record a pre-timed event (when context manager isn't suitable)."""
        if self._run_id is None:
            return
        event = _Event(
            run_id=self._run_id,
            image_id=image_id,
            module=module,
            phase=phase,
            event_type=event_type,
            start_ts=time.perf_counter() - self._run_start,
            duration_ms=duration_ms,
            thread_name=threading.current_thread().name,
            batch_size=batch_size,
            image_file_size=image_file_size,
            image_format=image_format,
            image_width=image_width,
            image_height=image_height,
        )
        with self._lock:
            self._events.append(event)

    def set_prefetch_depth(self, depth: int) -> None:
        """Update current prefetch queue depth for system snapshots."""
        self._prefetch_depth = depth

    # ── Flush to DB ───────────────────────────────────────────────────────

    def flush(self) -> None:
        """Write buffered events and snapshots to SQLite."""
        with self._lock:
            events = self._events
            snapshots = self._snapshots
            self._events = []
            self._snapshots = []

        if events:
            self.conn.executemany(
                """INSERT INTO profiler_events
                   (run_id, image_id, module, phase, event_type, start_ts,
                    duration_ms, thread_name, batch_size,
                    image_file_size, image_format, image_width, image_height)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (e.run_id, e.image_id, e.module, e.phase, e.event_type,
                     e.start_ts, e.duration_ms, e.thread_name, e.batch_size,
                     e.image_file_size, e.image_format, e.image_width, e.image_height)
                    for e in events
                ],
            )

        if snapshots:
            self.conn.executemany(
                """INSERT INTO profiler_snapshots
                   (run_id, ts, gpu_util_pct, gpu_mem_used_mb, gpu_mem_total_mb,
                    cpu_pct, ram_used_mb, prefetch_queue_depth)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (s.run_id, s.ts, s.gpu_util_pct, s.gpu_mem_used_mb,
                     s.gpu_mem_total_mb, s.cpu_pct, s.ram_used_mb,
                     s.prefetch_queue_depth)
                    for s in snapshots
                ],
            )

        if events or snapshots:
            self.conn.commit()

        self._last_flush = time.perf_counter()

    def maybe_flush(self) -> None:
        """Flush if enough time has passed since last flush."""
        if time.perf_counter() - self._last_flush >= self._flush_interval:
            self.flush()

    # ── System monitor ────────────────────────────────────────────────────

    def _monitor_loop(self) -> None:
        """Background thread: sample system resources every 5 seconds."""
        while not self._monitor_stop.wait(timeout=5.0):
            if self._run_id is None:
                break
            snapshot = self._sample_system()
            if snapshot:
                with self._lock:
                    self._snapshots.append(snapshot)

    def _sample_system(self) -> Optional[_Snapshot]:
        """Collect one system resource sample."""
        if self._run_id is None:
            return None

        ts = time.perf_counter() - self._run_start
        gpu_util: Optional[float] = None
        gpu_mem_used: Optional[float] = None
        gpu_mem_total: Optional[float] = None
        cpu_pct: Optional[float] = None
        ram_used: Optional[float] = None

        # GPU metrics via pynvml
        try:
            import pynvml  # type: ignore[import-untyped]
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = float(util.gpu)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_mem_used = mem_info.used / (1024 * 1024)
            gpu_mem_total = mem_info.total / (1024 * 1024)
        except Exception:
            # Try torch as fallback
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_mem_used = torch.cuda.memory_allocated(0) / (1024 * 1024)
                    gpu_mem_total = torch.cuda.get_device_properties(0).total_mem / (1024 * 1024)
            except Exception:
                pass

        # CPU/RAM via psutil
        try:
            import psutil  # type: ignore[import-untyped]
            cpu_pct = psutil.cpu_percent(interval=None)
            ram_used = psutil.virtual_memory().used / (1024 * 1024)
        except Exception:
            pass

        return _Snapshot(
            run_id=self._run_id,
            ts=ts,
            gpu_util_pct=gpu_util,
            gpu_mem_used_mb=gpu_mem_used,
            gpu_mem_total_mb=gpu_mem_total,
            cpu_pct=cpu_pct,
            ram_used_mb=ram_used,
            prefetch_queue_depth=self._prefetch_depth,
        )

    # ── Hardware detection ────────────────────────────────────────────────

    @staticmethod
    def _detect_gpu() -> tuple[str, float]:
        """Return (gpu_name, total_vram_gb)."""
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                return props.name, props.total_mem / (1024 ** 3)
        except Exception:
            pass
        return "unknown", 0.0

    @staticmethod
    def _detect_ram() -> float:
        """Return total system RAM in GB."""
        try:
            import psutil  # type: ignore[import-untyped]
            return psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            pass
        return 0.0


class NullProfiler:
    """No-op profiler with the same API as ProfileCollector.

    Used when profiling is disabled — all methods are zero-overhead no-ops.
    """

    @property
    def run_id(self) -> None:
        return None

    @property
    def enabled(self) -> bool:
        return False

    def start_run(self, total_images: int = 0) -> int:
        return 0

    def end_run(self) -> None:
        pass

    @contextmanager
    def span(self, event_type: str, **kwargs: Any) -> Generator[None, None, None]:
        yield

    def record_event(self, event_type: str, duration_ms: float, **kwargs: Any) -> None:
        pass

    def set_prefetch_depth(self, depth: int) -> None:
        pass

    def flush(self) -> None:
        pass

    def maybe_flush(self) -> None:
        pass


# Type alias for either real or null profiler
Profiler = ProfileCollector | NullProfiler
