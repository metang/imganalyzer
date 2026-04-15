"""Background pre-decode pipeline for populating the decoded image cache.

Reads images from the filesystem (NAS), decodes them (including RAW
demosaic), and stores the result in a :class:`DecodedImageStore`.  Runs
on the coordinator only, using a :class:`~concurrent.futures.ThreadPoolExecutor`
for parallel decoding.

Design principles:

* **Cache-gated dispatch** — jobs are only claimable after their image is
  cached.  Workers never trigger on-demand decode.
* **Parallel throughput** — RAW demosaic via rawpy releases the GIL during
  LibRaw C-level processing, so threads parallelise effectively.  At 8
  threads × ~2.5 s avg decode ≈ 3.2 images/s.
* **Adaptive scheduling** — a :class:`ResourceSampler` monitors CPU and
  disk I/O so the decode pipeline ramps up when idle and backs off under
  load, keeping the system responsive for GPU passes and the server.
* **Priority** — RAW files first (most expensive), then standard formats.
* **Idempotent** — already-cached images are skipped.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image

from imganalyzer.cache.decoded_store import DecodedImageStore

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Adaptive resource monitoring
# ---------------------------------------------------------------------------

# Thresholds for resource-aware scheduling
_CPU_HIGH_PCT = 70.0    # CPU% above which we reduce decode throughput
_CPU_LOW_PCT = 65.0     # CPU% below which we can ramp up aggressively
_DISK_BUSY_PCT = 80.0   # disk busy% above which we throttle feeding


class ResourceSampler:
    """Lightweight CPU and disk I/O sampler for adaptive decode scheduling.

    Uses ``psutil`` when available.  Falls back to a no-op that always
    reports zero utilisation (decode runs at full speed without gating).
    """

    def __init__(self) -> None:
        self._cpu_pct: float = 0.0
        self._disk_read_mbps: float = 0.0
        self._disk_busy_pct: float = 0.0
        self._last_disk_counters: Any = None
        self._last_sample_time: float = 0.0
        self._has_psutil = False
        try:
            import psutil  # type: ignore[import-untyped]

            self._has_psutil = True
            # Prime CPU measurement (first call always returns 0.0)
            psutil.cpu_percent(interval=None)
            self._last_disk_counters = psutil.disk_io_counters()
            self._last_sample_time = time.monotonic()
        except (ImportError, Exception):
            pass

    # -- Sampling ----------------------------------------------------------

    def sample(self) -> dict[str, float]:
        """Collect one resource snapshot.  Call periodically (e.g. every 5 s).

        Returns ``{cpu_pct, disk_read_mbps, disk_busy_pct}``.
        """
        if not self._has_psutil:
            return {"cpu_pct": 0.0, "disk_read_mbps": 0.0, "disk_busy_pct": 0.0}

        import psutil  # type: ignore[import-untyped]

        now = time.monotonic()
        cpu_pct = psutil.cpu_percent(interval=None)

        disk = psutil.disk_io_counters()
        disk_read_mbps = 0.0
        disk_busy_pct = 0.0

        if self._last_disk_counters is not None:
            dt = now - self._last_sample_time
            if dt > 0.5:
                read_delta = disk.read_bytes - self._last_disk_counters.read_bytes
                # read_time is cumulative milliseconds spent reading
                time_delta_ms = disk.read_time - self._last_disk_counters.read_time

                disk_read_mbps = (read_delta / dt) / (1024 * 1024)
                disk_busy_pct = min(100.0, (time_delta_ms / (dt * 1000)) * 100)

                self._last_disk_counters = disk
                self._last_sample_time = now

        self._cpu_pct = cpu_pct
        self._disk_read_mbps = disk_read_mbps
        self._disk_busy_pct = disk_busy_pct

        return {
            "cpu_pct": cpu_pct,
            "disk_read_mbps": disk_read_mbps,
            "disk_busy_pct": disk_busy_pct,
        }

    # -- Decision helpers --------------------------------------------------

    def should_feed(self, in_flight: int, min_buffer: int = 20) -> bool:
        """Whether the scheduler should queue more decode work.

        Always returns *True* when *in_flight* is below *min_buffer* to
        prevent worker starvation.  Otherwise checks CPU and disk load.
        """
        if in_flight < min_buffer:
            return True
        if not self._has_psutil:
            return True
        if self._cpu_pct > _CPU_HIGH_PCT and self._disk_busy_pct > _DISK_BUSY_PCT:
            return False
        if self._disk_busy_pct > 90.0:
            return False
        return True

    def recommended_batch_size(
        self, max_batch: int = 10, min_batch: int = 5
    ) -> int:
        """How many images to feed given current resource utilisation."""
        if not self._has_psutil:
            return max_batch
        if self._cpu_pct < _CPU_LOW_PCT and self._disk_busy_pct < 50.0:
            return max_batch
        if self._cpu_pct < _CPU_HIGH_PCT and self._disk_busy_pct < _DISK_BUSY_PCT:
            return max(min_batch, max_batch // 2)
        return min_batch

    @property
    def snapshot(self) -> dict[str, float]:
        """Last sampled values without taking a new sample."""
        return {
            "cpu_pct": self._cpu_pct,
            "disk_read_mbps": self._disk_read_mbps,
            "disk_busy_pct": self._disk_busy_pct,
        }

# Keys to exclude from the sidecar metadata (they're either the image
# pixels or objects that can't be JSON-serialised).
_EXCLUDE_META_KEYS = frozenset({"rgb_array", "pil_image", "raw_image"})


def _extract_sidecar_meta(image_data: dict[str, Any]) -> dict[str, Any]:
    """Extract JSON-serialisable metadata from a reader's image_data dict."""
    meta: dict[str, Any] = {}
    for k, v in image_data.items():
        if k in _EXCLUDE_META_KEYS:
            continue
        # Convert tuples/lists of numbers for JSON compatibility
        if isinstance(v, tuple):
            meta[k] = list(v)
        else:
            meta[k] = v
    return meta


class PreDecoder:
    """Background image decoder that populates a :class:`DecodedImageStore`.

    Supports two modes:

    * **Bulk** via :meth:`start` — decode a fixed list in one shot (legacy /
      CLI usage).
    * **Demand-driven** via :meth:`feed` — add small batches incrementally
      as workers need more cached images.  The coordinator's scheduler
      calls :meth:`feed` to keep a decode-ahead buffer for workers.

    Parameters
    ----------
    store:
        The decoded image cache to populate.
    max_workers:
        Number of threads in the decode pool.  Defaults to
        ``min(os.cpu_count() - 2, 6)`` (clamped ≥ 2), leaving headroom
        for GPU passes, the JSON-RPC server, and the OS.
    resource_sampler:
        Optional :class:`ResourceSampler` for adaptive scheduling.  When
        provided, the pre-decoder will check CPU and disk I/O before
        starting new decode tasks.
    """

    def __init__(
        self,
        store: DecodedImageStore,
        max_workers: int | None = None,
        resource_sampler: ResourceSampler | None = None,
    ) -> None:
        self._store = store
        # Scale to available cores, leaving 2 for server + GPU pipeline.
        # RAW demosaic via rawpy releases the GIL, so threads parallelise.
        cpu = os.cpu_count() or 4
        self._max_workers = max_workers or max(2, min(cpu - 2, 6))
        self._resource_sampler = resource_sampler or ResourceSampler()
        self._executor: ThreadPoolExecutor | None = None
        self._futures: list[Future[bool]] = []

        # Track which IDs have been submitted to avoid duplicates
        self._queued_ids: set[int] = set()

        # Progress tracking
        self._lock = threading.Lock()
        self._done = 0
        self._failed = 0
        self._total = 0
        self._running = False
        self._cancel_event = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(
        self,
        items: list[tuple[int, str]],
        *,
        raw_first: bool = True,
        pending_ids: set[int] | None = None,
    ) -> None:
        """Begin pre-decoding *items* in the background.

        Parameters
        ----------
        items:
            List of ``(image_id, file_path)`` pairs.
        raw_first:
            When True, RAW files are decoded before standard formats.
        pending_ids:
            Image IDs that have pending analysis jobs.  These are decoded
            first so that workers are unblocked as quickly as possible.
        """
        if self._running:
            log.warning("PreDecoder already running — ignoring start()")
            return

        # Filter out already-cached images
        cached = self._store.cached_image_ids()
        pending = [(iid, fp) for iid, fp in items if iid not in cached]

        if not pending:
            log.info("All %d images already cached — nothing to pre-decode", len(items))
            return

        pending = self._prioritize(pending, raw_first=raw_first, pending_ids=pending_ids)

        with self._lock:
            self._done = 0
            self._failed = 0
            self._total = len(pending)
            self._running = True
            self._queued_ids = {iid for iid, _ in pending}
        self._cancel_event.clear()

        log.info(
            "Starting pre-decode: %d images (%d already cached), %d threads",
            len(pending),
            len(items) - len(pending),
            self._max_workers,
        )

        self._executor = ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix="predecode",
        )
        self._futures = [
            self._executor.submit(self._decode_one, iid, fp)
            for iid, fp in pending
        ]

        # Monitor completion in a background thread
        threading.Thread(
            target=self._monitor,
            daemon=True,
            name="predecode-monitor",
        ).start()

    def feed(
        self,
        items: list[tuple[int, str]],
        *,
        raw_first: bool = True,
    ) -> int:
        """Add images to the decode pipeline incrementally.

        Unlike :meth:`start`, this can be called many times.  Images
        already cached or already queued are silently skipped.  The
        internal thread-pool is created on first call and reused.

        Returns the number of new images actually enqueued.
        """
        if self._cancel_event.is_set():
            return 0

        cached = self._store.cached_image_ids()
        with self._lock:
            new = [
                (iid, fp)
                for iid, fp in items
                if iid not in cached and iid not in self._queued_ids
            ]
        if not new:
            return 0

        if raw_first:
            new = self._sort_raw_first(new)

        # Create executor on first feed (lazy start)
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self._max_workers,
                thread_name_prefix="predecode",
            )
            self._cancel_event.clear()

        with self._lock:
            for iid, _ in new:
                self._queued_ids.add(iid)
            self._total += len(new)
            self._running = True

        for iid, fp in new:
            self._executor.submit(self._decode_one, iid, fp)

        log.info("Fed %d images to pre-decoder (total queued: %d)", len(new), self._total)
        return len(new)

    def stop(self) -> None:
        """Cancel any in-progress pre-decode work."""
        self._cancel_event.set()
        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        with self._lock:
            self._running = False
        log.info("PreDecoder stopped")

    def pause(self) -> None:
        """Signal threads to pause after current item."""
        self._cancel_event.set()

    def resume(self, items: list[tuple[int, str]] | None = None) -> None:
        """Resume (or restart with new items)."""
        if items:
            self._cancel_event.clear()
            self.start(items)

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def progress(self) -> dict[str, Any]:
        """Return ``{done, failed, total, running, resources}``."""
        with self._lock:
            result: dict[str, Any] = {
                "done": self._done,
                "failed": self._failed,
                "total": self._total,
                "running": self._running,
            }
        result["resources"] = self._resource_sampler.snapshot
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _prioritize(
        self,
        items: list[tuple[int, str]],
        *,
        raw_first: bool = True,
        pending_ids: set[int] | None = None,
    ) -> list[tuple[int, str]]:
        """Order items by priority for decoding.

        Priority tiers (highest first):
        1. Images with pending analysis jobs (``pending_ids``)
        2. All remaining images (background cache warming)

        Within each tier, RAW files come before standard formats when
        *raw_first* is True.  Within each RAW/standard partition, items
        are sorted by file path for disk locality — nearby files on an
        HDD have less seek overhead.
        """
        if pending_ids:
            tier1 = [(iid, fp) for iid, fp in items if iid in pending_ids]
            tier2 = [(iid, fp) for iid, fp in items if iid not in pending_ids]
        else:
            tier1 = []
            tier2 = items

        if raw_first:
            tier1 = self._sort_raw_first(tier1)
            tier2 = self._sort_raw_first(tier2)

        return tier1 + tier2

    def _sort_raw_first(
        self, items: list[tuple[int, str]]
    ) -> list[tuple[int, str]]:
        """Sort items: RAW files first, then standard formats.

        Within each group, sort by file path for disk locality on HDDs.
        """
        from imganalyzer.analyzer import RAW_EXTENSIONS

        raw: list[tuple[int, str]] = []
        std: list[tuple[int, str]] = []
        for iid, fp in items:
            suffix = Path(fp).suffix.lower()
            if suffix in RAW_EXTENSIONS:
                raw.append((iid, fp))
            else:
                std.append((iid, fp))
        raw.sort(key=lambda x: x[1])
        std.sort(key=lambda x: x[1])
        return raw + std

    def _decode_one(self, image_id: int, file_path: str) -> bool:
        """Decode a single image and store in cache.

        Returns True on success.  Thread-safe; uses per-image decode lock
        to prevent duplicate work.
        """
        if self._cancel_event.is_set():
            return False

        # Double-check (another thread may have cached it)
        if self._store.has(image_id):
            with self._lock:
                self._done += 1
            return True

        lock = self._store.get_decode_lock(image_id)
        with lock:
            # Re-check after acquiring lock
            if self._store.has(image_id):
                with self._lock:
                    self._done += 1
                return True

            path = Path(file_path)
            if not path.exists():
                log.warning("File not found for pre-decode: %s", file_path)
                with self._lock:
                    self._failed += 1
                return False

            try:
                t0 = time.monotonic()
                image_data = self._read_image(path)
                rgb = image_data.get("rgb_array")
                if rgb is None:
                    log.warning(
                        "No rgb_array from reader for %s", file_path
                    )
                    with self._lock:
                        self._failed += 1
                    return False

                pil_image = Image.fromarray(rgb)
                sidecar = _extract_sidecar_meta(image_data)

                # Also extract headers for metadata module
                try:
                    header_data = self._read_headers(path)
                    header_meta = _extract_sidecar_meta(header_data)
                    # Merge header-only fields (some may be richer)
                    for k, v in header_meta.items():
                        if k not in sidecar:
                            sidecar[k] = v
                except Exception:
                    pass  # Headers are optional enrichment

                # Pre-parse EXIF from the original file so remote workers
                # don't need the file on disk for metadata extraction.
                try:
                    import exifread

                    with open(path, "rb") as f:
                        tags = exifread.process_file(
                            f, details=False, strict=False,
                        )
                    from imganalyzer.analysis.metadata import MetadataExtractor

                    parsed = MetadataExtractor(
                        path, image_data,
                    )._parse_exifread_tags(tags)
                    if parsed:
                        sidecar["parsed_exif"] = parsed
                except Exception:
                    pass  # Best-effort; metadata module falls back to file

                self._store.put(image_id, pil_image, sidecar)

                elapsed = time.monotonic() - t0
                log.debug(
                    "Pre-decoded image %d (%s) in %.1fs",
                    image_id,
                    path.name,
                    elapsed,
                )

                with self._lock:
                    self._done += 1
                return True

            except Exception:
                log.exception(
                    "Failed to pre-decode image %d (%s)", image_id, file_path
                )
                with self._lock:
                    self._failed += 1
                return False

    def _read_image(self, path: Path) -> dict[str, Any]:
        """Read full image data (pixels + metadata)."""
        from imganalyzer.analyzer import RAW_EXTENSIONS

        suffix = path.suffix.lower()
        if suffix in RAW_EXTENSIONS:
            from imganalyzer.readers.raw import read
            return read(path, half_size=True)
        else:
            from imganalyzer.readers.standard import read
            return read(path)

    def _read_headers(self, path: Path) -> dict[str, Any]:
        """Read image headers only (no pixel decode)."""
        from imganalyzer.analyzer import RAW_EXTENSIONS

        suffix = path.suffix.lower()
        if suffix in RAW_EXTENSIONS:
            from imganalyzer.readers.raw import read_headers
            return read_headers(path)
        else:
            from imganalyzer.readers.standard import read_headers
            return read_headers(path)

    def _monitor(self) -> None:
        """Wait for all futures to complete and update running state."""
        for fut in self._futures:
            try:
                fut.result()
            except Exception:
                pass  # Already handled in _decode_one

        # Only mark not-running if no new work was fed while we waited.
        # feed() may have added items after start()'s initial batch.
        with self._lock:
            if self._done + self._failed >= self._total:
                self._running = False

        log.info(
            "Pre-decode batch complete: %d done, %d failed, %d total",
            self._done,
            self._failed,
            self._total,
        )

        # Don't shut down the executor — feed() may add more work later.
        # The executor is cleaned up in stop().
