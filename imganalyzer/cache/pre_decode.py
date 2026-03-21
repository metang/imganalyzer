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

    Parameters
    ----------
    store:
        The decoded image cache to populate.
    max_workers:
        Number of threads in the decode pool.  Defaults to ``os.cpu_count()``.
    """

    def __init__(
        self,
        store: DecodedImageStore,
        max_workers: int | None = None,
    ) -> None:
        self._store = store
        self._max_workers = max_workers or os.cpu_count() or 4
        self._executor: ThreadPoolExecutor | None = None
        self._futures: list[Future[bool]] = []

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
    ) -> None:
        """Begin pre-decoding *items* in the background.

        Parameters
        ----------
        items:
            List of ``(image_id, file_path)`` pairs.
        raw_first:
            When True, RAW files are decoded before standard formats.
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

        if raw_first:
            pending = self._sort_raw_first(pending)

        with self._lock:
            self._done = 0
            self._failed = 0
            self._total = len(pending)
            self._running = True
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
        """Return ``{done, failed, total, running}``."""
        with self._lock:
            return {
                "done": self._done,
                "failed": self._failed,
                "total": self._total,
                "running": self._running,
            }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sort_raw_first(
        self, items: list[tuple[int, str]]
    ) -> list[tuple[int, str]]:
        """Sort items: RAW files first, then standard formats."""
        from imganalyzer.analyzer import RAW_EXTENSIONS

        raw: list[tuple[int, str]] = []
        std: list[tuple[int, str]] = []
        for iid, fp in items:
            suffix = Path(fp).suffix.lower()
            if suffix in RAW_EXTENSIONS:
                raw.append((iid, fp))
            else:
                std.append((iid, fp))
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

        with self._lock:
            self._running = False

        log.info(
            "Pre-decode complete: %d done, %d failed, %d total",
            self._done,
            self._failed,
            self._total,
        )

        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None
