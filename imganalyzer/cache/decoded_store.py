"""Persistent disk-based LRU cache for decoded images.

Stores pre-decoded images as WebP/JPEG at a configurable resolution with
sidecar JSON metadata (EXIF, RAW sensor data, dimensions).  Designed so
that distributed workers and the lightbox can read decoded images without
touching the NAS.

Thread-safe.  Per-image decode locks prevent thundering-herd on cache miss.
A small SQLite index tracks entry sizes and last-accessed times for LRU
eviction.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image

log = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "imganalyzer" / "decoded"
_DEFAULT_MAX_BYTES = 300 * 1024 * 1024 * 1024  # 300 GB
_DEFAULT_RESOLUTION = 1024
_DEFAULT_FORMAT = "webp"
_DEFAULT_QUALITY = 95

_INDEX_SCHEMA = """\
CREATE TABLE IF NOT EXISTS entries (
    image_id      INTEGER PRIMARY KEY,
    file_size     INTEGER NOT NULL,
    meta_size     INTEGER NOT NULL DEFAULT 0,
    last_accessed REAL    NOT NULL,
    created_at    REAL    NOT NULL
);
"""


def _encode_binary_fields(d: dict[str, Any]) -> dict[str, Any]:
    """Recursively encode ``bytes`` values as base64 strings for JSON."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, bytes):
            out[k] = {"__b64__": base64.b64encode(v).decode("ascii")}
        elif isinstance(v, dict):
            out[k] = _encode_binary_fields(v)
        else:
            out[k] = v
    return out


class _MetaEncoder(json.JSONEncoder):
    """JSON encoder that handles EXIF types (IFDRational, Fraction, etc.)."""

    def default(self, o: Any) -> Any:
        # piexif IFDRational, fractions.Fraction, or any Rational-like
        if hasattr(o, "numerator") and hasattr(o, "denominator"):
            denom = o.denominator
            if denom == 0:
                return 0.0
            return o.numerator / denom
        if isinstance(o, bytes):
            return base64.b64encode(o).decode("ascii")
        return super().default(o)


def _decode_binary_fields(d: dict[str, Any]) -> dict[str, Any]:
    """Reverse of :func:`_encode_binary_fields`."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict) and "__b64__" in v:
            out[k] = base64.b64decode(v["__b64__"])
        elif isinstance(v, dict):
            out[k] = _decode_binary_fields(v)
        else:
            out[k] = v
    return out


class DecodedImageStore:
    """Persistent disk-based LRU cache for decoded images.

    Images are stored as WebP (or JPEG/PNG) at *resolution* px max on the
    long edge.  A sidecar ``{id}.meta.json`` stores EXIF, RAW sensor data,
    and anything else the caller passes as *metadata*.

    Parameters
    ----------
    cache_dir:
        Root directory for the cache.  Created if it does not exist.
    max_bytes:
        Maximum total cache size in bytes.  LRU eviction triggers when
        exceeded.
    resolution:
        Maximum long-edge pixel size for cached images.
    fmt:
        Image format — ``"webp"``, ``"jpeg"``, or ``"png"``.
    quality:
        Compression quality (1–100).  Ignored for PNG.
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        resolution: int = _DEFAULT_RESOLUTION,
        fmt: str = _DEFAULT_FORMAT,
        quality: int = _DEFAULT_QUALITY,
    ) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self._max_bytes = max_bytes
        self._resolution = resolution
        self._fmt = fmt.lower()
        self._quality = quality

        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # SQLite index -------------------------------------------------------
        self._index_path = self._cache_dir / "cache_index.db"
        self._conn = sqlite3.connect(
            str(self._index_path), check_same_thread=False
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_INDEX_SCHEMA)
        self._conn.commit()
        self._db_lock = threading.Lock()

        # In-memory set of cached IDs for fast lookup -------------------------
        self._ids_lock = threading.Lock()
        self._cached_ids: set[int] = self._load_cached_ids()

        # Per-image decode locks (prevent thundering-herd) --------------------
        self._decode_locks: dict[int, threading.Lock] = {}
        self._decode_locks_guard = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_cached_ids(self) -> set[int]:
        """Load cached IDs from the DB index, removing stale entries.

        An entry is stale if the DB row exists but the image file is missing
        on disk (e.g. after a crash during eviction, manual file deletion, or
        a cache directory change).
        """
        rows = self._conn.execute("SELECT image_id FROM entries").fetchall()
        valid: set[int] = set()
        stale: list[int] = []
        for r in rows:
            image_id = r[0]
            if self._image_path(image_id).exists():
                valid.add(image_id)
            else:
                stale.append(image_id)
        if stale:
            log.info(
                "Purging %d stale cache index entries (files missing on disk)",
                len(stale),
            )
            for image_id in stale:
                # Remove DB entry and any leftover sidecar
                meta = self._meta_path(image_id)
                if meta.exists():
                    meta.unlink(missing_ok=True)
                self._conn.execute(
                    "DELETE FROM entries WHERE image_id = ?", (image_id,)
                )
            self._conn.commit()
        return valid

    def _shard_dir(self, image_id: int) -> Path:
        level1 = image_id // 10000
        level2 = (image_id % 10000) // 100
        return self._cache_dir / str(level1) / str(level2)

    def _image_path(self, image_id: int) -> Path:
        return self._shard_dir(image_id) / f"{image_id}.{self._fmt}"

    def _meta_path(self, image_id: int) -> Path:
        return self._shard_dir(image_id) / f"{image_id}.meta.json"

    def _resize(self, pil_image: Image.Image) -> Image.Image:
        """Resize so longest edge <= *resolution*, convert to RGB."""
        img = pil_image
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        w, h = img.size
        if max(w, h) <= self._resolution:
            return img

        if w >= h:
            new_w = self._resolution
            new_h = max(1, int(h * (self._resolution / w)))
        else:
            new_h = self._resolution
            new_w = max(1, int(w * (self._resolution / h)))

        return img.resize((new_w, new_h), Image.LANCZOS)

    # ------------------------------------------------------------------
    # Decode lock (per-image)
    # ------------------------------------------------------------------

    def get_decode_lock(self, image_id: int) -> threading.Lock:
        """Return a per-image lock for serialising decode work.

        Multiple threads requesting the same uncached image will acquire
        this lock; only the first one performs the actual decode.
        """
        with self._decode_locks_guard:
            if image_id not in self._decode_locks:
                self._decode_locks[image_id] = threading.Lock()
            return self._decode_locks[image_id]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has(self, image_id: int) -> bool:
        """Fast O(1) check whether *image_id* is in the cache."""
        with self._ids_lock:
            return image_id in self._cached_ids

    def cached_image_ids(self) -> set[int]:
        """Return a snapshot of all cached image IDs.

        Used by cache-gated job dispatch to filter claimable jobs.
        """
        with self._ids_lock:
            return set(self._cached_ids)

    def get(
        self, image_id: int
    ) -> tuple[Image.Image, dict[str, Any]] | None:
        """Return ``(PIL.Image, metadata_dict)`` or *None* on cache miss.

        Updates the LRU timestamp on every successful read.
        """
        img_path = self._image_path(image_id)
        if not img_path.exists():
            # Stale index entry — clean up
            with self._ids_lock:
                self._cached_ids.discard(image_id)
            return None

        try:
            pil_image = Image.open(img_path)
            pil_image.load()

            metadata: dict[str, Any] = {}
            meta_path = self._meta_path(image_id)
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = _decode_binary_fields(json.load(f))

            now = time.time()
            with self._db_lock:
                self._conn.execute(
                    "UPDATE entries SET last_accessed = ? WHERE image_id = ?",
                    (now, image_id),
                )
                self._conn.commit()

            return pil_image, metadata
        except Exception:
            log.warning("Corrupted cache entry %d — removing", image_id)
            self.delete(image_id)
            return None

    def get_image_bytes(self, image_id: int) -> bytes | None:
        """Return raw image file bytes without PIL decode.

        Useful for HTTP serving where we send binary directly.
        Updates LRU timestamp.  Automatically purges stale index entries
        when the file is missing on disk.
        """
        img_path = self._image_path(image_id)
        if not img_path.exists():
            # Self-heal: remove stale DB/index entry so the cache gate
            # won't dispatch jobs for this image again.
            with self._ids_lock:
                if image_id in self._cached_ids:
                    self._cached_ids.discard(image_id)
                    with self._db_lock:
                        self._conn.execute(
                            "DELETE FROM entries WHERE image_id = ?",
                            (image_id,),
                        )
                        self._conn.commit()
                    log.debug(
                        "Purged stale cache entry %d (file missing)", image_id
                    )
            return None

        try:
            data = img_path.read_bytes()
            now = time.time()
            with self._db_lock:
                self._conn.execute(
                    "UPDATE entries SET last_accessed = ? WHERE image_id = ?",
                    (now, image_id),
                )
                self._conn.commit()
            return data
        except Exception:
            return None

    def get_metadata(self, image_id: int) -> dict[str, Any] | None:
        """Return sidecar metadata without loading the image pixels."""
        meta_path = self._meta_path(image_id)
        if not meta_path.exists():
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return _decode_binary_fields(json.load(f))
        except Exception:
            return None

    def put(
        self,
        image_id: int,
        pil_image: Image.Image,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Store a decoded image (and optional metadata) in the cache.

        The image is resized to *resolution* px max and saved in the
        configured format.  Returns the path to the cached image file.
        """
        shard_dir = self._shard_dir(image_id)
        shard_dir.mkdir(parents=True, exist_ok=True)

        img_path = self._image_path(image_id)
        meta_path = self._meta_path(image_id)

        resized = self._resize(pil_image)

        save_kwargs: dict[str, Any] = {}
        if self._fmt in ("webp", "jpeg"):
            save_kwargs["quality"] = self._quality
        if self._fmt == "webp":
            save_kwargs["method"] = 4  # speed / compression trade-off
        pil_format = "JPEG" if self._fmt == "jpeg" else self._fmt.upper()
        resized.save(str(img_path), format=pil_format, **save_kwargs)

        file_size = img_path.stat().st_size
        meta_size = 0
        if metadata:
            encoded = _encode_binary_fields(metadata)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(encoded, f, ensure_ascii=False, separators=(",", ":"),
                          cls=_MetaEncoder)
            meta_size = meta_path.stat().st_size

        now = time.time()
        with self._db_lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO entries "
                "(image_id, file_size, meta_size, last_accessed, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (image_id, file_size, meta_size, now, now),
            )
            self._conn.commit()

        with self._ids_lock:
            self._cached_ids.add(image_id)

        # Async eviction if needed
        if self.size_bytes > self._max_bytes:
            threading.Thread(
                target=self._evict_to_target,
                args=(int(self._max_bytes * 0.9),),
                daemon=True,
            ).start()

        return img_path

    def delete(self, image_id: int) -> None:
        """Remove an entry from the cache."""
        for p in (self._image_path(image_id), self._meta_path(image_id)):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass

        with self._db_lock:
            self._conn.execute(
                "DELETE FROM entries WHERE image_id = ?", (image_id,)
            )
            self._conn.commit()

        with self._ids_lock:
            self._cached_ids.discard(image_id)

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def evict_lru(self, target_bytes: int) -> int:
        """Evict LRU entries until total size ≤ *target_bytes*.

        Returns the number of bytes freed.
        """
        return self._evict_to_target(target_bytes)

    def _evict_to_target(self, target_bytes: int) -> int:
        freed = 0
        with self._db_lock:
            current = self._conn.execute(
                "SELECT COALESCE(SUM(file_size + meta_size), 0) FROM entries"
            ).fetchone()[0]

            if current <= target_bytes:
                return 0

            to_free = current - target_bytes
            rows = self._conn.execute(
                "SELECT image_id, file_size, meta_size FROM entries "
                "ORDER BY last_accessed ASC"
            ).fetchall()

        to_delete: list[int] = []
        for row in rows:
            if freed >= to_free:
                break
            to_delete.append(row["image_id"])
            freed += row["file_size"] + row["meta_size"]

        for image_id in to_delete:
            self.delete(image_id)

        if to_delete:
            log.info(
                "Evicted %d cache entries, freed %.1f MB",
                len(to_delete),
                freed / (1024 * 1024),
            )
        return freed

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def size_bytes(self) -> int:
        """Total size of cached files in bytes."""
        with self._db_lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(file_size + meta_size), 0) FROM entries"
            ).fetchone()
            return row[0]

    @property
    def entry_count(self) -> int:
        """Number of entries in the cache."""
        with self._ids_lock:
            return len(self._cached_ids)

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @property
    def resolution(self) -> int:
        return self._resolution

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def rebuild_index(self) -> int:
        """Rebuild the SQLite index by scanning the cache directory.

        Useful if files were added/removed externally.
        Returns the number of entries found.
        """
        count = 0
        with self._db_lock:
            self._conn.execute("DELETE FROM entries")
            for shard1 in self._cache_dir.iterdir():
                if not shard1.is_dir() or shard1.name.startswith((".", "cache")):
                    continue
                for shard2 in shard1.iterdir():
                    if not shard2.is_dir():
                        continue
                    for img_file in shard2.glob(f"*.{self._fmt}"):
                        try:
                            image_id = int(img_file.stem)
                            file_size = img_file.stat().st_size
                            meta_path = shard2 / f"{image_id}.meta.json"
                            meta_size = (
                                meta_path.stat().st_size
                                if meta_path.exists()
                                else 0
                            )
                            mtime = img_file.stat().st_mtime
                            self._conn.execute(
                                "INSERT OR REPLACE INTO entries "
                                "(image_id, file_size, meta_size, "
                                "last_accessed, created_at) "
                                "VALUES (?, ?, ?, ?, ?)",
                                (image_id, file_size, meta_size, mtime, mtime),
                            )
                            count += 1
                        except (ValueError, OSError):
                            continue
            self._conn.commit()

        with self._ids_lock:
            self._cached_ids = self._load_cached_ids()

        log.info("Rebuilt cache index: %d entries", count)
        return count

    def close(self) -> None:
        """Close the SQLite index connection."""
        try:
            self._conn.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"DecodedImageStore("
            f"dir={self._cache_dir}, "
            f"entries={self.entry_count}, "
            f"size={self.size_bytes / (1024**3):.1f}GB"
            f"/{self._max_bytes / (1024**3):.0f}GB)"
        )
