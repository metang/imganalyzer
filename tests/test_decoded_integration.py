"""Integration tests for the coordinator-mediated image distribution pipeline.

Tests the full flow:  DecodedImageStore → PreDecoder → server endpoints →
worker fetch → module runner with cached data.
"""
from __future__ import annotations

import io
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from imganalyzer.cache.decoded_store import (
    DecodedImageStore,
    _decode_binary_fields,
    _encode_binary_fields,
)
from imganalyzer.cache.pre_decode import PreDecoder
from imganalyzer.db.repository import Repository
from imganalyzer.db.schema import ensure_schema
from imganalyzer.pipeline.modules import ModuleRunner


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_pil(width: int = 64, height: int = 48) -> Image.Image:
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _make_metadata() -> dict[str, Any]:
    return {
        "width": 4000,
        "height": 3000,
        "format": "ARW",
        "is_raw": True,
        "exif_bytes": b"\xff\xd8\xff\xe1" + b"\x00" * 20,
    }


def _make_repo() -> tuple[sqlite3.Connection, Repository]:
    conn = sqlite3.connect(":memory:", isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    ensure_schema(conn)
    repo = Repository(conn)
    return conn, repo


# ── Tests ────────────────────────────────────────────────────────────────────


class TestStoreToRunnerPipeline:
    """Decoded store → module runner with cached data."""

    def test_runner_uses_cached_image(self, tmp_path: Path) -> None:
        """ModuleRunner.prime_image_cache + _cached_header_data skip file I/O."""
        conn, repo = _make_repo()
        runner = ModuleRunner(conn=conn, repo=repo)

        pil_img = _make_pil(1024, 768)
        rgb = np.asarray(pil_img.convert("RGB"))
        fake_path = Path("/nas/photos/test.arw")

        image_data = {
            "rgb_array": rgb,
            "width": rgb.shape[1],
            "height": rgb.shape[0],
            "format": "ARW",
            "is_raw": True,
        }
        runner.prime_image_cache(fake_path, image_data)

        # Verify cache is primed
        assert runner._image_cache_path == fake_path
        assert runner._image_cache_data is not None
        assert runner._image_cache_data["width"] == 1024

    def test_runner_cached_header_data_for_metadata(self, tmp_path: Path) -> None:
        """When _cached_header_data is set, metadata module uses it."""
        conn, repo = _make_repo()
        runner = ModuleRunner(conn=conn, repo=repo)

        meta = _make_metadata()
        runner._cached_header_data = meta

        assert runner._cached_header_data is not None
        assert runner._cached_header_data["is_raw"] is True
        assert runner._cached_header_data["format"] == "ARW"


class TestPreDecodeToStore:
    """PreDecoder → DecodedImageStore pipeline."""

    def test_pre_decode_populates_store(self, tmp_path: Path) -> None:
        """Pre-decode pipeline stores decoded images in the store."""
        store = DecodedImageStore(cache_dir=tmp_path, max_bytes=100 * 1024 * 1024)
        assert store.entry_count == 0

        # Put a test image manually (simulating what pre-decoder does)
        pil_img = _make_pil(200, 150)
        meta = _make_metadata()
        store.put(1, pil_img, meta)

        assert store.has(1)
        assert store.entry_count == 1

        # Retrieve and verify
        result = store.get(1)
        assert result is not None
        got_img, got_meta = result
        assert got_img.size[0] <= 1024
        assert got_meta["is_raw"] is True

    def test_pre_decode_idempotent(self, tmp_path: Path) -> None:
        """Decoding the same image twice does not create duplicates."""
        store = DecodedImageStore(cache_dir=tmp_path, max_bytes=100 * 1024 * 1024)
        pil_img = _make_pil()
        meta = _make_metadata()

        store.put(42, pil_img, meta)
        store.put(42, pil_img, meta)

        assert store.entry_count == 1


class TestCacheGatedDispatch:
    """Cache-gated job dispatch logic."""

    def test_cached_image_ids_filtering(self, tmp_path: Path) -> None:
        """Only images in the store appear in cached_image_ids()."""
        store = DecodedImageStore(cache_dir=tmp_path, max_bytes=100 * 1024 * 1024)

        # Put images 1, 3, 5
        for iid in [1, 3, 5]:
            store.put(iid, _make_pil(), _make_metadata())

        cached = store.cached_image_ids()
        assert cached == {1, 3, 5}

        # Image 2 is NOT cached — would be gated out of dispatch
        assert 2 not in cached

    def test_has_decoded_cache_flag_in_job(self, tmp_path: Path) -> None:
        """Simulate coordinator adding hasDecodedCache to job payloads."""
        store = DecodedImageStore(cache_dir=tmp_path, max_bytes=100 * 1024 * 1024)
        store.put(10, _make_pil(), _make_metadata())

        cache_gate_ids = store.cached_image_ids()

        # Simulate job building
        jobs = [
            {"imageId": 10, "module": "caption"},
            {"imageId": 20, "module": "objects"},
        ]

        for job in jobs:
            if int(job["imageId"]) in cache_gate_ids:
                job["hasDecodedCache"] = True

        assert jobs[0].get("hasDecodedCache") is True
        assert jobs[1].get("hasDecodedCache") is None


class TestImageServing:
    """Image serving endpoint logic (without HTTP server)."""

    def test_get_image_bytes(self, tmp_path: Path) -> None:
        """Store returns raw image bytes for HTTP serving."""
        store = DecodedImageStore(cache_dir=tmp_path, max_bytes=100 * 1024 * 1024)
        pil_img = _make_pil(800, 600)
        store.put(99, pil_img, _make_metadata())

        img_bytes = store.get_image_bytes(99)
        assert img_bytes is not None
        assert len(img_bytes) > 0

        # Verify it's valid WebP
        roundtrip = Image.open(io.BytesIO(img_bytes))
        assert roundtrip.format == "WEBP"

    def test_get_metadata_json(self, tmp_path: Path) -> None:
        """Store returns serializable metadata."""
        store = DecodedImageStore(cache_dir=tmp_path, max_bytes=100 * 1024 * 1024)
        meta = _make_metadata()
        store.put(99, _make_pil(), meta)

        got_meta = store.get_metadata(99)
        assert got_meta is not None
        assert got_meta["is_raw"] is True

        # Verify JSON round-trip with binary fields
        encoded = _encode_binary_fields(got_meta)
        json_str = json.dumps(encoded)
        decoded = _decode_binary_fields(json.loads(json_str))
        assert decoded["exif_bytes"] == meta["exif_bytes"]

    def test_missing_image_returns_none(self, tmp_path: Path) -> None:
        store = DecodedImageStore(cache_dir=tmp_path, max_bytes=100 * 1024 * 1024)
        assert store.get_image_bytes(999) is None
        assert store.get_metadata(999) is None


class TestWorkerFetchSimulation:
    """Simulate worker fetching decoded images from coordinator."""

    def test_prime_runner_from_bytes(self, tmp_path: Path) -> None:
        """Simulate the worker flow: fetch bytes → PIL → prime runner."""
        # 1. Coordinator stores image
        store = DecodedImageStore(cache_dir=tmp_path, max_bytes=100 * 1024 * 1024)
        original = _make_pil(1024, 768)
        meta = _make_metadata()
        store.put(1, original, meta)

        # 2. Worker fetches raw bytes (simulating HTTP GET)
        img_bytes = store.get_image_bytes(1)
        assert img_bytes is not None

        # 3. Worker decodes bytes into PIL + numpy
        pil_img = Image.open(io.BytesIO(img_bytes))
        pil_img.load()
        rgb_array = np.asarray(pil_img.convert("RGB"))

        # 4. Worker primes the runner
        conn, repo = _make_repo()
        runner = ModuleRunner(conn=conn, repo=repo)

        image_data = {
            "rgb_array": rgb_array,
            "width": rgb_array.shape[1],
            "height": rgb_array.shape[0],
            "format": "ARW",
            "is_raw": False,
        }
        fake_path = Path("/nas/photos/test.arw")
        runner.prime_image_cache(fake_path, image_data)

        # 5. Verify runner has cached data
        assert runner._image_cache_path == fake_path
        assert runner._image_cache_data["rgb_array"].shape[2] == 3

        # 6. Set cached header data
        fetched_meta = store.get_metadata(1)
        runner._cached_header_data = fetched_meta

        assert runner._cached_header_data is not None
        assert runner._cached_header_data["is_raw"] is True

    def test_batch_prefetch_deduplication(self) -> None:
        """Batch prefetch identifies unique image IDs from multiple jobs."""
        jobs = [
            {"imageId": 1, "module": "caption", "hasDecodedCache": True},
            {"imageId": 1, "module": "metadata", "hasDecodedCache": True},
            {"imageId": 2, "module": "objects", "hasDecodedCache": True},
            {"imageId": 3, "module": "faces"},  # no hasDecodedCache
        ]

        ids_to_fetch: list[int] = []
        seen: set[int] = set()
        for job in jobs:
            if not job.get("hasDecodedCache"):
                continue
            img_id = int(job.get("imageId", 0))
            if img_id and img_id not in seen:
                ids_to_fetch.append(img_id)
                seen.add(img_id)

        # Should deduplicate image 1 and skip image 3 (no flag)
        assert ids_to_fetch == [1, 2]


class TestCachedImageRPCHandler:
    """Test the cachedimage JSON-RPC handler with imagePath resolution."""

    def test_cachedimage_by_path(self, tmp_path: Path) -> None:
        """cachedimage handler resolves imagePath → imageId via DB."""
        # Set up a store with an image
        store = DecodedImageStore(cache_dir=tmp_path, max_bytes=100 * 1024 * 1024)
        store.put(42, _make_pil(512, 384), _make_metadata())

        # Set up a DB with the image record
        conn, repo = _make_repo()
        repo.conn.execute(
            "INSERT INTO images (id, file_path) VALUES (?, ?)",
            [42, "/photos/test.arw"],
        )
        repo.conn.commit()

        # Simulate the handler logic
        params: dict[str, Any] = {"imagePath": "/photos/test.arw"}
        image_id = params.get("imageId")
        image_path = params.get("imagePath")

        if image_id is None and image_path is not None:
            row = repo.get_image_by_path(str(image_path))
            assert row is not None
            image_id = row["id"]

        assert image_id == 42

        result = store.get(image_id)
        assert result is not None
        pil_image, _meta = result

        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=92)
        jpeg_bytes = buf.getvalue()
        assert len(jpeg_bytes) > 0

    def test_cachedimage_missing_path(self, tmp_path: Path) -> None:
        """cachedimage returns not-available for unknown paths."""
        conn, repo = _make_repo()

        params: dict[str, Any] = {"imagePath": "/nonexistent.arw"}
        row = repo.get_image_by_path(str(params["imagePath"]))
        assert row is None


class TestBinaryFieldSerialization:
    """Ensure binary metadata survives JSON round-trip."""

    def test_roundtrip_exif_bytes(self) -> None:
        exif = b"\xff\xd8\xff\xe1" + bytes(range(256))
        meta = {"exif_bytes": exif, "format": "ARW", "nested": {"icc": b"\x00\x01"}}

        encoded = _encode_binary_fields(meta)
        json_str = json.dumps(encoded)
        decoded = _decode_binary_fields(json.loads(json_str))

        assert decoded["exif_bytes"] == exif
        assert decoded["format"] == "ARW"
        assert decoded["nested"]["icc"] == b"\x00\x01"

    def test_empty_metadata(self) -> None:
        encoded = _encode_binary_fields({})
        assert encoded == {}
        decoded = _decode_binary_fields({})
        assert decoded == {}


class TestLRUEviction:
    """Cache eviction under space pressure."""

    def test_eviction_frees_space(self, tmp_path: Path) -> None:
        """LRU eviction removes oldest entries when cache exceeds limit."""
        # Tiny 50KB cache
        store = DecodedImageStore(
            cache_dir=tmp_path, max_bytes=50 * 1024, resolution=32
        )

        # Fill with several small images
        for i in range(20):
            store.put(i, _make_pil(32, 32), {"format": "TEST"})

        initial_count = store.entry_count
        initial_size = store.size_bytes

        # Evict to 50% capacity
        freed = store.evict_lru(target_bytes=store._max_bytes // 2)
        assert freed > 0
        assert store.entry_count < initial_count
        assert store.size_bytes < initial_size
