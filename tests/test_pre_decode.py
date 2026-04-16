"""Tests for the pre-decode pipeline."""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from imganalyzer.cache.decoded_store import DecodedImageStore
from imganalyzer.cache.pre_decode import PreDecoder, _extract_sidecar_meta


@pytest.fixture()
def store(tmp_path: Path) -> DecodedImageStore:
    return DecodedImageStore(
        cache_dir=tmp_path / "decoded",
        max_bytes=100 * 1024 * 1024,
        resolution=256,
        fmt="webp",
        quality=90,
    )


@pytest.fixture()
def image_dir(tmp_path: Path) -> Path:
    """Create a directory with test images."""
    d = tmp_path / "images"
    d.mkdir()
    for i in range(5):
        img = Image.new("RGB", (400, 300), color=(i * 50, 100, 200))
        img.save(str(d / f"img_{i}.jpg"), "JPEG")
    return d


class TestExtractSidecarMeta:
    def test_excludes_rgb_array(self) -> None:
        data = {
            "format": "JPEG",
            "width": 1024,
            "height": 768,
            "rgb_array": np.zeros((768, 1024, 3), dtype=np.uint8),
            "pil_image": Image.new("RGB", (1, 1)),
            "is_raw": False,
        }
        meta = _extract_sidecar_meta(data)
        assert "rgb_array" not in meta
        assert "pil_image" not in meta
        assert meta["format"] == "JPEG"
        assert meta["width"] == 1024
        assert meta["is_raw"] is False

    def test_converts_tuples(self) -> None:
        data = {"dpi": (72, 72), "camera_wb": (1.0, 2.0, 3.0, 4.0)}
        meta = _extract_sidecar_meta(data)
        assert meta["dpi"] == [72, 72]
        assert meta["camera_wb"] == [1.0, 2.0, 3.0, 4.0]


class TestPreDecoder:
    def test_decode_standard_images(
        self, store: DecodedImageStore, image_dir: Path
    ) -> None:
        items = [
            (i, str(image_dir / f"img_{i}.jpg")) for i in range(5)
        ]
        decoder = PreDecoder(store, max_workers=2)
        decoder.start(items)

        # Wait for completion
        timeout = 10.0
        t0 = time.time()
        while decoder.is_running and time.time() - t0 < timeout:
            time.sleep(0.1)

        assert not decoder.is_running
        prog = decoder.progress()
        assert prog["done"] == 5
        assert prog["failed"] == 0
        assert prog["total"] == 5

        # All images should be cached
        for i in range(5):
            assert store.has(i)
            result = store.get(i)
            assert result is not None
            img, meta = result
            assert isinstance(img, Image.Image)
            assert max(img.size) <= 256
            assert meta.get("format") == "JPEG"

    def test_skips_already_cached(
        self, store: DecodedImageStore, image_dir: Path
    ) -> None:
        # Pre-cache image 0
        img = Image.new("RGB", (100, 100))
        store.put(0, img, {"format": "pre-cached"})

        items = [
            (i, str(image_dir / f"img_{i}.jpg")) for i in range(3)
        ]
        decoder = PreDecoder(store, max_workers=2)
        decoder.start(items)

        timeout = 10.0
        t0 = time.time()
        while decoder.is_running and time.time() - t0 < timeout:
            time.sleep(0.1)

        prog = decoder.progress()
        # Only 2 new images decoded (0 was already cached)
        assert prog["total"] == 2
        assert prog["done"] == 2

    def test_handles_missing_files(
        self, store: DecodedImageStore, tmp_path: Path
    ) -> None:
        items = [
            (1, str(tmp_path / "nonexistent.jpg")),
        ]
        decoder = PreDecoder(store, max_workers=1)
        decoder.start(items)

        timeout = 5.0
        t0 = time.time()
        while decoder.is_running and time.time() - t0 < timeout:
            time.sleep(0.1)

        prog = decoder.progress()
        assert prog["failed"] == 1
        assert not store.has(1)

    def test_stop_cancels(
        self, store: DecodedImageStore, image_dir: Path
    ) -> None:
        items = [
            (i, str(image_dir / f"img_{i}.jpg")) for i in range(5)
        ]
        decoder = PreDecoder(store, max_workers=1)
        decoder.start(items)
        decoder.stop()

        assert not decoder.is_running

    def test_progress_tracking(
        self, store: DecodedImageStore, image_dir: Path
    ) -> None:
        items = [
            (i, str(image_dir / f"img_{i}.jpg")) for i in range(3)
        ]
        decoder = PreDecoder(store, max_workers=1)

        prog = decoder.progress()
        assert prog["total"] == 0
        assert not prog["running"]

        decoder.start(items)
        assert decoder.is_running

        timeout = 10.0
        t0 = time.time()
        while decoder.is_running and time.time() - t0 < timeout:
            time.sleep(0.1)

        prog = decoder.progress()
        assert prog["done"] == 3
        assert prog["total"] == 3
        assert not prog["running"]

    def test_all_cached_is_noop(
        self, store: DecodedImageStore, image_dir: Path
    ) -> None:
        # Pre-cache everything
        img = Image.new("RGB", (50, 50))
        for i in range(3):
            store.put(i, img)

        items = [(i, str(image_dir / f"img_{i}.jpg")) for i in range(3)]
        decoder = PreDecoder(store, max_workers=1)
        decoder.start(items)

        # Should complete immediately (nothing to do)
        assert not decoder.is_running

    def test_feed_marks_not_running_after_completion(
        self, store: DecodedImageStore, image_dir: Path
    ) -> None:
        items = [(0, str(image_dir / "img_0.jpg"))]
        decoder = PreDecoder(store, max_workers=1)

        assert decoder.feed(items) == 1

        timeout = 10.0
        t0 = time.time()
        while decoder.is_running and time.time() - t0 < timeout:
            time.sleep(0.1)

        assert not decoder.is_running
        assert store.has(0)

    def test_feed_retries_after_transient_missing_file(
        self, store: DecodedImageStore, tmp_path: Path
    ) -> None:
        late_path = tmp_path / "late-arrival.jpg"
        decoder = PreDecoder(store, max_workers=1)

        assert decoder.feed([(1, str(late_path))]) == 1

        timeout = 5.0
        t0 = time.time()
        while (
            (decoder.progress()["failed"] < 1 or decoder.is_running)
            and time.time() - t0 < timeout
        ):
            time.sleep(0.1)

        assert not store.has(1)

        Image.new("RGB", (64, 64), color=(10, 20, 30)).save(late_path, "JPEG")
        assert decoder.feed([(1, str(late_path))]) == 1

        timeout = 10.0
        t0 = time.time()
        while not store.has(1) and time.time() - t0 < timeout:
            time.sleep(0.1)

        assert store.has(1)

    def test_duplicate_start_ignored(
        self, store: DecodedImageStore, image_dir: Path
    ) -> None:
        items = [(0, str(image_dir / "img_0.jpg"))]
        decoder = PreDecoder(store, max_workers=1)
        decoder.start(items)

        # Second start while running should be ignored
        decoder.start(items)

        timeout = 5.0
        t0 = time.time()
        while decoder.is_running and time.time() - t0 < timeout:
            time.sleep(0.1)

    def test_metadata_stored_in_sidecar(
        self, store: DecodedImageStore, image_dir: Path
    ) -> None:
        items = [(0, str(image_dir / "img_0.jpg"))]
        decoder = PreDecoder(store, max_workers=1)
        decoder.start(items)

        timeout = 10.0
        t0 = time.time()
        while decoder.is_running and time.time() - t0 < timeout:
            time.sleep(0.1)

        meta = store.get_metadata(0)
        assert meta is not None
        assert "format" in meta
        assert "width" in meta
        assert "height" in meta


class TestPriorityOrdering:
    """Test that images with pending jobs are decoded before others."""

    def test_pending_ids_come_first(
        self, store: DecodedImageStore, image_dir: Path
    ) -> None:
        """Images with pending_ids are decoded before non-pending images."""
        decode_order: list[int] = []
        original_decode = PreDecoder._decode_one

        def tracking_decode(self_decoder, image_id: int, file_path: str) -> bool:
            decode_order.append(image_id)
            return original_decode(self_decoder, image_id, file_path)

        items = [(i, str(image_dir / f"img_{i}.jpg")) for i in range(5)]
        # Images 3 and 4 have pending jobs — should be decoded first
        pending_ids = {3, 4}

        decoder = PreDecoder(store, max_workers=1)
        with patch.object(PreDecoder, "_decode_one", tracking_decode):
            decoder.start(items, pending_ids=pending_ids)
            timeout = 10.0
            t0 = time.time()
            while decoder.is_running and time.time() - t0 < timeout:
                time.sleep(0.1)

        # Images 3 and 4 should appear before images 0, 1, 2
        assert len(decode_order) == 5
        pending_positions = [decode_order.index(i) for i in pending_ids]
        non_pending_positions = [decode_order.index(i) for i in range(3)]
        assert max(pending_positions) < min(non_pending_positions)

    def test_no_pending_ids_falls_back_to_default(
        self, store: DecodedImageStore, image_dir: Path
    ) -> None:
        """Without pending_ids, all images are treated equally."""
        items = [(i, str(image_dir / f"img_{i}.jpg")) for i in range(3)]
        decoder = PreDecoder(store, max_workers=2)
        decoder.start(items, pending_ids=None)

        timeout = 10.0
        t0 = time.time()
        while decoder.is_running and time.time() - t0 < timeout:
            time.sleep(0.1)

        assert decoder.progress()["done"] == 3

    def test_empty_pending_ids(
        self, store: DecodedImageStore, image_dir: Path
    ) -> None:
        """Empty pending_ids set behaves same as None."""
        items = [(i, str(image_dir / f"img_{i}.jpg")) for i in range(3)]
        decoder = PreDecoder(store, max_workers=2)
        decoder.start(items, pending_ids=set())

        timeout = 10.0
        t0 = time.time()
        while decoder.is_running and time.time() - t0 < timeout:
            time.sleep(0.1)

        assert decoder.progress()["done"] == 3
