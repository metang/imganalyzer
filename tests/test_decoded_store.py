"""Tests for DecodedImageStore."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from PIL import Image

from imganalyzer.cache.decoded_store import (
    DecodedImageStore,
    _decode_binary_fields,
    _encode_binary_fields,
)


@pytest.fixture()
def store(tmp_path: Path) -> DecodedImageStore:
    """Create a small store in a temp directory."""
    return DecodedImageStore(
        cache_dir=tmp_path / "decoded",
        max_bytes=10 * 1024 * 1024,  # 10 MB
        resolution=256,
        fmt="webp",
        quality=90,
    )


@pytest.fixture()
def sample_image() -> Image.Image:
    """A simple 500x400 RGB test image."""
    return Image.new("RGB", (500, 400), color=(128, 64, 32))


class TestBinaryFieldEncoding:
    def test_round_trip(self) -> None:
        original = {
            "name": "test",
            "data": b"\x00\x01\x02\xff",
            "nested": {"inner": b"hello"},
        }
        encoded = _encode_binary_fields(original)
        assert isinstance(encoded["data"], dict)
        assert "__b64__" in encoded["data"]

        decoded = _decode_binary_fields(encoded)
        assert decoded["data"] == b"\x00\x01\x02\xff"
        assert decoded["nested"]["inner"] == b"hello"
        assert decoded["name"] == "test"

    def test_json_serializable(self) -> None:
        original = {"exif": b"\xff\xd8\xff\xe1", "width": 1024}
        encoded = _encode_binary_fields(original)
        json_str = json.dumps(encoded)
        decoded = _decode_binary_fields(json.loads(json_str))
        assert decoded["exif"] == b"\xff\xd8\xff\xe1"
        assert decoded["width"] == 1024


class TestDecodedImageStore:
    def test_put_and_get(
        self, store: DecodedImageStore, sample_image: Image.Image
    ) -> None:
        metadata = {"format": "CR2", "width": 6000, "height": 4000, "is_raw": True}
        path = store.put(1, sample_image, metadata)
        assert path.exists()
        assert store.has(1)
        assert 1 in store.cached_image_ids()

        result = store.get(1)
        assert result is not None
        img, meta = result
        assert isinstance(img, Image.Image)
        assert meta["format"] == "CR2"
        assert meta["is_raw"] is True

    def test_get_miss(self, store: DecodedImageStore) -> None:
        assert store.get(999) is None
        assert not store.has(999)

    def test_put_resizes(
        self, store: DecodedImageStore, sample_image: Image.Image
    ) -> None:
        # store.resolution = 256, image is 500x400
        store.put(1, sample_image)
        result = store.get(1)
        assert result is not None
        img, _ = result
        assert max(img.size) <= 256

    def test_put_rgba_converts_to_rgb(self, store: DecodedImageStore) -> None:
        rgba = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        store.put(2, rgba)
        result = store.get(2)
        assert result is not None
        img, _ = result
        assert img.mode == "RGB"

    def test_delete(
        self, store: DecodedImageStore, sample_image: Image.Image
    ) -> None:
        store.put(1, sample_image)
        assert store.has(1)
        store.delete(1)
        assert not store.has(1)
        assert store.get(1) is None

    def test_entry_count(
        self, store: DecodedImageStore, sample_image: Image.Image
    ) -> None:
        assert store.entry_count == 0
        store.put(1, sample_image)
        store.put(2, sample_image)
        assert store.entry_count == 2
        store.delete(1)
        assert store.entry_count == 1

    def test_size_bytes_increases(
        self, store: DecodedImageStore, sample_image: Image.Image
    ) -> None:
        assert store.size_bytes == 0
        store.put(1, sample_image)
        assert store.size_bytes > 0

    def test_cached_image_ids(
        self, store: DecodedImageStore, sample_image: Image.Image
    ) -> None:
        store.put(10, sample_image)
        store.put(20, sample_image)
        ids = store.cached_image_ids()
        assert ids == {10, 20}
        # Returns a copy
        ids.add(999)
        assert 999 not in store.cached_image_ids()

    def test_get_metadata_only(
        self, store: DecodedImageStore, sample_image: Image.Image
    ) -> None:
        meta = {"format": "NEF", "camera": "Nikon D850"}
        store.put(1, sample_image, meta)
        result = store.get_metadata(1)
        assert result is not None
        assert result["camera"] == "Nikon D850"

    def test_get_metadata_miss(self, store: DecodedImageStore) -> None:
        assert store.get_metadata(999) is None

    def test_get_image_bytes(
        self, store: DecodedImageStore, sample_image: Image.Image
    ) -> None:
        store.put(1, sample_image)
        data = store.get_image_bytes(1)
        assert data is not None
        assert len(data) > 0
        # Should be a valid WebP
        assert data[:4] == b"RIFF"

    def test_get_image_bytes_miss(self, store: DecodedImageStore) -> None:
        assert store.get_image_bytes(999) is None

    def test_metadata_with_binary(
        self, store: DecodedImageStore, sample_image: Image.Image
    ) -> None:
        meta = {
            "exif_bytes": b"\xff\xd8\xff\xe1\x00\x00",
            "icc_profile": b"\x00\x01\x02",
            "format": "JPEG",
        }
        store.put(1, sample_image, meta)
        result = store.get(1)
        assert result is not None
        _, loaded_meta = result
        assert loaded_meta["exif_bytes"] == b"\xff\xd8\xff\xe1\x00\x00"
        assert loaded_meta["format"] == "JPEG"

    def test_shard_directory_structure(
        self, store: DecodedImageStore, sample_image: Image.Image
    ) -> None:
        # image_id=12345 -> shard1=1, shard2=23
        store.put(12345, sample_image)
        expected_dir = store.cache_dir / "1" / "23"
        assert expected_dir.exists()
        assert (expected_dir / "12345.webp").exists()

    def test_overwrite_existing(
        self, store: DecodedImageStore, sample_image: Image.Image
    ) -> None:
        store.put(1, sample_image, {"v": 1})
        store.put(1, sample_image, {"v": 2})
        assert store.entry_count == 1
        result = store.get(1)
        assert result is not None
        _, meta = result
        assert meta["v"] == 2


class TestLRUEviction:
    def test_evict_lru(self, tmp_path: Path) -> None:
        store = DecodedImageStore(
            cache_dir=tmp_path / "decoded",
            max_bytes=100 * 1024,  # 100 KB — very small
            resolution=64,
            fmt="webp",
            quality=50,
        )
        img = Image.new("RGB", (200, 200), color=(100, 100, 100))

        # Fill cache with several images
        for i in range(20):
            store.put(i, img)
            time.sleep(0.01)  # Ensure different timestamps

        # Access some images to make them recently used
        store.get(15)
        store.get(16)
        store.get(17)

        initial_size = store.size_bytes
        freed = store.evict_lru(target_bytes=initial_size // 2)
        assert freed > 0
        assert store.size_bytes <= initial_size // 2

        # Recently accessed images should survive
        assert store.has(15)
        assert store.has(16)
        assert store.has(17)

    def test_auto_eviction_on_put(self, tmp_path: Path) -> None:
        store = DecodedImageStore(
            cache_dir=tmp_path / "decoded",
            max_bytes=2 * 1024,  # 2 KB — very small
            resolution=64,
            fmt="jpeg",
            quality=90,
        )
        img = Image.new("RGB", (200, 200), color=(200, 150, 100))

        for i in range(30):
            store.put(i, img)
            time.sleep(0.01)

        # Wait for async eviction thread
        time.sleep(1.0)

        # Should have evicted old entries to stay under limit
        assert store.entry_count < 30


class TestDecodeLock:
    def test_same_lock_returned(self, store: DecodedImageStore) -> None:
        lock1 = store.get_decode_lock(42)
        lock2 = store.get_decode_lock(42)
        assert lock1 is lock2

    def test_different_ids_different_locks(
        self, store: DecodedImageStore
    ) -> None:
        lock1 = store.get_decode_lock(1)
        lock2 = store.get_decode_lock(2)
        assert lock1 is not lock2

    def test_concurrent_decode_serialized(
        self, store: DecodedImageStore
    ) -> None:
        results: list[int] = []
        img = Image.new("RGB", (100, 100))

        def decode_and_put(thread_id: int) -> None:
            lock = store.get_decode_lock(99)
            with lock:
                if not store.has(99):
                    time.sleep(0.05)
                    store.put(99, img)
                    results.append(thread_id)

        threads = [
            threading.Thread(target=decode_and_put, args=(i,))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Only one thread should have done the actual put
        assert len(results) == 1
        assert store.has(99)


class TestThreadSafety:
    def test_concurrent_put_get(
        self, store: DecodedImageStore
    ) -> None:
        errors: list[Exception] = []
        img = Image.new("RGB", (100, 100), color=(50, 50, 50))

        def writer(start_id: int) -> None:
            try:
                for i in range(start_id, start_id + 10):
                    store.put(i, img, {"id": i})
            except Exception as e:
                errors.append(e)

        def reader(start_id: int) -> None:
            try:
                for i in range(start_id, start_id + 10):
                    store.get(i)  # May be None if not yet written
            except Exception as e:
                errors.append(e)

        threads = []
        for base in range(0, 50, 10):
            threads.append(threading.Thread(target=writer, args=(base,)))
            threads.append(threading.Thread(target=reader, args=(base,)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestRebuildIndex:
    def test_rebuild_after_external_changes(
        self, store: DecodedImageStore, sample_image: Image.Image
    ) -> None:
        store.put(1, sample_image)
        store.put(2, sample_image)
        assert store.entry_count == 2

        # Simulate clearing the index
        store._conn.execute("DELETE FROM entries")
        store._conn.commit()

        count = store.rebuild_index()
        assert count == 2
        assert store.has(1)
        assert store.has(2)


class TestJPEGFormat:
    def test_jpeg_store(self, tmp_path: Path) -> None:
        store = DecodedImageStore(
            cache_dir=tmp_path / "decoded",
            max_bytes=10 * 1024 * 1024,
            resolution=256,
            fmt="jpeg",
            quality=90,
        )
        img = Image.new("RGB", (300, 200))
        store.put(1, img)
        assert store.has(1)

        result = store.get(1)
        assert result is not None
        loaded, _ = result
        assert loaded.mode == "RGB"

        # File should be JPEG
        img_path = store._image_path(1)
        assert img_path.suffix == ".jpeg"
        data = img_path.read_bytes()
        assert data[:2] == b"\xff\xd8"  # JPEG magic bytes
