"""Tests for the cache module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_eval.cache import CACHE_DIR, clear_cache, get_cached, store_result


@pytest.fixture(autouse=True)
def _clean_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect cache to a temp directory for testing."""
    test_cache = tmp_path / ".cache"
    monkeypatch.setattr("model_eval.cache.CACHE_DIR", test_cache)


def _make_image(tmp_path: Path, name: str = "test.jpg") -> Path:
    """Create a dummy image file."""
    p = tmp_path / name
    p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # Minimal JPEG header
    return p


class TestCache:
    def test_store_and_retrieve(self, tmp_path: Path) -> None:
        img = _make_image(tmp_path)
        result = {"score": 7.5, "model": "test-model"}

        store_result("test-model", img, result)
        cached = get_cached("test-model", img)

        assert cached is not None
        assert cached["score"] == 7.5
        assert cached["model"] == "test-model"

    def test_cache_miss(self, tmp_path: Path) -> None:
        img = _make_image(tmp_path)
        assert get_cached("nonexistent", img) is None

    def test_different_models_different_keys(self, tmp_path: Path) -> None:
        img = _make_image(tmp_path)

        store_result("model-a", img, {"score": 1.0})
        store_result("model-b", img, {"score": 2.0})

        assert get_cached("model-a", img)["score"] == 1.0
        assert get_cached("model-b", img)["score"] == 2.0

    def test_clear_cache(self, tmp_path: Path) -> None:
        img = _make_image(tmp_path)
        store_result("model-x", img, {"score": 5.0})

        count = clear_cache()
        assert count == 1
        assert get_cached("model-x", img) is None

    def test_clear_empty_cache(self) -> None:
        count = clear_cache()
        assert count == 0
