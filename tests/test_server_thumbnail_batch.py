from __future__ import annotations

import sqlite3
import sys
from concurrent.futures import Future
from typing import Any

import pytest

import imganalyzer.server as server

# server.py redirects stdout at import time for JSON-RPC mode; restore test stdout.
sys.stdout = server._real_stdout


class _NoCacheStore:
    def has(self, _image_id: int) -> bool:
        return False

    def get_image_bytes(self, _image_id: int) -> bytes | None:
        return None


class _CorruptCacheStore:
    def has(self, _image_id: int) -> bool:
        return True

    def get_image_bytes(self, _image_id: int) -> bytes | None:
        return b"not a real image"


class _NoopDecoder:
    def feed_priority(self, _items: list[tuple[int, str]]) -> None:
        return None


class _ImmediatePool:
    def submit(self, fn: Any, *args: Any, **kwargs: Any) -> Future:
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:
            future.set_exception(exc)
        return future

    def map(self, fn: Any, items: list[tuple[str, int]]) -> list[Any]:
        return [fn(item) for item in items]


@pytest.fixture()
def thumbnail_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE images (id INTEGER PRIMARY KEY, file_path TEXT NOT NULL)")
    try:
        yield conn
    finally:
        conn.close()


def test_thumbnails_batch_resolves_id_only_items_against_current_db_path(
    thumbnail_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_path = r"E:\Pic\renamed\current.jpg"
    thumbnail_db.execute(
        "INSERT INTO images (id, file_path) VALUES (?, ?)",
        (7, current_path),
    )
    captured_paths: list[str] = []

    def fake_thumbnail(params: dict[str, str]) -> dict[str, str]:
        captured_paths.append(params["imagePath"])
        return {"data": "ZmFrZQ=="}

    monkeypatch.setattr(server, "_get_db", lambda: thumbnail_db)
    monkeypatch.setattr(server, "_get_decoded_store", lambda: _NoCacheStore())
    monkeypatch.setattr(server, "_get_pre_decoder", lambda: _NoopDecoder())
    monkeypatch.setattr(server, "_get_thumb_slow_pool", lambda: _ImmediatePool())
    monkeypatch.setattr(server, "_handle_thumbnail", fake_thumbnail)

    result = server._handle_thumbnails_batch({"items": [{"image_id": 7}]})

    assert captured_paths == [current_path]
    assert result == {
        "thumbnails": {"7": "data:image/jpeg;base64,ZmFrZQ=="},
        "errors": {},
    }


def test_thumbnails_batch_reports_corrupt_cached_bytes_without_slow_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = r"E:\Pic\cached\broken.jpg"

    def fail_slow_pool() -> _ImmediatePool:
        raise AssertionError("corrupt cached bytes should not enter the slow path")

    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_get_decoded_store", lambda: _CorruptCacheStore())
    monkeypatch.setattr(server, "_get_thumb_cached_pool", lambda: _ImmediatePool())
    monkeypatch.setattr(server, "_get_thumb_slow_pool", fail_slow_pool)

    result = server._handle_thumbnails_batch(
        {"items": [{"image_id": 3, "file_path": image_path}]}
    )

    assert result["thumbnails"] == {}
    assert set(result["errors"]) == {image_path}
    assert result["errors"][image_path]
