"""Tests for the ``decode/enqueue_missing`` RPC handler."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from imganalyzer.cache.decoded_store import DecodedImageStore
from imganalyzer.db.schema import ensure_schema


def _seed_images(conn: sqlite3.Connection, paths: list[str]) -> list[int]:
    ids: list[int] = []
    for p in paths:
        cur = conn.execute(
            "INSERT INTO images (file_path, file_hash) VALUES (?, ?)",
            (p, f"hash_{p}"),
        )
        ids.append(int(cur.lastrowid))
    conn.commit()
    return ids


def _put(store: DecodedImageStore, image_id: int) -> None:
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    store.put(image_id, img, metadata={"width": 8, "height": 8})


def test_enqueue_missing_feeds_only_uncached(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)

    ids = _seed_images(conn, ["/a.jpg", "/b.jpg", "/c.jpg"])
    conn.close()

    store = DecodedImageStore(cache_dir=tmp_path / "cache", max_bytes=10 * 1024 * 1024)
    _put(store, ids[0])  # only the first image is cached

    fake_decoder = MagicMock()
    fake_decoder.feed.return_value = 2

    from imganalyzer import server

    with patch.object(server, "_get_decoded_store", return_value=store), \
         patch.object(server, "_get_pre_decoder", return_value=fake_decoder), \
         patch.object(server, "create_connection", create=True), \
         patch("imganalyzer.db.connection.create_connection") as cc:
        cc.return_value = sqlite3.connect(str(db_path), check_same_thread=False)
        cc.return_value.row_factory = sqlite3.Row
        result = server._handle_decode_enqueue_missing({})

    assert result["total_images"] == 3
    assert result["cached"] == 1
    assert result["missing"] == 2
    assert result["requested"] == 2
    assert result["fed"] == 2

    fake_decoder.feed.assert_called_once()
    fed_items = fake_decoder.feed.call_args[0][0]
    fed_ids = sorted(iid for iid, _ in fed_items)
    assert fed_ids == sorted([ids[1], ids[2]])


def test_enqueue_missing_respects_limit(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    _seed_images(conn, ["/a.jpg", "/b.jpg", "/c.jpg", "/d.jpg"])
    conn.close()

    store = DecodedImageStore(cache_dir=tmp_path / "cache", max_bytes=10 * 1024 * 1024)
    fake_decoder = MagicMock()
    fake_decoder.feed.return_value = 2

    from imganalyzer import server

    with patch.object(server, "_get_decoded_store", return_value=store), \
         patch.object(server, "_get_pre_decoder", return_value=fake_decoder), \
         patch("imganalyzer.db.connection.create_connection") as cc:
        cc.return_value = sqlite3.connect(str(db_path), check_same_thread=False)
        cc.return_value.row_factory = sqlite3.Row
        result = server._handle_decode_enqueue_missing({"limit": 2})

    assert result["total_images"] == 4
    assert result["missing"] == 4
    assert result["requested"] == 2
    fed_items = fake_decoder.feed.call_args[0][0]
    assert len(fed_items) == 2


def test_enqueue_missing_when_all_cached(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    ids = _seed_images(conn, ["/a.jpg", "/b.jpg"])
    conn.close()

    store = DecodedImageStore(cache_dir=tmp_path / "cache", max_bytes=10 * 1024 * 1024)
    for iid in ids:
        _put(store, iid)

    fake_decoder = MagicMock()
    from imganalyzer import server

    with patch.object(server, "_get_decoded_store", return_value=store), \
         patch.object(server, "_get_pre_decoder", return_value=fake_decoder), \
         patch("imganalyzer.db.connection.create_connection") as cc:
        cc.return_value = sqlite3.connect(str(db_path), check_same_thread=False)
        cc.return_value.row_factory = sqlite3.Row
        result = server._handle_decode_enqueue_missing({})

    assert result["missing"] == 0
    assert result["fed"] == 0
    fake_decoder.feed.assert_not_called()
