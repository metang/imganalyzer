"""Regression tests for the cached SearchEngine in server.py (perf bug B1).

The cache must survive across different thread-local SQLite connections
(it is keyed by DB path, not by connection identity) so that the ~1.5 GB
embedding matrix is not rebuilt on every RPC call.
"""
from __future__ import annotations

import sqlite3
import sys

import imganalyzer.server as server

# server.py redirects stdout for JSON-RPC; restore it for test output.
sys.stdout = server._real_stdout


def _fresh_conn(tmp_path) -> sqlite3.Connection:
    """Return a new sqlite connection to the same (empty) test DB."""
    db_path = tmp_path / "imganalyzer.db"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    # SearchEngine only needs an ``embeddings`` table to exist; create a
    # minimal schema that lets ``_EmbeddingMatrix.get()`` execute.
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            image_id INTEGER,
            embedding_type TEXT,
            vector BLOB
        );
        """
    )
    return conn


def test_search_engine_cached_across_connections(tmp_path, monkeypatch):
    """Two consecutive ``_get_search_engine`` calls with *different* connection
    objects must return the **same** SearchEngine instance, and the embedding
    matrix caches must remain the same objects (not rebuilt)."""
    # Point the cache key at an isolated DB.
    monkeypatch.setenv("IMGANALYZER_DB_PATH", str(tmp_path / "imganalyzer.db"))
    # Clear any previously-cached engine from other tests.
    server._cached_search_engine = None
    server._cached_search_engine_key = None

    conn1 = _fresh_conn(tmp_path)
    conn2 = _fresh_conn(tmp_path)
    assert conn1 is not conn2

    try:
        engine1 = server._get_search_engine(conn1)
        try:
            # The first acquisition must have swapped ``engine.conn`` to conn1.
            assert engine1.conn is conn1
            image_cache_before = engine1._image_clip_cache
            desc_cache_before = engine1._desc_clip_cache
        finally:
            server._release_search_engine()

        engine2 = server._get_search_engine(conn2)
        try:
            assert engine2 is engine1, "SearchEngine should be cached across connections"
            assert engine2.conn is conn2, "engine.conn should be swapped to the caller's connection"
            assert engine2.repo.conn is conn2, "repo.conn should be swapped too"
            # Embedding matrix caches must be the SAME objects — not rebuilt.
            assert engine2._image_clip_cache is image_cache_before
            assert engine2._desc_clip_cache is desc_cache_before
        finally:
            server._release_search_engine()
    finally:
        conn1.close()
        conn2.close()
        server._cached_search_engine = None
        server._cached_search_engine_key = None


def test_search_engine_ctx_releases_lock(tmp_path, monkeypatch):
    """The context manager must release the lock even when the body raises."""
    monkeypatch.setenv("IMGANALYZER_DB_PATH", str(tmp_path / "imganalyzer.db"))
    server._cached_search_engine = None
    server._cached_search_engine_key = None

    conn = _fresh_conn(tmp_path)
    try:
        try:
            with server._search_engine_ctx(conn) as engine:
                assert engine.conn is conn
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        # Lock must now be free — a subsequent acquire must succeed immediately.
        assert server._search_engine_lock.acquire(timeout=1.0)
        server._search_engine_lock.release()
    finally:
        conn.close()
        server._cached_search_engine = None
        server._cached_search_engine_key = None
