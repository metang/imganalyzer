from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from imganalyzer.rpc.handler_registry import RpcHandlerRegistry
from imganalyzer.rpc.search_engine_cache import SearchEngineCache


def test_rpc_handler_registry_retries_configured_transient_errors() -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(params: dict[str, Any]) -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise sqlite3.OperationalError("database is locked")
        return {"ok": params["ok"]}

    registry = RpcHandlerRegistry(
        sync_methods={"status": handler},
        async_methods={},
        stdio_async_methods={},
        lock_retryable_methods={"status"},
        is_transient_lock_error=lambda exc: isinstance(exc, sqlite3.OperationalError),
        retry_attempts=4,
        retry_initial_delay_s=0.15,
        sleeper=sleeps.append,
    )

    assert registry.call_sync("status", {"ok": True}) == {"ok": True}
    assert attempts == 3
    assert sleeps == [0.15, 0.3]


def test_rpc_handler_registry_does_not_retry_non_retryable_methods() -> None:
    attempts = 0

    def handler(_params: dict[str, Any]) -> None:
        nonlocal attempts
        attempts += 1
        raise sqlite3.OperationalError("database is locked")

    registry = RpcHandlerRegistry(
        sync_methods={"search": handler},
        async_methods={},
        stdio_async_methods={},
        lock_retryable_methods={"status"},
        is_transient_lock_error=lambda exc: True,
        retry_attempts=4,
        retry_initial_delay_s=0.15,
    )

    with pytest.raises(sqlite3.OperationalError):
        registry.call_sync("search", {})
    assert attempts == 1


def test_search_engine_cache_reuses_engine_and_rebinds_connections(tmp_path: Path) -> None:
    class FakeSearchEngine:
        def __init__(self, conn: object) -> None:
            self.conn = conn
            self.repo = SimpleNamespace(conn=conn)

    search_module = SimpleNamespace(SearchEngine=FakeSearchEngine)
    cache = SearchEngineCache(
        search_module_loader=lambda: search_module,  # type: ignore[arg-type]
        db_path_getter=lambda: tmp_path / "imganalyzer.db",
    )
    conn1 = object()
    conn2 = object()

    engine1 = cache.acquire(conn1)
    try:
        assert engine1.conn is conn1
        assert engine1.repo.conn is conn1
    finally:
        cache.release()

    engine2 = cache.acquire(conn2)
    try:
        assert engine2 is engine1
        assert engine2.conn is conn2
        assert engine2.repo.conn is conn2
    finally:
        cache.release()


def test_search_engine_cache_resets_when_search_class_changes(tmp_path: Path) -> None:
    class FirstSearchEngine:
        def __init__(self, conn: object) -> None:
            self.conn = conn

    class SecondSearchEngine:
        def __init__(self, conn: object) -> None:
            self.conn = conn

    search_module = SimpleNamespace(SearchEngine=FirstSearchEngine)
    cache = SearchEngineCache(
        search_module_loader=lambda: search_module,  # type: ignore[arg-type]
        db_path_getter=lambda: tmp_path / "imganalyzer.db",
    )

    first = cache.acquire(object())
    cache.release()

    search_module.SearchEngine = SecondSearchEngine
    second = cache.acquire(object())
    cache.release()

    assert second is not first
    assert isinstance(second, SecondSearchEngine)
