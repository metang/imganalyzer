from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any


def test_create_connection_enables_wal_foreign_keys_and_cross_thread_use(
    tmp_path: Path,
) -> None:
    from imganalyzer.db.connection import create_connection

    conn = create_connection(tmp_path / "threaded.db", busy_timeout_ms=1234)
    try:
        assert conn.row_factory is sqlite3.Row
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 1234

        conn.execute("CREATE TABLE smoke (value TEXT NOT NULL)")
        errors: list[Exception] = []
        values: list[str] = []

        def use_from_worker_thread() -> None:
            try:
                conn.execute("INSERT INTO smoke (value) VALUES (?)", ("ok",))
                row = conn.execute("SELECT value FROM smoke").fetchone()
                values.append(str(row["value"]))
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=use_from_worker_thread)
        thread.start()
        thread.join(timeout=5)

        assert not thread.is_alive()
        assert errors == []
        assert values == ["ok"]
    finally:
        conn.close()


def test_begin_immediate_retries_transient_operational_errors(
    monkeypatch,
) -> None:
    from imganalyzer.db import connection

    class FlakyConnection:
        def __init__(self) -> None:
            self.attempts = 0

        def execute(self, sql: str) -> Any:
            assert sql == "BEGIN IMMEDIATE"
            self.attempts += 1
            if self.attempts < 3:
                raise sqlite3.OperationalError("database is locked")
            return None

    sleeps: list[float] = []
    monkeypatch.setattr(connection._time, "sleep", sleeps.append)

    conn = FlakyConnection()
    connection.begin_immediate(conn)  # type: ignore[arg-type]

    assert conn.attempts == 3
    assert sleeps == [
        connection._BEGIN_RETRY_DELAY,
        connection._BEGIN_RETRY_DELAY * 2,
    ]
