"""Database connection management — SQLite with WAL mode."""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

_DEFAULT_DB_PATH = Path.home() / ".cache" / "imganalyzer" / "imganalyzer.db"

_connection: sqlite3.Connection | None = None


def get_db_path() -> Path:
    """Return the database file path from env or default."""
    raw = os.getenv("IMGANALYZER_DB_PATH", "")
    if raw:
        return Path(raw).expanduser()
    return _DEFAULT_DB_PATH


def get_db(path: Path | None = None) -> sqlite3.Connection:
    """Return a singleton SQLite connection (creates DB + schema on first call).

    WAL mode is enabled for concurrent read/write performance.
    """
    global _connection
    if _connection is not None:
        return _connection

    db_path = path or get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # isolation_level=None disables Python's implicit transaction management
    # so we can use explicit BEGIN IMMEDIATE / COMMIT in our _transaction ctx mgr.
    conn = sqlite3.connect(str(db_path), timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")

    # Run schema migrations
    from imganalyzer.db.schema import ensure_schema
    ensure_schema(conn)

    _connection = conn
    return conn


def close_db() -> None:
    """Close the singleton connection if open."""
    global _connection
    if _connection is not None:
        _connection.close()
        _connection = None
