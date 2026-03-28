"""Database connection management — SQLite with WAL mode."""
from __future__ import annotations

import os
import sqlite3
import time as _time
from pathlib import Path

_DEFAULT_DB_PATH = Path.home() / ".cache" / "imganalyzer" / "imganalyzer.db"

_connection: sqlite3.Connection | None = None

# Retry parameters for BEGIN IMMEDIATE during WAL auto-checkpoints.
# Checkpoints bypass busy_timeout, causing immediate SQLITE_BUSY.
_BEGIN_MAX_RETRIES = 3
_BEGIN_RETRY_DELAY = 0.05  # 50ms base


def begin_immediate(conn: sqlite3.Connection) -> None:
    """Execute ``BEGIN IMMEDIATE`` with retry for checkpoint-triggered SQLITE_BUSY.

    SQLite WAL auto-checkpoints bypass ``busy_timeout`` entirely, causing
    immediate ``SQLITE_BUSY`` errors.  This helper retries a few times with
    brief sleeps to ride out the checkpoint.
    """
    for attempt in range(1, _BEGIN_MAX_RETRIES + 1):
        try:
            conn.execute("BEGIN IMMEDIATE")
            return
        except sqlite3.OperationalError:
            if attempt >= _BEGIN_MAX_RETRIES:
                raise
            _time.sleep(_BEGIN_RETRY_DELAY * attempt)


def get_db_path() -> Path:
    """Return the database file path from env or default."""
    raw = os.getenv("IMGANALYZER_DB_PATH", "")
    if raw:
        return Path(raw).expanduser()
    return _DEFAULT_DB_PATH


def create_connection(
    path: Path | None = None,
    busy_timeout_ms: int = 30000,
) -> sqlite3.Connection:
    """Create a properly configured SQLite connection for worker/server use.

    All connections get WAL mode, NORMAL sync, foreign keys, and the
    specified busy_timeout.  Uses ``isolation_level=None`` for explicit
    transaction control.
    """
    db_path = path or get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(db_path),
        timeout=busy_timeout_ms / 1000,
        isolation_level=None,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
    # Raise autocheckpoint threshold to reduce checkpoint-triggered
    # SQLITE_BUSY errors (checkpoints bypass busy_timeout entirely).
    # Default is 1000 WAL frames; 2000 halves checkpoint frequency
    # during heavy batch processing at the cost of a larger WAL file.
    conn.execute("PRAGMA wal_autocheckpoint=2000")
    return conn


def get_db(path: Path | None = None) -> sqlite3.Connection:
    """Return a singleton SQLite connection (creates DB + schema on first call).

    WAL mode is enabled for concurrent read/write performance.
    """
    global _connection
    if _connection is not None:
        return _connection

    # CLI connections use shorter busy timeout
    conn = create_connection(path=path, busy_timeout_ms=5000)

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
