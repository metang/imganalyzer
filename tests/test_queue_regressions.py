from __future__ import annotations

import sqlite3
from pathlib import Path


def _make_test_db(tmp_path: Path) -> sqlite3.Connection:
    from imganalyzer.db.schema import ensure_schema

    db_path = tmp_path / "queue-regressions.db"
    conn = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    ensure_schema(conn)
    return conn


def test_recover_stale_requeues_failures_and_removes_non_running_leases(
    tmp_path: Path,
) -> None:
    from imganalyzer.db.queue import JobQueue
    from imganalyzer.db.repository import Repository

    conn = _make_test_db(tmp_path)
    repo = Repository(conn)
    queue = JobQueue(conn)

    conn.execute(
        """INSERT INTO worker_nodes (id, display_name, platform, status)
           VALUES (?, ?, ?, ?)""",
        ("worker-1", "Worker 1", "linux", "online"),
    )
    retry_image_id = repo.register_image(file_path="/photos/stale-retry.jpg")
    fail_image_id = repo.register_image(file_path="/photos/stale-fail.jpg")
    retry_job_id = queue.enqueue(retry_image_id, "objects")
    fail_job_id = queue.enqueue(fail_image_id, "caption")
    assert retry_job_id is not None
    assert fail_job_id is not None

    claimed = queue.claim_leased(worker_id="worker-1", lease_ttl_seconds=120, batch_size=2)
    assert {int(job["id"]) for job in claimed} == {retry_job_id, fail_job_id}
    conn.execute(
        "UPDATE job_queue SET attempts = 3, max_attempts = 3 WHERE id = ?",
        (fail_job_id,),
    )
    conn.commit()

    recovered = queue.recover_stale(timeout_minutes=0)

    assert recovered == 1
    rows = {
        int(row["id"]): row
        for row in conn.execute(
            "SELECT id, status, attempts, error_message FROM job_queue ORDER BY id"
        ).fetchall()
    }
    assert rows[retry_job_id]["status"] == "pending"
    assert rows[retry_job_id]["attempts"] == 1
    assert rows[fail_job_id]["status"] == "failed"
    assert rows[fail_job_id]["attempts"] == 4
    assert "stale recovery" in rows[fail_job_id]["error_message"]

    lease_count = conn.execute("SELECT COUNT(*) AS cnt FROM job_leases").fetchone()
    assert lease_count["cnt"] == 0
