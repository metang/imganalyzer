"""Job queue — SQLite-backed task queue with atomic claim, retry, pause/resume.

The queue stores one row per (image, module) combination.  Workers claim jobs
atomically using UPDATE ... RETURNING to avoid double-processing.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class JobQueue:
    """Priority queue backed by the ``job_queue`` table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    # ── Enqueue ────────────────────────────────────────────────────────────

    def enqueue(
        self,
        image_id: int,
        module: str,
        priority: int = 0,
        force: bool = False,
        _auto_commit: bool = True,
    ) -> int | None:
        """Add a job unless an identical pending/running/done job exists.

        With *force=True*, re-enqueue even if a ``done`` job exists (for
        rebuilds).  Returns the job_id or None if skipped.

        When *force=False* a ``failed`` or ``skipped`` job row is also treated
        as "already handled" — it is only re-enqueued when force is True.
        This prevents normal ingest runs from re-queuing jobs that were
        intentionally skipped (e.g. has_people) or failed but whose analysis
        data was since populated by another means.

        When *_auto_commit* is False, the caller is responsible for committing
        (used by batched ingest to avoid per-job fsync overhead).
        """
        existing = self.conn.execute(
            "SELECT id, status FROM job_queue WHERE image_id = ? AND module = ?",
            [image_id, module],
        ).fetchone()

        if existing:
            if existing["status"] in ("pending", "running"):
                return None  # already queued
            if not force:
                # done / failed / skipped — don't re-enqueue without explicit force
                return None
            # force=True: reset the row back to pending
            self.conn.execute(
                """UPDATE job_queue
                   SET status = 'pending', attempts = 0, error_message = NULL,
                       skip_reason = NULL, started_at = NULL, completed_at = NULL,
                       queued_at = ?, priority = ?
                   WHERE id = ?""",
                [_now(), priority, existing["id"]],
            )
            if _auto_commit:
                self.conn.commit()
            return existing["id"]

        cur = self.conn.execute(
            """INSERT INTO job_queue (image_id, module, priority, status, queued_at)
               VALUES (?, ?, ?, 'pending', ?)""",
            [image_id, module, priority, _now()],
        )
        if _auto_commit:
            self.conn.commit()
        return cur.lastrowid

    def enqueue_batch(
        self,
        image_ids: list[int],
        modules: list[str],
        priority: int = 0,
        force: bool = False,
    ) -> int:
        """Enqueue multiple (image, module) pairs.  Returns count enqueued."""
        count = 0
        for image_id in image_ids:
            for module in modules:
                job_id = self.enqueue(image_id, module, priority=priority, force=force)
                if job_id is not None:
                    count += 1
        return count

    # ── Claim (atomic) ─────────────────────────────────────────────────────

    def claim(self, batch_size: int = 1, module: str | None = None) -> list[dict[str, Any]]:
        """Atomically claim up to *batch_size* pending jobs.

        Returns list of job dicts with keys: id, image_id, module, attempts.
        Jobs are returned in priority order (highest first, then oldest).
        """
        where = "WHERE status = 'pending'"
        params: list[Any] = []
        if module:
            where += " AND module = ?"
            params.append(module)
        params.append(batch_size)

        # Use BEGIN IMMEDIATE to prevent concurrent claims from selecting
        # the same jobs (atomic SELECT + UPDATE).
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            rows = self.conn.execute(
                f"""SELECT id, image_id, module, attempts
                    FROM job_queue
                    {where}
                    ORDER BY priority DESC, queued_at ASC
                    LIMIT ?""",
                params,
            ).fetchall()

            if not rows:
                self.conn.rollback()
                return []

            job_ids = [r["id"] for r in rows]
            placeholders = ",".join("?" * len(job_ids))
            self.conn.execute(
                f"""UPDATE job_queue
                    SET status = 'running', started_at = ?
                    WHERE id IN ({placeholders})""",
                [_now()] + job_ids,
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        return [dict(r) for r in rows]

    # ── Complete / Fail / Skip ─────────────────────────────────────────────

    def mark_done(self, job_id: int) -> None:
        self.conn.execute(
            "UPDATE job_queue SET status = 'done', completed_at = ? WHERE id = ?",
            [_now(), job_id],
        )
        self.conn.commit()

    def mark_failed(self, job_id: int, error: str) -> None:
        self.conn.execute(
            """UPDATE job_queue
               SET status = 'failed', error_message = ?,
                   attempts = attempts + 1, completed_at = ?
               WHERE id = ?""",
            [error, _now(), job_id],
        )
        self.conn.commit()

    def mark_skipped(self, job_id: int, reason: str) -> None:
        self.conn.execute(
            """UPDATE job_queue
               SET status = 'skipped', skip_reason = ?, completed_at = ?
               WHERE id = ?""",
            [reason, _now(), job_id],
        )
        self.conn.commit()

    def mark_pending(self, job_id: int) -> None:
        """Reset a claimed (running) job back to pending.

        Used during shutdown when a job was claimed but could not be
        submitted to the thread pool — ensures it is retried next run.
        """
        self.conn.execute(
            """UPDATE job_queue
               SET status = 'pending', started_at = NULL
               WHERE id = ?""",
            [job_id],
        )
        self.conn.commit()

    # ── Retry failed jobs ──────────────────────────────────────────────────

    def retry_failed(self, max_attempts: int = 3) -> int:
        """Re-enqueue failed jobs that haven't exceeded max_attempts."""
        cur = self.conn.execute(
            """UPDATE job_queue
               SET status = 'pending', error_message = NULL,
                   started_at = NULL, completed_at = NULL
               WHERE status = 'failed' AND attempts < ?""",
            [max_attempts],
        )
        self.conn.commit()
        return cur.rowcount

    # ── Recover stale running jobs (crash recovery) ────────────────────────

    def recover_stale(self, timeout_minutes: int = 10) -> int:
        """Reset jobs stuck in 'running' state (e.g. after a crash)."""
        cur = self.conn.execute(
            """UPDATE job_queue
               SET status = 'pending', attempts = attempts + 1
               WHERE status = 'running'
               AND started_at < datetime('now', '-' || ? || ' minutes')""",
            [timeout_minutes],
        )
        self.conn.commit()
        return cur.rowcount

    # ── Status / stats ─────────────────────────────────────────────────────

    def stats(self) -> dict[str, dict[str, int]]:
        """Return {module: {status: count}} for all modules."""
        rows = self.conn.execute(
            """SELECT module, status, COUNT(*) as cnt
               FROM job_queue
               GROUP BY module, status
               ORDER BY module, status"""
        ).fetchall()
        result: dict[str, dict[str, int]] = {}
        for r in rows:
            result.setdefault(r["module"], {})[r["status"]] = r["cnt"]
        return result

    def total_stats(self) -> dict[str, int]:
        """Return {status: count} across all modules."""
        rows = self.conn.execute(
            "SELECT status, COUNT(*) as cnt FROM job_queue GROUP BY status"
        ).fetchall()
        return {r["status"]: r["cnt"] for r in rows}

    def pending_count(self, module: str | None = None) -> int:
        if module:
            row = self.conn.execute(
                "SELECT COUNT(*) as cnt FROM job_queue WHERE status = 'pending' AND module = ?",
                [module],
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT COUNT(*) as cnt FROM job_queue WHERE status = 'pending'"
            ).fetchone()
        return row["cnt"]

    # ── Cleanup ────────────────────────────────────────────────────────────

    def clear_module(self, module: str) -> int:
        """Delete all jobs for a module (used before rebuild re-enqueue)."""
        cur = self.conn.execute(
            "DELETE FROM job_queue WHERE module = ?", [module]
        )
        self.conn.commit()
        return cur.rowcount

    def clear_all(self) -> int:
        cur = self.conn.execute("DELETE FROM job_queue")
        self.conn.commit()
        return cur.rowcount

    def clear_by_folder(
        self,
        folder_prefix: str,
        statuses: list[str] | None = None,
    ) -> int:
        """Delete jobs for images whose file_path starts with *folder_prefix*.

        If *statuses* is given, only jobs with those statuses are deleted
        (e.g. ``['pending', 'running']``).  Defaults to all statuses.
        """
        prefix = folder_prefix.rstrip("/\\") + "%"
        if statuses:
            placeholders = ",".join("?" * len(statuses))
            cur = self.conn.execute(
                f"""DELETE FROM job_queue
                    WHERE status IN ({placeholders})
                    AND image_id IN (
                        SELECT id FROM images WHERE file_path LIKE ?
                    )""",
                [*statuses, prefix],
            )
        else:
            cur = self.conn.execute(
                """DELETE FROM job_queue
                   WHERE image_id IN (
                       SELECT id FROM images WHERE file_path LIKE ?
                   )""",
                [prefix],
            )
        self.conn.commit()
        return cur.rowcount
