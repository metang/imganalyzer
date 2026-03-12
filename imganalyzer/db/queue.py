"""Job queue — SQLite-backed task queue with atomic claim, retry, pause/resume.

The queue stores one row per (image, module) combination.  Workers claim jobs
atomically using UPDATE ... RETURNING to avoid double-processing.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _now_plus(seconds: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).strftime("%Y-%m-%d %H:%M:%S")


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
                       queued_at = ?, priority = ?,
                       last_node_id = NULL, last_node_role = NULL
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

    def get_pending_image_ids(
        self,
        modules: list[str] | None = None,
    ) -> list[int]:
        """Return distinct image_ids with pending jobs, ordered by image_id.

        If *modules* is given, only consider jobs for those modules.
        """
        where = "WHERE status = 'pending'"
        params: list[Any] = []
        if modules:
            placeholders = ",".join("?" * len(modules))
            where += f" AND module IN ({placeholders})"
            params.extend(modules)
        rows = self.conn.execute(
            f"SELECT DISTINCT image_id FROM job_queue {where} ORDER BY image_id",
            params,
        ).fetchall()
        return [r["image_id"] for r in rows]

    def claim(
        self,
        batch_size: int = 1,
        module: str | None = None,
        node_id: str = "master",
        node_role: str = "master",
        image_ids: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Atomically claim up to *batch_size* pending jobs.

        Returns list of job dicts with keys: id, image_id, module, attempts.
        Jobs are returned in priority order (highest first, then oldest).

        If *image_ids* is given, only claim jobs for those images (chunking).
        """
        where = "WHERE status = 'pending'"
        params: list[Any] = []
        if module:
            where += " AND module = ?"
            params.append(module)
        if image_ids is not None:
            placeholders = ",".join("?" * len(image_ids))
            where += f" AND image_id IN ({placeholders})"
            params.extend(image_ids)
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
                    SET status = 'running', started_at = ?,
                        last_node_id = ?, last_node_role = ?
                    WHERE id IN ({placeholders})""",
                [_now(), node_id, node_role] + job_ids,
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        return [dict(r) for r in rows]

    def claim_leased(
        self,
        worker_id: str,
        lease_ttl_seconds: int = 120,
        batch_size: int = 1,
        module: str | None = None,
        modules: list[str] | None = None,
        exclude_modules: list[str] | None = None,
        prefer_module: str | None = None,
    ) -> list[dict[str, Any]]:
        """Atomically claim jobs and create leases for distributed workers.

        If *prefer_module* is given, jobs for that module are sorted first
        (module affinity) to minimize model switching across claims.
        """
        where = "WHERE status = 'pending'"
        params: list[Any] = []
        if module:
            where += " AND module = ?"
            params.append(module)
        elif modules:
            placeholders = ",".join("?" * len(modules))
            where += f" AND module IN ({placeholders})"
            params.extend(modules)
        if exclude_modules:
            excl_ph = ",".join("?" * len(exclude_modules))
            where += f" AND module NOT IN ({excl_ph})"
            params.extend(exclude_modules)

        # Module affinity: prefer same module to minimize model switching
        if prefer_module and not module:
            order = "(CASE WHEN module = ? THEN 0 ELSE 1 END), priority DESC, queued_at ASC"
            params.append(prefer_module)
        else:
            order = "priority DESC, queued_at ASC"
        params.append(batch_size)

        self.conn.execute("BEGIN IMMEDIATE")
        try:
            rows = self.conn.execute(
                f"""SELECT id, image_id, module, attempts
                    FROM job_queue
                    {where}
                    ORDER BY {order}
                    LIMIT ?""",
                params,
            ).fetchall()
            if not rows:
                self.conn.rollback()
                return []

            now = _now()
            expires_at = _now_plus(max(1, lease_ttl_seconds))
            job_ids = [r["id"] for r in rows]
            placeholders = ",".join("?" * len(job_ids))
            self.conn.execute(
                f"""UPDATE job_queue
                    SET status = 'running', started_at = ?,
                        last_node_id = ?, last_node_role = ?
                    WHERE id IN ({placeholders})""",
                [now, worker_id, "worker"] + job_ids,
            )

            claimed: list[dict[str, Any]] = []
            for row in rows:
                token = str(uuid4())
                self.conn.execute(
                    """INSERT INTO job_leases
                       (job_id, worker_id, lease_token, leased_at, heartbeat_at, lease_expires_at)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ON CONFLICT(job_id) DO UPDATE SET
                         worker_id = excluded.worker_id,
                         lease_token = excluded.lease_token,
                         leased_at = excluded.leased_at,
                         heartbeat_at = excluded.heartbeat_at,
                         lease_expires_at = excluded.lease_expires_at""",
                    [row["id"], worker_id, token, now, now, expires_at],
                )
                item = dict(row)
                item["lease_token"] = token
                item["lease_expires_at"] = expires_at
                claimed.append(item)

            self.conn.commit()
            return claimed
        except Exception:
            self.conn.rollback()
            raise

    def release_expired_leases(self) -> int:
        """Return expired leased jobs to pending state."""
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            rows = self.conn.execute(
                """SELECT job_id FROM job_leases
                   WHERE lease_expires_at <= datetime('now')"""
            ).fetchall()
            if not rows:
                self.conn.rollback()
                return 0

            job_ids = [r["job_id"] for r in rows]
            placeholders = ",".join("?" * len(job_ids))
            cur = self.conn.execute(
                f"""UPDATE job_queue
                    SET status = 'pending', started_at = NULL, attempts = attempts + 1,
                        last_node_id = NULL, last_node_role = NULL
                    WHERE id IN ({placeholders})""",
                job_ids,
            )
            self.conn.execute(
                f"DELETE FROM job_leases WHERE job_id IN ({placeholders})",
                job_ids,
            )
            self.conn.commit()
            return cur.rowcount
        except Exception:
            self.conn.rollback()
            raise

    def release_worker_leases(self, worker_id: str) -> int:
        """Return all leases held by a worker to pending state."""
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            rows = self.conn.execute(
                "SELECT job_id FROM job_leases WHERE worker_id = ?",
                [worker_id],
            ).fetchall()
            if not rows:
                self.conn.rollback()
                return 0
            job_ids = [r["job_id"] for r in rows]
            placeholders = ",".join("?" * len(job_ids))
            cur = self.conn.execute(
                f"""UPDATE job_queue
                    SET status = 'pending', started_at = NULL, attempts = attempts + 1,
                        last_node_id = NULL, last_node_role = NULL
                    WHERE id IN ({placeholders})""",
                job_ids,
            )
            self.conn.execute(
                f"DELETE FROM job_leases WHERE job_id IN ({placeholders})",
                job_ids,
            )
            self.conn.commit()
            return cur.rowcount
        except Exception:
            self.conn.rollback()
            raise

    def heartbeat_lease(
        self,
        job_id: int,
        lease_token: str,
        extend_ttl_seconds: int = 120,
    ) -> bool:
        """Refresh a lease heartbeat and extend its expiry if the token matches."""
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            row = self.conn.execute(
                "SELECT 1 FROM job_leases WHERE job_id = ? AND lease_token = ?",
                [job_id, lease_token],
            ).fetchone()
            if row is None:
                self.conn.rollback()
                return False
            now = _now()
            expires_at = _now_plus(max(5, extend_ttl_seconds))
            self.conn.execute(
                """UPDATE job_leases
                   SET heartbeat_at = ?, lease_expires_at = ?
                   WHERE job_id = ?""",
                [now, expires_at, job_id],
            )
            self.conn.commit()
            return True
        except Exception:
            self.conn.rollback()
            raise

    def release_leased(self, job_id: int, lease_token: str) -> bool:
        """Return a claimed leased job to pending if the token matches."""
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            row = self.conn.execute(
                "SELECT 1 FROM job_leases WHERE job_id = ? AND lease_token = ?",
                [job_id, lease_token],
            ).fetchone()
            if row is None:
                self.conn.rollback()
                return False
            self.conn.execute(
                """UPDATE job_queue
                   SET status = 'pending',
                       started_at = NULL,
                       queued_at = ?,
                       attempts = attempts + 1,
                       last_node_id = NULL,
                       last_node_role = NULL
                    WHERE id = ?""",
                [_now(), job_id],
            )
            self.conn.execute("DELETE FROM job_leases WHERE job_id = ?", [job_id])
            self.conn.commit()
            return True
        except Exception:
            self.conn.rollback()
            raise

    def mark_done_leased(self, job_id: int, lease_token: str) -> bool:
        """Mark a leased job done if the lease token matches."""
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            row = self.conn.execute(
                "SELECT 1 FROM job_leases WHERE job_id = ? AND lease_token = ?",
                [job_id, lease_token],
            ).fetchone()
            if row is None:
                self.conn.rollback()
                return False
            self.conn.execute(
                "UPDATE job_queue SET status = 'done', completed_at = ? WHERE id = ?",
                [_now(), job_id],
            )
            self.conn.execute("DELETE FROM job_leases WHERE job_id = ?", [job_id])
            self.conn.commit()
            return True
        except Exception:
            self.conn.rollback()
            raise

    def mark_failed_leased(self, job_id: int, lease_token: str, error: str) -> bool:
        """Mark a leased job failed if the lease token matches."""
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            row = self.conn.execute(
                "SELECT 1 FROM job_leases WHERE job_id = ? AND lease_token = ?",
                [job_id, lease_token],
            ).fetchone()
            if row is None:
                self.conn.rollback()
                return False
            self.conn.execute(
                """UPDATE job_queue
                   SET status = 'failed', error_message = ?,
                       attempts = attempts + 1, completed_at = ?
                   WHERE id = ?""",
                [error, _now(), job_id],
            )
            self.conn.execute("DELETE FROM job_leases WHERE job_id = ?", [job_id])
            self.conn.commit()
            return True
        except Exception:
            self.conn.rollback()
            raise

    def mark_skipped_leased(self, job_id: int, lease_token: str, reason: str) -> bool:
        """Mark a leased job skipped if the lease token matches."""
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            row = self.conn.execute(
                "SELECT 1 FROM job_leases WHERE job_id = ? AND lease_token = ?",
                [job_id, lease_token],
            ).fetchone()
            if row is None:
                self.conn.rollback()
                return False
            self.conn.execute(
                """UPDATE job_queue
                   SET status = 'skipped', skip_reason = ?, completed_at = ?
                   WHERE id = ?""",
                [reason, _now(), job_id],
            )
            self.conn.execute("DELETE FROM job_leases WHERE job_id = ?", [job_id])
            self.conn.commit()
            return True
        except Exception:
            self.conn.rollback()
            raise

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

    def mark_image_pending_jobs_skipped(self, image_id: int, reason: str) -> int:
        """Skip pending jobs for an image after a fatal file-level failure."""
        cur = self.conn.execute(
            """UPDATE job_queue
               SET status = 'skipped', skip_reason = ?, completed_at = ?
               WHERE image_id = ? AND status = 'pending'""",
            [reason, _now(), image_id],
        )
        self.conn.commit()
        return cur.rowcount

    def mark_image_pending_modules_skipped(
        self,
        image_id: int,
        modules: list[str],
        reason: str,
    ) -> int:
        """Skip pending jobs for specific modules on one image."""
        if not modules:
            return 0
        placeholders = ",".join("?" * len(modules))
        cur = self.conn.execute(
            f"""UPDATE job_queue
                SET status = 'skipped', skip_reason = ?, completed_at = ?
                WHERE image_id = ? AND status = 'pending' AND module IN ({placeholders})""",
            [reason, _now(), image_id, *modules],
        )
        self.conn.commit()
        return cur.rowcount

    def get_image_module_job_status(self, image_id: int, module: str) -> str | None:
        """Return the queue status for an image/module, if a job row exists."""
        row = self.conn.execute(
            "SELECT status FROM job_queue WHERE image_id = ? AND module = ?",
            [image_id, module],
        ).fetchone()
        if row is None:
            return None
        return str(row["status"])

    def mark_pending(self, job_id: int) -> None:
        """Reset a claimed (running) job back to pending.

        Used during shutdown when a job was claimed but could not be
        submitted to the thread pool — ensures it is retried next run.
        """
        self.conn.execute(
            """UPDATE job_queue
               SET status = 'pending', started_at = NULL,
                   last_node_id = NULL, last_node_role = NULL
               WHERE id = ?""",
            [job_id],
        )
        self.conn.commit()

    def defer(self, job_id: int, seconds: int = 30) -> None:
        """Re-queue a job with a bumped ``queued_at`` timestamp.

        Used when a job's prerequisite isn't met yet.  Pushing
        ``queued_at`` forward prevents the same ineligible job from
        being claimed repeatedly while eligible jobs starve.
        """
        self.conn.execute(
            """UPDATE job_queue
               SET status = 'pending', started_at = NULL,
                   queued_at = ?,
                   last_node_id = NULL, last_node_role = NULL
               WHERE id = ?""",
            [_now_plus(seconds), job_id],
        )
        self.conn.commit()

    # ── Retry failed jobs ──────────────────────────────────────────────────

    def retry_failed(self, max_attempts: int = 3) -> int:
        """Re-enqueue failed jobs that haven't exceeded max_attempts."""
        cur = self.conn.execute(
            """UPDATE job_queue
               SET status = 'pending', error_message = NULL,
                   started_at = NULL, completed_at = NULL,
                   last_node_id = NULL, last_node_role = NULL
                WHERE status = 'failed' AND attempts < ?""",
            [max_attempts],
        )
        self.conn.commit()
        return cur.rowcount

    # ── Recover stale running jobs (crash recovery) ────────────────────────

    def recover_stale(self, timeout_minutes: int = 10) -> int:
        """Reset jobs stuck in 'running' state (e.g. after a crash)."""
        if timeout_minutes <= 0:
            cur = self.conn.execute(
                """UPDATE job_queue
                   SET status = 'pending', attempts = attempts + 1,
                       last_node_id = NULL, last_node_role = NULL
                    WHERE status = 'running'"""
            )
        else:
            cur = self.conn.execute(
                """UPDATE job_queue
                   SET status = 'pending', attempts = attempts + 1,
                       last_node_id = NULL, last_node_role = NULL
                    WHERE status = 'running'
                    AND started_at <= datetime('now', '-' || ? || ' minutes')""",
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

    def remaining_image_count(self) -> int:
        """Return the number of distinct images with pending or running work."""
        row = self.conn.execute(
            """SELECT COUNT(DISTINCT image_id) as cnt
               FROM job_queue
               WHERE status IN ('pending', 'running')"""
        ).fetchone()
        return int(row["cnt"] if row is not None else 0)

    def pending_count(
        self,
        module: str | None = None,
        modules: list[str] | None = None,
        image_ids: set[int] | None = None,
    ) -> int:
        where = "WHERE status = 'pending'"
        params: list[Any] = []
        if module:
            where += " AND module = ?"
            params.append(module)
        elif modules:
            placeholders = ",".join("?" * len(modules))
            where += f" AND module IN ({placeholders})"
            params.extend(modules)
        if image_ids is not None:
            id_ph = ",".join("?" * len(image_ids))
            where += f" AND image_id IN ({id_ph})"
            params.extend(image_ids)
        row = self.conn.execute(
            f"SELECT COUNT(*) as cnt FROM job_queue {where}", params
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
