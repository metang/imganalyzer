"""Job queue — SQLite-backed task queue with atomic claim, retry, pause/resume.

The queue stores one row per (image, module) combination.  Workers claim jobs
atomically using UPDATE ... RETURNING to avoid double-processing.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from imganalyzer.db.connection import begin_immediate as _begin_immediate


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
        if _auto_commit:
            _begin_immediate(self.conn)
        try:
            existing = self.conn.execute(
                "SELECT id, status FROM job_queue WHERE image_id = ? AND module = ?",
                [image_id, module],
            ).fetchone()

            if existing:
                if existing["status"] in ("pending", "running"):
                    if _auto_commit:
                        self.conn.rollback()
                    return None  # already queued
                if not force:
                    # done / failed / skipped — don't re-enqueue without explicit force
                    if _auto_commit:
                        self.conn.rollback()
                    return None
                # force=True: reset the row back to pending
                self.conn.execute(
                    """UPDATE job_queue
                       SET status = 'pending', attempts = 0, error_message = NULL,
                            skip_reason = NULL, started_at = NULL, completed_at = NULL,
                            queued_at = ?, priority = ?,
                            last_node_id = NULL, last_node_role = 'force'
                        WHERE id = ?""",
                    [_now(), priority, existing["id"]],
                )
                if _auto_commit:
                    self.conn.commit()
                return existing["id"]

            cur = self.conn.execute(
                """INSERT INTO job_queue (image_id, module, priority, status, queued_at, last_node_role)
                   VALUES (?, ?, ?, 'pending', ?, ?)""",
                [image_id, module, priority, _now(), "force" if force else None],
            )
            if _auto_commit:
                self.conn.commit()
            return cur.lastrowid
        except Exception:
            if _auto_commit:
                self.conn.rollback()
            raise

    def enqueue_batch(
        self,
        image_ids: list[int],
        modules: list[str],
        priority: int = 0,
        force: bool = False,
    ) -> int:
        """Enqueue multiple (image, module) pairs.  Returns count enqueued."""
        _begin_immediate(self.conn)
        try:
            count = 0
            for image_id in image_ids:
                for module in modules:
                    job_id = self.enqueue(
                        image_id, module, priority=priority, force=force,
                        _auto_commit=False,
                    )
                    if job_id is not None:
                        count += 1
            self.conn.commit()
            return count
        except Exception:
            self.conn.rollback()
            raise

    # ── Claim (atomic) ─────────────────────────────────────────────────────

    def get_pending_image_ids(
        self,
        modules: list[str] | None = None,
    ) -> list[int]:
        """Return distinct image_ids with pending jobs.

        Images are ordered by number of pending jobs **descending** so that
        images needing the most work appear first.  This ensures that the
        first chunk contains fresh images requiring all pipeline modules
        rather than nearly-complete images that only need one deferred
        module (e.g. perception).

        If *modules* is given, only consider jobs for those modules.
        """
        where = "WHERE status = 'pending' AND attempts <= max_attempts"
        params: list[Any] = []
        if modules:
            placeholders = ",".join("?" * len(modules))
            where += f" AND module IN ({placeholders})"
            params.extend(modules)
        rows = self.conn.execute(
            f"""SELECT image_id, COUNT(*) AS cnt
                FROM job_queue {where}
                GROUP BY image_id
                ORDER BY cnt DESC, image_id ASC""",
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
        where = "WHERE status = 'pending' AND attempts <= max_attempts"
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
        _begin_immediate(self.conn)
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
        prefer_image_ids: set[int] | None = None,
        restrict_image_ids: set[int] | None = None,
        master_reserve: int = 0,
    ) -> list[dict[str, Any]]:
        """Atomically claim jobs and create leases for distributed workers.

        If *prefer_module* is given, jobs for that module are sorted first
        (module affinity) to minimize model switching across claims.

        If *prefer_image_ids* is given, jobs for those images are sorted
        first (chunk affinity) so distributed workers focus on the same
        chunk as the coordinator.

        If *restrict_image_ids* is given, ONLY jobs for those image IDs are
        eligible (cache-gated dispatch).

        If *master_reserve* > 0, the claim leaves at least that many pending
        jobs unclaimed so the master device can process them.  This check
        is performed inside the BEGIN IMMEDIATE transaction for atomicity.
        """
        where = "WHERE status = 'pending' AND attempts <= max_attempts"
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
        if restrict_image_ids is not None:
            if not restrict_image_ids:
                return []  # empty set — nothing eligible
            rid_ph = ",".join("?" * len(restrict_image_ids))
            where += f" AND image_id IN ({rid_ph})"
            params.extend(restrict_image_ids)
        where_params = list(params)

        # Build ORDER BY with optional chunk and module affinity
        order_parts: list[str] = []

        # Chunk affinity: prefer images from coordinator's current chunk
        if prefer_image_ids:
            chunk_ph = ",".join("?" * len(prefer_image_ids))
            order_parts.append(f"(CASE WHEN image_id IN ({chunk_ph}) THEN 0 ELSE 1 END)")
            params.extend(prefer_image_ids)

        # Module affinity: prefer same module to minimize model switching
        if prefer_module and not module:
            order_parts.append("(CASE WHEN module = ? THEN 0 ELSE 1 END)")
            params.append(prefer_module)

        order_parts.extend(["priority DESC", "queued_at ASC"])
        order = ", ".join(order_parts)
        # Effective limit: respect master_reserve within the transaction
        # so the reservation cannot be bypassed by concurrent worker claims.
        effective_limit = batch_size
        params.append(effective_limit)

        _begin_immediate(self.conn)
        try:
            if master_reserve > 0:
                # Re-count pending inside the exclusive transaction to enforce
                # the reservation atomically.  Without this, multiple workers
                # checking concurrently can each think there are enough pending
                # jobs and collectively drain past the reservation.
                pending_now = self.conn.execute(
                    f"SELECT COUNT(*) AS cnt FROM job_queue {where}",
                    where_params,
                ).fetchone()["cnt"]
                claimable = max(0, pending_now - master_reserve)
                if claimable <= 0:
                    self.conn.rollback()
                    return []
                effective_limit = min(batch_size, claimable)
                # Replace the last param (LIMIT value) with the clamped limit
                params[-1] = effective_limit

            rows = self.conn.execute(
                f"""SELECT id, image_id, module, attempts, last_node_role
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
        _begin_immediate(self.conn)
        try:
            rows = self.conn.execute(
                """SELECT job_id FROM job_leases
                   WHERE lease_expires_at <= ?""",
                [_now()],
            ).fetchall()
            if not rows:
                self.conn.rollback()
                return 0

            job_ids = [r["job_id"] for r in rows]
            placeholders = ",".join("?" * len(job_ids))

            # Permanently fail jobs that exceeded max_attempts (check BEFORE
            # re-queue to avoid double-update on boundary: a job at
            # attempts = max_attempts-1 would be re-queued by query 1 then
            # immediately matched by query 2 if checked after).
            self.conn.execute(
                f"""UPDATE job_queue
                    SET status = 'failed', attempts = attempts + 1,
                        error_message = 'Exceeded max attempts (lease expired)',
                        completed_at = ?
                    WHERE id IN ({placeholders})
                      AND status = 'running'
                      AND attempts >= max_attempts""",
                [_now()] + job_ids,
            )
            # Re-queue jobs that still have retries left
            self.conn.execute(
                f"""UPDATE job_queue
                    SET status = 'pending', started_at = NULL, attempts = attempts + 1,
                        last_node_id = NULL, last_node_role = NULL
                    WHERE id IN ({placeholders})
                      AND status = 'running'
                      AND attempts < max_attempts""",
                job_ids,
            )
            self.conn.execute(
                f"DELETE FROM job_leases WHERE job_id IN ({placeholders})",
                job_ids,
            )
            self.conn.commit()
            return len(job_ids)
        except Exception:
            self.conn.rollback()
            raise

    def release_worker_leases(self, worker_id: str, delay_seconds: int = 0) -> int:
        """Return all leases held by a worker to pending state.

        This is used for graceful worker shutdown / drain, so it does *not*
        consume retry attempts. Hard failures are handled separately by
        ``mark_failed_leased()`` and ``release_expired_leases()``.
        """
        _begin_immediate(self.conn)
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
            queued_at = _now_plus(max(0, int(delay_seconds))) if delay_seconds > 0 else _now()
            self.conn.execute(
                f"""UPDATE job_queue
                    SET status = 'pending', started_at = NULL, queued_at = ?,
                        last_node_id = NULL, last_node_role = NULL
                    WHERE id IN ({placeholders})
                      AND status = 'running'""",
                [queued_at] + job_ids,
            )
            self.conn.execute(
                f"DELETE FROM job_leases WHERE job_id IN ({placeholders})",
                job_ids,
            )
            self.conn.commit()
            return len(job_ids)
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
        _begin_immediate(self.conn)
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

    def release_leased(self, job_id: int, lease_token: str, delay_seconds: int = 0) -> bool:
        """Return a claimed leased job to pending if the token matches.

        ``delay_seconds`` can be used to defer immediate re-claim of a
        temporarily unrunnable job (for example, waiting on prerequisites or
        decoded-cache warmup). This is a soft release and does not consume a
        retry attempt.
        """
        _begin_immediate(self.conn)
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
                       last_node_id = NULL,
                       last_node_role = NULL
                     WHERE id = ?""",
                [_now_plus(max(0, int(delay_seconds))) if delay_seconds > 0 else _now(), job_id],
            )
            self.conn.execute("DELETE FROM job_leases WHERE job_id = ?", [job_id])
            self.conn.commit()
            return True
        except Exception:
            self.conn.rollback()
            raise

    def mark_done_leased(self, job_id: int, lease_token: str, processing_ms: int | None = None) -> bool:
        """Mark a leased job done if the lease token matches."""
        _begin_immediate(self.conn)
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
                   SET status = 'done', completed_at = ?,
                       processing_ms = ?
                   WHERE id = ?""",
                [_now(), processing_ms, job_id],
            )
            self.conn.execute("DELETE FROM job_leases WHERE job_id = ?", [job_id])
            self.conn.commit()
            return True
        except Exception:
            self.conn.rollback()
            raise

    def mark_failed_leased(self, job_id: int, lease_token: str, error: str) -> bool:
        """Mark a leased job failed if the lease token matches."""
        _begin_immediate(self.conn)
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
        _begin_immediate(self.conn)
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

    def batch_skip_release_leased(
        self,
        skips: list[tuple[int, str, str]],
        releases: list[tuple[int, str]],
        defers: list[tuple[int, str, int]] | None = None,
    ) -> None:
        """Batch skip and release multiple leased jobs in a single transaction.

        *skips* is a list of ``(job_id, lease_token, reason)`` tuples.
        *releases* is a list of ``(job_id, lease_token)`` tuples.
        *defers* is a list of ``(job_id, lease_token, delay_seconds)`` tuples.
        """
        if not skips and not releases and not defers:
            return
        _begin_immediate(self.conn)
        try:
            now = _now()
            for job_id, lease_token, reason in skips:
                row = self.conn.execute(
                    "SELECT 1 FROM job_leases WHERE job_id = ? AND lease_token = ?",
                    [job_id, lease_token],
                ).fetchone()
                if row is None:
                    continue
                self.conn.execute(
                    "UPDATE job_queue SET status = 'skipped', skip_reason = ?, completed_at = ? "
                    "WHERE id = ?",
                    [reason, now, job_id],
                )
                self.conn.execute("DELETE FROM job_leases WHERE job_id = ?", [job_id])
            for job_id, lease_token, delay_seconds in defers or []:
                row = self.conn.execute(
                    "SELECT 1 FROM job_leases WHERE job_id = ? AND lease_token = ?",
                    [job_id, lease_token],
                ).fetchone()
                if row is None:
                    continue
                defer_until = _now_plus(max(0, int(delay_seconds)))
                self.conn.execute(
                    "UPDATE job_queue SET status = 'pending', started_at = NULL, queued_at = ?, "
                    "last_node_id = NULL, last_node_role = NULL "
                    "WHERE id = ?",
                    [defer_until, job_id],
                )
                self.conn.execute("DELETE FROM job_leases WHERE job_id = ?", [job_id])
            for job_id, lease_token in releases:
                row = self.conn.execute(
                    "SELECT 1 FROM job_leases WHERE job_id = ? AND lease_token = ?",
                    [job_id, lease_token],
                ).fetchone()
                if row is None:
                    continue
                # A coordinator-side release here means "not dispatched after
                # claim scan" or "defer until ready", not a real processing
                # failure. Do not burn retry attempts for these soft releases.
                self.conn.execute(
                    "UPDATE job_queue SET status = 'pending', started_at = NULL, queued_at = ?, "
                    "last_node_id = NULL, last_node_role = NULL "
                    "WHERE id = ?",
                    [now, job_id],
                )
                self.conn.execute("DELETE FROM job_leases WHERE job_id = ?", [job_id])
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # ── Complete / Fail / Skip ─────────────────────────────────────────────

    def mark_done(self, job_id: int, processing_ms: int | None = None) -> None:
        self.conn.execute(
            """UPDATE job_queue
               SET status = 'done', completed_at = ?,
                   processing_ms = ?
               WHERE id = ?""",
            [_now(), processing_ms, job_id],
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
               WHERE id = ? AND status = 'running'""",
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
               WHERE id = ? AND status = 'running'""",
            [_now_plus(seconds), job_id],
        )
        self.conn.commit()

    # ── Retry failed jobs ──────────────────────────────────────────────────

    def fail_exhausted_pending(self) -> int:
        """Permanently fail pending jobs whose attempts exceeded max_attempts.

        These "zombie" jobs can accumulate when older recovery logic reset
        jobs to pending without checking the attempt count.
        """
        cur = self.conn.execute(
            """UPDATE job_queue
               SET status = 'failed',
                   error_message = 'Exceeded max attempts (exhausted pending)',
                   completed_at = ?
               WHERE status = 'pending' AND attempts >= max_attempts""",
            [_now()],
        )
        if cur.rowcount:
            self.conn.commit()
        return cur.rowcount

    def retry_failed(self, max_attempts: int = 3) -> int:
        """Re-enqueue retryable failed jobs.

        Jobs failed with ``Exceeded max attempts (exhausted pending)`` are
        synthetic queue-management failures rather than real processing
        failures, so they are safe to restore and get their attempt count
        reset.
        """
        cur = self.conn.execute(
            """UPDATE job_queue
                SET status = 'pending', error_message = NULL,
                    skip_reason = NULL,
                    started_at = NULL, completed_at = NULL,
                    last_node_id = NULL, last_node_role = NULL,
                    attempts = CASE
                        WHEN error_message = 'Exceeded max attempts (exhausted pending)' THEN 0
                        ELSE attempts
                    END
                 WHERE status = 'failed'
                   AND (
                       attempts < ?
                       OR error_message = 'Exceeded max attempts (exhausted pending)'
                   )""",
            [max_attempts],
        )
        self.conn.commit()
        return cur.rowcount

    def remap_pending_modules(
        self,
        mapping: dict[str, str],
        statuses: tuple[str, ...] = ("pending", "running"),
    ) -> dict[str, int]:
        """Remap legacy module keys for active queue rows.

        For each ``source -> target`` mapping:
        1) Delete source rows when a target row already exists for the same image.
        2) Rename remaining source rows to target.

        This lets newer workers resume old queues (e.g. ``local_ai`` -> ``caption``)
        without hitting UNIQUE(image_id, module) conflicts.
        """
        normalized = {
            str(source): str(target)
            for source, target in mapping.items()
            if source and target and source != target
        }
        if not normalized:
            return {"updated": 0, "deleted": 0}
        if not statuses:
            return {"updated": 0, "deleted": 0}

        status_placeholders = ",".join("?" * len(statuses))
        updated = 0
        deleted = 0

        _begin_immediate(self.conn)
        try:
            for source, target in normalized.items():
                deleted_cur = self.conn.execute(
                    f"""DELETE FROM job_queue
                        WHERE module = ?
                          AND status IN ({status_placeholders})
                          AND EXISTS (
                              SELECT 1
                              FROM job_queue q2
                              WHERE q2.image_id = job_queue.image_id
                                AND q2.module = ?
                          )""",
                    [source, *statuses, target],
                )
                deleted += deleted_cur.rowcount

                updated_cur = self.conn.execute(
                    f"""UPDATE job_queue
                        SET module = ?,
                            started_at = NULL,
                            last_node_id = NULL,
                            last_node_role = NULL
                        WHERE module = ?
                          AND status IN ({status_placeholders})""",
                    [target, source, *statuses],
                )
                updated += updated_cur.rowcount

            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

        return {"updated": updated, "deleted": deleted}

    def reconcile_runtime_state(self, recover_master_jobs: bool = True) -> dict[str, int]:
        """Repair queue/lease runtime invariants after interruptions.

        This reconciles stale coordinator state without touching healthy in-flight
        distributed work:

        1) Remove dangling lease rows whose jobs are no longer ``running``.
        2) Re-queue ``running`` worker jobs that no longer have an active lease.
        3) Optionally re-queue ``running`` master jobs with no lease
           (used on coordinator startup when no local run is active).
        """
        _begin_immediate(self.conn)
        try:
            dangling_cur = self.conn.execute(
                """DELETE FROM job_leases
                   WHERE job_id IN (
                       SELECT jl.job_id
                       FROM job_leases jl
                       JOIN job_queue jq ON jq.id = jl.job_id
                       WHERE jq.status <> 'running'
                   )"""
            )

            worker_orphans_cur = self.conn.execute(
                """UPDATE job_queue
                   SET status = 'pending',
                       started_at = NULL,
                       attempts = attempts + 1,
                       last_node_id = NULL,
                       last_node_role = NULL
                   WHERE status = 'running'
                     AND COALESCE(last_node_role, 'master') <> 'master'
                     AND NOT EXISTS (
                         SELECT 1 FROM job_leases jl WHERE jl.job_id = job_queue.id
                     )"""
            )

            master_orphans = 0
            if recover_master_jobs:
                master_orphans_cur = self.conn.execute(
                    """UPDATE job_queue
                       SET status = 'pending',
                           started_at = NULL,
                           attempts = attempts + 1,
                           last_node_id = NULL,
                           last_node_role = NULL
                       WHERE status = 'running'
                         AND COALESCE(last_node_role, 'master') = 'master'
                         AND NOT EXISTS (
                             SELECT 1 FROM job_leases jl WHERE jl.job_id = job_queue.id
                         )"""
                )
                master_orphans = int(master_orphans_cur.rowcount)

            self.conn.commit()
            return {
                "dangling_leases": int(dangling_cur.rowcount),
                "worker_orphans": int(worker_orphans_cur.rowcount),
                "master_orphans": master_orphans,
            }
        except Exception:
            self.conn.rollback()
            raise

    # ── Recover stale running jobs (crash recovery) ────────────────────────

    def recover_stale(self, timeout_minutes: int = 10) -> int:
        """Reset jobs stuck in 'running' state (e.g. after a crash).

        Jobs that have exceeded ``max_attempts`` are permanently failed
        instead of being re-queued to prevent infinite retry cycling.
        """
        time_filter = ""
        params: list[Any] = []
        if timeout_minutes > 0:
            time_filter = " AND started_at <= datetime('now', '-' || ? || ' minutes')"
            params = [timeout_minutes]

        # Re-queue jobs that still have retries left
        cur = self.conn.execute(
            f"""UPDATE job_queue
                SET status = 'pending', attempts = attempts + 1,
                    last_node_id = NULL, last_node_role = NULL
                WHERE status = 'running' AND attempts < max_attempts
                {time_filter}""",
            params,
        )
        requeued = cur.rowcount

        # Permanently fail jobs that exceeded max_attempts
        self.conn.execute(
            f"""UPDATE job_queue
                SET status = 'failed', attempts = attempts + 1,
                    error_message = 'Exceeded max attempts (stale recovery)',
                    completed_at = ?
                WHERE status = 'running' AND attempts >= max_attempts
                {time_filter}""",
            [_now()] + params,
        )
        # Clean up orphaned leases — any lease whose job is no longer running
        self.conn.execute(
            """DELETE FROM job_leases WHERE job_id IN (
                SELECT jl.job_id FROM job_leases jl
                JOIN job_queue jq ON jq.id = jl.job_id
                WHERE jq.status NOT IN ('running')
            )"""
        )
        self.conn.commit()
        return requeued

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

    def module_avg_processing_ms(self, last_n: int = 100) -> dict[str, float]:
        """Average processing time per module from the last *last_n* done jobs.

        Prefers the explicit ``processing_ms`` column (actual elapsed time
        reported by the worker) over ``completed_at − started_at`` (which is
        inflated for batch-claimed jobs where ``started_at`` is set at claim
        time for the whole batch, not per-job start).

        Returns ``{module: avg_ms}``.
        """
        rows = self.conn.execute(
            """SELECT module,
                      AVG(
                          CASE
                              WHEN processing_ms IS NOT NULL AND processing_ms > 0
                                  THEN processing_ms
                              ELSE (julianday(completed_at) - julianday(started_at)) * 86400000
                          END
                      ) AS avg_ms
                FROM (
                    SELECT module, started_at, completed_at, processing_ms,
                           ROW_NUMBER() OVER (
                               PARTITION BY module ORDER BY completed_at DESC
                           ) AS rn
                    FROM job_queue
                    WHERE status = 'done'
                      AND (processing_ms IS NOT NULL
                           OR (started_at IS NOT NULL AND completed_at IS NOT NULL))
                )
                WHERE rn <= ?
                GROUP BY module""",
            [last_n],
        ).fetchall()
        return {r["module"]: round(r["avg_ms"], 0) for r in rows if r["avg_ms"] is not None}

    def pending_count(
        self,
        module: str | None = None,
        modules: list[str] | None = None,
        image_ids: set[int] | None = None,
    ) -> int:
        where = "WHERE status = 'pending' AND attempts <= max_attempts"
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

    def running_count(
        self,
        module: str | None = None,
        image_ids: set[int] | None = None,
    ) -> int:
        where = "WHERE status = 'running'"
        params: list[Any] = []
        if module:
            where += " AND module = ?"
            params.append(module)
        if image_ids is not None:
            id_ph = ",".join("?" * len(image_ids))
            where += f" AND image_id IN ({id_ph})"
            params.extend(image_ids)
        row = self.conn.execute(
            f"SELECT COUNT(*) as cnt FROM job_queue {where}", params
        ).fetchone()
        return row["cnt"]

    def leased_running_count(
        self,
        module: str | None = None,
        image_ids: set[int] | None = None,
    ) -> int:
        """Count running jobs held by distributed workers (with active leases)."""
        where = "WHERE jq.status = 'running' AND jl.job_id IS NOT NULL"
        params: list[Any] = []
        if module:
            where += " AND jq.module = ?"
            params.append(module)
        if image_ids is not None:
            id_ph = ",".join("?" * len(image_ids))
            where += f" AND jq.image_id IN ({id_ph})"
            params.extend(image_ids)
        row = self.conn.execute(
            f"""SELECT COUNT(*) AS cnt
                FROM job_queue jq
                INNER JOIN job_leases jl ON jl.job_id = jq.id
                {where}""",
            params,
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
