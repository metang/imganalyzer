"""Unified scheduling policy helpers for local and distributed workers.

This module centralizes scheduler policy decisions (claim controls, worker
pause state, module affinity, ETA-aware module preference) so the policy is
not scattered across server/worker implementations.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from imganalyzer.db.queue import _now, _now_plus

ACTIVE_STATE = "active"
PAUSED_STATES: frozenset[str] = frozenset({"pause-drain", "pause-immediate", "paused"})
VALID_CONTROL_STATES: frozenset[str] = frozenset({ACTIVE_STATE, *PAUSED_STATES})

MAX_ACTIVE_LEASES_PER_WORKER = 3
DEFAULT_EPOCH_SECONDS = 90
EMA_ALPHA = 0.20
# Conservative fallbacks when no per-worker timing history exists yet.
_DEFAULT_MODULE_MS: dict[str, float] = {
    "metadata": 250.0,
    "technical": 500.0,
    "objects": 2_500.0,
    "caption": 9_000.0,
    "faces": 1_200.0,
    "perception": 25_000.0,
    "embedding": 1_000.0,
}


@dataclass(frozen=True)
class ClaimPolicy:
    """Computed claim controls for one worker claim request."""

    allow_claims: bool
    requested: int
    scan_size: int
    module_filter: str | None
    modules_filter: list[str] | None
    prefer_module: str | None
    prefer_image_ids: set[int] | None
    reason: str | None = None
    master_reserve: int = 0


def _normalize_control_state(desired_state: Any) -> tuple[str, bool]:
    state = str(desired_state or ACTIVE_STATE).strip().lower()
    if state == "resume":
        return (ACTIVE_STATE, True)
    if state in VALID_CONTROL_STATES:
        return (state, True)
    return ("paused", False)


def _decode_capabilities(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def get_worker_capabilities(conn: sqlite3.Connection, worker_id: str) -> dict[str, Any]:
    row = conn.execute(
        "SELECT capabilities FROM worker_nodes WHERE id = ?",
        [worker_id],
    ).fetchone()
    if row is None:
        return {}
    return _decode_capabilities(row["capabilities"])


def get_worker_control_state(
    conn: sqlite3.Connection,
    worker_id: str,
) -> tuple[str, str | None]:
    row = conn.execute(
        "SELECT desired_state, state_reason FROM worker_nodes WHERE id = ?",
        [worker_id],
    ).fetchone()
    if row is None:
        return (ACTIVE_STATE, None)
    desired_state, is_valid = _normalize_control_state(row["desired_state"])
    state_reason = str(row["state_reason"]) if row["state_reason"] else None
    if not is_valid:
        raw_state = str(row["desired_state"] or "").strip() or "<empty>"
        invalid_reason = f"invalid desired_state '{raw_state}'"
        state_reason = f"{state_reason}; {invalid_reason}" if state_reason else invalid_reason
    return (desired_state, state_reason)


def set_worker_control_state(
    conn: sqlite3.Connection,
    worker_id: str,
    desired_state: str,
    reason: str | None = None,
) -> None:
    state, is_valid = _normalize_control_state(desired_state)
    if not is_valid:
        raise ValueError(f"Unsupported worker desired_state: {desired_state}")
    now = _now()
    row = conn.execute(
        "SELECT id FROM worker_nodes WHERE id = ?",
        [worker_id],
    ).fetchone()
    if row is None:
        conn.execute(
            """INSERT INTO worker_nodes
               (id, display_name, platform, capabilities, status, last_heartbeat, created_at, updated_at,
                desired_state, state_reason, state_updated_at)
               VALUES (?, ?, '', '{}', 'offline', ?, ?, ?, ?, ?, ?)""",
            [worker_id, worker_id, now, now, now, state, reason, now],
        )
    else:
        conn.execute(
            """UPDATE worker_nodes
               SET desired_state = ?,
                   state_reason = ?,
                   state_updated_at = ?,
                   updated_at = ?
               WHERE id = ?""",
            [state, reason, now, now, worker_id],
        )
    conn.commit()


def _module_allowed_by_caps(module_name: str, caps: dict[str, Any]) -> bool:
    # Hard constraint: perception requires CUDA-capable node.
    # For backward compatibility, only enforce when the worker explicitly
    # reports a CUDA capability flag.
    if module_name == "perception" and "cuda" in caps and not bool(caps.get("cuda", False)):
        return False
    supported = caps.get("supportedModules")
    if isinstance(supported, list) and supported:
        return module_name in {str(m) for m in supported}
    return True


def _normalize_module_filters(
    module: Any,
    modules_list: Any,
    caps: dict[str, Any],
) -> tuple[str | None, list[str] | None]:
    module_filter: str | None = str(module) if module is not None else None
    modules_filter: list[str] | None = None
    if module_filter is None:
        if isinstance(modules_list, list) and modules_list:
            modules_filter = [str(m) for m in modules_list]
        else:
            supported = caps.get("supportedModules")
            if isinstance(supported, list) and supported:
                modules_filter = [str(m) for m in supported]

    if module_filter is not None and not _module_allowed_by_caps(module_filter, caps):
        return (None, [])
    if modules_filter is not None:
        modules_filter = [m for m in modules_filter if _module_allowed_by_caps(m, caps)]
    return (module_filter, modules_filter)


def _pending_count(
    conn: sqlite3.Connection,
    module_filter: str | None,
    modules_filter: list[str] | None,
) -> int:
    where = "WHERE status = 'pending' AND attempts <= max_attempts"
    params: list[Any] = []
    if module_filter:
        where += " AND module = ?"
        params.append(module_filter)
    elif modules_filter:
        placeholders = ",".join("?" * len(modules_filter))
        where += f" AND module IN ({placeholders})"
        params.extend(modules_filter)
    row = conn.execute(
        f"SELECT COUNT(*) AS cnt FROM job_queue {where}",
        params,
    ).fetchone()
    return int(row["cnt"]) if row else 0


def _pending_count_by_module(
    conn: sqlite3.Connection,
    modules_filter: list[str] | None,
    prefer_image_ids: set[int] | None,
) -> dict[str, int]:
    where = "WHERE status = 'pending' AND attempts <= max_attempts"
    params: list[Any] = []
    if modules_filter:
        placeholders = ",".join("?" * len(modules_filter))
        where += f" AND module IN ({placeholders})"
        params.extend(modules_filter)
    if prefer_image_ids:
        id_ph = ",".join("?" * len(prefer_image_ids))
        where += f" AND image_id IN ({id_ph})"
        params.extend(prefer_image_ids)
    rows = conn.execute(
        f"""SELECT module, COUNT(*) AS cnt
            FROM job_queue
            {where}
            GROUP BY module""",
        params,
    ).fetchall()
    return {str(row["module"]): int(row["cnt"]) for row in rows}


def _worker_module_ms(conn: sqlite3.Connection, worker_id: str) -> dict[str, float]:
    rows = conn.execute(
        "SELECT module, avg_ms FROM worker_module_stats WHERE worker_id = ?",
        [worker_id],
    ).fetchall()
    return {
        str(row["module"]): float(row["avg_ms"])
        for row in rows
        if row["avg_ms"] is not None
    }


def _master_running_count(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        """SELECT COUNT(*) AS cnt
           FROM job_queue jq
           LEFT JOIN job_leases jl ON jl.job_id = jq.id
           WHERE jq.status = 'running'
             AND jl.job_id IS NULL"""
    ).fetchone()
    return int(row["cnt"]) if row else 0


def _count_online_workers(conn: sqlite3.Connection) -> int:
    """Count distributed workers that are online (excludes the master node)."""
    try:
        row = conn.execute(
            """SELECT COUNT(*) AS cnt FROM worker_nodes
               WHERE id != 'master'
                 AND status IN ('online', 'active')
                 AND last_heartbeat > datetime('now', '-5 minutes')"""
        ).fetchone()
        return int(row["cnt"]) if row else 0
    except Exception:
        return 0


def choose_preferred_module(
    conn: sqlite3.Connection,
    worker_id: str,
    module_filter: str | None,
    modules_filter: list[str] | None,
    prefer_image_ids: set[int] | None,
) -> str | None:
    """Choose a preferred module for this worker (affinity + ETA aware)."""
    if module_filter:
        return module_filter
    if modules_filter is not None and not modules_filter:
        return None

    row = conn.execute(
        """SELECT epoch_module
           FROM worker_nodes
           WHERE id = ?
             AND epoch_module IS NOT NULL
             AND epoch_expires_at IS NOT NULL
             AND epoch_expires_at > datetime('now')""",
        [worker_id],
    ).fetchone()
    if row and row["epoch_module"]:
        epoch_module = str(row["epoch_module"])
        if modules_filter is None or epoch_module in set(modules_filter):
            epoch_pending = _pending_count_by_module(conn, [epoch_module], prefer_image_ids)
            if epoch_pending.get(epoch_module, 0) > 0:
                return epoch_module

    pending_by_module = _pending_count_by_module(conn, modules_filter, prefer_image_ids)
    if not pending_by_module and prefer_image_ids:
        # Fallback to global pending when the active chunk has no claimable jobs.
        pending_by_module = _pending_count_by_module(conn, modules_filter, None)
    if not pending_by_module:
        return None

    module_ms = _worker_module_ms(conn, worker_id)
    last_module_row = conn.execute(
        "SELECT last_module FROM worker_nodes WHERE id = ?",
        [worker_id],
    ).fetchone()
    last_module = str(last_module_row["last_module"]) if last_module_row and last_module_row["last_module"] else None

    best_module: str | None = None
    best_score = -1.0
    for module_name, pending_count in pending_by_module.items():
        expected_ms = module_ms.get(module_name, _DEFAULT_MODULE_MS.get(module_name, 1000.0))
        score = float(pending_count) * float(expected_ms)
        if module_name == last_module:
            score *= 1.15
        if score > best_score:
            best_score = score
            best_module = module_name
    return best_module


def compute_claim_policy(
    conn: sqlite3.Connection,
    *,
    worker_id: str,
    batch_size: int,
    module: Any,
    modules_list: Any,
    active_chunk_ids: set[int] | None,
    coordinator_run_active: bool,
) -> ClaimPolicy:
    """Compute scheduler claim policy for one worker claim request."""
    desired_state, reason = get_worker_control_state(conn, worker_id)
    if desired_state in PAUSED_STATES:
        return ClaimPolicy(
            allow_claims=False,
            requested=0,
            scan_size=0,
            module_filter=None,
            modules_filter=None,
            prefer_module=None,
            prefer_image_ids=None,
            reason=reason or desired_state,
        )

    try:
        active_row = conn.execute(
            """SELECT COUNT(*) AS cnt FROM job_leases
               WHERE worker_id = ?
                 AND lease_expires_at > datetime('now')""",
            [worker_id],
        ).fetchone()
        active_leases = int(active_row["cnt"]) if active_row else 0
    except Exception:
        active_leases = 0
    if active_leases >= MAX_ACTIVE_LEASES_PER_WORKER:
        return ClaimPolicy(False, 0, 0, None, None, None, None, "active_lease_cap")

    requested = max(1, int(batch_size or 1))
    requested = min(requested, MAX_ACTIVE_LEASES_PER_WORKER - active_leases)
    scan_size = min(max(requested * 4, requested), 32)

    caps = get_worker_capabilities(conn, worker_id)
    module_filter, modules_filter = _normalize_module_filters(module, modules_list, caps)
    if modules_filter == []:
        return ClaimPolicy(False, 0, 0, None, [], None, None, "unsupported_modules")

    pending_eligible = _pending_count(conn, module_filter, modules_filter)
    if pending_eligible <= 0:
        return ClaimPolicy(False, 0, 0, module_filter, modules_filter, None, None, "no_pending")

    # Reserve a proportional fair share of pending jobs for the master when
    # the run is active and the master currently has no running local jobs.
    # Without this, concurrent distributed workers drain the pending queue
    # before the master can claim, leaving its GPU idle.
    master_reserve = 0
    if coordinator_run_active:
        master_running = _master_running_count(conn)
        if master_running <= 0:
            num_workers = _count_online_workers(conn)
            if num_workers > 0:
                # Master gets 1/(N+1) of pending, at least one worker batch_size.
                master_share = max(requested, pending_eligible // (num_workers + 1))
                master_reserve = master_share
                worker_budget = pending_eligible - master_share
                if worker_budget <= 0:
                    return ClaimPolicy(
                        False, 0, 0, module_filter, modules_filter,
                        None, None, "reserved_for_master",
                    )
                per_worker = max(1, worker_budget // num_workers)
                requested = min(requested, per_worker)
                scan_size = min(max(requested * 4, requested), 32)

    prefer_image_ids = active_chunk_ids if active_chunk_ids else None
    prefer_module = choose_preferred_module(
        conn,
        worker_id=worker_id,
        module_filter=module_filter,
        modules_filter=modules_filter,
        prefer_image_ids=prefer_image_ids,
    )

    return ClaimPolicy(
        allow_claims=True,
        requested=requested,
        scan_size=scan_size,
        module_filter=module_filter,
        modules_filter=modules_filter,
        prefer_module=prefer_module,
        prefer_image_ids=prefer_image_ids,
        reason=None,
        master_reserve=master_reserve,
    )


def record_worker_affinity(
    conn: sqlite3.Connection,
    *,
    worker_id: str,
    module: str,
    epoch_seconds: int = DEFAULT_EPOCH_SECONDS,
    auto_commit: bool = True,
) -> None:
    """Update worker module affinity / epoch after a successful claim."""
    now = _now()
    epoch_until = _now_plus(max(1, int(epoch_seconds)))
    conn.execute(
        """UPDATE worker_nodes
           SET last_module = ?,
               epoch_module = ?,
               epoch_expires_at = ?,
               updated_at = ?
           WHERE id = ?""",
        [module, module, epoch_until, now, worker_id],
    )
    if auto_commit:
        conn.commit()


def record_worker_module_timing(
    conn: sqlite3.Connection,
    *,
    worker_id: str,
    module: str,
    processing_ms: int,
    auto_commit: bool = True,
) -> None:
    """Update worker/module runtime EWMA used by ETA-aware balancing."""
    if not worker_id or processing_ms <= 0:
        return

    row = conn.execute(
        """SELECT avg_ms, samples
           FROM worker_module_stats
           WHERE worker_id = ? AND module = ?""",
        [worker_id, module],
    ).fetchone()
    now = _now()
    if row is None:
        conn.execute(
            """INSERT INTO worker_module_stats
               (worker_id, module, avg_ms, samples, updated_at)
               VALUES (?, ?, ?, 1, ?)""",
            [worker_id, module, float(processing_ms), now],
        )
    else:
        current_avg = float(row["avg_ms"] or processing_ms)
        samples = int(row["samples"] or 1)
        updated_avg = (current_avg * (1.0 - EMA_ALPHA)) + (float(processing_ms) * EMA_ALPHA)
        conn.execute(
            """UPDATE worker_module_stats
               SET avg_ms = ?,
                   samples = ?,
                   updated_at = ?
               WHERE worker_id = ? AND module = ?""",
            [updated_avg, min(samples + 1, 1_000_000), now, worker_id, module],
        )
    if auto_commit:
        conn.commit()
