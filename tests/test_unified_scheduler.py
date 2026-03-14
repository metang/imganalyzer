from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from imganalyzer.db.queue import JobQueue
from imganalyzer.db.repository import Repository
from imganalyzer.db.schema import ensure_schema
from imganalyzer.pipeline.unified_scheduler import (
    compute_claim_policy,
    get_worker_control_state,
    record_worker_module_timing,
)


def _make_test_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    return conn


def test_compute_claim_policy_filters_perception_without_cuda(tmp_path):
    conn = _make_test_db(tmp_path / "policy-no-cuda.db")
    repo = Repository(conn)
    queue = JobQueue(conn)

    image_id = repo.register_image(file_path="/photos/no-cuda.jpg")
    queue.enqueue(image_id, "perception")
    queue.enqueue(image_id, "metadata")

    conn.execute(
        """INSERT INTO worker_nodes
           (id, display_name, platform, capabilities, status, last_heartbeat, created_at, updated_at, desired_state)
           VALUES (?, ?, ?, ?, 'online', datetime('now'), datetime('now'), datetime('now'), 'active')""",
        [
            "worker-1",
            "Worker 1",
            "linux",
            json.dumps({"supportedModules": ["metadata", "perception"], "cuda": False}),
        ],
    )
    conn.commit()

    policy = compute_claim_policy(
        conn,
        worker_id="worker-1",
        batch_size=2,
        module=None,
        modules_list=None,
        active_chunk_ids=None,
        coordinator_run_active=False,
    )

    assert policy.allow_claims
    assert policy.modules_filter is not None
    assert "metadata" in policy.modules_filter
    assert "perception" not in policy.modules_filter

    conn.close()


def test_compute_claim_policy_prefers_eta_weighted_module_and_epoch(tmp_path):
    conn = _make_test_db(tmp_path / "policy-eta.db")
    repo = Repository(conn)
    queue = JobQueue(conn)

    for idx in range(10):
        image_id = repo.register_image(file_path=f"/photos/eta-metadata-{idx}.jpg")
        queue.enqueue(image_id, "metadata")
    for idx in range(2):
        image_id = repo.register_image(file_path=f"/photos/eta-caption-{idx}.jpg")
        queue.enqueue(image_id, "caption")

    conn.execute(
        """INSERT INTO worker_nodes
           (id, display_name, platform, capabilities, status, last_heartbeat, created_at, updated_at, desired_state)
           VALUES (?, ?, ?, ?, 'online', datetime('now'), datetime('now'), datetime('now'), 'active')""",
        [
            "worker-1",
            "Worker 1",
            "linux",
            json.dumps({"supportedModules": ["metadata", "caption"], "cuda": False}),
        ],
    )
    conn.commit()

    # Make caption far slower than metadata so ETA-weighted policy prefers it.
    record_worker_module_timing(
        conn,
        worker_id="worker-1",
        module="metadata",
        processing_ms=200,
    )
    record_worker_module_timing(
        conn,
        worker_id="worker-1",
        module="caption",
        processing_ms=15_000,
    )

    policy = compute_claim_policy(
        conn,
        worker_id="worker-1",
        batch_size=2,
        module=None,
        modules_list=None,
        active_chunk_ids=None,
        coordinator_run_active=False,
    )
    assert policy.allow_claims
    assert policy.prefer_module == "caption"

    # Active epoch should pin the preferred module while it has pending work.
    conn.execute(
        """UPDATE worker_nodes
           SET epoch_module = 'metadata',
               epoch_expires_at = datetime('now', '+5 minutes')
           WHERE id = 'worker-1'"""
    )
    conn.commit()
    epoch_policy = compute_claim_policy(
        conn,
        worker_id="worker-1",
        batch_size=2,
        module=None,
        modules_list=None,
        active_chunk_ids=None,
        coordinator_run_active=False,
    )
    assert epoch_policy.allow_claims
    assert epoch_policy.prefer_module == "metadata"

    conn.close()


def test_invalid_worker_control_state_fails_closed(tmp_path):
    conn = _make_test_db(tmp_path / "policy-invalid-state.db")
    repo = Repository(conn)
    queue = JobQueue(conn)

    image_id = repo.register_image(file_path="/photos/invalid-state.jpg")
    queue.enqueue(image_id, "metadata")

    conn.execute(
        """INSERT INTO worker_nodes
           (id, display_name, platform, capabilities, status, last_heartbeat, created_at, updated_at,
            desired_state, state_reason)
           VALUES (?, ?, ?, ?, 'online', datetime('now'), datetime('now'), datetime('now'), ?, ?)""",
        [
            "worker-1",
            "Worker 1",
            "linux",
            json.dumps({"supportedModules": ["metadata"]}),
            "totally-unknown",
            "manual override",
        ],
    )
    conn.commit()

    state, reason = get_worker_control_state(conn, "worker-1")
    assert state == "paused"
    assert reason is not None
    assert "invalid desired_state" in reason

    policy = compute_claim_policy(
        conn,
        worker_id="worker-1",
        batch_size=1,
        module=None,
        modules_list=None,
        active_chunk_ids=None,
        coordinator_run_active=False,
    )
    assert not policy.allow_claims
    assert policy.reason is not None
    assert "invalid desired_state" in policy.reason

    conn.close()
