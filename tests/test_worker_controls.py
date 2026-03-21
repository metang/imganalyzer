from __future__ import annotations

import sqlite3
import threading

from imganalyzer.db.queue import JobQueue
from imganalyzer.db.repository import Repository
from imganalyzer.db.schema import ensure_schema
from imganalyzer.pipeline.distributed_worker import DistributedWorker


def test_server_worker_pause_resume_controls_claim(tmp_path, monkeypatch):
    db_path = tmp_path / "worker-control.db"
    conn = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    repo = Repository(conn)
    queue = JobQueue(conn)

    image_id = repo.register_image(file_path="/photos/paused-worker.jpg")
    queue.enqueue(image_id, "metadata")
    conn.close()

    monkeypatch.setenv("IMGANALYZER_DB_PATH", str(db_path))
    import imganalyzer.server as server

    server._db_local = threading.local()
    server._decoded_store = None
    server._active_worker = None

    server._handle_workers_register(
        {
            "workerId": "worker-1",
            "displayName": "Worker 1",
            "capabilities": {"supportedModules": ["metadata"], "cuda": False},
        }
    )
    paused = server._handle_workers_pause(
        {
            "workerId": "worker-1",
            "mode": "pause-drain",
            "reason": "maintenance window",
        }
    )
    assert paused["ok"] is True
    assert paused["desiredState"] == "pause-drain"

    heartbeat = server._handle_workers_heartbeat({"workerId": "worker-1"})
    assert heartbeat["desiredState"] == "pause-drain"
    assert heartbeat["stateReason"] == "maintenance window"

    paused_claim = server._handle_jobs_claim({"workerId": "worker-1", "batchSize": 1})
    assert paused_claim["jobs"] == []

    resumed = server._handle_workers_resume({"workerId": "worker-1"})
    assert resumed["ok"] is True
    assert resumed["desiredState"] == "active"

    resumed_claim = server._handle_jobs_claim({"workerId": "worker-1", "batchSize": 1})
    assert len(resumed_claim["jobs"]) == 1
    assert resumed_claim["jobs"][0]["module"] == "metadata"


def test_server_worker_pause_immediate_releases_leases(tmp_path, monkeypatch):
    db_path = tmp_path / "worker-pause-immediate.db"
    conn = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    repo = Repository(conn)
    queue = JobQueue(conn)

    image_id = repo.register_image(file_path="/photos/pause-immediate.jpg")
    queue.enqueue(image_id, "metadata")
    conn.close()

    monkeypatch.setenv("IMGANALYZER_DB_PATH", str(db_path))
    import imganalyzer.server as server

    server._db_local = threading.local()
    server._decoded_store = None
    server._active_worker = None
    server._handle_workers_register({"workerId": "worker-1", "displayName": "Worker 1"})

    claimed = server._handle_jobs_claim({"workerId": "worker-1", "batchSize": 1})
    assert len(claimed["jobs"]) == 1

    paused = server._handle_workers_pause(
        {
            "workerId": "worker-1",
            "mode": "pause-immediate",
            "reason": "emergency stop",
        }
    )
    assert paused["ok"] is True
    assert paused["previousState"] == "active"
    assert paused["desiredState"] == "pause-immediate"
    assert paused["transitioned"] is True
    assert paused["releasedLeases"] == 1

    check = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
    check.row_factory = sqlite3.Row
    row = check.execute("SELECT status FROM job_queue WHERE id = ?", [claimed["jobs"][0]["id"]]).fetchone()
    lease = check.execute("SELECT 1 FROM job_leases WHERE job_id = ?", [claimed["jobs"][0]["id"]]).fetchone()
    check.close()
    assert row is not None
    assert row["status"] == "pending"
    assert lease is None


def test_distributed_worker_heartbeat_updates_pause_state(monkeypatch):
    worker = DistributedWorker(coordinator_url="http://127.0.0.1:8765/", worker_id="worker-1")
    worker.supported_modules = None  # bypass startup dep checks
    worker.heartbeat_interval_seconds = 2.0
    worker._mark_all_active([{"id": 42, "leaseToken": "lease-42"}])
    worker._shutdown.set()

    calls: list[str] = []
    sleep_calls: list[float] = []

    def _fake_call(method: str, _params: dict[str, object]) -> dict[str, object]:
        calls.append(method)
        if method == "workers/heartbeat":
            return {
                "releasedExpired": 0,
                "desiredState": "pause-drain",
                "stateReason": "maintenance",
            }
        if method == "jobs/heartbeat":
            worker._clear_active(42)
            return {"ok": True}
        raise AssertionError(f"Unexpected coordinator method: {method}")

    monkeypatch.setattr(worker, "_coordinator_call", _fake_call)
    monkeypatch.setattr(
        "imganalyzer.pipeline.distributed_worker.time.sleep",
        lambda seconds: sleep_calls.append(seconds),
    )

    worker._heartbeat_loop()

    assert calls == ["workers/heartbeat", "jobs/heartbeat"]
    assert worker._desired_state == "pause-drain"
    assert worker._pause_reason == "maintenance"
    assert sleep_calls == [1.0]


def test_server_master_worker_lifecycle_visible_in_status_and_workers_list(tmp_path, monkeypatch):
    db_path = tmp_path / "master-lifecycle.db"
    conn = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    conn.close()

    monkeypatch.setenv("IMGANALYZER_DB_PATH", str(db_path))
    import imganalyzer.server as server

    server._db_local = threading.local()
    server._decoded_store = None
    server._active_worker = None
    server._run_thread = None

    status = server._handle_status({})
    master = status["nodes"]["master"]
    assert master["id"] == "master"
    assert master["role"] == "master"
    assert master["status"] == "offline"
    assert master["desiredState"] == "active"
    assert isinstance(master["lastHeartbeat"], str)
    assert master["capabilities"]["coordinator"] is True
    assert all(item["id"] != "master" for item in status["nodes"]["workers"])

    server._handle_workers_register({"workerId": "worker-1", "displayName": "Worker 1"})
    workers_list = server._handle_workers_list({})
    assert workers_list["master"]["id"] == "master"
    assert workers_list["master"]["status"] == "offline"
    assert all(item["id"] != "master" for item in workers_list["workers"])
    assert any(item["id"] == "worker-1" for item in workers_list["workers"])

    class ActiveWorkerStub:
        current_chunk_ids = None
        current_chunk_index = 0
        total_chunks = 1

    server._active_worker = ActiveWorkerStub()
    try:
        running = server._handle_workers_list({})
        assert running["master"]["status"] == "online"
    finally:
        server._active_worker = None

    idle_again = server._handle_workers_list({})
    assert idle_again["master"]["status"] == "offline"


def test_server_master_pause_blocks_new_runs_until_resumed(tmp_path, monkeypatch):
    db_path = tmp_path / "master-pause-run.db"
    conn = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    conn.close()

    monkeypatch.setenv("IMGANALYZER_DB_PATH", str(db_path))
    import imganalyzer.server as server

    server._db_local = threading.local()
    server._decoded_store = None
    server._active_worker = None
    server._run_thread = None

    paused = server._handle_workers_pause(
        {
            "workerId": "master",
            "mode": "pause-drain",
            "reason": "maintenance window",
        }
    )
    assert paused["ok"] is True
    assert paused["desiredState"] == "pause-drain"
    assert paused["worker"]["desiredState"] == "pause-drain"

    sent_errors: list[tuple[int | str | None, int, str]] = []
    sent_results: list[tuple[int | str, object]] = []
    monkeypatch.setattr(
        server,
        "_send_error",
        lambda req_id, code, message: sent_errors.append((req_id, code, message)),
    )
    monkeypatch.setattr(
        server,
        "_send_result",
        lambda req_id, result: sent_results.append((req_id, result)),
    )

    server._handle_run(77, {})

    assert sent_results == []
    assert len(sent_errors) == 1
    assert sent_errors[0][0] == 77
    assert sent_errors[0][1] == -3
    assert "workers/resume" in sent_errors[0][2]
    assert server._run_thread is None

    resumed = server._handle_workers_resume({"workerId": "master"})
    assert resumed["ok"] is True
    assert resumed["desiredState"] == "active"
    assert resumed["worker"]["desiredState"] == "active"
