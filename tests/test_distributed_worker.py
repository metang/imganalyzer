from __future__ import annotations

import sqlite3
import threading
from unittest.mock import patch

from imganalyzer.db.queue import JobQueue
from imganalyzer.db.repository import Repository
from imganalyzer.db.schema import ensure_schema
from imganalyzer.pipeline.distributed_worker import (
    CoordinatorClient,
    DistributedWorker,
    _should_bypass_proxy,
)
from imganalyzer.pipeline.modules import _rewrite_path_with_mappings


def _make_test_db(tmp_path):
    db_path = tmp_path / "distributed-worker.db"
    conn = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    ensure_schema(conn)
    return conn


def test_process_claimed_job_reports_completion(tmp_path):
    conn = _make_test_db(tmp_path)
    repo = Repository(conn)
    image_id = repo.register_image(file_path="/photos/distributed-metadata.jpg")

    worker = DistributedWorker(
        coordinator_url="http://127.0.0.1:8765/",
        worker_id="worker-a",
        batch_size=1,
        write_xmp=False,
    )

    class FakeRunner:
        def run(self, _image_id: int, _module: str):
            return {"description": "ok"}

    recorded: list[tuple[str, dict[str, object]]] = []

    def fake_call(method: str, params: dict[str, object]) -> dict[str, object]:
        recorded.append((method, dict(params)))
        return {"ok": True}

    worker._open_job_sandbox = lambda _job: (conn, repo, FakeRunner())  # type: ignore[method-assign]
    worker._coordinator_call = fake_call  # type: ignore[method-assign]

    with patch(
        "imganalyzer.pipeline.distributed_worker.extract_result_payload",
        return_value={"data": {"camera_make": "TestCam"}},
    ):
        status = worker._process_claimed_job(
            {
                "id": 5,
                "imageId": image_id,
                "module": "metadata",
                "leaseToken": "lease-123",
                "filePath": "/photos/distributed-metadata.jpg",
            }
        )

    assert status == "done"
    assert recorded == [
        (
            "jobs/complete",
            {
                "jobId": 5,
                "leaseToken": "lease-123",
                "payload": {"data": {"camera_make": "TestCam"}},
                "noXmp": True,
            },
        )
    ]


def test_should_bypass_proxy_for_private_coordinator_urls():
    assert _should_bypass_proxy("http://10.0.0.215:8765/jsonrpc") is True
    assert _should_bypass_proxy("http://127.0.0.1:8765/jsonrpc") is True
    assert _should_bypass_proxy("http://coordinator.local:8765/jsonrpc") is True
    assert _should_bypass_proxy("https://example.com/jsonrpc") is False


def test_coordinator_client_bypasses_env_proxy_for_private_hosts(monkeypatch):
    built_with: list[object] = []

    def _fake_build_opener(*handlers: object):
        built_with.extend(handlers)

        class _FakeOpener:
            def open(self, _req, timeout: float):
                raise TimeoutError(timeout)

        return _FakeOpener()

    monkeypatch.setattr("imganalyzer.pipeline.distributed_worker.request.build_opener", _fake_build_opener)
    client = CoordinatorClient("http://10.0.0.215:8765/jsonrpc")

    assert client is not None
    assert len(built_with) == 1
    proxy_handler = built_with[0]
    assert type(proxy_handler).__name__ == "ProxyHandler"
    assert getattr(proxy_handler, "proxies", None) == {}


def test_server_get_db_is_thread_local(tmp_path, monkeypatch):
    db_path = tmp_path / "server-thread-local.db"
    conn = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    conn.close()
    monkeypatch.setenv("IMGANALYZER_DB_PATH", str(db_path))

    import imganalyzer.server as server

    server._db_local = threading.local()
    conn_ids: list[int] = []
    errors: list[Exception] = []

    def _use_connection() -> None:
        try:
            conn = server._get_db()
            conn.execute("SELECT 1").fetchone()
            conn_ids.append(id(conn))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_use_connection) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert len(conn_ids) == 2
    assert conn_ids[0] != conn_ids[1]


def test_rewrite_path_with_mappings_windows_to_posix():
    original = r"Z:\photos\trip\image.jpg"
    rewritten = _rewrite_path_with_mappings(
        original,
        [(r"Z:\photos", "/Volumes/photos")],
    )
    assert rewritten == "/Volumes/photos/trip/image.jpg"


def test_rewrite_path_with_mappings_posix_to_posix():
    original = "/mnt/photos/trip/image.jpg"
    rewritten = _rewrite_path_with_mappings(
        original,
        [("/mnt/photos", "/Volumes/photos")],
    )
    assert rewritten == "/Volumes/photos/trip/image.jpg"


def test_rewrite_path_with_mappings_no_match_returns_original():
    original = r"Z:\photos-archive\image.jpg"
    rewritten = _rewrite_path_with_mappings(
        original,
        [(r"Z:\photos", "/Volumes/photos")],
    )
    assert rewritten == original


def test_distributed_worker_passes_path_mappings_to_module_runner(tmp_path, monkeypatch):
    worker = DistributedWorker(
        coordinator_url="http://127.0.0.1:8765/",
        path_mappings=[(r"Z:\photos", "/Volumes/photos")],
    )

    conn, _, runner = worker._open_job_sandbox(
        {
            "imageId": 1,
            "filePath": r"Z:\photos\trip\image.jpg",
            "image": {},
            "context": {},
        }
    )
    assert runner.path_mappings == [(r"Z:\photos", "/Volumes/photos")]
    conn.close()


def test_open_job_sandbox_uses_claim_context_without_shared_db(monkeypatch):
    monkeypatch.delenv("IMGANALYZER_DB_PATH", raising=False)
    worker = DistributedWorker(coordinator_url="http://127.0.0.1:8765/")

    conn, repo, _runner = worker._open_job_sandbox(
        {
            "imageId": 7,
            "filePath": "/nas/photos/image.jpg",
            "image": {"width": 640, "height": 480, "format": "jpeg"},
            "context": {
                "modules": {
                    "local_ai": {
                        "description": "sunset over water",
                        "scene_type": "landscape",
                        "main_subject": "harbor",
                    }
                }
            },
        }
    )

    assert repo.get_image(7)["file_path"] == "/nas/photos/image.jpg"
    assert repo.get_analysis(7, "local_ai")["description"] == "sunset over water"
    conn.close()


def test_server_jobs_claim_packages_embedding_context(tmp_path, monkeypatch):
    db_path = tmp_path / "server-claim.db"
    conn = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    repo = Repository(conn)
    queue = JobQueue(conn)

    image_id = repo.register_image(file_path="/nas/photos/image.jpg")
    repo.upsert_local_ai(image_id, {"description": "red boat", "scene_type": "harbor"})
    repo.upsert_cloud_ai(image_id, "openai", {"description": "dock at sunset"})
    queue.enqueue(image_id, "embedding")
    conn.close()

    monkeypatch.setenv("IMGANALYZER_DB_PATH", str(db_path))
    import imganalyzer.server as server

    server._db_local = threading.local()
    server._handle_workers_register({"workerId": "worker-1", "displayName": "Worker 1"})
    result = server._handle_jobs_claim({"workerId": "worker-1", "batchSize": 1})

    assert len(result["jobs"]) == 1
    job = result["jobs"][0]
    assert job["context"]["modules"]["local_ai"]["description"] == "red boat"
    assert job["context"]["modules"]["cloud_ai"]["providers"][0]["description"] == "dock at sunset"


def test_server_jobs_complete_persists_embedding_payload(tmp_path, monkeypatch):
    db_path = tmp_path / "server-complete.db"
    conn = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    repo = Repository(conn)
    queue = JobQueue(conn)

    image_id = repo.register_image(file_path="/nas/photos/image.jpg")
    job_id = queue.enqueue(image_id, "embedding")
    assert job_id is not None
    claimed = queue.claim_leased("worker-1", batch_size=1)
    assert claimed
    lease = claimed[0]
    conn.close()

    monkeypatch.setenv("IMGANALYZER_DB_PATH", str(db_path))
    import imganalyzer.server as server

    server._db_local = threading.local()
    result = server._handle_jobs_complete(
        {
            "jobId": lease["id"],
            "leaseToken": lease["lease_token"],
            "payload": {
                "image": {"width": 1024, "height": 768, "format": "jpeg"},
                "embeddings": [
                    {
                        "embeddingType": "description_clip",
                        "vector": "AQIDBA==",
                        "modelVersion": "clip-test",
                    }
                ],
            },
            "noXmp": True,
        }
    )

    assert result == {"ok": True}

    check_conn = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
    check_conn.row_factory = sqlite3.Row
    check_repo = Repository(check_conn)
    image = check_repo.get_image(image_id)
    assert image["width"] == 1024
    rows = check_conn.execute(
        "SELECT embedding_type, vector, model_version FROM embeddings WHERE image_id = ?",
        [image_id],
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["embedding_type"] == "description_clip"
    assert rows[0]["vector"] == b"\x01\x02\x03\x04"
    assert rows[0]["model_version"] == "clip-test"
    status_row = check_conn.execute("SELECT status FROM job_queue WHERE id = ?", [job_id]).fetchone()
    assert status_row["status"] == "done"
    check_conn.close()


def test_server_status_reports_node_progress_and_recent_results(tmp_path, monkeypatch):
    db_path = tmp_path / "server-status.db"
    conn = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    repo = Repository(conn)
    queue = JobQueue(conn)

    conn.execute(
        """INSERT INTO worker_nodes (id, display_name, platform, status)
           VALUES (?, ?, ?, ?)""",
        ["worker-1", "Worker 1", "linux", "online"],
    )

    master_done_image = repo.register_image(file_path="/photos/master-done.jpg")
    master_done_job = queue.enqueue(master_done_image, "metadata")
    assert master_done_job is not None
    queue.claim(batch_size=1, module="metadata")
    queue.mark_done(master_done_job)

    master_running_image = repo.register_image(file_path="/photos/master-running.jpg")
    master_running_job = queue.enqueue(master_running_image, "technical")
    assert master_running_job is not None
    queue.claim(batch_size=1, module="technical")

    worker_running_image = repo.register_image(file_path="/photos/worker-running.jpg")
    worker_running_job = queue.enqueue(worker_running_image, "objects")
    assert worker_running_job is not None
    running_claim = queue.claim_leased(worker_id="worker-1", batch_size=1, module="objects")
    assert len(running_claim) == 1

    worker_failed_image = repo.register_image(file_path="/photos/worker-failed.jpg")
    worker_failed_job = queue.enqueue(worker_failed_image, "faces")
    assert worker_failed_job is not None
    failed_claim = queue.claim_leased(worker_id="worker-1", batch_size=1, module="faces")
    assert len(failed_claim) == 1
    assert queue.mark_failed_leased(worker_failed_job, failed_claim[0]["lease_token"], "boom") is True

    conn.close()

    monkeypatch.setenv("IMGANALYZER_DB_PATH", str(db_path))
    import imganalyzer.server as server

    server._db_local = threading.local()
    status = server._handle_status({})

    assert status["remaining_images"] == 2
    assert status["nodes"]["master"]["runningJobs"] == 1
    assert status["nodes"]["master"]["activeModules"] == [
        {"module": "technical", "count": 1}
    ]

    worker = next(item for item in status["nodes"]["workers"] if item["id"] == "worker-1")
    assert worker["displayName"] == "Worker 1"
    assert worker["runningJobs"] == 1
    assert worker["activeModules"] == [{"module": "objects", "count": 1}]

    recent_by_job = {item["jobId"]: item for item in status["recent_results"]}
    assert recent_by_job[master_done_job]["nodeRole"] == "master"
    assert recent_by_job[master_done_job]["nodeId"] == "master"
    assert recent_by_job[master_done_job]["path"] == "/photos/master-done.jpg"
    assert recent_by_job[worker_failed_job]["nodeRole"] == "worker"
    assert recent_by_job[worker_failed_job]["nodeId"] == "worker-1"
    assert recent_by_job[worker_failed_job]["nodeLabel"] == "Worker 1"
    assert recent_by_job[worker_failed_job]["error"] == "boom"

    workers_list = server._handle_workers_list({})
    listed_worker = next(item for item in workers_list["workers"] if item["id"] == "worker-1")
    assert listed_worker["runningJobs"] == 1
    assert listed_worker["activeModules"] == [{"module": "objects", "count": 1}]


def test_run_forever_prints_progress_summary(monkeypatch):
    worker = DistributedWorker(coordinator_url="http://127.0.0.1:8765/", worker_id="worker-1")

    printed: list[str] = []
    monkeypatch.setattr("imganalyzer.pipeline.distributed_worker.console.print", lambda *args, **_kwargs: printed.append(" ".join(str(arg) for arg in args)))
    monkeypatch.setattr(worker, "_heartbeat_loop", lambda: None)
    monkeypatch.setattr(worker, "_claim_jobs", lambda: [{
        "id": 11,
        "imageId": 5,
        "module": "metadata",
        "leaseToken": "lease-11",
        "filePath": "/photos/job.jpg",
    }] if not worker._shutdown.is_set() else [])

    def _fake_process(_job: dict[str, object]) -> str:
        worker._clear_active(11)
        worker._shutdown.set()
        return "done"

    monkeypatch.setattr(worker, "_process_claimed_job", _fake_process)
    monkeypatch.setattr(worker, "_coordinator_call", lambda _method, _params: {"ok": True})

    stats = worker.run_forever()

    assert stats == {"done": 1, "failed": 0, "skipped": 0}
    assert any("Connected to coordinator as" in line for line in printed)
    assert any("Progress:" in line and "1 processed" in line for line in printed)


def test_run_forever_retries_worker_registration_after_timeout(monkeypatch):
    worker = DistributedWorker(coordinator_url="http://127.0.0.1:8765/", worker_id="worker-1")
    worker.poll_interval_seconds = 0

    printed: list[str] = []
    register_attempts = 0

    monkeypatch.setattr("imganalyzer.pipeline.distributed_worker.console.print", lambda *args, **_kwargs: printed.append(" ".join(str(arg) for arg in args)))
    monkeypatch.setattr(worker, "_heartbeat_loop", lambda: None)
    monkeypatch.setattr(worker, "_coordinator_call", lambda _method, _params: {"ok": True})

    def _fake_register() -> None:
        nonlocal register_attempts
        register_attempts += 1
        if register_attempts == 1:
            raise RuntimeError("Coordinator request failed: timed out")

    def _fake_claim_jobs() -> list[dict[str, object]]:
        worker._shutdown.set()
        return []

    monkeypatch.setattr(worker, "_register_worker", _fake_register)
    monkeypatch.setattr(worker, "_claim_jobs", _fake_claim_jobs)

    stats = worker.run_forever()

    assert stats == {"done": 0, "failed": 0, "skipped": 0}
    assert register_attempts == 2
    assert any("Coordinator unavailable during registration" in line for line in printed)
    assert any("Connected to coordinator as" in line for line in printed)


def test_run_forever_retries_claim_after_timeout(monkeypatch):
    worker = DistributedWorker(coordinator_url="http://127.0.0.1:8765/", worker_id="worker-1")
    worker.poll_interval_seconds = 0

    printed: list[str] = []
    register_attempts = 0
    claim_attempts = 0

    monkeypatch.setattr("imganalyzer.pipeline.distributed_worker.console.print", lambda *args, **_kwargs: printed.append(" ".join(str(arg) for arg in args)))
    monkeypatch.setattr(worker, "_heartbeat_loop", lambda: None)
    monkeypatch.setattr(worker, "_coordinator_call", lambda _method, _params: {"ok": True})

    def _fake_register() -> None:
        nonlocal register_attempts
        register_attempts += 1

    def _fake_claim_jobs() -> list[dict[str, object]]:
        nonlocal claim_attempts
        claim_attempts += 1
        if claim_attempts == 1:
            raise RuntimeError("Coordinator request failed: timed out")
        worker._shutdown.set()
        return []

    monkeypatch.setattr(worker, "_register_worker", _fake_register)
    monkeypatch.setattr(worker, "_claim_jobs", _fake_claim_jobs)

    stats = worker.run_forever()

    assert stats == {"done": 0, "failed": 0, "skipped": 0}
    assert register_attempts == 2
    assert claim_attempts == 2
    assert any("Coordinator unavailable while claiming jobs" in line for line in printed)
    assert any("Reconnected to coordinator as" in line for line in printed)
