from __future__ import annotations

import os
import sqlite3
import threading
import time
from unittest.mock import patch

import pytest

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


def test_process_claimed_job_retries_transient_db_lock_on_complete(tmp_path):
    conn = _make_test_db(tmp_path)
    repo = Repository(conn)
    image_id = repo.register_image(file_path="/photos/distributed-metadata.jpg")

    worker = DistributedWorker(
        coordinator_url="http://127.0.0.1:8765/",
        worker_id="worker-a",
        batch_size=1,
        write_xmp=False,
    )
    worker._db_lock_retry_attempts = 4
    worker._db_lock_retry_base_seconds = 0.0

    class FakeRunner:
        def run(self, _image_id: int, _module: str):
            return {"description": "ok"}

    call_counts = {"complete": 0}

    def fake_call(method: str, _params: dict[str, object]) -> dict[str, object]:
        if method != "jobs/complete":
            raise AssertionError(f"Unexpected coordinator method: {method}")
        call_counts["complete"] += 1
        if call_counts["complete"] < 3:
            raise RuntimeError("database is locked")
        return {"ok": True}

    worker._open_job_sandbox = lambda _job: (conn, repo, FakeRunner())  # type: ignore[method-assign]
    worker._coordinator_call = fake_call  # type: ignore[method-assign]

    with patch(
        "imganalyzer.pipeline.distributed_worker.extract_result_payload",
        return_value={"data": {"camera_make": "TestCam"}},
    ):
        status = worker._process_claimed_job(
            {
                "id": 8,
                "imageId": image_id,
                "module": "metadata",
                "leaseToken": "lease-999",
                "filePath": "/photos/distributed-metadata.jpg",
            }
        )

    assert status == "done"
    assert call_counts["complete"] == 3


def test_process_claimed_job_does_not_crash_when_fail_update_raises(tmp_path):
    conn = _make_test_db(tmp_path)
    repo = Repository(conn)
    image_id = repo.register_image(file_path="/photos/distributed-metadata.jpg")

    worker = DistributedWorker(
        coordinator_url="http://127.0.0.1:8765/",
        worker_id="worker-a",
        batch_size=1,
        write_xmp=False,
    )
    worker._db_lock_retry_attempts = 2
    worker._db_lock_retry_base_seconds = 0.0

    class FailingRunner:
        def run(self, _image_id: int, _module: str):
            raise RuntimeError("boom")

    fail_calls = {"count": 0}

    def fake_call(method: str, _params: dict[str, object]) -> dict[str, object]:
        if method != "jobs/fail":
            raise AssertionError(f"Unexpected coordinator method: {method}")
        fail_calls["count"] += 1
        raise RuntimeError("database is locked")

    worker._open_job_sandbox = lambda _job: (conn, repo, FailingRunner())  # type: ignore[method-assign]
    worker._coordinator_call = fake_call  # type: ignore[method-assign]

    status = worker._process_claimed_job(
        {
            "id": 9,
            "imageId": image_id,
            "module": "metadata",
            "leaseToken": "lease-1000",
            "filePath": "/photos/distributed-metadata.jpg",
        }
    )

    assert status == "failed"
    assert fail_calls["count"] == 2


@pytest.mark.parametrize(
    ("runner_error", "reason"),
    [
        (ImportError("module torch is not installed"), "missing_dependency"),
        (ValueError("Pillow cannot decode broken.jpg: bad bytes"), "corrupt_file"),
    ],
)
def test_process_claimed_job_skip_rejection_is_nonfatal(tmp_path, runner_error, reason):
    conn = _make_test_db(tmp_path)
    repo = Repository(conn)
    image_id = repo.register_image(file_path="/photos/distributed-metadata.jpg")

    worker = DistributedWorker(
        coordinator_url="http://127.0.0.1:8765/",
        worker_id="worker-a",
        batch_size=1,
        write_xmp=False,
    )

    class FailingRunner:
        def run(self, _image_id: int, _module: str):
            raise runner_error

    calls: list[tuple[str, dict[str, object]]] = []

    def fake_call(method: str, params: dict[str, object]) -> dict[str, object]:
        calls.append((method, dict(params)))
        if method != "jobs/skip":
            raise AssertionError(f"Unexpected coordinator method: {method}")
        return {"ok": False}

    worker._open_job_sandbox = lambda _job: (conn, repo, FailingRunner())  # type: ignore[method-assign]
    worker._coordinator_call = fake_call  # type: ignore[method-assign]

    status = worker._process_claimed_job(
        {
            "id": 10,
            "imageId": image_id,
            "module": "metadata",
            "leaseToken": "lease-1001",
            "filePath": "/photos/distributed-metadata.jpg",
        }
    )

    assert status == "failed"
    assert calls == [
        (
            "jobs/skip",
            {
                "jobId": 10,
                "leaseToken": "lease-1001",
                "reason": reason,
                "details": str(runner_error),
            },
        )
    ]


def test_process_claimed_job_non_corrupt_value_error_reports_failed(tmp_path):
    conn = _make_test_db(tmp_path)
    repo = Repository(conn)
    image_id = repo.register_image(file_path="/photos/distributed-metadata.jpg")

    worker = DistributedWorker(
        coordinator_url="http://127.0.0.1:8765/",
        worker_id="worker-a",
        batch_size=1,
        write_xmp=False,
    )

    class FailingRunner:
        def run(self, _image_id: int, _module: str):
            raise ValueError("unexpected metadata parse error")

    calls: list[tuple[str, dict[str, object]]] = []

    def fake_call(method: str, params: dict[str, object]) -> dict[str, object]:
        calls.append((method, dict(params)))
        if method != "jobs/fail":
            raise AssertionError(f"Unexpected coordinator method: {method}")
        return {"ok": True}

    worker._open_job_sandbox = lambda _job: (conn, repo, FailingRunner())  # type: ignore[method-assign]
    worker._coordinator_call = fake_call  # type: ignore[method-assign]

    status = worker._process_claimed_job(
        {
            "id": 12,
            "imageId": image_id,
            "module": "metadata",
            "leaseToken": "lease-1002",
            "filePath": "/photos/distributed-metadata.jpg",
        }
    )

    assert status == "failed"
    assert calls == [
        (
            "jobs/fail",
            {
                "jobId": 12,
                "leaseToken": "lease-1002",
                "error": "ValueError: unexpected metadata parse error",
            },
        )
    ]


def test_process_claimed_job_malformed_payload_returns_failed(monkeypatch):
    worker = DistributedWorker(
        coordinator_url="http://127.0.0.1:8765/",
        worker_id="worker-a",
        batch_size=1,
        write_xmp=False,
    )

    called = {"count": 0}

    def fake_call(_method: str, _params: dict[str, object]) -> dict[str, object]:
        called["count"] += 1
        return {"ok": True}

    worker._coordinator_call = fake_call  # type: ignore[method-assign]
    monkeypatch.setattr(
        "imganalyzer.pipeline.distributed_worker._emit_result",
        lambda *_args, **_kwargs: None,
    )

    status = worker._process_claimed_job(
        {
            "id": 13,
            "module": "metadata",
            "leaseToken": "lease-1003",
            "filePath": "/photos/distributed-metadata.jpg",
        }
    )

    assert status == "failed"
    assert called["count"] == 1


def test_process_claimed_job_logs_slow_diagnostics(tmp_path, monkeypatch):
    conn = _make_test_db(tmp_path)
    repo = Repository(conn)
    image_id = repo.register_image(file_path="/photos/distributed-metadata.jpg")

    worker = DistributedWorker(
        coordinator_url="http://127.0.0.1:8765/",
        worker_id="worker-a",
        batch_size=1,
        write_xmp=False,
        slow_job_log_seconds=0.05,
        running_log_interval_seconds=0.0,
    )

    class FakeRunner:
        def run(self, _image_id: int, _module: str):
            return {"description": "ok"}

    printed: list[str] = []
    monkeypatch.setattr(
        "imganalyzer.pipeline.distributed_worker.console.print",
        lambda *args, **_kwargs: printed.append(" ".join(str(arg) for arg in args)),
    )

    mono_values = [0.0, 0.0, 0.01, 0.01, 0.12, 0.12, 0.13, 0.13, 0.20, 0.20, 0.20]
    mono_idx = {"value": 0}

    def _fake_monotonic() -> float:
        idx = mono_idx["value"]
        if idx < len(mono_values):
            value = mono_values[idx]
        else:
            value = mono_values[-1]
        mono_idx["value"] = idx + 1
        return value

    monkeypatch.setattr("imganalyzer.pipeline.distributed_worker.time.monotonic", _fake_monotonic)

    worker._open_job_sandbox = lambda _job: (conn, repo, FakeRunner())  # type: ignore[method-assign]
    worker._coordinator_call = lambda _method, _params: {"ok": True}  # type: ignore[method-assign]

    with patch(
        "imganalyzer.pipeline.distributed_worker.extract_result_payload",
        return_value={"data": {"camera_make": "TestCam"}},
    ):
        status = worker._process_claimed_job(
            {
                "id": 6,
                "imageId": image_id,
                "module": "metadata",
                "leaseToken": "lease-456",
                "filePath": "/photos/distributed-metadata.jpg",
            }
        )

    assert status == "done"
    assert any("Slow job diagnostic" in line for line in printed)


def test_process_claimed_job_logs_still_running(tmp_path, monkeypatch):
    conn = _make_test_db(tmp_path)
    repo = Repository(conn)
    image_id = repo.register_image(file_path="/photos/distributed-metadata.jpg")

    worker = DistributedWorker(
        coordinator_url="http://127.0.0.1:8765/",
        worker_id="worker-a",
        batch_size=1,
        write_xmp=False,
        slow_job_log_seconds=0.0,
        running_log_interval_seconds=0.01,
    )

    class SlowRunner:
        def run(self, _image_id: int, _module: str):
            time.sleep(0.05)
            return {"description": "ok"}

    printed: list[str] = []
    monkeypatch.setattr(
        "imganalyzer.pipeline.distributed_worker.console.print",
        lambda *args, **_kwargs: printed.append(" ".join(str(arg) for arg in args)),
    )

    worker._open_job_sandbox = lambda _job: (conn, repo, SlowRunner())  # type: ignore[method-assign]
    worker._coordinator_call = lambda _method, _params: {"ok": True}  # type: ignore[method-assign]

    with patch(
        "imganalyzer.pipeline.distributed_worker.extract_result_payload",
        return_value={"data": {"camera_make": "TestCam"}},
    ):
        status = worker._process_claimed_job(
            {
                "id": 7,
                "imageId": image_id,
                "module": "metadata",
                "leaseToken": "lease-789",
                "filePath": "/photos/distributed-metadata.jpg",
            }
        )

    assert status == "done"
    assert any("still running" in line for line in printed)


def test_should_bypass_proxy_for_private_coordinator_urls():
    assert _should_bypass_proxy("http://10.0.0.215:8765/jsonrpc") is True
    assert _should_bypass_proxy("http://127.0.0.1:8765/jsonrpc") is True
    assert _should_bypass_proxy("http://coordinator.local:8765/jsonrpc") is True
    assert _should_bypass_proxy("https://example.com/jsonrpc") is False


def test_distributed_worker_init_sets_mps_fallback_even_with_module_filter(monkeypatch):
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    DistributedWorker(
        coordinator_url="http://127.0.0.1:8765/",
        module_filter="objects",
    )
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"


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
    repo.upsert_objects(
        image_id, {"detected_objects": [], "has_person": False, "has_text": False}
    )
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


def test_server_jobs_skip_cascades_corrupt_file_to_pending_sibling_jobs(tmp_path, monkeypatch):
    db_path = tmp_path / "server-corrupt-skip.db"
    conn = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    repo = Repository(conn)
    queue = JobQueue(conn)

    image_path = tmp_path / "broken.tiff"
    image_id = repo.register_image(file_path=str(image_path))
    first_job = queue.enqueue(image_id, "metadata")
    second_job = queue.enqueue(image_id, "technical")
    assert first_job is not None
    assert second_job is not None

    claimed = queue.claim_leased(worker_id="worker-1", lease_ttl_seconds=120, batch_size=1)
    assert len(claimed) == 1

    monkeypatch.setenv("IMGANALYZER_DB_PATH", str(db_path))
    import imganalyzer.server as server

    server._db_local = threading.local()

    result = server._handle_jobs_skip(
        {
            "jobId": claimed[0]["id"],
            "leaseToken": claimed[0]["lease_token"],
            "reason": "corrupt_file",
            "details": "Pillow cannot decode broken.tiff: Invalid value for samples per pixel",
        }
    )

    assert result == {"ok": True}

    rows = conn.execute(
        "SELECT id, status, skip_reason FROM job_queue WHERE image_id = ? ORDER BY id",
        [image_id],
    ).fetchall()
    assert [(row["id"], row["status"], row["skip_reason"]) for row in rows] == [
        (first_job, "skipped", "corrupt_file"),
        (second_job, "skipped", "corrupt_file"),
    ]

    corrupt = conn.execute(
        "SELECT file_path, error_msg FROM corrupt_files WHERE image_id = ?",
        [image_id],
    ).fetchone()
    assert corrupt is not None
    assert corrupt["file_path"] == str(image_path)
    assert "Pillow cannot decode broken.tiff" in corrupt["error_msg"]


def test_server_jobs_claim_scans_past_prereq_blocked_jobs(tmp_path, monkeypatch):
    db_path = tmp_path / "server-claim-scan.db"
    conn = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    repo = Repository(conn)
    queue = JobQueue(conn)

    for idx in range(300):
        image_id = repo.register_image(file_path=f"/nas/photos/blocked-{idx}.jpg")
        job_id = queue.enqueue(image_id, "faces")
        assert job_id is not None
        conn.execute(
            "UPDATE job_queue SET queued_at = ? WHERE id = ?",
            [f"2024-01-01 00:{idx // 60:02d}:{idx % 60:02d}", job_id],
        )

    eligible_image_id = repo.register_image(file_path="/nas/photos/eligible.jpg")
    eligible_job_id = queue.enqueue(eligible_image_id, "faces")
    assert eligible_job_id is not None
    conn.execute(
        "UPDATE job_queue SET queued_at = ? WHERE id = ?",
        ["2024-01-01 05:59:59", eligible_job_id],
    )
    repo.upsert_objects(
        eligible_image_id,
        {"detected_objects": [], "has_person": False, "has_text": False},
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("IMGANALYZER_DB_PATH", str(db_path))
    import imganalyzer.server as server

    server._db_local = threading.local()
    server._handle_workers_register({"workerId": "worker-1", "displayName": "Worker 1"})
    result = server._handle_jobs_claim({"workerId": "worker-1", "batchSize": 1, "module": "faces"})

    assert len(result["jobs"]) == 1
    job = result["jobs"][0]
    assert job["id"] == eligible_job_id
    assert job["imageId"] == eligible_image_id


def test_server_jobs_claim_skips_when_prerequisite_failed(tmp_path, monkeypatch):
    db_path = tmp_path / "server-claim-prereq-failed.db"
    conn = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    repo = Repository(conn)
    queue = JobQueue(conn)

    image_id = repo.register_image(file_path="/nas/photos/prereq-failed.jpg")
    objects_job = queue.enqueue(image_id, "objects")
    faces_job = queue.enqueue(image_id, "faces")
    assert objects_job is not None
    assert faces_job is not None
    queue.mark_failed(objects_job, "ImportError: PyTorch not installed")
    conn.close()

    monkeypatch.setenv("IMGANALYZER_DB_PATH", str(db_path))
    import imganalyzer.server as server

    server._db_local = threading.local()
    server._handle_workers_register({"workerId": "worker-1", "displayName": "Worker 1"})
    result = server._handle_jobs_claim({"workerId": "worker-1", "batchSize": 1, "module": "faces"})
    assert result["jobs"] == []

    check = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
    check.row_factory = sqlite3.Row
    row = check.execute(
        "SELECT status, skip_reason FROM job_queue WHERE id = ?",
        [faces_job],
    ).fetchone()
    check.close()
    assert row is not None
    assert row["status"] == "skipped"
    assert row["skip_reason"] == "prerequisite_objects_failed"


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


def test_persist_cloud_ai_payload_ignores_provider_primary_key(tmp_path):
    from imganalyzer.pipeline.distributed_payloads import persist_result_payload

    conn = _make_test_db(tmp_path)
    repo = Repository(conn)

    first_image_id = repo.register_image(file_path="/nas/photos/existing.jpg")
    repo.upsert_cloud_ai(first_image_id, "openai", {"description": "existing row"})

    second_image_id = repo.register_image(file_path="/nas/photos/new.jpg")
    persist_result_payload(
        conn,
        repo,
        image_id=second_image_id,
        module="cloud_ai",
        payload={
            "data": {
                "providers": [
                    {
                        "id": 1,
                        "provider": "openai",
                        "description": "new row",
                        "keywords": ["sunset"],
                    }
                ]
            }
        },
    )

    cloud_rows = conn.execute(
        """SELECT image_id, provider, description
           FROM analysis_cloud_ai
           ORDER BY image_id""",
    ).fetchall()
    assert [(row["image_id"], row["provider"], row["description"]) for row in cloud_rows] == [
        (first_image_id, "openai", "existing row"),
        (second_image_id, "openai", "new row"),
    ]
    conn.close()


def test_extract_cloud_ai_payload_includes_blip2(tmp_path):
    from imganalyzer.pipeline.distributed_payloads import extract_result_payload

    conn = _make_test_db(tmp_path)
    repo = Repository(conn)
    image_id = repo.register_image(file_path="/nas/photos/cloud-sync-source.jpg")

    repo.upsert_cloud_ai(image_id, "openai", {
        "description": "provider description",
        "keywords": ["provider", "tags"],
    })
    repo.upsert_blip2(image_id, {
        "description": "caption from cloud pass",
        "scene_type": "indoor",
        "keywords": ["caption", "keyword"],
    })

    payload = extract_result_payload(conn, repo, image_id=image_id, module="cloud_ai")
    assert "blip2" in payload
    assert payload["blip2"]["description"] == "caption from cloud pass"
    assert payload["blip2"]["scene_type"] == "indoor"
    conn.close()


def test_persist_cloud_ai_payload_writes_blip2_caption_fields(tmp_path):
    from imganalyzer.pipeline.distributed_payloads import persist_result_payload

    conn = _make_test_db(tmp_path)
    repo = Repository(conn)
    image_id = repo.register_image(file_path="/nas/photos/cloud-sync-target.jpg")

    persist_result_payload(
        conn,
        repo,
        image_id=image_id,
        module="cloud_ai",
        payload={
            "data": {
                "providers": [
                    {
                        "provider": "openai",
                        "description": "provider description",
                        "keywords": ["provider", "tags"],
                    }
                ]
            },
            "blip2": {
                "description": "caption from worker",
                "scene_type": "portrait",
                "keywords": ["boy", "computer"],
            },
        },
    )

    blip2_row = conn.execute(
        "SELECT description, scene_type, keywords FROM analysis_blip2 WHERE image_id = ?",
        [image_id],
    ).fetchone()
    assert blip2_row is not None
    assert blip2_row["description"] == "caption from worker"
    assert blip2_row["scene_type"] == "portrait"
    assert "boy" in str(blip2_row["keywords"])
    conn.close()


def test_extract_aesthetic_payload_includes_perception(tmp_path):
    from imganalyzer.pipeline.distributed_payloads import extract_result_payload

    conn = _make_test_db(tmp_path)
    repo = Repository(conn)

    image_id = repo.register_image(file_path="/nas/photos/aesthetic-sync.jpg")
    repo.upsert_aesthetic(image_id, {
        "aesthetic_score": 6.4,
        "aesthetic_label": "Good",
        "aesthetic_reason": "",
        "provider": "siglip-v2.5",
    })
    repo.upsert_perception(image_id, {
        "perception_iaa": 6.9,
        "perception_iaa_label": "Good",
        "perception_iqa": 5.8,
        "perception_iqa_label": "Average",
        "perception_ista": 7.1,
        "perception_ista_label": "Very Good",
    })

    payload = extract_result_payload(conn, repo, image_id=image_id, module="aesthetic")
    assert payload["data"]["aesthetic_score"] == 6.4
    assert payload["perception"]["perception_iaa"] == 6.9
    assert payload["perception"]["perception_iqa"] == 5.8
    assert payload["perception"]["perception_ista"] == 7.1
    conn.close()


def test_persist_aesthetic_payload_persists_perception(tmp_path):
    from imganalyzer.pipeline.distributed_payloads import persist_result_payload

    conn = _make_test_db(tmp_path)
    repo = Repository(conn)

    image_id = repo.register_image(file_path="/nas/photos/aesthetic-sync-apply.jpg")
    persist_result_payload(
        conn,
        repo,
        image_id=image_id,
        module="aesthetic",
        payload={
            "data": {
                "aesthetic_score": 5.7,
                "aesthetic_label": "Average",
                "aesthetic_reason": "",
            },
            "perception": {
                "perception_iaa": 6.0,
                "perception_iaa_label": "Good",
                "perception_iqa": 5.1,
                "perception_iqa_label": "Average",
                "perception_ista": 4.6,
                "perception_ista_label": "Average",
            },
        },
    )

    stored_aesthetic = repo.get_analysis(image_id, "aesthetic")
    stored_perception = repo.get_analysis(image_id, "perception")
    assert stored_aesthetic is not None
    assert stored_perception is not None
    assert stored_aesthetic["aesthetic_score"] == 5.7
    assert stored_perception["perception_iaa"] == 6.0
    assert stored_perception["perception_iqa"] == 5.1
    assert stored_perception["perception_ista"] == 4.6
    conn.close()


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


def test_heartbeat_loop_keeps_leases_alive_during_shutdown(monkeypatch):
    worker = DistributedWorker(coordinator_url="http://127.0.0.1:8765/", worker_id="worker-1")
    worker.supported_modules = None  # bypass startup dep check in tests
    worker.heartbeat_interval_seconds = 2.0
    worker._mark_all_active([{"id": 99, "leaseToken": "lease-99"}])
    worker._shutdown.set()

    calls: list[tuple[str, dict[str, object]]] = []
    sleep_calls: list[float] = []

    def _fake_call(method: str, params: dict[str, object]) -> dict[str, object]:
        calls.append((method, dict(params)))
        if method == "workers/heartbeat":
            return {"releasedExpired": 0}
        if method == "jobs/heartbeat":
            worker._clear_active(99)
            return {"ok": True}
        raise AssertionError(f"Unexpected coordinator method: {method}")

    monkeypatch.setattr(worker, "_coordinator_call", _fake_call)
    monkeypatch.setattr(
        "imganalyzer.pipeline.distributed_worker.time.sleep",
        lambda seconds: sleep_calls.append(seconds),
    )

    worker._heartbeat_loop()

    assert [method for method, _params in calls] == ["workers/heartbeat", "jobs/heartbeat"]
    assert calls[1][1] == {"jobId": 99, "leaseToken": "lease-99", "extendTtlSeconds": 120}
    assert sleep_calls == [1.0]
    assert worker._snapshot_active() == []


def test_run_forever_prints_progress_summary(monkeypatch):
    worker = DistributedWorker(coordinator_url="http://127.0.0.1:8765/", worker_id="worker-1")
    worker.supported_modules = None  # bypass startup dep check in tests

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
    worker.supported_modules = None  # bypass startup dep check in tests
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
    worker.supported_modules = None  # bypass startup dep check in tests
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
