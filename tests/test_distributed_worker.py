from __future__ import annotations

import sqlite3
import threading

from imganalyzer.db.repository import Repository
from imganalyzer.db.schema import ensure_schema
from imganalyzer.pipeline.distributed_worker import DistributedWorker
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
        def should_run(self, _image_id: int, _module: str) -> bool:
            return True

        def run(self, _image_id: int, _module: str):
            return {"description": "ok"}

    recorded: list[tuple[str, dict[str, object]]] = []

    def fake_call(method: str, params: dict[str, object]) -> dict[str, object]:
        recorded.append((method, dict(params)))
        return {"ok": True}

    worker._get_thread_db = lambda: (conn, repo, None, FakeRunner())  # type: ignore[method-assign]
    worker._coordinator_call = fake_call  # type: ignore[method-assign]

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
            },
        )
    ]


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
    conn = _make_test_db(tmp_path)
    db_path = tmp_path / "distributed-worker.db"
    conn.close()
    monkeypatch.setenv("IMGANALYZER_DB_PATH", str(db_path))

    worker = DistributedWorker(
        coordinator_url="http://127.0.0.1:8765/",
        path_mappings=[(r"Z:\photos", "/Volumes/photos")],
    )

    _, _, _, runner = worker._get_thread_db()
    assert runner.path_mappings == [(r"Z:\photos", "/Volumes/photos")]
    worker._close_thread_db()


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

