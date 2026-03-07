from __future__ import annotations

import sqlite3
import threading

from imganalyzer.db.repository import Repository
from imganalyzer.db.schema import ensure_schema
from imganalyzer.pipeline.distributed_worker import DistributedWorker


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
