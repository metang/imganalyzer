from __future__ import annotations

import sys
import threading
from typing import Any

import imganalyzer.server as server

# server.py redirects stdout at import time for JSON-RPC mode; restore test stdout.
sys.stdout = server._real_stdout


def test_expensive_stdio_methods_have_backpressure_policies() -> None:
    stdio_expensive_methods = {
        "search",
        "gallery/listFolders",
        "gallery/listImagesChunk",
        "thumbnail",
        "thumbnails/batch",
        "fullimage",
        "cachedimage",
    }

    assert stdio_expensive_methods <= set(server._STDIO_ASYNC_METHODS)
    assert stdio_expensive_methods <= set(server._RPC_BACKPRESSURE_POLICIES)
    assert "faces/crop-batch" in server._ASYNC_METHODS
    assert "faces/crop-batch" in server._RPC_BACKPRESSURE_POLICIES


def test_dispatch_rejects_expensive_requests_when_backpressure_queue_is_full(
    monkeypatch,
) -> None:
    method = "test/slow-expensive"
    started = threading.Event()
    release = threading.Event()
    results_ready = threading.Event()
    lock = threading.Lock()
    results: list[tuple[int | str | None, Any]] = []
    errors: list[tuple[int | str | None, int, str]] = []

    def slow_handler(req_id: int | str | None, _params: dict[str, Any]) -> None:
        started.set()
        release.wait(timeout=5)
        server._send_result(req_id, {"ok": True})

    def record_result(req_id: int | str | None, result: Any) -> None:
        with lock:
            results.append((req_id, result))
            if len(results) >= 2:
                results_ready.set()

    def record_error(req_id: int | str | None, code: int, message: str) -> None:
        with lock:
            errors.append((req_id, code, message))

    monkeypatch.setitem(server._STDIO_ASYNC_METHODS, method, slow_handler)
    monkeypatch.setitem(
        server._RPC_BACKPRESSURE_POLICIES,
        method,
        server._RpcBackpressurePolicy(max_workers=1, max_pending=1),
    )
    monkeypatch.setattr(server, "_send_result", record_result)
    monkeypatch.setattr(server, "_send_error", record_error)

    old_queue = server._rpc_backpressure_queues.pop(method, None)
    if old_queue is not None:
        old_queue.shutdown(cancel_pending=True)

    try:
        server._dispatch({"jsonrpc": "2.0", "id": 1, "method": method, "params": {}})
        assert started.wait(timeout=1)

        server._dispatch({"jsonrpc": "2.0", "id": 2, "method": method, "params": {}})
        for req_id in range(3, 8):
            server._dispatch(
                {"jsonrpc": "2.0", "id": req_id, "method": method, "params": {}}
            )

        rpc_queue = server._rpc_backpressure_queues[method]
        assert rpc_queue.thread_count == 1
        assert [err[0] for err in errors] == [3, 4, 5, 6, 7]
        assert all(err[1] == server._RPC_BACKPRESSURE_ERROR_CODE for err in errors)
        assert all("Server is busy" in err[2] and method in err[2] for err in errors)
        assert results == []

        release.set()
        assert results_ready.wait(timeout=2)
        assert [item[0] for item in results] == [1, 2]
    finally:
        release.set()
        rpc_queue = server._rpc_backpressure_queues.pop(method, None)
        if rpc_queue is not None:
            rpc_queue.shutdown(cancel_pending=True)
