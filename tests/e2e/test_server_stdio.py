from __future__ import annotations

from typing import Any

import pytest

from tests.e2e.jsonrpc_harness import JsonRpcServerProcess


pytestmark = pytest.mark.e2e


def test_server_stdio_ready_status_and_shutdown(jsonrpc_server: JsonRpcServerProcess) -> None:
    status = jsonrpc_server.call("status", {"lite": True, "cache": False})

    assert status["jsonrpc"] == "2.0"
    assert "error" not in status
    result = status["result"]
    assert result["total_images"] == 0
    assert result["totals"] == {
        "pending": 0,
        "running": 0,
        "done": 0,
        "failed": 0,
        "skipped": 0,
    }
    assert result["remaining_images"] == 0
    assert result["nodes"]["master"]["id"] == "master"
    assert result["nodes"]["master"]["role"] == "master"

    shutdown = jsonrpc_server.call("shutdown", {})
    assert shutdown == {"jsonrpc": "2.0", "id": shutdown["id"], "result": {"ok": True}}
    assert jsonrpc_server.process is not None
    assert jsonrpc_server.process.wait(timeout=5) == 0


def test_server_stdio_reports_protocol_errors(jsonrpc_server: JsonRpcServerProcess) -> None:
    missing = jsonrpc_server.call("e2e/does-not-exist", {})
    assert missing["error"]["code"] == -32601
    assert "Method not found" in missing["error"]["message"]

    invalid_request: dict[str, Any] = {"jsonrpc": "1.0", "id": "bad-version", "method": "status"}
    jsonrpc_server.send(invalid_request)
    invalid_response = jsonrpc_server.read_message()
    assert invalid_response["id"] == "bad-version"
    assert invalid_response["error"]["code"] == -32600

    jsonrpc_server.send_raw("{not json")
    parse_error = jsonrpc_server.read_message()
    assert parse_error["id"] is None
    assert parse_error["error"]["code"] == -32700
    assert "Parse error" in parse_error["error"]["message"]


def test_server_stdio_stdout_contains_only_json(tmp_path) -> None:
    server = JsonRpcServerProcess(tmp_path)
    server.start()
    try:
        server.call("status", {"lite": True, "cache": False})
        server.call("shutdown", {})
        assert server.process is not None
        server.process.wait(timeout=5)
    finally:
        if server.process is not None and server.process.poll() is None:
            server.process.kill()
            server.process.wait(timeout=5)

    assert server.process is not None
    assert server.process.returncode == 0
    assert not any("Traceback" in line for line in server.stderr_text().splitlines())
