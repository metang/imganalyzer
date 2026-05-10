from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e.jsonrpc_harness import JsonRpcServerProcess


@pytest.fixture
def jsonrpc_server(tmp_path: Path) -> JsonRpcServerProcess:
    server = JsonRpcServerProcess(tmp_path)
    server.start()
    try:
        yield server
    finally:
        server.close()
