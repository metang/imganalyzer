from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any


class JsonRpcServerProcess:
    """Line-oriented JSON-RPC client for a real imganalyzer.server subprocess."""

    def __init__(self, tmp_path: Path, timeout_s: float = 30.0) -> None:
        self.tmp_path = tmp_path
        self.timeout_s = timeout_s
        self.process: subprocess.Popen[str] | None = None
        self._next_id = 1
        self._stdout_lines: queue.Queue[str | None] = queue.Queue()
        self._stderr_lines: queue.Queue[str | None] = queue.Queue()
        self.notifications: list[dict[str, Any]] = []

    def start(self) -> None:
        db_path = self.tmp_path / "imganalyzer-e2e.db"
        cache_dir = self.tmp_path / "decoded-cache"
        model_cache_dir = self.tmp_path / "model-cache"
        env = os.environ.copy()
        env.update(
            {
                "PYTHONUNBUFFERED": "1",
                "IMGANALYZER_DB_PATH": str(db_path),
                "IMGANALYZER_DECODED_CACHE_DIR": str(cache_dir),
                "IMGANALYZER_MODEL_CACHE": str(model_cache_dir),
                "IMGANALYZER_PRE_DECODE_WORKERS": "1",
            }
        )
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "imganalyzer.server",
                "--decoded-cache-dir",
                str(cache_dir),
                "--pre-decode-workers",
                "1",
            ],
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert self.process.stdout is not None
        assert self.process.stderr is not None
        threading.Thread(
            target=self._drain_stream,
            args=(self.process.stdout, self._stdout_lines),
            daemon=True,
            name="imganalyzer-e2e-stdout",
        ).start()
        threading.Thread(
            target=self._drain_stream,
            args=(self.process.stderr, self._stderr_lines),
            daemon=True,
            name="imganalyzer-e2e-stderr",
        ).start()

        ready = self.read_message()
        assert ready["jsonrpc"] == "2.0"
        assert ready["method"] == "server/ready"
        assert ready["params"]["pid"] == self.process.pid
        self.notifications.append(ready)

    def close(self) -> None:
        proc = self.process
        if proc is None:
            return
        if proc.poll() is None:
            try:
                self.call("shutdown", {})
            except Exception:
                proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        self.process = None

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        req_id = self._next_id
        self._next_id += 1
        self.send(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": method,
                "params": params or {},
            }
        )

        while True:
            message = self.read_message()
            if message.get("id") == req_id:
                return message
            if "method" in message and "id" not in message:
                self.notifications.append(message)
                continue
            raise AssertionError(f"Unexpected JSON-RPC message while waiting for {req_id}: {message!r}")

    def send(self, message: dict[str, Any]) -> None:
        self.send_raw(json.dumps(message, separators=(",", ":")))

    def send_raw(self, raw: str) -> None:
        proc = self._running_process()
        assert proc.stdin is not None
        proc.stdin.write(raw + "\n")
        proc.stdin.flush()

    def read_message(self, timeout_s: float | None = None) -> dict[str, Any]:
        line = self.read_raw_line(timeout_s=timeout_s)
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"Server wrote non-JSON stdout line: {line!r}\nStderr:\n{self.stderr_text()}"
            ) from exc
        assert isinstance(message, dict), f"Expected JSON object from server, got {message!r}"
        return message

    def read_raw_line(self, timeout_s: float | None = None) -> str:
        timeout = self.timeout_s if timeout_s is None else timeout_s
        try:
            line = self._stdout_lines.get(timeout=timeout)
        except queue.Empty as exc:
            proc = self.process
            code = proc.poll() if proc is not None else None
            raise AssertionError(
                f"Timed out waiting for server stdout line; exit={code}\nStderr:\n{self.stderr_text()}"
            ) from exc
        if line is None:
            raise AssertionError(f"Server stdout closed unexpectedly\nStderr:\n{self.stderr_text()}")
        return line

    def stderr_text(self) -> str:
        lines: list[str] = []
        while True:
            try:
                line = self._stderr_lines.get_nowait()
            except queue.Empty:
                break
            if line is not None:
                lines.append(line)
        return "\n".join(lines)

    def _running_process(self) -> subprocess.Popen[str]:
        proc = self.process
        if proc is None:
            raise AssertionError("Server process was not started")
        if proc.poll() is not None:
            raise AssertionError(f"Server process exited with {proc.returncode}\nStderr:\n{self.stderr_text()}")
        return proc

    @staticmethod
    def _drain_stream(stream, target: queue.Queue[str | None]) -> None:  # type: ignore[no-untyped-def]
        try:
            for line in stream:
                target.put(line.rstrip("\r\n"))
        finally:
            target.put(None)
