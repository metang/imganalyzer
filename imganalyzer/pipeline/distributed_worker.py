"""Distributed worker agent that leases jobs from a remote coordinator."""
from __future__ import annotations

import json
import ipaddress
import os
import platform
import signal
import socket
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib import parse
from urllib import error, request

from rich.console import Console
from rich.markup import escape

from imganalyzer.db.repository import Repository
from imganalyzer.db.schema import ensure_schema
from imganalyzer.pipeline.distributed_payloads import extract_result_payload, seed_job_context
from imganalyzer.pipeline.modules import ModuleRunner
from imganalyzer.pipeline.worker import _emit_result

console = Console()

# Modules that require local-AI packages (torch, transformers, etc.)
_LOCAL_AI_MODULES = {"objects", "blip2", "local_ai", "ocr", "faces", "embedding"}

# Modules that always work (pure Python / stdlib)
_ALWAYS_AVAILABLE_MODULES = {"metadata", "technical"}

# Modules that require a cloud SDK (checked separately per provider)
_CLOUD_MODULES = {"cloud_ai", "aesthetic"}


def _ensure_torch_runtime_env() -> None:
    """Set torch runtime env vars that must be present before importing torch."""
    # Some ops used by GroundingDINO are still missing on MPS; this enables
    # CPU fallback for unsupported kernels instead of raising hard failures.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _current_python_info() -> str:
    """Return a short description of the active Python interpreter."""
    exe = sys.executable or "unknown"
    ver = platform.python_version()
    env = os.environ.get("CONDA_DEFAULT_ENV", "")
    parts = [f"{exe} (Python {ver})"]
    if env:
        parts.append(f"conda env={env}")
    return ", ".join(parts)


def _probe_available_modules(cloud_provider: str = "copilot") -> list[str]:
    """Probe which analysis modules this worker can actually run.

    Checks for required dependencies at startup so the worker only claims
    jobs it can execute, avoiding wasteful claim-then-skip cycles.
    """
    _ensure_torch_runtime_env()
    available = list(_ALWAYS_AVAILABLE_MODULES)

    # Check local-AI deps (torch + transformers)
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        available.extend(["objects", "blip2", "local_ai", "ocr"])
    except ImportError:
        pass

    # faces needs insightface
    try:
        import insightface  # noqa: F401

        # insightface also needs torch (already checked above)
        if "objects" in available:
            available.append("faces")
    except ImportError:
        pass

    # embedding needs open_clip + torch
    try:
        import open_clip  # noqa: F401

        if "objects" in available:  # implies torch is present
            available.append("embedding")
    except ImportError:
        pass

    # cloud_ai / aesthetic need a cloud SDK
    provider = (cloud_provider or "").lower()
    cloud_ok = False
    if provider == "copilot":
        cloud_ok = True  # copilot uses CLI subprocess, no extra package needed
    elif provider == "openai":
        try:
            import openai  # noqa: F401
            cloud_ok = True
        except ImportError:
            pass
    elif provider == "anthropic":
        try:
            import anthropic  # noqa: F401
            cloud_ok = True
        except ImportError:
            pass
    elif provider == "google":
        try:
            import google.cloud.vision  # noqa: F401
            cloud_ok = True
        except ImportError:
            pass
    else:
        cloud_ok = True  # unknown provider — optimistic, let it fail at runtime

    if cloud_ok:
        available.extend(["cloud_ai", "aesthetic"])

    return sorted(set(available))


def _should_bypass_proxy(url: str) -> bool:
    """Return True when coordinator traffic should bypass environment proxies."""
    host = parse.urlparse(url).hostname
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return host.endswith(".local")
    return addr.is_private or addr.is_loopback or addr.is_link_local


class CoordinatorClient:
    """Minimal JSON-RPC-over-HTTP client for the distributed coordinator."""

    def __init__(
        self,
        url: str,
        auth_token: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.url = url
        self.auth_token = auth_token
        self.timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._next_id = 1
        self.bypass_proxy = _should_bypass_proxy(url)
        self._opener = request.build_opener(
            request.ProxyHandler({}) if self.bypass_proxy else request.ProxyHandler()
        )

    def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON-RPC request and return the result object."""
        with self._lock:
            request_id = self._next_id
            self._next_id += 1
        payload = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
        ).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        req = request.Request(self.url, data=payload, headers=headers, method="POST")
        started = time.monotonic()
        try:
            with self._opener.open(req, timeout=self.timeout_seconds) as resp:
                raw = resp.read()
        except TimeoutError as exc:
            elapsed = time.monotonic() - started
            raise RuntimeError(
                f"Coordinator request timed out for {method} after {elapsed:.1f}s "
                f"(timeout={self.timeout_seconds:.1f}s, url={self.url}, bypass_proxy={self.bypass_proxy})"
            ) from exc
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            elapsed = time.monotonic() - started
            raise RuntimeError(
                f"Coordinator HTTP {exc.code} for {method} after {elapsed:.1f}s: {body}"
            ) from exc
        except error.URLError as exc:
            elapsed = time.monotonic() - started
            reason = exc.reason
            reason_type = type(reason).__name__
            raise RuntimeError(
                f"Coordinator request failed for {method} after {elapsed:.1f}s "
                f"(timeout={self.timeout_seconds:.1f}s, url={self.url}, bypass_proxy={self.bypass_proxy}): "
                f"{reason} [{reason_type}]"
            ) from exc

        try:
            message = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Coordinator returned invalid JSON: {exc}") from exc

        if message.get("id") != request_id:
            raise RuntimeError(
                f"Coordinator response id mismatch for {method}: "
                f"expected {request_id}, got {message.get('id')}"
            )
        if "error" in message:
            err = message["error"] or {}
            raise RuntimeError(str(err.get("message") or err))
        result = message.get("result")
        if not isinstance(result, dict):
            raise RuntimeError(f"Coordinator returned invalid result for {method}")
        return result


class DistributedWorker:
    """Lease jobs from a coordinator and execute them locally."""

    def __init__(
        self,
        coordinator_url: str,
        worker_id: str | None = None,
        display_name: str | None = None,
        auth_token: str | None = None,
        batch_size: int = 1,
        poll_interval_seconds: float = 5.0,
        heartbeat_interval_seconds: float = 30.0,
        lease_ttl_seconds: int = 120,
        module_filter: str | None = None,
        force: bool = False,
        cloud_provider: str = "copilot",
        detection_prompt: str | None = None,
        detection_threshold: float | None = None,
        face_match_threshold: float | None = None,
        verbose: bool = False,
        write_xmp: bool = True,
        path_mappings: list[tuple[str, str]] | None = None,
        slow_job_log_seconds: float = 45.0,
        running_log_interval_seconds: float = 30.0,
    ) -> None:
        _ensure_torch_runtime_env()
        self.client = CoordinatorClient(coordinator_url, auth_token=auth_token)
        self.worker_id = worker_id or platform.node() or "imganalyzer-worker"
        self.display_name = display_name or self.worker_id
        self.batch_size = max(1, batch_size)
        self.poll_interval_seconds = max(0.5, poll_interval_seconds)
        self.heartbeat_interval_seconds = max(1.0, heartbeat_interval_seconds)
        self.lease_ttl_seconds = max(5, lease_ttl_seconds)
        self.module_filter = module_filter.strip() if module_filter else None
        self.force = force
        self.cloud_provider = cloud_provider
        self.detection_prompt = detection_prompt
        self.detection_threshold = detection_threshold
        self.face_match_threshold = face_match_threshold
        self.verbose = verbose
        self.write_xmp = write_xmp
        self.path_mappings = path_mappings or []
        self.slow_job_log_seconds = max(0.0, slow_job_log_seconds)
        self.running_log_interval_seconds = max(0.0, running_log_interval_seconds)

        # Probe which modules this worker can actually execute.
        # When a single --module filter is set, trust the user; otherwise
        # auto-detect to avoid claiming jobs that will always be skipped.
        if self.module_filter:
            self.supported_modules: list[str] | None = None  # single-module mode
        else:
            self.supported_modules = _probe_available_modules(cloud_provider)

        self._shutdown = threading.Event()
        self._active_leases: dict[int, str] = {}
        self._active_lock = threading.Lock()
        self._registration_attempts = 0
        self._claim_attempts = 0
        self._empty_claim_polls = 0

    def _job_runtime_logger(
        self,
        *,
        module: str,
        short_path: str,
        job_id: int,
        started_monotonic: float,
        finished: threading.Event,
    ) -> None:
        """Emit periodic logs while a claimed job is still executing."""
        interval = self.running_log_interval_seconds
        if interval <= 0:
            return

        while not finished.wait(interval):
            elapsed = time.monotonic() - started_monotonic
            console.print(
                f"[dim]  … {module} still running ({elapsed:.1f}s) ← "
                f"{escape(short_path)} (job {job_id})[/dim]"
            )

    def _proxy_env_summary(self) -> str:
        """Summarize proxy-related environment variables for diagnostics."""
        keys = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")
        values: list[str] = []
        for key in keys:
            raw = os.getenv(key) or os.getenv(key.lower())
            if raw:
                values.append(f"{key}={raw}")
        return "; ".join(values) if values else "<none>"

    def _resolve_host_summary(self) -> str:
        """Resolve coordinator host to IPs for debugging connectivity issues."""
        host = parse.urlparse(self.client.url).hostname
        if not host:
            return "<unparseable coordinator host>"
        try:
            infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        except OSError as exc:
            return f"{host}: resolution failed ({exc})"
        addrs = sorted({info[4][0] for info in infos if info[4]})
        return f"{host} -> {', '.join(addrs) if addrs else '<no addresses>'}"

    def _log_connectivity_context(self) -> None:
        """Print one-shot connectivity context at worker startup."""
        parsed = parse.urlparse(self.client.url)
        console.print(
            "[dim]Coordinator endpoint:[/dim] "
            f"{self.client.url} "
            f"[dim](scheme={parsed.scheme or '<none>'}, host={parsed.hostname or '<none>'}, "
            f"port={parsed.port or ('443' if parsed.scheme == 'https' else '80')}, "
            f"bypass_proxy={self.client.bypass_proxy}, timeout={self.client.timeout_seconds:.1f}s)[/dim]"
        )
        console.print(f"[dim]Host resolution:[/dim] {self._resolve_host_summary()}")
        console.print(f"[dim]Proxy env:[/dim] {self._proxy_env_summary()}")

    def _tcp_probe_summary(self, timeout_seconds: float = 3.0) -> str:
        """Perform a direct TCP probe to the coordinator host/port."""
        parsed = parse.urlparse(self.client.url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if not host:
            return "unavailable (invalid coordinator URL host)"
        start = time.monotonic()
        try:
            with socket.create_connection((host, port), timeout=timeout_seconds):
                elapsed = time.monotonic() - start
                return f"ok in {elapsed:.2f}s to {host}:{port}"
        except OSError as exc:
            elapsed = time.monotonic() - start
            return f"failed in {elapsed:.2f}s to {host}:{port}: {type(exc).__name__}: {exc}"

    def _handle_sigint(self, _signum: int, _frame: Any) -> None:
        """Request a graceful shutdown on Ctrl+C."""
        if not self._shutdown.is_set():
            console.print("[yellow]Stopping distributed worker after current job(s)...[/yellow]")
            self._shutdown.set()

    def _open_job_sandbox(self, job: dict[str, Any]) -> tuple[sqlite3.Connection, Repository, ModuleRunner]:
        """Build a temporary SQLite sandbox for one claimed job."""
        image_id = int(job["imageId"])
        file_path = str(job.get("filePath") or "")
        if not file_path:
            raise ValueError("Claimed job is missing filePath")

        conn = sqlite3.connect(":memory:", isolation_level=None, check_same_thread=False)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            ensure_schema(conn)
            repo = Repository(conn)
            seed_job_context(
                conn,
                repo,
                image_id=image_id,
                file_path=file_path,
                image_info=job.get("image") if isinstance(job.get("image"), dict) else None,
                context=job.get("context") if isinstance(job.get("context"), dict) else None,
            )
            runner = ModuleRunner(
                conn=conn,
                repo=repo,
                force=self.force,
                cloud_provider=self.cloud_provider,
                detection_prompt=self.detection_prompt,
                detection_threshold=self.detection_threshold,
                face_match_threshold=self.face_match_threshold,
                verbose=self.verbose,
                path_mappings=self.path_mappings,
            )
            return conn, repo, runner
        except Exception:
            conn.close()
            raise

    def _coordinator_call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Delegate a request to the coordinator client."""
        return self.client.call(method, params)

    def _register_worker(self) -> None:
        """Register this worker with the coordinator."""
        capabilities: dict[str, Any] = {
            "pid": os.getpid(),
            "moduleFilter": self.module_filter,
        }
        if self.supported_modules is not None:
            capabilities["supportedModules"] = self.supported_modules
        self._coordinator_call(
            "workers/register",
            {
                "workerId": self.worker_id,
                "displayName": self.display_name,
                "platform": platform.platform(),
                "capabilities": capabilities,
            },
        )

    def _mark_all_active(self, jobs: list[dict[str, Any]]) -> None:
        """Track currently leased jobs so the heartbeat thread can extend them."""
        with self._active_lock:
            for job in jobs:
                self._active_leases[int(job["id"])] = str(job["leaseToken"])

    def _clear_active(self, job_id: int) -> None:
        """Stop heartbeating a completed or released lease."""
        with self._active_lock:
            self._active_leases.pop(job_id, None)

    def _snapshot_active(self) -> list[tuple[int, str]]:
        """Return the currently active leases as a stable snapshot."""
        with self._active_lock:
            return list(self._active_leases.items())

    def _heartbeat_loop(self) -> None:
        """Refresh worker and lease liveness until shutdown."""
        while not self._shutdown.is_set():
            try:
                heartbeat = self._coordinator_call("workers/heartbeat", {"workerId": self.worker_id})
                released = int(heartbeat.get("releasedExpired", 0))
                if released and self.verbose:
                    console.print(
                        f"[dim]Coordinator released {released} expired lease(s) during heartbeat[/dim]"
                    )
            except Exception as exc:
                console.print(f"[red]Worker heartbeat failed:[/red] {escape(str(exc))}")

            for job_id, lease_token in self._snapshot_active():
                try:
                    result = self._coordinator_call(
                        "jobs/heartbeat",
                        {
                            "jobId": job_id,
                            "leaseToken": lease_token,
                            "extendTtlSeconds": self.lease_ttl_seconds,
                        },
                    )
                    if not bool(result.get("ok")):
                        console.print(
                            f"[yellow]Lease heartbeat rejected for job {job_id}; "
                            "completion may fail if another worker reclaimed it.[/yellow]"
                        )
                except Exception as exc:
                    console.print(
                        f"[red]Lease heartbeat failed for job {job_id}:[/red] {escape(str(exc))}"
                    )

            self._shutdown.wait(self.heartbeat_interval_seconds)

    def _claim_jobs(self) -> list[dict[str, Any]]:
        """Lease the next batch of jobs from the coordinator."""
        params: dict[str, Any] = {
            "workerId": self.worker_id,
            "batchSize": self.batch_size,
            "leaseTtlSeconds": self.lease_ttl_seconds,
            "force": self.force,
        }
        if self.module_filter:
            params["module"] = self.module_filter
        elif self.supported_modules is not None:
            params["modules"] = self.supported_modules
        result = self._coordinator_call("jobs/claim", params)
        jobs = result.get("jobs", [])
        return jobs if isinstance(jobs, list) else []

    def _coordinator_queue_summary(self) -> str | None:
        """Return a short queue snapshot for empty-poll diagnostics."""
        try:
            status = self._coordinator_call("status", {})
        except Exception:
            return None

        totals = status.get("totals", {})
        pending = int(totals.get("pending", 0) or 0)
        running = int(totals.get("running", 0) or 0)
        if pending <= 0 and running <= 0:
            return "queue empty"

        modules = status.get("modules", {})
        active: list[str] = []
        if isinstance(modules, dict):
            for module_name, module_stats in modules.items():
                if not isinstance(module_stats, dict):
                    continue
                pending_count = int(module_stats.get("pending", 0) or 0)
                running_count = int(module_stats.get("running", 0) or 0)
                if pending_count <= 0 and running_count <= 0:
                    continue
                summary = f"{module_name}"
                if running_count > 0:
                    summary += f" r{running_count}"
                if pending_count > 0:
                    summary += f" p{pending_count}"
                active.append(summary)
                if len(active) >= 5:
                    break

        summary = f"queue pending={pending}, running={running}"
        if active:
            summary += f"; active={', '.join(active)}"
        return summary

    def _process_claimed_job(self, job: dict[str, Any]) -> str:
        """Process one leased job and report its final state to the coordinator."""
        image_id = int(job["imageId"])
        module = str(job["module"])
        job_id = int(job["id"])
        lease_token = str(job["leaseToken"])
        path = str(job.get("filePath") or f"id={image_id}")
        short_path = Path(path).name
        start_ms = int(time.time() * 1000)
        started_monotonic = time.monotonic()
        conn: sqlite3.Connection | None = None
        sandbox_ms = 0
        run_ms = 0
        payload_ms = 0
        complete_ms = 0
        finished = threading.Event()
        runtime_thread: threading.Thread | None = None

        if self.running_log_interval_seconds > 0:
            runtime_thread = threading.Thread(
                target=self._job_runtime_logger,
                kwargs={
                    "module": module,
                    "short_path": short_path,
                    "job_id": job_id,
                    "started_monotonic": started_monotonic,
                    "finished": finished,
                },
                name=f"{self.worker_id}-job-{job_id}-runtime",
                daemon=True,
            )
            runtime_thread.start()

        console.print(
            f"  [blue]▶[/blue] [bold]{module}[/bold] ← {short_path} [dim](job {job_id})[/dim]"
        )

        try:
            sandbox_started = time.monotonic()
            conn, repo, runner = self._open_job_sandbox(job)
            sandbox_ms = int((time.monotonic() - sandbox_started) * 1000)

            run_started = time.monotonic()
            result = runner.run(image_id, module)
            run_ms = int((time.monotonic() - run_started) * 1000)

            payload_started = time.monotonic()
            payload = extract_result_payload(conn, repo, image_id=image_id, module=module)
            payload_ms = int((time.monotonic() - payload_started) * 1000)

            complete_started = time.monotonic()
            done = self._coordinator_call(
                "jobs/complete",
                {
                    "jobId": job_id,
                    "leaseToken": lease_token,
                    "payload": payload,
                    "noXmp": not self.write_xmp,
                },
            )
            complete_ms = int((time.monotonic() - complete_started) * 1000)
            elapsed = int((time.monotonic() - started_monotonic) * 1000)
            if not bool(done.get("ok")):
                raise RuntimeError(f"Coordinator rejected completion for job {job_id}")
            keywords = result.get("keywords") if module == "cloud_ai" and result else None
            _emit_result(path, module, "done", elapsed, keywords=keywords)
            console.print(
                f"  [green]✓[/green] [bold]{module}[/bold] done in {elapsed / 1000:.1f}s ← {short_path}"
            )
            if (
                self.slow_job_log_seconds > 0
                and elapsed >= int(self.slow_job_log_seconds * 1000)
            ):
                console.print(
                    "[yellow]  Slow job diagnostic:[/yellow] "
                    f"{module} job {job_id} total {elapsed / 1000:.1f}s "
                    f"(sandbox {sandbox_ms / 1000:.2f}s, "
                    f"run {run_ms / 1000:.2f}s, "
                    f"payload {payload_ms / 1000:.2f}s, "
                    f"complete {complete_ms / 1000:.2f}s)"
                )
            return "done"

        except ImportError as exc:
            elapsed = int(time.time() * 1000) - start_ms
            skipped = self._coordinator_call(
                "jobs/skip",
                {
                    "jobId": job_id,
                    "leaseToken": lease_token,
                    "reason": "missing_dependency",
                    "details": str(exc),
                },
            )
            if not bool(skipped.get("ok")):
                raise RuntimeError("Coordinator rejected missing_dependency skip") from exc
            _emit_result(path, module, "skipped", elapsed, f"missing dependency: {exc}")
            console.print(
                f"  [yellow]⊘[/yellow] [bold]{module}[/bold] skipped (missing dependency) in "
                f"{elapsed / 1000:.1f}s ← {short_path}"
            )
            return "skipped"

        except ValueError as exc:
            err_lower = str(exc).lower()
            if (
                "libraw cannot decode" in err_lower
                or "libraw postprocess failed" in err_lower
                or "pillow cannot decode" in err_lower
            ):
                elapsed = int(time.time() * 1000) - start_ms
                skipped = self._coordinator_call(
                    "jobs/skip",
                    {
                        "jobId": job_id,
                        "leaseToken": lease_token,
                        "reason": "corrupt_file",
                        "details": str(exc),
                    },
                )
                if not bool(skipped.get("ok")):
                    raise RuntimeError("Coordinator rejected corrupt_file skip") from exc
                _emit_result(path, module, "skipped", elapsed, f"corrupt file: {exc}")
                console.print(
                    f"  [yellow]⊘[/yellow] [bold]{module}[/bold] skipped (corrupt) in "
                    f"{elapsed / 1000:.1f}s ← {short_path}"
                )
                return "skipped"
            raise

        except Exception as exc:
            elapsed = int(time.time() * 1000) - start_ms
            error_msg = f"{type(exc).__name__}: {exc}"
            failed = self._coordinator_call(
                "jobs/fail",
                {"jobId": job_id, "leaseToken": lease_token, "error": error_msg},
            )
            if not bool(failed.get("ok")) and self.verbose:
                console.print(
                    f"[yellow]Coordinator rejected failure update for job {job_id}; "
                    "the lease may have expired.[/yellow]"
                )
            _emit_result(path, module, "failed", elapsed, error_msg)
            console.print(
                f"  [red]✗[/red] [bold]{module}[/bold] failed in {elapsed / 1000:.1f}s ← "
                f"{escape(short_path)}: {escape(error_msg)}"
            )
            return "failed"

        finally:
            finished.set()
            if runtime_thread is not None:
                runtime_thread.join(timeout=0.1)
            if conn is not None:
                conn.close()
            self._clear_active(job_id)

    def _release_all_active_leases(self) -> None:
        """Return any still-active leases to pending on shutdown."""
        for job_id, lease_token in self._snapshot_active():
            try:
                self._coordinator_call(
                    "jobs/release",
                    {"jobId": job_id, "leaseToken": lease_token},
                )
            except Exception as exc:
                if self.verbose:
                    console.print(
                        f"[yellow]Failed to release job {job_id} during shutdown:[/yellow] "
                        f"{escape(str(exc))}"
                    )
            finally:
                self._clear_active(job_id)

    def run_forever(self) -> dict[str, int]:
        """Run until interrupted, returning summary stats."""
        stats = {"done": 0, "failed": 0, "skipped": 0}
        original_handler = None
        registered = False
        heartbeat_started = False
        has_connected = False
        is_main = threading.current_thread() is threading.main_thread()
        if is_main:
            original_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._handle_sigint)

        heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"{self.worker_id}-heartbeat",
            daemon=True,
        )

        try:
            self._log_connectivity_context()
            if self.supported_modules is not None:
                from imganalyzer.db.repository import ALL_MODULES

                missing = sorted(set(ALL_MODULES) - set(self.supported_modules))
                console.print(
                    f"[dim]Supported modules:[/dim] {', '.join(self.supported_modules)}"
                )
                if missing:
                    console.print(
                        f"[yellow]Unavailable modules (missing deps):[/yellow] {', '.join(missing)}"
                    )
                # Fail fast when core local-AI deps are absent — the worker
                # would just claim and skip every objects/blip2/ocr/faces job.
                local_ai_available = _LOCAL_AI_MODULES & set(self.supported_modules)
                if not local_ai_available:
                    console.print(
                        "\n[bold red]ERROR: torch / transformers are not installed in "
                        "this Python environment.[/bold red]\n"
                        "[red]The worker cannot run any local-AI modules "
                        "(objects, blip2, ocr, faces, embedding).[/red]\n\n"
                        "You are likely running from the wrong conda environment.\n"
                        "  [bold]Current python:[/bold] "
                        + _current_python_info()
                        + "\n\n"
                        "[green]Fix:[/green] activate the environment that has "
                        "local-AI dependencies:\n"
                        "  [bold]conda activate imganalyzer312[/bold]\n"
                        "  imganalyzer run-distributed-worker ...\n\n"
                        "Or, to set up a fresh worker environment:\n"
                        "  [bold]bash scripts/setup_worker_env.sh[/bold]\n\n"
                        "[dim]To intentionally run only cloud modules, pass "
                        "--module cloud_ai[/dim]"
                    )
                    raise SystemExit(1)
            while not self._shutdown.is_set():
                if not registered:
                    self._registration_attempts += 1
                    try:
                        self._register_worker()
                    except Exception as exc:
                        exc_text = escape(str(exc))
                        hint = ""
                        reason = str(exc).lower()
                        probe = self._tcp_probe_summary()
                        if "timed out" in reason or "timeout" in reason:
                            hint = (
                                "\n[dim]  Hint: if the coordinator is on a remote host, "
                                "ensure firewall allows both coordinator port and python.exe inbound.\n"
                                "  On Windows (port):   New-NetFirewallRule -DisplayName 'imganalyzer Coordinator TCP' "
                                "-Direction Inbound -Protocol TCP -LocalPort <PORT> -Action Allow\n"
                                "  On Windows (python): New-NetFirewallRule -DisplayName 'imganalyzer Python Inbound' "
                                "-Direction Inbound -Program '<python.exe>' -Protocol TCP -Action Allow\n"
                                "  If still blocked: disable inbound 'TCP Query User ... python.exe' BLOCK rules.\n"
                                "  On Linux:   sudo ufw allow <PORT>/tcp[/dim]"
                            )
                        console.print(
                            "[yellow]Coordinator unavailable during registration; "
                            f"attempt {self._registration_attempts}, "
                            f"retrying in {self.poll_interval_seconds:g}s:[/yellow] {exc_text}\n"
                            f"[dim]  TCP probe: {probe}[/dim]{hint}"
                        )
                        self._shutdown.wait(self.poll_interval_seconds)
                        continue

                    registered = True
                    self._registration_attempts = 0
                    if not heartbeat_started:
                        heartbeat_thread.start()
                        heartbeat_started = True
                    if has_connected:
                        console.print(
                            f"[green]Reconnected to coordinator as[/green] {self.display_name} "
                            f"[dim]({self.worker_id})[/dim]"
                        )
                    else:
                        console.print(
                            f"[cyan]Connected to coordinator as[/cyan] {self.display_name} "
                            f"[dim]({self.worker_id})[/dim]"
                        )
                        has_connected = True

                try:
                    self._claim_attempts += 1
                    jobs = self._claim_jobs()
                except Exception as exc:
                    registered = False
                    probe = self._tcp_probe_summary()
                    exc_text = escape(str(exc))
                    console.print(
                        "[yellow]Coordinator unavailable while claiming jobs; "
                        f"attempt {self._claim_attempts}, "
                        f"retrying in {self.poll_interval_seconds:g}s:[/yellow] {exc_text}\n"
                        f"[dim]  TCP probe: {probe}[/dim]"
                    )
                    self._shutdown.wait(self.poll_interval_seconds)
                    continue

                self._claim_attempts = 0
                if not jobs:
                    self._empty_claim_polls += 1
                    poll_interval = max(self.poll_interval_seconds, 0.5)
                    idle_log_every = max(1, int(30 / poll_interval))
                    if self._empty_claim_polls % idle_log_every == 0:
                        queue_summary = self._coordinator_queue_summary()
                        queue_hint = f", {queue_summary}" if queue_summary else ""
                        console.print(
                            "[dim]No jobs claimed yet; "
                            f"still polling every {self.poll_interval_seconds:g}s "
                            f"(empty polls={self._empty_claim_polls}, "
                            f"active leases={len(self._snapshot_active())}{queue_hint})[/dim]"
                        )
                    self._shutdown.wait(self.poll_interval_seconds)
                    continue

                self._empty_claim_polls = 0
                console.print(
                    f"[cyan]Claimed {len(jobs)} job(s)[/cyan]"
                )
                self._mark_all_active(jobs)
                for job in jobs:
                    if self._shutdown.is_set():
                        break
                    status = self._process_claimed_job(job)
                    stats[status] = stats.get(status, 0) + 1
                    processed = stats.get("done", 0) + stats.get("failed", 0) + stats.get("skipped", 0)
                    console.print(
                        f"[dim]Progress:[/dim] {processed} processed — "
                        f"{stats.get('done', 0)} done, "
                        f"{stats.get('failed', 0)} failed, "
                        f"{stats.get('skipped', 0)} skipped"
                    )
        finally:
            self._shutdown.set()
            self._release_all_active_leases()
            try:
                self._coordinator_call("jobs/release-worker", {"workerId": self.worker_id})
            except Exception as exc:
                if self.verbose:
                    console.print(
                        f"[yellow]Worker lease release failed on shutdown:[/yellow] "
                        f"{escape(str(exc))}"
                    )
            if is_main and original_handler is not None:
                signal.signal(signal.SIGINT, original_handler)

        return stats
