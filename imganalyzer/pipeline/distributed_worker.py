"""Distributed worker agent that leases jobs from a remote coordinator."""
from __future__ import annotations

import json
import ipaddress
import os
import platform
import signal
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any
from urllib import parse
from urllib import error, request

from rich.console import Console

from imganalyzer.db.repository import Repository
from imganalyzer.db.schema import ensure_schema
from imganalyzer.pipeline.distributed_payloads import extract_result_payload, seed_job_context
from imganalyzer.pipeline.modules import ModuleRunner
from imganalyzer.pipeline.worker import _emit_result

console = Console()


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
        self._opener = request.build_opener(
            request.ProxyHandler({}) if _should_bypass_proxy(url) else request.ProxyHandler()
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
        try:
            with self._opener.open(req, timeout=self.timeout_seconds) as resp:
                raw = resp.read()
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Coordinator HTTP {exc.code}: {body}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Coordinator request failed: {exc.reason}") from exc

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
        cloud_provider: str = "openai",
        detection_prompt: str | None = None,
        detection_threshold: float | None = None,
        face_match_threshold: float | None = None,
        verbose: bool = False,
        write_xmp: bool = True,
        path_mappings: list[tuple[str, str]] | None = None,
    ) -> None:
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
        self._shutdown = threading.Event()
        self._active_leases: dict[int, str] = {}
        self._active_lock = threading.Lock()

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
        self._coordinator_call(
            "workers/register",
            {
                "workerId": self.worker_id,
                "displayName": self.display_name,
                "platform": platform.platform(),
                "capabilities": {
                    "pid": os.getpid(),
                    "moduleFilter": self.module_filter,
                },
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
                console.print(f"[red]Worker heartbeat failed:[/red] {exc}")

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
                    console.print(f"[red]Lease heartbeat failed for job {job_id}:[/red] {exc}")

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
        result = self._coordinator_call("jobs/claim", params)
        jobs = result.get("jobs", [])
        return jobs if isinstance(jobs, list) else []

    def _process_claimed_job(self, job: dict[str, Any]) -> str:
        """Process one leased job and report its final state to the coordinator."""
        image_id = int(job["imageId"])
        module = str(job["module"])
        job_id = int(job["id"])
        lease_token = str(job["leaseToken"])
        path = str(job.get("filePath") or f"id={image_id}")
        short_path = Path(path).name
        start_ms = int(time.time() * 1000)
        conn: sqlite3.Connection | None = None

        console.print(
            f"  [blue]▶[/blue] [bold]{module}[/bold] ← {short_path} [dim](job {job_id})[/dim]"
        )

        try:
            conn, repo, runner = self._open_job_sandbox(job)
            result = runner.run(image_id, module)
            payload = extract_result_payload(conn, repo, image_id=image_id, module=module)
            elapsed = int(time.time() * 1000) - start_ms
            done = self._coordinator_call(
                "jobs/complete",
                {
                    "jobId": job_id,
                    "leaseToken": lease_token,
                    "payload": payload,
                    "noXmp": not self.write_xmp,
                },
            )
            if not bool(done.get("ok")):
                raise RuntimeError(f"Coordinator rejected completion for job {job_id}")
            keywords = result.get("keywords") if module == "cloud_ai" and result else None
            _emit_result(path, module, "done", elapsed, keywords=keywords)
            console.print(
                f"  [green]✓[/green] [bold]{module}[/bold] done in {elapsed / 1000:.1f}s ← {short_path}"
            )
            return "done"

        except ValueError as exc:
            err_lower = str(exc).lower()
            if "libraw cannot decode" in err_lower or "libraw postprocess failed" in err_lower:
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
                f"{short_path}: {error_msg}"
            )
            return "failed"

        finally:
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
                        f"[yellow]Failed to release job {job_id} during shutdown:[/yellow] {exc}"
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
            while not self._shutdown.is_set():
                if not registered:
                    try:
                        self._register_worker()
                    except Exception as exc:
                        hint = ""
                        reason = str(exc).lower()
                        if "timed out" in reason or "timeout" in reason:
                            hint = (
                                "\n[dim]  Hint: if the coordinator is on a remote host, "
                                "ensure the firewall allows inbound TCP on the coordinator port.\n"
                                "  On Windows: New-NetFirewallRule -DisplayName 'imganalyzer' "
                                "-Direction Inbound -Protocol TCP -LocalPort <PORT> -Action Allow\n"
                                "  On Linux:   sudo ufw allow <PORT>/tcp[/dim]"
                            )
                        console.print(
                            "[yellow]Coordinator unavailable during registration; "
                            f"retrying in {self.poll_interval_seconds:g}s:[/yellow] {exc}{hint}"
                        )
                        self._shutdown.wait(self.poll_interval_seconds)
                        continue

                    registered = True
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
                    jobs = self._claim_jobs()
                except Exception as exc:
                    registered = False
                    console.print(
                        "[yellow]Coordinator unavailable while claiming jobs; "
                        f"retrying in {self.poll_interval_seconds:g}s:[/yellow] {exc}"
                    )
                    self._shutdown.wait(self.poll_interval_seconds)
                    continue

                if not jobs:
                    self._shutdown.wait(self.poll_interval_seconds)
                    continue

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
                    console.print(f"[yellow]Worker lease release failed on shutdown:[/yellow] {exc}")
            if is_main and original_handler is not None:
                signal.signal(signal.SIGINT, original_handler)

        return stats
