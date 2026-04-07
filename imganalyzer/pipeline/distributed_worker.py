"""Distributed worker agent that leases jobs from a remote coordinator."""
from __future__ import annotations

import json
import ipaddress
import os
import platform
import signal
import socket
import sqlite3
import subprocess
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
_LOCAL_AI_MODULES = {"objects", "caption", "faces", "embedding", "perception"}

# Modules that always work (pure Python / stdlib)
_ALWAYS_AVAILABLE_MODULES = {"metadata"}

# Modules that must run on the master device because they need
# the original full-resolution image for accurate results:
#   technical — noise/sharpness estimates are destroyed by resizing
#   faces — small faces lost at 1024px, embedding quality degrades
_MASTER_ONLY_MODULES = frozenset({"technical", "faces"})
_TRANSIENT_DB_LOCK_MARKERS = (
    "database is locked",
    "database table is locked",
    "database schema is locked",
)


def _detect_git_repo() -> Path | None:
    """Return the git repo root containing this package, or None."""
    try:
        pkg_dir = Path(__file__).resolve().parent.parent
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=pkg_dir, capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            return Path(r.stdout.strip())
    except Exception:
        pass
    return None


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


def _ensure_ollama_running() -> None:
    """Start Ollama if it isn't already running.

    Tries ``ollama serve`` as a detached background process.  Safe to call
    even if Ollama is already running — Ollama exits immediately with a
    non-zero code if the port is already bound, so this is a no-op in that
    case.  Waits up to 10 s for the server to become responsive.
    """
    import os as _os
    import shutil
    import subprocess
    import sys as _sys
    from urllib import request as _req
    from urllib.error import URLError

    ollama_url = _os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")

    # Already running?
    try:
        req = _req.Request(f"{ollama_url}/api/tags", method="GET")
        with _req.urlopen(req, timeout=3) as resp:
            resp.read()
        return  # Ollama is reachable
    except Exception:
        pass

    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        _sys.stderr.write("[init] Ollama binary not found in PATH; skipping auto-start\n")
        return

    _sys.stderr.write("[init] Starting Ollama server…\n")
    try:
        kwargs: dict[str, Any] = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        if _os.name == "nt":
            kwargs["creationflags"] = (
                subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
            )
        else:
            kwargs["start_new_session"] = True
        subprocess.Popen([ollama_bin, "serve"], **kwargs)
    except Exception as exc:
        _sys.stderr.write(f"[init] Failed to start Ollama: {exc}\n")
        return

    # Wait for Ollama to become responsive
    import time
    for _ in range(20):
        time.sleep(0.5)
        try:
            req = _req.Request(f"{ollama_url}/api/tags", method="GET")
            with _req.urlopen(req, timeout=2) as resp:
                resp.read()
            _sys.stderr.write("[init] Ollama server is ready\n")
            return
        except Exception:
            pass
    _sys.stderr.write("[init] Ollama started but not responsive after 10s\n")


def _probe_available_modules(_cloud_provider: str | None = None) -> list[str]:
    """Probe which analysis modules this worker can actually run.

    Checks for required dependencies at startup so the worker only claims
    jobs it can execute, avoiding wasteful claim-then-skip cycles.

    The optional cloud provider arg is accepted for backward compatibility
    with older setup scripts that passed it positionally.
    """
    _ensure_torch_runtime_env()
    _ensure_ollama_running()
    available = list(_ALWAYS_AVAILABLE_MODULES)

    # Check local-AI deps (torch + transformers)
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        available.append("objects")
    except ImportError:
        pass

    # caption needs Ollama server running with a vision model
    try:
        from urllib import request as _req
        from urllib.error import URLError
        import os as _os

        ollama_url = _os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        req = _req.Request(f"{ollama_url}/api/tags", method="GET")
        with _req.urlopen(req, timeout=5) as resp:
            resp.read()
        available.append("caption")
    except (URLError, OSError) as exc:
        import sys as _sys

        _sys.stderr.write(
            f"[probe] caption unavailable: Ollama not reachable at "
            f"{_os.environ.get('OLLAMA_URL', 'http://localhost:11434')} ({exc})\n"
        )
    except Exception as exc:
        import sys as _sys

        _sys.stderr.write(f"[probe] caption unavailable: {exc}\n")

    # faces needs insightface — but is master-only (needs full-res
    # image for small face detection and accurate embeddings).
    # Only probe so we can log if it *would* be available.
    try:
        import insightface  # noqa: F401

        if "objects" in available:
            import sys as _sys

            _sys.stderr.write(
                "[probe] faces available but master-only "
                "(needs full-resolution image)\n"
            )
    except ImportError:
        pass

    # embedding needs open_clip + torch
    try:
        import open_clip  # noqa: F401

        if "objects" in available:  # implies torch is present
            available.append("embedding")
    except ImportError:
        pass

    # perception needs UniPercept dependency and CUDA runtime.
    try:
        import torch
        import transformers  # noqa: F401

        if torch.cuda.is_available():
            import unipercept_reward  # noqa: F401

            available.append("perception")
    except ImportError:
        pass

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


def _is_transient_db_lock_error(exc: Exception) -> bool:
    """Return True when an exception looks like a transient SQLite lock conflict."""
    msg = str(exc).lower()
    return any(marker in msg for marker in _TRANSIENT_DB_LOCK_MARKERS)


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

    def fetch_decoded_image(self, image_id: int) -> bytes | None:
        """Fetch a pre-decoded image from the coordinator's HTTP endpoint.

        Returns raw image bytes (WebP/JPEG) or None if not available.
        Retries once on transient network errors.
        """
        parsed = parse.urlparse(self.url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        img_url = f"{base_url}/images/decoded/{image_id}"

        headers: dict[str, str] = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        last_exc: Exception | None = None
        for attempt in range(2):
            req = request.Request(img_url, headers=headers, method="GET")
            try:
                with self._opener.open(req, timeout=self.timeout_seconds) as resp:
                    if resp.status == 200:
                        return resp.read()
                    # 404 = image not cached; don't retry
                    if resp.status == 404:
                        return None
                    sys.stderr.write(
                        f"[worker] HTTP {resp.status} fetching decoded image"
                        f" {image_id} (attempt {attempt + 1})\n"
                    )
            except error.HTTPError as exc:
                if exc.code == 404:
                    return None
                last_exc = exc
                sys.stderr.write(
                    f"[worker] HTTP {exc.code} fetching decoded image"
                    f" {image_id} (attempt {attempt + 1}): {exc}\n"
                )
            except (error.URLError, TimeoutError, OSError) as exc:
                last_exc = exc
                sys.stderr.write(
                    f"[worker] Network error fetching decoded image"
                    f" {image_id} (attempt {attempt + 1}): {exc}\n"
                )
            if attempt == 0:
                time.sleep(0.5)

        if last_exc is not None:
            sys.stderr.write(
                f"[worker] Giving up on decoded image {image_id}"
                f" after 2 attempts: {last_exc}\n"
            )
        return None

    def fetch_decoded_metadata(self, image_id: int) -> dict[str, Any] | None:
        """Fetch sidecar metadata for a pre-decoded image."""
        parsed = parse.urlparse(self.url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        meta_url = f"{base_url}/images/decoded/{image_id}/meta"

        headers: dict[str, str] = {"Accept": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        req = request.Request(meta_url, headers=headers, method="GET")
        try:
            with self._opener.open(req, timeout=self.timeout_seconds) as resp:
                if resp.status == 200:
                    raw = resp.read()
                    meta = json.loads(raw.decode("utf-8"))
                    # Decode base64 binary fields
                    from imganalyzer.cache.decoded_store import _decode_binary_fields
                    return _decode_binary_fields(meta)
                return None
        except (error.HTTPError, error.URLError, TimeoutError, json.JSONDecodeError):
            return None


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
        detection_prompt: str | None = None,
        detection_threshold: float | None = None,
        face_match_threshold: float | None = None,
        verbose: bool = False,
        write_xmp: bool = True,
        path_mappings: list[tuple[str, str]] | None = None,
        slow_job_log_seconds: float = 45.0,
        running_log_interval_seconds: float = 30.0,
        auto_update: bool = False,
        auto_update_interval_seconds: float = 60.0,
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
        self.detection_prompt = detection_prompt
        self.detection_threshold = detection_threshold
        self.face_match_threshold = face_match_threshold
        self.verbose = verbose
        self.write_xmp = write_xmp
        self.path_mappings = path_mappings or []
        self.slow_job_log_seconds = max(0.0, slow_job_log_seconds)
        self.running_log_interval_seconds = max(0.0, running_log_interval_seconds)
        self._db_lock_retry_attempts = 4
        self._db_lock_retry_base_seconds = 0.25
        self._coordinator_timeout_retry_attempts = 3
        self._coordinator_timeout_retry_base_seconds = 0.75

        # Probe which modules this worker can actually execute.
        # When a single --module filter is set, trust the user; otherwise
        # auto-detect to avoid claiming jobs that will always be skipped.
        if self.module_filter:
            self.supported_modules: list[str] | None = None  # single-module mode
        else:
            self.supported_modules = _probe_available_modules()

        self._shutdown = threading.Event()
        self._active_leases: dict[int, str] = {}
        self._active_lock = threading.Lock()
        self._desired_state = "active"
        self._pause_reason: str | None = None
        self._pause_notice_emitted = False
        self._registration_attempts = 0
        self._claim_attempts = 0
        self._empty_claim_polls = 0
        self._startup_lease_recovery_done = False

        # In-memory cache for pre-fetched decoded images from the coordinator.
        # Key: image_id, Value: (image_bytes, metadata_dict | None).
        # Avoids re-downloading when multiple jobs share the same image.
        self._prefetch_cache: dict[int, tuple[bytes, dict[str, Any] | None]] = {}
        self._prefetch_cache_max = 100

        # Auto-update: check git for new commits and restart when found.
        self._auto_update = auto_update
        self._auto_update_interval = max(60.0, auto_update_interval_seconds)
        self._last_update_check = 0.0
        self._update_pending = False
        self._repo_dir: Path | None = None
        if auto_update:
            self._repo_dir = _detect_git_repo()

    # ── Module re-probing ────────────────────────────────────────────────

    _REPROBE_INTERVAL = 60.0  # seconds between re-probe attempts

    def _maybe_reprobe_modules(self) -> None:
        """Re-probe for modules that were missing at startup.

        Ollama may not have been running when the worker started, so
        ``caption`` would be absent from ``supported_modules``.  Re-probe
        periodically so the worker picks up caption once Ollama is ready.
        """
        if self.module_filter or self.supported_modules is None:
            return
        if "caption" in self.supported_modules:
            return  # nothing to re-probe
        now = time.monotonic()
        if now - getattr(self, "_last_reprobe", 0.0) < self._REPROBE_INTERVAL:
            return
        self._last_reprobe = now
        new_modules = _probe_available_modules()
        added = set(new_modules) - set(self.supported_modules)
        if added:
            self.supported_modules = new_modules
            console.print(
                f"[green]Re-probe discovered new modules:[/green] "
                f"{', '.join(sorted(added))}"
            )

    # ── Auto-update ───────────────────────────────────────────────────────

    def _maybe_check_for_update(self) -> None:
        """Check git for new commits if enough time has passed."""
        if not self._auto_update or self._repo_dir is None:
            return
        now = time.monotonic()
        if now - self._last_update_check < self._auto_update_interval:
            return
        self._last_update_check = now
        if self._has_remote_update():
            console.print(
                "[yellow]Code update detected on remote; "
                "finishing current work and restarting…[/yellow]"
            )
            self._update_pending = True
            self._shutdown.set()

    def _has_remote_update(self) -> bool:
        """Return True if the remote branch has commits ahead of local HEAD."""
        try:
            subprocess.run(
                ["git", "fetch", "origin"],
                cwd=self._repo_dir, capture_output=True, timeout=30,
            )
            local = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self._repo_dir, capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            upstream = subprocess.run(
                ["git", "rev-parse", "@{u}"],
                cwd=self._repo_dir, capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            return bool(upstream) and local != upstream
        except Exception:
            return False

    def _pull_and_restart(self) -> None:
        """Pull latest code and re-exec the worker process."""
        assert self._repo_dir is not None
        # Record HEAD before pulling so we can tell if anything changed.
        old_head = ""
        try:
            old_head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self._repo_dir, capture_output=True, text=True, timeout=10,
            ).stdout.strip()
        except Exception:
            pass

        console.print("[cyan]Pulling latest code…[/cyan]")
        result = subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=self._repo_dir, capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            # ff-only failed (local diverged) — force-reset to upstream.
            console.print(
                "[yellow]Fast-forward failed; resetting to upstream…[/yellow]"
            )
            branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self._repo_dir, capture_output=True, text=True, timeout=10,
            ).stdout.strip() or "main"
            reset = subprocess.run(
                ["git", "reset", "--hard", f"origin/{branch}"],
                cwd=self._repo_dir, capture_output=True, text=True, timeout=30,
            )
            if reset.returncode != 0:
                console.print(
                    f"[red]git reset failed (exit {reset.returncode}):[/red] "
                    f"{reset.stderr.strip()}"
                )
                self._update_pending = False
                return

        new_head = ""
        try:
            new_head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self._repo_dir, capture_output=True, text=True, timeout=10,
            ).stdout.strip()
        except Exception:
            pass

        if old_head and new_head and old_head == new_head:
            console.print("[dim]Already up to date — skipping restart.[/dim]")
            self._update_pending = False
            return

        console.print(f"[green]Updated:[/green] {old_head[:8]}→{new_head[:8]}")
        console.print("[cyan]Restarting worker…[/cyan]")
        os.execv(sys.executable, [sys.executable] + sys.argv)

    # ── Job runtime ───────────────────────────────────────────────────────

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
                detection_prompt=self.detection_prompt,
                detection_threshold=self.detection_threshold,
                face_match_threshold=self.face_match_threshold,
                verbose=self.verbose,
                path_mappings=self.path_mappings,
            )

            # If coordinator signals a decoded cache is available, fetch the
            # pre-decoded image and prime the runner's cache.  This removes
            # the worker's dependency on NAS.
            if job.get("hasDecodedCache"):
                primed = self._prime_from_coordinator(runner, image_id, file_path)
                if not primed:
                    raise FileNotFoundError(
                        f"Could not fetch decoded image {image_id} from coordinator"
                    )

            return conn, repo, runner
        except Exception:
            conn.close()
            raise

    def _batch_prefetch_decoded(self, jobs: list[dict[str, Any]]) -> None:
        """Prefetch decoded images for a batch of claimed jobs.

        Identifies unique image IDs that have ``hasDecodedCache`` set and
        downloads their bytes + metadata in parallel before the sequential
        processing loop begins.  Results are stored in ``_prefetch_cache``.
        """
        ids_to_fetch: list[int] = []
        seen: set[int] = set()
        for job in jobs:
            if not job.get("hasDecodedCache"):
                continue
            img_id = int(job.get("imageId", 0))
            if img_id and img_id not in seen and img_id not in self._prefetch_cache:
                ids_to_fetch.append(img_id)
                seen.add(img_id)

        if not ids_to_fetch:
            return

        import concurrent.futures

        console.print(
            f"  [dim]Prefetching {len(ids_to_fetch)} decoded image(s) from coordinator…[/dim]"
        )

        def _fetch_one(image_id: int) -> tuple[int, bytes | None, dict[str, Any] | None]:
            img_bytes = self.client.fetch_decoded_image(image_id)
            meta: dict[str, Any] | None = None
            if img_bytes is not None:
                meta = self.client.fetch_decoded_metadata(image_id)
            return image_id, img_bytes, meta

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_fetch_one, iid): iid for iid in ids_to_fetch}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    image_id, img_bytes, meta = fut.result()
                    if img_bytes is not None:
                        # Evict oldest entries if cache is full
                        while len(self._prefetch_cache) >= self._prefetch_cache_max:
                            first_key = next(iter(self._prefetch_cache))
                            del self._prefetch_cache[first_key]
                        self._prefetch_cache[image_id] = (img_bytes, meta)
                except Exception as exc:
                    iid = futures[fut]
                    sys.stderr.write(
                        f"[worker] Prefetch failed for image {iid}: {exc}\n"
                    )

    def _prime_from_coordinator(
        self, runner: ModuleRunner, image_id: int, file_path: str
    ) -> bool:
        """Fetch a pre-decoded image from the coordinator and prime the runner cache.

        Returns True if the cache was successfully primed.
        """
        import io
        import numpy as np
        from PIL import Image as PILImage

        try:
            # Check prefetch cache first
            cached = self._prefetch_cache.pop(image_id, None)
            if cached is not None:
                img_bytes, meta = cached
            else:
                img_bytes = self.client.fetch_decoded_image(image_id)
                meta = None

            if img_bytes is None:
                sys.stderr.write(
                    f"[worker] Decoded image not available from coordinator"
                    f" for image {image_id} ({file_path})\n"
                )
                return False

            pil_img = PILImage.open(io.BytesIO(img_bytes))
            pil_img.load()
            rgb_array = np.asarray(pil_img.convert("RGB"))

            path = Path(file_path)
            image_data = {
                "rgb_array": rgb_array,
                "width": rgb_array.shape[1],
                "height": rgb_array.shape[0],
                "format": path.suffix.lstrip(".").upper(),
                "is_raw": False,
            }
            runner.prime_image_cache(path, image_data)

            # Also fetch sidecar metadata for the metadata module
            if meta is None:
                meta = self.client.fetch_decoded_metadata(image_id)
            if meta:
                runner._cached_header_data = meta

            return True

        except Exception as exc:
            sys.stderr.write(
                f"[worker] Failed to prime image {image_id}: {exc}\n"
            )
            return False

    def _coordinator_call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Delegate a request to the coordinator client."""
        return self.client.call(method, params)

    def _coordinator_call_with_lock_retry(
        self,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Retry coordinator calls for transient SQLite lock conflicts/timeouts."""
        lock_attempts = max(1, self._db_lock_retry_attempts)
        lock_base_delay = max(0.0, self._db_lock_retry_base_seconds)
        timeout_attempts = max(1, self._coordinator_timeout_retry_attempts)
        timeout_base_delay = max(0.0, self._coordinator_timeout_retry_base_seconds)
        total_attempts = max(lock_attempts, timeout_attempts)
        for attempt in range(1, total_attempts + 1):
            try:
                return self._coordinator_call(method, params)
            except Exception as exc:
                message = str(exc).lower()
                is_lock = _is_transient_db_lock_error(exc)
                is_timeout = "timed out" in message or "timeout" in message
                if is_lock and attempt < lock_attempts:
                    delay = min(2.0, lock_base_delay * (2 ** (attempt - 1)))
                    if self.verbose:
                        console.print(
                            "[yellow]Coordinator database is locked; retrying "
                            f"{method} ({attempt}/{lock_attempts - 1}) in {delay:.2f}s[/yellow]"
                        )
                    if delay > 0:
                        self._shutdown.wait(delay)
                    continue
                if is_timeout and attempt < timeout_attempts:
                    delay = min(4.0, timeout_base_delay * (2 ** (attempt - 1)))
                    if self.verbose:
                        console.print(
                            "[yellow]Coordinator request timed out; retrying "
                            f"{method} ({attempt}/{timeout_attempts - 1}) in {delay:.2f}s[/yellow]"
                        )
                    if delay > 0:
                        self._shutdown.wait(delay)
                    continue
                raise
        raise RuntimeError(f"Coordinator call failed unexpectedly for {method}")

    def _is_method_missing_error(self, exc: Exception) -> bool:
        """Return True when the coordinator reports an unknown RPC method."""
        text = str(exc).lower()
        return "method not found" in text or "unknown method" in text

    def _apply_worker_heartbeat(self, heartbeat: dict[str, Any]) -> int:
        """Apply coordinator heartbeat state and return reclaimed lease count."""
        released = int(heartbeat.get("releasedExpired", 0) or 0)
        desired_state = self._normalize_desired_state(heartbeat.get("desiredState"))
        self._desired_state = desired_state
        raw_reason = heartbeat.get("stateReason")
        reason = str(raw_reason).strip() if raw_reason else ""
        self._pause_reason = reason or None
        return released

    def _refresh_worker_control_state(self) -> int:
        """Fetch worker control state once before claim/resume transitions."""
        heartbeat = self._coordinator_call_with_lock_retry(
            "workers/heartbeat",
            {"workerId": self.worker_id},
        )
        released = self._apply_worker_heartbeat(heartbeat)
        if released and self.verbose:
            console.print(
                f"[dim]Coordinator released {released} expired lease(s) during startup heartbeat[/dim]"
            )
        return released

    def _recover_startup_leases(self) -> int:
        """Release stale leases for this worker id from previous interrupted sessions."""
        try:
            result = self._coordinator_call_with_lock_retry(
                "jobs/release-worker",
                {"workerId": self.worker_id},
            )
        except Exception as exc:
            if self._is_method_missing_error(exc):
                if self.verbose:
                    console.print(
                        "[yellow]Coordinator does not support jobs/release-worker; "
                        "skipping startup lease recovery.[/yellow]"
                    )
                return 0
            raise
        released = int(result.get("released", 0) or 0)
        if released:
            console.print(
                f"[yellow]Recovered {released} stale lease(s) for worker {self.worker_id}[/yellow]"
            )
        return released

    def _try_report_skip(
        self,
        *,
        job_id: int,
        lease_token: str,
        reason: str,
        details: str,
    ) -> bool:
        """Best-effort skip reporting; never raises to the caller."""
        try:
            skipped = self._coordinator_call_with_lock_retry(
                "jobs/skip",
                {
                    "jobId": job_id,
                    "leaseToken": lease_token,
                    "reason": reason,
                    "details": details,
                },
            )
            if bool(skipped.get("ok")):
                return True
            console.print(
                f"[yellow]Coordinator rejected {reason} skip for job {job_id}; "
                "the lease may have expired.[/yellow]"
            )
            return False
        except Exception as exc:
            console.print(
                f"[yellow]Failed to report {reason} skip for job {job_id}:[/yellow] "
                f"{escape(str(exc))}"
            )
            return False

    def _safe_emit_result(
        self,
        *,
        path: str,
        module: str,
        status: str,
        elapsed_ms: int,
        error: str = "",
        keywords: list[str] | None = None,
    ) -> None:
        """Emit a result notification without allowing secondary crashes."""
        try:
            _emit_result(path, module, status, elapsed_ms, error, keywords=keywords)
        except Exception as exc:
            console.print(
                "[yellow]Failed to emit run/result notification:[/yellow] "
                f"{escape(str(exc))}"
            )

    def _report_failure(
        self,
        *,
        job_id: int,
        lease_token: str,
        path: str,
        module: str,
        short_path: str,
        started_monotonic: float,
        exc: Exception,
    ) -> str:
        """Best-effort failure reporting that never raises to the caller."""
        elapsed = int((time.monotonic() - started_monotonic) * 1000)
        error_msg = f"{type(exc).__name__}: {exc}"
        try:
            failed = self._coordinator_call_with_lock_retry(
                "jobs/fail",
                {"jobId": job_id, "leaseToken": lease_token, "error": error_msg},
            )
            if not bool(failed.get("ok")) and self.verbose:
                console.print(
                    f"[yellow]Coordinator rejected failure update for job {job_id}; "
                    "the lease may have expired.[/yellow]"
                )
        except Exception as mark_exc:
            if self.verbose:
                console.print(
                    f"[yellow]Failed to report job {job_id} as failed:[/yellow] "
                    f"{escape(str(mark_exc))}"
                )
        self._safe_emit_result(
            path=path,
            module=module,
            status="failed",
            elapsed_ms=elapsed,
            error=error_msg,
        )
        console.print(
            f"  [red]✗[/red] [bold]{module}[/bold] failed in {elapsed / 1000:.1f}s ← "
            f"{escape(short_path)}: {escape(error_msg)}"
        )
        return "failed"

    def _register_worker(self) -> None:
        """Register this worker with the coordinator."""
        capabilities: dict[str, Any] = {
            "pid": os.getpid(),
            "moduleFilter": self.module_filter,
        }
        try:
            import torch

            capabilities["cuda"] = bool(torch.cuda.is_available())
            capabilities["mps"] = bool(
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            )
        except Exception:
            capabilities["cuda"] = False
            capabilities["mps"] = False
        if self.supported_modules is not None:
            capabilities["supportedModules"] = self.supported_modules
        self._coordinator_call_with_lock_retry(
            "workers/register",
            {
                "workerId": self.worker_id,
                "displayName": self.display_name,
                "platform": platform.platform(),
                "capabilities": capabilities,
            },
        )

    def _normalize_desired_state(self, desired_state: Any) -> str:
        """Normalize coordinator control state values."""
        state = str(desired_state or "active").strip().lower()
        if state == "resume":
            return "active"
        return state or "active"

    def _is_paused(self) -> bool:
        return self._normalize_desired_state(self._desired_state) != "active"

    def _mark_all_active(self, jobs: list[dict[str, Any]]) -> None:
        """Track currently leased jobs so the heartbeat thread can extend them."""
        with self._active_lock:
            for job in jobs:
                try:
                    self._active_leases[int(job["id"])] = str(job["leaseToken"])
                except (KeyError, TypeError, ValueError):
                    if self.verbose:
                        console.print(
                            f"[yellow]Ignoring malformed claimed job lease payload:[/yellow] {job!r}"
                        )

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
        while True:
            active_leases = self._snapshot_active()
            if self._shutdown.is_set() and not active_leases:
                break
            try:
                heartbeat = self._coordinator_call_with_lock_retry(
                    "workers/heartbeat",
                    {"workerId": self.worker_id},
                )
                released = self._apply_worker_heartbeat(heartbeat)
                if released and self.verbose:
                    console.print(
                        f"[dim]Coordinator released {released} expired lease(s) during heartbeat[/dim]"
                    )
            except Exception as exc:
                console.print(f"[red]Worker heartbeat failed:[/red] {escape(str(exc))}")

            for job_id, lease_token in active_leases:
                try:
                    result = self._coordinator_call_with_lock_retry(
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
                            "completion may fail if another worker reclaimed it. "
                            "Dropping local lease tracking.[/yellow]"
                        )
                        self._clear_active(job_id)
                except Exception as exc:
                    console.print(
                        f"[red]Lease heartbeat failed for job {job_id}:[/red] {escape(str(exc))}"
                    )

            if self._shutdown.is_set():
                # Keep leases alive while draining in-flight jobs after Ctrl+C.
                time.sleep(min(self.heartbeat_interval_seconds, 1.0))
            else:
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
        result = self._coordinator_call_with_lock_retry("jobs/claim", params)
        jobs = result.get("jobs", [])
        return jobs if isinstance(jobs, list) else []

    def _coordinator_queue_summary(self) -> str | None:
        """Return a short queue snapshot for empty-poll diagnostics."""
        try:
            status = self._coordinator_call_with_lock_retry(
                "status",
                {
                    "lite": True,
                    "cache": True,
                    "cache_ttl_ms": 1000,
                    "include_recent_results": False,
                    "include_module_avg_ms": False,
                },
            )
        except Exception:
            return None

        totals = status.get("totals", {})
        pending = int(totals.get("pending", 0) or 0)
        running = int(totals.get("running", 0) or 0)
        if pending <= 0 and running <= 0:
            return "queue empty"

        modules = status.get("modules", {})
        active: list[str] = []
        pending_modules: list[str] = []
        if isinstance(modules, dict):
            for module_name, module_stats in modules.items():
                if not isinstance(module_stats, dict):
                    continue
                pending_count = int(module_stats.get("pending", 0) or 0)
                running_count = int(module_stats.get("running", 0) or 0)
                if pending_count <= 0 and running_count <= 0:
                    continue
                if pending_count > 0:
                    pending_modules.append(module_name)
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
        claimable_modules: set[str] | None = None
        if self.module_filter:
            claimable_modules = {self.module_filter}
        elif self.supported_modules is not None:
            claimable_modules = set(self.supported_modules)
        if claimable_modules is not None and pending_modules:
            claimable_pending = sorted(set(pending_modules) & claimable_modules)
            if not claimable_pending:
                pending_preview = ", ".join(sorted(set(pending_modules))[:4])
                if self.module_filter:
                    summary += (
                        f"; no claimable pending modules for worker filter "
                        f"'{self.module_filter}' (pending: {pending_preview})"
                    )
                else:
                    summary += (
                        "; no claimable pending modules for this worker "
                        f"(pending: {pending_preview})"
                    )
        return summary

    def _process_claimed_job(self, job: dict[str, Any]) -> str:
        """Process one leased job and report its final state to the coordinator."""
        started_monotonic = time.monotonic()
        image_id_raw = job.get("imageId")
        path = str(job.get("filePath") or f"id={image_id_raw}")
        short_path = Path(path).name
        try:
            image_id = int(job["imageId"])
            module = str(job["module"])
            job_id = int(job["id"])
            lease_token = str(job["leaseToken"])
        except (KeyError, TypeError, ValueError) as exc:
            elapsed = int((time.monotonic() - started_monotonic) * 1000)
            console.print(
                "[yellow]Malformed claimed job payload; marking as failed if possible:[/yellow] "
                f"{escape(str(exc))} payload={job!r}"
            )
            job_id_raw = job.get("id")
            lease_token_raw = str(job.get("leaseToken", "")).strip()
            module_name = str(job.get("module", "unknown"))
            if isinstance(job_id_raw, int) and lease_token_raw:
                self._report_failure(
                    job_id=job_id_raw,
                    lease_token=lease_token_raw,
                    path=path,
                    module=module_name,
                    short_path=short_path,
                    started_monotonic=started_monotonic,
                    exc=ValueError(f"Malformed claimed job payload: {exc}"),
                )
            else:
                self._safe_emit_result(
                    path=path,
                    module=module_name,
                    status="failed",
                    elapsed_ms=elapsed,
                    error=f"Malformed claimed job payload: {exc}",
                )
            return "failed"

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

            processing_ms = int((time.monotonic() - started_monotonic) * 1000)
            complete_started = time.monotonic()
            done = self._coordinator_call_with_lock_retry(
                "jobs/complete",
                {
                    "jobId": job_id,
                    "leaseToken": lease_token,
                    "payload": payload,
                    "noXmp": not self.write_xmp,
                    "processingMs": processing_ms,
                },
            )
            complete_ms = int((time.monotonic() - complete_started) * 1000)
            elapsed = int((time.monotonic() - started_monotonic) * 1000)
            if not bool(done.get("ok")):
                raise RuntimeError(f"Coordinator rejected completion for job {job_id}")
            keywords = result.get("keywords") if module == "caption" and result else None
            self._safe_emit_result(
                path=path,
                module=module,
                status="done",
                elapsed_ms=elapsed,
                keywords=keywords,
            )
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

        except FileNotFoundError as exc:
            # Prime failed — decoded image not available from coordinator.
            # Release the job back to pending so it can be retried once the
            # image is cached, instead of marking it as permanently failed.
            elapsed = int((time.monotonic() - started_monotonic) * 1000)
            try:
                self._coordinator_call_with_lock_retry(
                    "jobs/release",
                    {"jobId": job_id, "leaseToken": lease_token},
                )
            except Exception:
                pass
            self._safe_emit_result(
                path=path,
                module=module,
                status="skipped",
                elapsed_ms=elapsed,
                error=f"decoded image unavailable: {exc}",
            )
            console.print(
                f"  [yellow]⊘[/yellow] [bold]{module}[/bold] released (image not cached) ← {short_path}"
            )
            return "skipped"

        except ImportError as exc:
            elapsed = int((time.monotonic() - started_monotonic) * 1000)
            reported = self._try_report_skip(
                job_id=job_id,
                lease_token=lease_token,
                reason="missing_dependency",
                details=str(exc),
            )
            if not reported:
                return "failed"
            self._safe_emit_result(
                path=path,
                module=module,
                status="skipped",
                elapsed_ms=elapsed,
                error=f"missing dependency: {exc}",
            )
            console.print(
                f"  [yellow]⊘[/yellow] [bold]{module}[/bold] skipped (missing dependency) in "
                f"{elapsed / 1000:.1f}s ← {short_path} ({exc})"
            )
            return "skipped"

        except ValueError as exc:
            err_lower = str(exc).lower()
            if (
                "libraw cannot decode" in err_lower
                or "libraw postprocess failed" in err_lower
                or "pillow cannot decode" in err_lower
            ):
                elapsed = int((time.monotonic() - started_monotonic) * 1000)
                reported = self._try_report_skip(
                    job_id=job_id,
                    lease_token=lease_token,
                    reason="corrupt_file",
                    details=str(exc),
                )
                if not reported:
                    return "failed"
                self._safe_emit_result(
                    path=path,
                    module=module,
                    status="skipped",
                    elapsed_ms=elapsed,
                    error=f"corrupt file: {exc}",
                )
                console.print(
                    f"  [yellow]⊘[/yellow] [bold]{module}[/bold] skipped (corrupt) in "
                    f"{elapsed / 1000:.1f}s ← {short_path}"
                )
                return "skipped"
            return self._report_failure(
                job_id=job_id,
                lease_token=lease_token,
                path=path,
                module=module,
                short_path=short_path,
                started_monotonic=started_monotonic,
                exc=exc,
            )

        except Exception as exc:
            return self._report_failure(
                job_id=job_id,
                lease_token=lease_token,
                path=path,
                module=module,
                short_path=short_path,
                started_monotonic=started_monotonic,
                exc=exc,
            )

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
                self._coordinator_call_with_lock_retry(
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

    def _log_registration_retry(self, exc: Exception) -> None:
        """Log a transient worker-registration failure."""
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

    def _log_claim_retry(self, exc: Exception) -> None:
        """Log a transient job-claim failure."""
        probe = self._tcp_probe_summary()
        exc_text = escape(str(exc))
        console.print(
            "[yellow]Coordinator unavailable while claiming jobs; "
            f"attempt {self._claim_attempts}, "
            f"retrying in {self.poll_interval_seconds:g}s:[/yellow] {exc_text}\n"
            f"[dim]  TCP probe: {probe}[/dim]"
        )

    def _handle_pause_state(self) -> bool:
        """Handle paused/resumed coordinator state; return True when paused."""
        if self._is_paused():
            if not self._pause_notice_emitted:
                reason = f": {self._pause_reason}" if self._pause_reason else ""
                console.print(
                    f"[yellow]Worker paused by coordinator ({self._desired_state}){reason}. "
                    "Waiting for resume...[/yellow]"
                )
                self._pause_notice_emitted = True
            self._shutdown.wait(self.poll_interval_seconds)
            self._maybe_check_for_update()
            return True
        if self._pause_notice_emitted:
            console.print("[green]Worker resumed by coordinator.[/green]")
            self._pause_notice_emitted = False
        return False

    def _handle_empty_claim_poll(self) -> None:
        """Handle an empty claim result and poll again."""
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
        self._maybe_reprobe_modules()
        self._shutdown.wait(self.poll_interval_seconds)
        self._maybe_check_for_update()

    def _teardown_runtime_session(self) -> None:
        """Release worker leases and mark runtime session as stopped."""
        self._shutdown.set()
        self._release_all_active_leases()
        try:
            self._coordinator_call_with_lock_retry("jobs/release-worker", {"workerId": self.worker_id})
        except Exception as exc:
            if self.verbose:
                console.print(
                    f"[yellow]Worker lease release failed on shutdown:[/yellow] "
                    f"{escape(str(exc))}"
                )

    def _run_worker_session(self, *, stats: dict[str, int], has_connected: bool) -> bool:
        """Run one claim/process/poll session until shutdown."""
        registered = False
        heartbeat_started = False
        heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"{self.worker_id}-heartbeat",
            daemon=True,
        )
        try:
            while not self._shutdown.is_set():
                if not registered:
                    self._registration_attempts += 1
                    try:
                        self._register_worker()
                    except Exception as exc:
                        self._log_registration_retry(exc)
                        self._shutdown.wait(self.poll_interval_seconds)
                        continue

                    try:
                        if not self._startup_lease_recovery_done:
                            self._recover_startup_leases()
                            self._startup_lease_recovery_done = True
                        self._refresh_worker_control_state()
                    except Exception as exc:
                        self._log_registration_retry(exc)
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

                if self._handle_pause_state():
                    continue

                try:
                    self._claim_attempts += 1
                    jobs = self._claim_jobs()
                except Exception as exc:
                    registered = False
                    self._log_claim_retry(exc)
                    self._shutdown.wait(self.poll_interval_seconds)
                    continue

                self._claim_attempts = 0
                if not jobs:
                    self._handle_empty_claim_poll()
                    continue

                self._empty_claim_polls = 0
                console.print(f"[cyan]Claimed {len(jobs)} job(s)[/cyan]")
                self._mark_all_active(jobs)
                self._batch_prefetch_decoded(jobs)
                for job in jobs:
                    if self._shutdown.is_set() or self._is_paused():
                        self._release_all_active_leases()
                        break
                    status = self._process_claimed_job(job)
                    stats[status] = stats.get(status, 0) + 1
                    processed = (
                        stats.get("done", 0)
                        + stats.get("failed", 0)
                        + stats.get("skipped", 0)
                    )
                    console.print(
                        f"[dim]Progress:[/dim] {processed} processed — "
                        f"{stats.get('done', 0)} done, "
                        f"{stats.get('failed', 0)} failed, "
                        f"{stats.get('skipped', 0)} skipped"
                    )
                self._maybe_check_for_update()
        finally:
            self._teardown_runtime_session()
        return has_connected

    def run_forever(self) -> dict[str, int]:
        """Run until interrupted, returning summary stats."""
        stats = {"done": 0, "failed": 0, "skipped": 0}
        original_handler = None
        has_connected = False
        started_session = False
        is_main = threading.current_thread() is threading.main_thread()
        if is_main:
            original_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._handle_sigint)

        try:
            self._log_connectivity_context()
            if self.supported_modules is not None:
                from imganalyzer.db.repository import ALL_MODULES

                missing = sorted(set(ALL_MODULES) - set(self.supported_modules))
                console.print(
                    f"[dim]Supported modules:[/dim] {', '.join(self.supported_modules)}"
                )
                is_apple_silicon = (
                    platform.system() == "Darwin"
                    and platform.machine().lower() in {"arm64", "aarch64"}
                )
                if is_apple_silicon and "perception" not in self.supported_modules:
                    console.print(
                        "[dim]Apple Silicon worker detected: "
                        "perception is CUDA-only and will be handled by "
                        "CUDA workers or the coordinator.[/dim]"
                    )
                if missing:
                    console.print(
                        f"[yellow]Unavailable modules (missing deps):[/yellow] {', '.join(missing)}"
                    )
                if self._auto_update:
                    if self._repo_dir:
                        console.print(
                            f"[dim]Auto-update enabled "
                            f"(checking every {self._auto_update_interval:g}s, "
                            f"repo: {self._repo_dir})[/dim]"
                        )
                    else:
                        console.print(
                            "[yellow]Auto-update requested but no git repo found; disabled.[/yellow]"
                        )
                        self._auto_update = False
                # Fail fast when core local-AI deps are absent — the worker
                # would just claim and skip every GPU job.
                local_ai_available = _LOCAL_AI_MODULES & set(self.supported_modules)
                if not local_ai_available:
                    console.print(
                        "\n[bold red]ERROR: torch / transformers are not installed in "
                        "this Python environment.[/bold red]\n"
                        "[red]The worker cannot run any GPU modules "
                        "(caption, objects, faces, embedding, perception).[/red]\n\n"
                        "You are likely running from the wrong conda environment.\n"
                        "  [bold]Current python:[/bold] "
                        + _current_python_info()
                        + "\n\n"
                        "[green]Fix:[/green] activate the environment that has "
                        "local-AI dependencies:\n"
                        "  [bold]conda activate imganalyzer312[/bold]\n"
                        "  imganalyzer run-distributed-worker ...\n\n"
                        "Or, to set up a fresh worker environment:\n"
                        "  [bold]bash scripts/setup_worker_env.sh[/bold]"
                    )
                    raise SystemExit(1)

            while True:
                started_session = True
                has_connected = self._run_worker_session(
                    stats=stats,
                    has_connected=has_connected,
                )
                if not (self._update_pending and self._repo_dir is not None):
                    break

                self._pull_and_restart()
                # _pull_and_restart returns only if pull had no new commits.
                # Resume the main loop without full re-init.
                self._shutdown.clear()
                self._update_pending = False
                self._empty_claim_polls = 0
                console.print("[dim]Resuming worker…[/dim]")
        finally:
            if not started_session:
                self._teardown_runtime_session()
            if is_main and original_handler is not None:
                signal.signal(signal.SIGINT, original_handler)

        return stats
