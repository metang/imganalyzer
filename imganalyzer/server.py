"""JSON-RPC 2.0 stdio server for persistent Python process.

Eliminates the 1-3s conda subprocess overhead by keeping a long-lived
process that the Electron GUI communicates with via line-delimited
JSON-RPC over stdin/stdout.

Protocol:
    - One JSON object per line on stdin (requests)
    - One JSON object per line on stdout (responses + notifications)
    - stderr is reserved for logging / diagnostics (not parsed by Electron)

Request format:
    {"jsonrpc": "2.0", "id": 1, "method": "status", "params": {}}

Response format:
    {"jsonrpc": "2.0", "id": 1, "result": {...}}

Error format:
    {"jsonrpc": "2.0", "id": 1, "error": {"code": -1, "message": "..."}}

Notification format (streaming, no id):
    {"jsonrpc": "2.0", "method": "progress", "params": {...}}

Supported methods:
    status          - Get queue stats (one-shot)
    ingest          - Scan folder, register images, enqueue jobs (streaming)
    run             - Start processing queue (streaming)
    queue_clear     - Clear jobs from queue (one-shot)
    rebuild         - Re-enqueue module jobs (one-shot)
    workers/register - Register/update distributed worker metadata (one-shot)
    workers/heartbeat - Refresh worker heartbeat and recover expired leases (one-shot)
    workers/list    - List workers plus coordinator master metadata (one-shot)
    workers/pause   - Pause a specific worker (`pause-drain` or `pause-immediate`) (one-shot)
    workers/resume  - Resume a previously paused worker (one-shot)
    jobs/claim      - Lease pending jobs to a worker (one-shot)
    jobs/release-expired - Return expired leases to pending (one-shot)
    jobs/heartbeat  - Extend an active job lease (one-shot)
    jobs/release    - Return a leased job to pending (one-shot)
    jobs/complete   - Mark leased job complete (one-shot)
    jobs/fail       - Mark leased job failed (one-shot)
    jobs/skip       - Mark leased job skipped (one-shot)
    jobs/release-worker - Release all leases held by a worker (one-shot)
    search          - Search the image database (one-shot)
    gallery/listFolders - List folder tree nodes for DB-backed gallery (one-shot)
    gallery/listImagesChunk - List progressive gallery image chunk (one-shot)
    analyze         - Analyze a single image (streaming)
    thumbnail       - Generate thumbnail JPEG (one-shot, base64)
    fullimage       - Generate full-res JPEG for RAW/HEIC (one-shot, base64)
    cancel_run      - Cancel a running batch (one-shot)
    cancel_analyze  - Cancel a running single analysis (one-shot)
    faces/list      - List all face identities with image counts (one-shot)
    faces/images    - Get images containing a face identity (one-shot)
    faces/set-alias - Set display name (alias) for a face identity (one-shot)
    faces/clusters  - List face clusters with counts (one-shot)
    faces/cluster-link-suggestions - Suggest likely person/alias targets for a cluster (one-shot)
    faces/person-link-suggestions - Suggest likely unlinked clusters for a person (one-shot)
    faces/cluster-images - Get face occurrences for a cluster (one-shot)
    faces/crop      - Crop a face from source image by occurrence ID (one-shot)
    faces/run-clustering - Run embedding-based face clustering (one-shot)
    shutdown        - Gracefully shut down the server (one-shot)

HTTP coordinator mode:
    python -m imganalyzer.server --transport http --host 127.0.0.1 --port 8765
    # Optional bearer token (recommended for LAN):
    python -m imganalyzer.server --transport http --auth-token <token>
"""
from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import io
import json
import os
import re
import sqlite3
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from imganalyzer.readers.raw import _suppress_c_stderr

# Redirect stdout early so print() from library imports goes to stderr
_real_stdout = sys.stdout
sys.stdout = sys.stderr

# Lock protecting writes to _real_stdout so concurrent threads
# (ThreadPoolExecutor workers emitting [RESULT] lines) cannot
# interleave partial JSON lines.
_send_lock = threading.Lock()

# Set to True when serving over HTTP; _send() becomes a no-op because
# the Electron stdout pipe isn't used for communication in that mode and
# writing to it can block indefinitely if the pipe buffer fills.
_http_transport = False


def _send(obj: dict[str, Any]) -> None:
    """Write a JSON-RPC message to the real stdout (one line, thread-safe)."""
    if _http_transport:
        return
    line = json.dumps(obj, default=str, separators=(",", ":"))
    with _send_lock:
        try:
            _real_stdout.write(line + "\n")
            _real_stdout.flush()
        except (BrokenPipeError, OSError):
            pass


def _send_result(req_id: int | str, result: Any) -> None:
    _send({"jsonrpc": "2.0", "id": req_id, "result": result})


def _send_error(req_id: int | str | None, code: int, message: str) -> None:
    _send({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})


def _send_notification(method: str, params: Any) -> None:
    _send({"jsonrpc": "2.0", "method": method, "params": params})


# ── Thread-local DB connections ──────────────────────────────────────────────

_db_local = threading.local()
_schema_init_lock = threading.Lock()
_schema_ready_paths: set[str] = set()
_runtime_reconcile_lock = threading.Lock()
_runtime_reconciled = False


def _open_server_db() -> sqlite3.Connection:
    """Open a fresh SQLite connection for the current server thread."""
    from imganalyzer.db.connection import get_db_path
    from imganalyzer.db.schema import ensure_schema

    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_key = str(db_path.resolve())
    if db_key not in _schema_ready_paths:
        with _schema_init_lock:
            if db_key not in _schema_ready_paths:
                bootstrap = sqlite3.connect(
                    str(db_path),
                    timeout=30,
                    isolation_level=None,
                    check_same_thread=False,
                )
                bootstrap.row_factory = sqlite3.Row
                bootstrap.execute("PRAGMA journal_mode=WAL")
                bootstrap.execute("PRAGMA synchronous=NORMAL")
                bootstrap.execute("PRAGMA foreign_keys=ON")
                bootstrap.execute("PRAGMA busy_timeout=5000")
                ensure_schema(bootstrap)
                bootstrap.close()
                _schema_ready_paths.add(db_key)
    conn = sqlite3.connect(
        str(db_path),
        timeout=30,
        isolation_level=None,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _get_db() -> sqlite3.Connection:
    """Return a thread-local SQLite connection for the current server thread."""
    conn = getattr(_db_local, "conn", None)
    if conn is None:
        conn = _open_server_db()
        _db_local.conn = conn
    return conn


# ── State for cancellable operations ─────────────────────────────────────────

_run_cancel = threading.Event()
_run_thread: threading.Thread | None = None
_active_worker: Any = None  # Worker | None — set while a run is active

_analyze_cancel: dict[str, threading.Event] = {}  # imagePath -> cancel event

_MASTER_WORKER_ID = "master"
_MASTER_WORKER_LABEL = "Master device"
_LEGACY_QUEUE_MODULE_MAP: dict[str, str] = {
    "blip2": "caption",
    "cloud_ai": "caption",
    "local_ai": "caption",
    "aesthetic": "perception",
}
_PERSON_LINK_SUGGESTION_CACHE_TTL_SECONDS = 30.0
_person_link_suggestion_cache: dict[tuple[int, int], tuple[float, list[dict[str, Any]]]] = {}

# ── Decoded image cache (coordinator-mediated distribution) ──────────────────

_decoded_store: Any = None  # DecodedImageStore | None
_decoded_store_lock = threading.Lock()
_pre_decoder: Any = None    # PreDecoder | None


def _get_decoded_store() -> Any:
    """Return the global DecodedImageStore, creating it lazily on first use."""
    global _decoded_store
    if _decoded_store is not None:
        return _decoded_store
    with _decoded_store_lock:
        if _decoded_store is not None:
            return _decoded_store
        from imganalyzer.cache.decoded_store import DecodedImageStore
        cache_dir = os.getenv("IMGANALYZER_DECODED_CACHE_DIR", "")
        max_gb = float(os.getenv("IMGANALYZER_DECODED_CACHE_MAX_GB", "300"))
        resolution = int(os.getenv("IMGANALYZER_DECODED_CACHE_RESOLUTION", "1024"))
        fmt = os.getenv("IMGANALYZER_DECODED_CACHE_FORMAT", "webp")
        quality = int(os.getenv("IMGANALYZER_DECODED_CACHE_QUALITY", "95"))
        from pathlib import Path
        _decoded_store = DecodedImageStore(
            cache_dir=Path(cache_dir) if cache_dir else None,
            max_bytes=int(max_gb * 1024 * 1024 * 1024),
            resolution=resolution,
            fmt=fmt,
            quality=quality,
        )
        sys.stderr.write(f"[server] decoded image store: {_decoded_store}\n")
    return _decoded_store


def _get_pre_decoder() -> Any:
    """Return the global PreDecoder, creating it lazily on first use."""
    global _pre_decoder
    if _pre_decoder is None:
        from imganalyzer.cache.pre_decode import PreDecoder
        store = _get_decoded_store()
        max_workers_str = os.getenv("IMGANALYZER_PRE_DECODE_WORKERS", "")
        max_workers = int(max_workers_str) if max_workers_str else None
        _pre_decoder = PreDecoder(store, max_workers=max_workers)
    return _pre_decoder


_pre_decode_auto_triggered = False

# Demand-driven decode buffer: decode enough images to keep workers fed
# without queuing the entire image set upfront.
# _DECODE_BUFFER_SIZE is the *maximum* batch; actual feed size is adapted
# based on CPU and disk I/O via ResourceSampler.
_DECODE_BUFFER_SIZE = int(os.getenv("IMGANALYZER_DECODE_BUFFER_SIZE", "100"))
_MIN_DECODE_BUFFER = 20  # never let in-flight drop below this
_last_replenish_time: float = 0.0
_REPLENISH_INTERVAL: float = 5.0  # seconds between buffer checks


def _auto_trigger_pre_decode() -> None:
    """Kept for backward compatibility — calls _replenish_decode_buffer."""
    _replenish_decode_buffer()


def _replenish_decode_buffer() -> None:
    """Feed images to the pre-decoder based on CPU and disk availability.

    Called from ``_handle_jobs_claim`` on every request.  Uses a
    :class:`~imganalyzer.cache.pre_decode.ResourceSampler` to check
    system utilisation before deciding whether (and how much) to feed.

    * **High capacity** (CPU < 40 %, disk < 50 %): feed up to
      ``_DECODE_BUFFER_SIZE`` images.
    * **Medium capacity** (CPU < 70 %, disk < 80 %): feed half the
      buffer.
    * **Low capacity** (CPU ≥ 70 % AND disk ≥ 80 %): feed nothing —
      *unless* in-flight is below ``_MIN_DECODE_BUFFER``, in which
      case we feed the minimum to prevent worker starvation.

    Throttled to run at most every ``_REPLENISH_INTERVAL`` seconds.
    """
    global _last_replenish_time

    now = time.monotonic()
    if now - _last_replenish_time < _REPLENISH_INTERVAL:
        return
    _last_replenish_time = now

    try:
        decoder = _get_pre_decoder()
        prog = decoder.progress()
        in_flight = prog["total"] - prog["done"] - prog["failed"]

        # Sample resources for adaptive scheduling
        sampler = decoder._resource_sampler
        res = sampler.sample()

        if not sampler.should_feed(in_flight, min_buffer=_MIN_DECODE_BUFFER):
            sys.stderr.write(
                f"[server] decode throttled: CPU {res['cpu_pct']:.0f}%"
                f" disk {res['disk_busy_pct']:.0f}%"
                f" in-flight {in_flight}\n"
            )
            return

        # Adaptive batch size: large when idle, small when busy
        batch_size = sampler.recommended_batch_size(
            max_batch=_DECODE_BUFFER_SIZE,
            min_batch=_MIN_DECODE_BUFFER,
        )

        need = batch_size - max(in_flight, 0)
        if need <= 0:
            return

        store = _get_decoded_store()
        cached_ids = store.cached_image_ids()

        from imganalyzer.db.connection import get_db_path as _gdp
        conn2 = sqlite3.connect(str(_gdp()), timeout=10, check_same_thread=False)
        conn2.row_factory = sqlite3.Row
        conn2.execute("PRAGMA journal_mode=WAL")

        # Get image IDs with pending jobs, ordered by number of pending
        # modules descending (images needing more work are more valuable
        # to decode first).
        pending_rows = conn2.execute(
            "SELECT image_id, COUNT(*) AS cnt "
            "FROM job_queue WHERE status = 'pending' "
            "GROUP BY image_id ORDER BY cnt DESC"
        ).fetchall()
        pending_ids = [int(r["image_id"]) for r in pending_rows]

        if not pending_ids:
            conn2.close()
            return

        uncached = [iid for iid in pending_ids if iid not in cached_ids][:need]

        if not uncached:
            conn2.close()
            return

        # Fetch file paths for the batch
        ph = ",".join("?" * len(uncached))
        rows = conn2.execute(
            f"SELECT id, file_path FROM images WHERE id IN ({ph})",
            uncached,
        ).fetchall()
        conn2.close()

        items = [(int(r["id"]), str(r["file_path"])) for r in rows]
        if items:
            added = decoder.feed(items)
            if added:
                sys.stderr.write(
                    f"[server] decode buffer: fed {added} images"
                    f" (in-flight {in_flight}, batch {batch_size},"
                    f" CPU {res['cpu_pct']:.0f}%,"
                    f" disk {res['disk_busy_pct']:.0f}%,"
                    f" cached {len(cached_ids)})\n"
                )
    except Exception as exc:
        sys.stderr.write(f"[server] decode buffer replenish failed: {exc}\n")


def _invalidate_person_link_suggestion_cache() -> None:
    _person_link_suggestion_cache.clear()


def _master_worker_runtime_status() -> str:
    if _active_worker is not None:
        return "online"
    if _run_thread is not None and _run_thread.is_alive():
        return "online"
    return "offline"


def _sync_master_worker_node(conn: sqlite3.Connection) -> None:
    _upsert_master_worker_node(conn, status=_master_worker_runtime_status())


def _mark_stale_worker_nodes_offline(
    conn: sqlite3.Connection,
    *,
    stale_seconds: int = 180,
) -> int:
    """Mark stale worker rows offline so lifecycle state is rebuilt on startup."""
    cur = conn.execute(
        """UPDATE worker_nodes
           SET status = 'offline',
               updated_at = datetime('now')
           WHERE id <> ?
             AND status = 'online'
             AND (
                 last_heartbeat IS NULL
                 OR last_heartbeat <= datetime('now', '-' || ? || ' seconds')
             )""",
        [_MASTER_WORKER_ID, max(30, int(stale_seconds))],
    )
    if cur.rowcount:
        conn.commit()
    return int(cur.rowcount)


def _reconcile_runtime_queue_state(
    conn: sqlite3.Connection,
    *,
    recover_master_jobs: bool,
) -> dict[str, int]:
    """Reconcile queue/runtime invariants to support crash-safe resume."""
    from imganalyzer.db.queue import JobQueue

    queue = JobQueue(conn)
    remapped = queue.remap_pending_modules(_LEGACY_QUEUE_MODULE_MAP)
    reconciled = queue.reconcile_runtime_state(recover_master_jobs=recover_master_jobs)
    stale_workers = _mark_stale_worker_nodes_offline(conn)
    return {
        "remapped_updated": int(remapped.get("updated", 0)),
        "remapped_deleted": int(remapped.get("deleted", 0)),
        "dangling_leases": int(reconciled.get("dangling_leases", 0)),
        "worker_orphans": int(reconciled.get("worker_orphans", 0)),
        "master_orphans": int(reconciled.get("master_orphans", 0)),
        "stale_workers": stale_workers,
    }


def _ensure_runtime_state_reconciled(
    *,
    context: str,
    recover_master_jobs: bool,
) -> None:
    """Ensure queue/runtime state is reconciled exactly once per server process."""
    global _runtime_reconciled
    if _runtime_reconciled:
        return

    with _runtime_reconcile_lock:
        if _runtime_reconciled:
            return
        try:
            conn = _get_db()
            _sync_master_worker_node(conn)
            summary = _reconcile_runtime_queue_state(
                conn,
                recover_master_jobs=recover_master_jobs,
            )
            changed = {k: v for k, v in summary.items() if int(v) > 0}
            if changed:
                items = ", ".join(f"{key}={value}" for key, value in sorted(changed.items()))
                sys.stderr.write(
                    f"[coordinator] Runtime reconciliation ({context}): {items}\n"
                )
            _runtime_reconciled = True
        except Exception as exc:
            sys.stderr.write(
                f"[coordinator] Runtime reconciliation failed during {context}: {exc}\n"
            )


# ── Method handlers ──────────────────────────────────────────────────────────

def _handle_status(params: dict) -> dict:
    """Return queue stats as JSON."""
    from imganalyzer.db.queue import JobQueue
    from imganalyzer.db.repository import Repository

    conn = _get_db()
    _sync_master_worker_node(conn)
    queue = JobQueue(conn)
    repo = Repository(conn)

    total_images = repo.count_images()
    module_stats = queue.stats()
    totals = queue.total_stats()
    remaining_images = queue.remaining_image_count()
    module_avg_ms = queue.module_avg_processing_ms(last_n=100)

    from imganalyzer.db.repository import ALL_MODULES
    queue_modules = list(module_stats.keys())
    ordered = [m for m in ALL_MODULES if m in module_stats]
    extra = [m for m in queue_modules if m not in ALL_MODULES]
    all_modules = ordered + extra

    modules_out: dict[str, dict[str, int]] = {}
    for mod in all_modules:
        s = module_stats.get(mod, {})
        if not any(s.values()) and mod not in module_stats:
            continue
        modules_out[mod] = {
            "pending": s.get("pending", 0),
            "running": s.get("running", 0),
            "done": s.get("done", 0),
            "failed": s.get("failed", 0),
            "skipped": s.get("skipped", 0),
        }

    running_modules = _get_running_modules_by_node(conn)
    master_item = _build_master_item(conn, running_modules=running_modules)
    worker_items = _build_worker_items(conn, running_modules=running_modules)

    # Chunk-level stats: pending counts per module within the current chunk.
    chunk_modules: dict[str, int] = {}
    chunk_info: dict[str, Any] | None = None
    if _active_worker is not None and _active_worker.current_chunk_ids is not None:
        chunk_ids = _active_worker.current_chunk_ids
        for mod in modules_out:
            cp = queue.pending_count(module=mod, image_ids=chunk_ids)
            cr = queue.running_count(module=mod, image_ids=chunk_ids)
            if cp > 0 or cr > 0:
                chunk_modules[mod] = cp + cr
        chunk_info = {
            "size": len(chunk_ids),
            "index": _active_worker.current_chunk_index,
            "total": _active_worker.total_chunks,
            "modules": chunk_modules,
        }

    result = {
        "total_images": total_images,
        "modules": modules_out,
        "module_avg_ms": module_avg_ms,
        "chunk": chunk_info,
        "totals": {
            "pending": totals.get("pending", 0),
            "running": totals.get("running", 0),
            "done": totals.get("done", 0),
            "failed": totals.get("failed", 0),
            "skipped": totals.get("skipped", 0),
        },
        "remaining_images": remaining_images,
        "nodes": {
            "master": master_item,
            "workers": worker_items,
        },
        "recent_results": _recent_queue_results(conn),
    }

    # Pre-decode / image cache progress.  Always report the overall cache
    # fill state (cached vs total images in DB) so the dashboard progress
    # bar is meaningful even with incremental demand-driven decoding.
    try:
        store = _get_decoded_store()
        cached_count = store.entry_count
        decoder = _get_pre_decoder()
        is_running = decoder.is_running

        if total_images > 0:
            pre_decode_info: dict[str, Any] = {
                "done": min(cached_count, total_images),
                "failed": 0,
                "total": total_images,
                "running": is_running,
            }
            # Include resource utilisation so the dashboard can show load
            res = decoder._resource_sampler.snapshot
            pre_decode_info["resources"] = {
                "cpu_pct": round(res["cpu_pct"], 1),
                "disk_read_mbps": round(res["disk_read_mbps"], 1),
                "disk_busy_pct": round(res["disk_busy_pct"], 1),
            }
            result["pre_decode"] = pre_decode_info
    except Exception as exc:
        sys.stderr.write(f"[server] pre_decode status error: {exc}\n")

    return result


def _handle_ingest(req_id: int | str, params: dict) -> None:
    """Ingest folders — streaming progress notifications, then final result."""
    from imganalyzer.pipeline.batch import BatchProcessor
    from imganalyzer.db.connection import get_db_path

    # Open a FRESH connection for this thread — _handle_ingest runs in a
    # daemon thread (async RPC method) and cannot reuse the main-thread
    # singleton returned by _get_db().
    db_path = get_db_path()
    conn = sqlite3.connect(
        str(db_path),
        timeout=30,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    processor = BatchProcessor(conn)

    folders = [Path(f) for f in params.get("folders", [])]
    modules = params.get("modules")
    if isinstance(modules, str):
        modules = [m.strip() for m in modules.split(",")]
    force = params.get("force", False)
    recursive = params.get("recursive", True)
    compute_hash = params.get("computeHash", True)
    verbose = params.get("verbose", False)

    # Monkey-patch the print function to intercept [PROGRESS] lines
    # and forward them as JSON-RPC notifications instead.
    import builtins
    _orig_print = builtins.print

    def _patched_print(*args, **kwargs):
        text = " ".join(str(a) for a in args)
        if text.startswith("[PROGRESS] "):
            try:
                payload = json.loads(text[len("[PROGRESS] "):])
                _send_notification("ingest/progress", payload)
                return
            except Exception:
                pass
        # All other prints go to stderr (normal stdout is already redirected)
        _orig_print(*args, **kwargs)

    builtins.print = _patched_print
    try:
        ingest_stats = processor.ingest(
            folders=folders,
            modules=modules,
            force=force,
            recursive=recursive,
            compute_hash=compute_hash,
            verbose=verbose,
        )
    except Exception as exc:
        builtins.print = _orig_print
        conn.close()
        _send_error(req_id, -1, f"Ingest failed: {exc}")
        return
    finally:
        builtins.print = _orig_print

    conn.close()

    # Trigger demand-driven pre-decode: feed an initial batch so
    # distributed workers can start receiving cached images immediately.
    # The buffer will be replenished automatically on subsequent claims.
    try:
        _replenish_decode_buffer()
    except Exception as exc:
        sys.stderr.write(f"[server] pre-decode trigger failed: {exc}\n")

    _send_result(req_id, {
        "ok": True,
        "registered": ingest_stats.get("registered", 0),
        "enqueued": ingest_stats.get("enqueued", 0),
        "skipped": ingest_stats.get("skipped", 0),
    })


def _handle_run(req_id: int | str, params: dict) -> None:
    """Start processing the queue — streaming [RESULT] notifications."""
    global _run_thread
    from imganalyzer.pipeline.unified_scheduler import PAUSED_STATES, get_worker_control_state

    if _run_thread is not None and _run_thread.is_alive():
        # The previous worker may still be winding down after cancel_run.
        # Wait briefly for it to finish before rejecting.
        _run_thread.join(timeout=10)
        if _run_thread.is_alive():
            _send_error(req_id, -2, "A run is already in progress")
            return

    conn = _get_db()
    _sync_master_worker_node(conn)
    desired_state, state_reason = get_worker_control_state(conn, _MASTER_WORKER_ID)
    if desired_state in PAUSED_STATES:
        reason_suffix = f" ({state_reason})" if state_reason else ""
        _send_error(
            req_id,
            -3,
            f"Master worker is {desired_state}{reason_suffix}; call workers/resume before run.",
        )
        return

    _run_cancel.clear()

    workers = params.get("workers", 1)
    verbose = params.get("verbose", True)
    write_xmp = not params.get("noXmp", True)
    force = params.get("force", False)
    batch_size = params.get("batchSize", 10)
    chunk_size = params.get("chunkSize", 500)
    detection_prompt = params.get("detectionPrompt")
    detection_threshold = params.get("detectionThreshold")
    face_threshold = params.get("faceThreshold")
    # In Electron/JSON-RPC mode we run a single worker at a time, so any
    # leftover `running` rows are stale from an interrupted previous run.
    # Recover them immediately by default unless the caller overrides it.
    stale_timeout = params.get("staleTimeout", 0)
    profile = params.get("profile", False)

    def _run_worker():
        global _active_worker
        try:
            from imganalyzer.pipeline.worker import Worker
            from imganalyzer.db.connection import get_db_path

            # Open a FRESH connection for this thread — SQLite connections
            # cannot be shared across threads (check_same_thread=True by default).
            db_path = get_db_path()
            conn = sqlite3.connect(
                str(db_path),
                timeout=30,
                isolation_level=None,
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=5000")
            _upsert_master_worker_node(conn, status="online")
            worker_kwargs = dict(
                conn=conn,
                workers=workers,
                force=force,
                detection_prompt=detection_prompt,
                detection_threshold=detection_threshold,
                face_match_threshold=face_threshold,
                verbose=verbose,
                write_xmp=write_xmp,
                profile=profile,
            )
            if stale_timeout is not None:
                worker_kwargs["stale_timeout_minutes"] = stale_timeout
            worker = Worker(**worker_kwargs)
            _active_worker = worker

            # Wire up direct result-notification callback (bypasses print)
            from imganalyzer.pipeline import worker as worker_mod
            worker_mod._result_notify = lambda payload: _send_notification("run/result", payload)
            worker_mod._chunk_notify = lambda payload: _send_notification("run/chunk_done", payload)
            try:
                result = worker.run(batch_size=batch_size, chunk_size=chunk_size)
                _send_notification("run/done", result)
            except Exception as exc:
                _send_notification("run/error", {"error": str(exc)})
            finally:
                worker_mod._result_notify = None
                worker_mod._chunk_notify = None
                _active_worker = None
                try:
                    _upsert_master_worker_node(conn, status="offline")
                except Exception:
                    pass
                conn.close()
        except Exception as exc:
            _active_worker = None
            _send_notification("run/error", {"error": str(exc), "traceback": traceback.format_exc()})

    _run_thread = threading.Thread(target=_run_worker, daemon=True, name="rpc-run")
    _run_thread.start()
    _send_result(req_id, {"started": True})


def _handle_cancel_run(params: dict) -> dict:
    """Cancel a running batch by setting the Worker's shutdown event."""
    _run_cancel.set()
    # Directly signal the active worker to stop (graceful shutdown).
    # The worker checks _shutdown.is_set() between jobs and will finish
    # the current batch then exit.
    if _active_worker is not None:
        _active_worker._shutdown.set()
    return {"cancelled": True}


def _handle_queue_clear(params: dict) -> dict:
    """Clear jobs from the queue."""
    from imganalyzer.db.queue import JobQueue

    conn = _get_db()
    queue = JobQueue(conn)

    folder = params.get("folder")
    status_filter = params.get("status", "pending,running")

    if status_filter.strip().lower() == "all":
        statuses = None
    else:
        statuses = [s.strip() for s in status_filter.split(",") if s.strip()]

    if folder:
        deleted = queue.clear_by_folder(folder, statuses)
    else:
        if statuses:
            placeholders = ",".join("?" * len(statuses))
            cur = conn.execute(
                f"DELETE FROM job_queue WHERE status IN ({placeholders})", statuses
            )
            conn.commit()
            deleted = cur.rowcount
        else:
            deleted = queue.clear_all()

    return {"deleted": deleted}


def _handle_workers_register(params: dict) -> dict:
    """Register or update a distributed worker node."""
    from imganalyzer.db.queue import _now

    worker_id = str(params.get("workerId", "")).strip()
    if not worker_id:
        raise ValueError("workerId is required")
    display_name = str(params.get("displayName") or worker_id)
    platform = str(params.get("platform") or "")
    capabilities = params.get("capabilities") or {}
    capabilities_json = json.dumps(capabilities, separators=(",", ":"))
    now = _now()

    conn = _get_db()
    conn.execute(
        """INSERT INTO worker_nodes
           (id, display_name, platform, capabilities, status, last_heartbeat, created_at, updated_at)
           VALUES (?, ?, ?, ?, 'online', ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
             display_name = excluded.display_name,
             platform = excluded.platform,
             capabilities = excluded.capabilities,
             status = 'online',
             last_heartbeat = excluded.last_heartbeat,
             updated_at = excluded.updated_at""",
        [worker_id, display_name, platform, capabilities_json, now, now, now],
    )
    conn.commit()
    return {"workerId": worker_id, "registered": True}


def _handle_workers_heartbeat(params: dict) -> dict:
    """Refresh worker heartbeat timestamp and return basic lease recovery stats."""
    from imganalyzer.db.queue import JobQueue, _now
    from imganalyzer.pipeline.unified_scheduler import get_worker_control_state

    worker_id = str(params.get("workerId", "")).strip()
    if not worker_id:
        raise ValueError("workerId is required")

    conn = _get_db()
    if worker_id == _MASTER_WORKER_ID:
        _sync_master_worker_node(conn)
    else:
        now = _now()
        cur = conn.execute(
            """UPDATE worker_nodes
               SET status = 'online', last_heartbeat = ?, updated_at = ?
               WHERE id = ?""",
            [now, now, worker_id],
        )
        if cur.rowcount == 0:
            conn.execute(
                """INSERT INTO worker_nodes (id, display_name, platform, capabilities, status, last_heartbeat, created_at, updated_at)
                   VALUES (?, ?, '', '{}', 'online', ?, ?, ?)""",
                [worker_id, worker_id, now, now, now],
            )
        conn.commit()

    queue = JobQueue(conn)
    released = queue.release_expired_leases()
    desired_state, state_reason = get_worker_control_state(conn, worker_id)
    return {
        "workerId": worker_id,
        "ok": True,
        "releasedExpired": released,
        "desiredState": desired_state,
        "stateReason": state_reason,
    }


def _get_worker_running_counts(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        """SELECT jl.worker_id, COUNT(*) AS cnt
           FROM job_leases jl
           JOIN job_queue jq ON jq.id = jl.job_id
           WHERE jq.status = 'running'
           GROUP BY jl.worker_id"""
    ).fetchall()
    return {
        str(row["worker_id"]): int(row["cnt"])
        for row in rows
        if row["worker_id"] is not None
    }


def _get_running_modules_by_node(conn: sqlite3.Connection) -> dict[tuple[str, str], list[dict[str, Any]]]:
    rows = conn.execute(
        """SELECT COALESCE(jq.last_node_role, CASE WHEN jl.worker_id IS NOT NULL THEN 'worker' ELSE 'master' END) AS node_role,
                  COALESCE(jq.last_node_id, jl.worker_id, 'master') AS node_id,
                  jq.module,
                  COUNT(*) AS cnt
           FROM job_queue jq
           LEFT JOIN job_leases jl ON jl.job_id = jq.id
           WHERE jq.status = 'running'
           GROUP BY node_role, node_id, jq.module"""
    ).fetchall()

    items: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        node_role = str(row["node_role"] or "master")
        node_id = str(row["node_id"] or "master")
        key = (node_role, node_id)
        items.setdefault(key, []).append({
            "module": str(row["module"]),
            "count": int(row["cnt"]),
        })

    for modules in items.values():
        modules.sort(key=lambda item: (-int(item["count"]), str(item["module"])))

    return items


def _decode_worker_capabilities(raw: Any) -> dict[str, Any]:
    caps_raw = raw or "{}"
    try:
        caps = json.loads(caps_raw)
        if isinstance(caps, dict):
            return caps
    except Exception:
        pass
    return {}


def _master_running_jobs(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        """SELECT COUNT(*) AS cnt
           FROM job_queue jq
           LEFT JOIN job_leases jl ON jl.job_id = jq.id
           WHERE jq.status = 'running'
             AND jl.job_id IS NULL"""
    ).fetchone()
    return int(row["cnt"] if row is not None else 0)


def _build_master_item(
    conn: sqlite3.Connection,
    *,
    running_modules: dict[tuple[str, str], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    row = conn.execute(
        """SELECT id, display_name, platform, capabilities, status, last_heartbeat,
                  created_at, updated_at, desired_state, state_reason
           FROM worker_nodes
           WHERE id = ?""",
        [_MASTER_WORKER_ID],
    ).fetchone()
    active_modules = (
        running_modules
        if running_modules is not None
        else _get_running_modules_by_node(conn)
    )
    if row is None:
        return {
            "id": _MASTER_WORKER_ID,
            "role": "master",
            "displayName": _MASTER_WORKER_LABEL,
            "platform": sys.platform,
            "capabilities": {},
            "status": _master_worker_runtime_status(),
            "lastHeartbeat": None,
            "createdAt": None,
            "updatedAt": None,
            "desiredState": "active",
            "stateReason": None,
            "runningJobs": _master_running_jobs(conn),
            "activeModules": active_modules.get(("master", _MASTER_WORKER_ID), []),
        }

    return {
        "id": _MASTER_WORKER_ID,
        "role": "master",
        "displayName": row["display_name"] or _MASTER_WORKER_LABEL,
        "platform": row["platform"] or sys.platform,
        "capabilities": _decode_worker_capabilities(row["capabilities"]),
        "status": row["status"] or _master_worker_runtime_status(),
        "lastHeartbeat": row["last_heartbeat"],
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
        "desiredState": row["desired_state"] or "active",
        "stateReason": row["state_reason"],
        "runningJobs": _master_running_jobs(conn),
        "activeModules": active_modules.get(("master", _MASTER_WORKER_ID), []),
    }


def _build_worker_items(
    conn: sqlite3.Connection,
    *,
    running_modules: dict[tuple[str, str], list[dict[str, Any]]] | None = None,
    include_master: bool = False,
) -> list[dict[str, Any]]:
    running_counts = _get_worker_running_counts(conn)
    active_modules = (
        running_modules
        if running_modules is not None
        else _get_running_modules_by_node(conn)
    )
    where = ""
    params: list[Any] = []
    if not include_master:
        where = "WHERE id <> ?"
        params.append(_MASTER_WORKER_ID)
    rows = conn.execute(
        f"""SELECT id, display_name, platform, capabilities, status, last_heartbeat,
                   created_at, updated_at, desired_state, state_reason
            FROM worker_nodes
            {where}
            ORDER BY updated_at DESC, id ASC""",
        params,
    ).fetchall()
    items = []
    for row in rows:
        worker_id = str(row["id"])
        items.append({
            "id": worker_id,
            "displayName": row["display_name"],
            "platform": row["platform"],
            "capabilities": _decode_worker_capabilities(row["capabilities"]),
            "status": row["status"],
            "lastHeartbeat": row["last_heartbeat"],
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
            "desiredState": row["desired_state"] or "active",
            "stateReason": row["state_reason"],
            "runningJobs": running_counts.get(worker_id, 0),
            "activeModules": active_modules.get(("worker", worker_id), []),
        })
    return items


def _recent_queue_results(conn: sqlite3.Connection, limit: int = 200) -> list[dict[str, Any]]:
    rows = conn.execute(
        """SELECT jq.id,
                  jq.module,
                  jq.status,
                  jq.error_message,
                  jq.skip_reason,
                  jq.completed_at,
                  jq.last_node_id,
                  jq.last_node_role,
                  i.file_path,
                  wn.display_name AS worker_display_name,
                  COALESCE(
                      CAST(ROUND((julianday(jq.completed_at) - julianday(jq.started_at)) * 86400000.0) AS INTEGER),
                      0
                  ) AS duration_ms
           FROM job_queue jq
           JOIN images i ON i.id = jq.image_id
           LEFT JOIN worker_nodes wn ON wn.id = jq.last_node_id
           WHERE jq.status IN ('done', 'failed', 'skipped')
             AND jq.completed_at IS NOT NULL
           ORDER BY jq.completed_at DESC, jq.id DESC
           LIMIT ?""",
        [max(1, int(limit))],
    ).fetchall()

    items: list[dict[str, Any]] = []
    for row in rows:
        node_role = str(row["last_node_role"] or "master")
        node_id = str(row["last_node_id"] or "master")
        node_label = (
            str(row["worker_display_name"])
            if node_role == "worker" and row["worker_display_name"]
            else _MASTER_WORKER_LABEL
        )
        message = row["error_message"] or row["skip_reason"]
        items.append({
            "jobId": int(row["id"]),
            "path": row["file_path"],
            "module": row["module"],
            "status": row["status"],
            "durationMs": max(0, int(row["duration_ms"] or 0)),
            "completedAt": row["completed_at"],
            "error": str(message) if message else None,
            "nodeId": node_id,
            "nodeRole": node_role,
            "nodeLabel": node_label,
        })
    return items


def _handle_workers_list(_params: dict) -> dict:
    """List registered workers and include the coordinator master node."""
    conn = _get_db()
    _sync_master_worker_node(conn)
    running_modules = _get_running_modules_by_node(conn)
    return {
        "master": _build_master_item(conn, running_modules=running_modules),
        "workers": _build_worker_items(conn, running_modules=running_modules),
    }


def _handle_workers_pause(params: dict) -> dict:
    """Pause a specific worker (drain or immediate mode)."""
    from imganalyzer.db.queue import JobQueue
    from imganalyzer.pipeline.unified_scheduler import (
        get_worker_control_state,
        set_worker_control_state,
    )

    worker_id = str(params.get("workerId", "")).strip()
    if not worker_id:
        raise ValueError("workerId is required")
    mode = str(params.get("mode") or "pause-drain").strip().lower()
    if mode not in {"pause-drain", "pause-immediate", "paused"}:
        raise ValueError("mode must be one of: pause-drain, pause-immediate, paused")
    reason_raw = params.get("reason")
    reason = str(reason_raw).strip() if reason_raw is not None else None
    if reason == "":
        reason = None

    conn = _get_db()
    if worker_id == _MASTER_WORKER_ID:
        _sync_master_worker_node(conn)
    previous_state, _ = get_worker_control_state(conn, worker_id)
    set_worker_control_state(conn, worker_id, mode, reason=reason)

    released_leases = 0
    if mode == "pause-immediate" and worker_id != _MASTER_WORKER_ID:
        queue = JobQueue(conn)
        released_leases = queue.release_worker_leases(worker_id)

    # Master pause maps to run cancellation.
    if worker_id == _MASTER_WORKER_ID and _active_worker is not None:
        _active_worker._shutdown.set()

    result: dict[str, Any] = {
        "workerId": worker_id,
        "ok": True,
        "previousState": previous_state,
        "transitioned": previous_state != mode,
        "desiredState": mode,
        "stateReason": reason,
        "releasedLeases": released_leases,
    }
    if worker_id == _MASTER_WORKER_ID:
        _sync_master_worker_node(conn)
        result["worker"] = _build_master_item(conn)
    return result


def _handle_workers_resume(params: dict) -> dict:
    """Resume a specific worker so it can claim new jobs again."""
    from imganalyzer.pipeline.unified_scheduler import (
        get_worker_control_state,
        set_worker_control_state,
    )

    worker_id = str(params.get("workerId", "")).strip()
    if not worker_id:
        raise ValueError("workerId is required")

    conn = _get_db()
    if worker_id == _MASTER_WORKER_ID:
        _sync_master_worker_node(conn)
    previous_state, _ = get_worker_control_state(conn, worker_id)
    set_worker_control_state(conn, worker_id, "active", reason=None)
    result: dict[str, Any] = {
        "workerId": worker_id,
        "ok": True,
        "previousState": previous_state,
        "transitioned": previous_state != "active",
        "desiredState": "active",
        "stateReason": None,
    }
    if worker_id == _MASTER_WORKER_ID:
        _sync_master_worker_node(conn)
        result["worker"] = _build_master_item(conn)
    return result


def _handle_workers_remove(params: dict) -> dict:
    """Remove (deregister) a distributed worker node.

    Releases any leased jobs back to pending and deletes the worker row
    from ``worker_nodes``.  Cannot remove the master/coordinator node.
    """
    from imganalyzer.db.queue import JobQueue

    worker_id = str(params.get("workerId", "")).strip()
    if not worker_id:
        raise ValueError("workerId is required")
    if worker_id == _MASTER_WORKER_ID:
        raise ValueError("Cannot remove the coordinator node")

    conn = _get_db()

    # Release any leased jobs so they return to pending
    queue = JobQueue(conn)
    released_leases = queue.release_worker_leases(worker_id)

    # Delete the worker record
    conn.execute("DELETE FROM worker_nodes WHERE id = ?", [worker_id])
    conn.commit()

    return {
        "workerId": worker_id,
        "ok": True,
        "releasedLeases": released_leases,
    }


def _upsert_master_worker_node(conn: sqlite3.Connection, status: str = "online") -> None:
    """Keep the coordinator represented in ``worker_nodes`` as a local worker."""
    from imganalyzer.db.queue import _now

    now = _now()
    caps: dict[str, Any] = {
        "pid": os.getpid(),
        "coordinator": True,
        "workerType": "local-master",
    }
    try:
        import torch

        caps["cuda"] = bool(torch.cuda.is_available())
        caps["mps"] = bool(
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
    except Exception:
        caps["cuda"] = False
        caps["mps"] = False
    capabilities_json = json.dumps(caps, separators=(",", ":"))

    conn.execute(
        """INSERT INTO worker_nodes
           (id, display_name, platform, capabilities, status, last_heartbeat, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
              display_name = excluded.display_name,
              platform = excluded.platform,
              capabilities = excluded.capabilities,
              status = excluded.status,
              last_heartbeat = excluded.last_heartbeat,
              updated_at = excluded.updated_at""",
        [_MASTER_WORKER_ID, _MASTER_WORKER_LABEL, sys.platform, capabilities_json, status, now, now, now],
    )
    conn.commit()


_DISTRIBUTED_CONTEXT_MODULES: dict[str, tuple[str, ...]] = {
    "faces": ("objects",),
    "embedding": ("caption",),
}
_DISTRIBUTED_SEARCH_MODULES = {"metadata", "caption", "faces"}


def _build_distributed_job_context(repo: Any, image_id: int, module: str) -> dict[str, Any]:
    modules: dict[str, Any] = {}
    for context_module in _DISTRIBUTED_CONTEXT_MODULES.get(module, ()):
        data = repo.get_analysis(image_id, context_module)
        if data:
            modules[context_module] = data
    return {"modules": modules}


def _handle_jobs_claim(params: dict) -> dict:
    """Lease pending jobs to a worker for distributed execution."""
    from imganalyzer.db.queue import JobQueue
    from imganalyzer.db.repository import Repository
    from imganalyzer.pipeline.unified_scheduler import (
        compute_claim_policy,
        record_worker_affinity,
    )
    from imganalyzer.pipeline.worker import _PREREQUISITES

    worker_id = str(params.get("workerId", "")).strip()
    if not worker_id:
        raise ValueError("workerId is required")
    batch_size = int(params.get("batchSize", 1))
    module = params.get("module")
    modules_list = params.get("modules")  # list of supported modules
    force = bool(params.get("force", False))
    lease_ttl_seconds = int(params.get("leaseTtlSeconds", 120))

    conn = _get_db()
    queue = JobQueue(conn)
    repo = Repository(conn)

    active_chunk_ids: set[int] | None = None
    if _active_worker is not None:
        active_chunk_ids = _active_worker.current_chunk_ids
    policy = compute_claim_policy(
        conn,
        worker_id=worker_id,
        batch_size=batch_size,
        module=module,
        modules_list=modules_list,
        active_chunk_ids=active_chunk_ids,
        coordinator_run_active=_active_worker is not None,
    )
    if not policy.allow_claims:
        return {"jobs": []}

    requested = policy.requested
    scan_size = policy.scan_size
    module_filter = policy.module_filter
    modules_filter = policy.modules_filter
    prefer_module = policy.prefer_module
    prefer_image_ids = policy.prefer_image_ids
    pending_eligible = queue.pending_count(module_filter, modules=modules_filter)
    pending_candidates = max(requested, pending_eligible)
    # Scan far enough in one request to rotate past large blocked backlogs
    # (for example hundreds of faces/ocr jobs waiting on objects) instead of
    # forcing remote workers to idle through many empty polls.
    max_scan_candidates = max(
        scan_size,
        min(pending_candidates, max(2048, requested * 256)),
    )
    jobs: list[dict[str, Any]] = []
    scanned_candidates = 0
    # Cache-gated dispatch: only activate when the decoded store has been
    # initialised (by server startup or status poll).  Jobs are restricted
    # to cached images so workers never need NAS access.  When the cache is
    # empty the gate returns an empty set → no jobs dispatched → workers
    # keep polling until the pre-decoder populates the cache.
    cache_gate_ids: set[int] | None = None
    if _decoded_store is not None:
        try:
            cache_gate_ids = _decoded_store.cached_image_ids()
            _auto_trigger_pre_decode()
        except Exception:
            pass
    # When a module yields only invalid jobs for several consecutive batches,
    # exclude it so lower-priority modules (e.g. embedding) can be reached.
    exhausted_modules: set[str] = set()
    module_miss_streak: dict[str, int] = {}
    _MISS_THRESHOLD = scan_size * 2  # skip module after this many consecutive misses
    while len(jobs) < requested and scanned_candidates < max_scan_candidates:
        claim_size = min(scan_size, max_scan_candidates - scanned_candidates)
        excl = sorted(exhausted_modules) if exhausted_modules else None
        claimed = queue.claim_leased(
            worker_id=worker_id,
            lease_ttl_seconds=max(5, lease_ttl_seconds),
            batch_size=claim_size,
            module=module_filter,
            modules=modules_filter,
            exclude_modules=excl,
            prefer_module=prefer_module,
            prefer_image_ids=prefer_image_ids,
            restrict_image_ids=cache_gate_ids,
        )
        if not claimed:
            break
        scanned_candidates += len(claimed)

        batch_valid_modules: set[str] = set()

        for job in claimed:
            if len(jobs) >= requested:
                queue.release_leased(job["id"], job["lease_token"])
                continue

            image = repo.get_image(job["image_id"])
            if image is None:
                queue.mark_failed_leased(job["id"], job["lease_token"], "image_not_found")
                continue

            image_id = int(job["image_id"])
            module_name = str(job["module"])

            job_force = force or str(job.get("last_node_role") or "").lower() == "force"

            already_analyzed = repo.is_analyzed(image_id, module_name)
            if not job_force and already_analyzed:
                queue.mark_skipped_leased(job["id"], job["lease_token"], "already_analyzed")
                continue

            prereq = _PREREQUISITES.get(module_name)
            if prereq and not repo.is_analyzed(image_id, prereq):
                prereq_status = queue.get_image_module_job_status(image_id, prereq)
                if prereq_status in ("failed", "skipped"):
                    queue.mark_skipped_leased(
                        job["id"],
                        job["lease_token"],
                        f"prerequisite_{prereq}_{prereq_status}",
                    )
                else:
                    queue.release_leased(job["id"], job["lease_token"])
                continue

            batch_valid_modules.add(module_name)
            job_entry: dict[str, Any] = {
                "id": job["id"],
                "imageId": image_id,
                "module": module_name,
                "attempts": job["attempts"],
                "leaseToken": job["lease_token"],
                "leaseExpiresAt": job["lease_expires_at"],
                "filePath": image["file_path"],
                "image": {
                    "width": image.get("width"),
                    "height": image.get("height"),
                    "format": image.get("format"),
                },
                "context": _build_distributed_job_context(repo, image_id, module_name),
            }
            if cache_gate_ids is not None:
                job_entry["hasDecodedCache"] = True
            jobs.append(job_entry)

        # Track per-module miss streaks to detect exhausted modules.
        # Only in multi-module mode — single-module filter is explicit.
        if not module_filter:
            for mod in {str(j["module"]) for j in claimed}:
                if mod in batch_valid_modules:
                    module_miss_streak.pop(mod, None)
                else:
                    miss_count = sum(1 for j in claimed if str(j["module"]) == mod)
                    module_miss_streak[mod] = module_miss_streak.get(mod, 0) + miss_count
                    if module_miss_streak[mod] >= _MISS_THRESHOLD:
                        exhausted_modules.add(mod)

        if len(claimed) < claim_size:
            break

    # Update worker affinity epoch for sticky-module scheduling.
    if jobs:
        last_mod = jobs[-1]["module"]
        try:
            record_worker_affinity(
                conn,
                worker_id=worker_id,
                module=str(last_mod),
            )
        except Exception:
            pass  # non-critical

    return {"jobs": jobs}


def _handle_jobs_release_expired(_params: dict) -> dict:
    """Requeue expired leases (coordinator safety sweep)."""
    from imganalyzer.db.queue import JobQueue

    conn = _get_db()
    queue = JobQueue(conn)
    released = queue.release_expired_leases()
    return {"released": released}


def _handle_jobs_heartbeat(params: dict) -> dict:
    """Extend the lease for an active distributed job."""
    from imganalyzer.db.queue import JobQueue

    job_id = int(params.get("jobId", 0))
    lease_token = str(params.get("leaseToken", "")).strip()
    extend_ttl_seconds = int(params.get("extendTtlSeconds", 120))
    if job_id <= 0 or not lease_token:
        raise ValueError("jobId and leaseToken are required")

    conn = _get_db()
    queue = JobQueue(conn)
    ok = queue.heartbeat_lease(job_id, lease_token, extend_ttl_seconds)
    return {"ok": ok}


def _handle_jobs_release(params: dict) -> dict:
    """Return a leased job to pending when a worker defers it."""
    from imganalyzer.db.queue import JobQueue

    job_id = int(params.get("jobId", 0))
    lease_token = str(params.get("leaseToken", "")).strip()
    if job_id <= 0 or not lease_token:
        raise ValueError("jobId and leaseToken are required")

    conn = _get_db()
    queue = JobQueue(conn)
    ok = queue.release_leased(job_id, lease_token)
    return {"ok": ok}


def _worker_node_info(conn: Any, job_id: int) -> tuple[str, str, str]:
    """Return (nodeId, nodeRole, nodeLabel) for a completed job."""
    row = conn.execute(
        """SELECT jq.last_node_id, jq.last_node_role, wn.display_name
           FROM job_queue jq
           LEFT JOIN worker_nodes wn ON jq.last_node_id = wn.id
           WHERE jq.id = ?""",
        [job_id],
    ).fetchone()
    if row is None:
        return (_MASTER_WORKER_ID, "master", _MASTER_WORKER_LABEL)
    node_role = str(row["last_node_role"] or "master")
    node_id = str(row["last_node_id"] or _MASTER_WORKER_ID)
    node_label = (
        str(row["display_name"])
        if node_role == "worker" and row["display_name"]
        else _MASTER_WORKER_LABEL
    )
    return (node_id, node_role, node_label)


def _job_duration_ms(conn: Any, job_id: int) -> int:
    """Compute wall-clock duration in ms from started_at to completed_at."""
    row = conn.execute(
        """SELECT CAST(
               ROUND((julianday(completed_at) - julianday(started_at)) * 86400000.0)
           AS INTEGER) AS ms
           FROM job_queue WHERE id = ?""",
        [job_id],
    ).fetchone()
    return max(0, int(row["ms"] or 0)) if row else 0


def _normalize_result_keywords(value: Any) -> list[str] | None:
    """Normalize keyword payload values to a non-empty string list."""
    if value is None:
        return None
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return cleaned or None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.startswith("[") and text.endswith("]"):
            try:
                decoded = json.loads(text)
                if isinstance(decoded, list):
                    cleaned = [str(item).strip() for item in decoded if str(item).strip()]
                    return cleaned or None
            except Exception:
                pass
        cleaned = [part.strip() for part in text.split(",") if part.strip()]
        return cleaned or None
    return None


def _handle_jobs_complete(params: dict) -> dict:
    """Persist a worker result payload and mark the leased job complete."""
    from imganalyzer.db.repository import Repository
    from imganalyzer.pipeline.distributed_payloads import persist_result_payload
    from imganalyzer.pipeline.modules import write_xmp_from_db
    from imganalyzer.pipeline.unified_scheduler import record_worker_module_timing

    job_id = int(params.get("jobId", 0))
    lease_token = str(params.get("leaseToken", "")).strip()
    payload = params.get("payload", {})
    no_xmp = bool(params.get("noXmp", False))
    processing_ms_raw = params.get("processingMs")
    if job_id <= 0 or not lease_token:
        raise ValueError("jobId and leaseToken are required")
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    try:
        processing_ms = max(0, int(processing_ms_raw)) if processing_ms_raw is not None else 0
    except (TypeError, ValueError):
        processing_ms = 0

    conn = _get_db()
    repo = Repository(conn)
    worker_id_for_timing = ""

    conn.execute("BEGIN IMMEDIATE")
    try:
        lease = conn.execute(
            "SELECT worker_id FROM job_leases WHERE job_id = ? AND lease_token = ?",
            [job_id, lease_token],
        ).fetchone()
        if lease is None:
            conn.rollback()
            return {"ok": False}
        worker_id_for_timing = str(lease["worker_id"] or "")

        job_row = conn.execute(
            "SELECT image_id, module FROM job_queue WHERE id = ?",
            [job_id],
        ).fetchone()
        if job_row is None:
            conn.rollback()
            raise ValueError(f"Job {job_id} not found")

        image_id = int(job_row["image_id"])
        module_name = str(job_row["module"])
        persist_result_payload(
            conn,
            repo,
            image_id=image_id,
            module=module_name,
            payload=payload,
        )
        if module_name in _DISTRIBUTED_SEARCH_MODULES:
            repo.update_search_index(image_id)

        conn.execute(
            """UPDATE job_queue
               SET status = 'done',
                   processing_ms = CASE WHEN ? > 0 THEN ? ELSE NULL END,
                   started_at = CASE
                       WHEN ? > 0 THEN strftime(
                           '%Y-%m-%d %H:%M:%f',
                           julianday('now') - (? / 86400000.0)
                       )
                       ELSE started_at
                   END,
                   completed_at = strftime('%Y-%m-%d %H:%M:%f', 'now')
               WHERE id = ?""",
            [processing_ms, processing_ms, processing_ms, processing_ms, job_id],
        )
        conn.execute("DELETE FROM job_leases WHERE job_id = ?", [job_id])
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    if worker_id_for_timing and processing_ms > 0:
        try:
            record_worker_module_timing(
                conn,
                worker_id=worker_id_for_timing,
                module=module_name,
                processing_ms=processing_ms,
            )
        except Exception:
            pass

    if not no_xmp:
        remaining = conn.execute(
            """SELECT 1
               FROM job_queue
               WHERE image_id = ? AND status IN ('pending', 'running')
               LIMIT 1""",
            [image_id],
        ).fetchone()
        if remaining is None:
            try:
                write_xmp_from_db(repo, image_id)
            except Exception:
                pass

    # Emit a run/result notification so the frontend live-feed shows this
    # worker result immediately (same as the master-side _emit_result path).
    try:
        image = repo.get_image(image_id)
        file_path = image["file_path"] if image else ""
        node_id, node_role, node_label = _worker_node_info(conn, job_id)
        duration_ms = _job_duration_ms(conn, job_id)
        kw: list[str] | None = None
        if module_name == "caption":
            payload_data = payload.get("data")
            if isinstance(payload_data, dict):
                for key in ("keywords", "keyword"):
                    kw = _normalize_result_keywords(payload_data.get(key))
                    if kw:
                        break
        note: dict[str, Any] = {
            "path": file_path,
            "module": module_name,
            "status": "done",
            "ms": duration_ms,
            "nodeId": node_id,
            "nodeRole": node_role,
            "nodeLabel": node_label,
        }
        if kw:
            note["keywords"] = kw
        _send_notification("run/result", note)
    except Exception:
        pass  # notification failure is non-fatal

    return {"ok": True}


def _handle_jobs_fail(params: dict) -> dict:
    """Mark a leased job as failed when worker reports an error."""
    from imganalyzer.db.queue import JobQueue

    job_id = int(params.get("jobId", 0))
    lease_token = str(params.get("leaseToken", "")).strip()
    error = str(params.get("error", "remote worker error"))
    if job_id <= 0 or not lease_token:
        raise ValueError("jobId and leaseToken are required")

    conn = _get_db()
    queue = JobQueue(conn)
    ok = queue.mark_failed_leased(job_id, lease_token, error)
    return {"ok": ok}


def _handle_jobs_skip(params: dict) -> dict:
    """Mark a leased job skipped when a worker intentionally bypasses it."""
    from imganalyzer.db.queue import JobQueue
    from imganalyzer.db.repository import Repository

    job_id = int(params.get("jobId", 0))
    lease_token = str(params.get("leaseToken", "")).strip()
    reason = str(params.get("reason", "skipped")).strip()
    details = str(params.get("details", "")).strip()
    if job_id <= 0 or not lease_token:
        raise ValueError("jobId and leaseToken are required")

    conn = _get_db()
    queue = JobQueue(conn)
    ok = queue.mark_skipped_leased(job_id, lease_token, reason)
    if ok and reason == "corrupt_file" and details:
        repo = Repository(conn)
        job_row = conn.execute("SELECT image_id FROM job_queue WHERE id = ?", [job_id]).fetchone()
        if job_row is not None:
            queue.mark_image_pending_jobs_skipped(int(job_row["image_id"]), reason)
            image = repo.get_image(int(job_row["image_id"]))
            if image is not None:
                conn.execute(
                    """INSERT OR IGNORE INTO corrupt_files (image_id, file_path, error_msg)
                       VALUES (?, ?, ?)""",
                    [job_row["image_id"], image["file_path"], details],
                )
                conn.commit()
    return {"ok": ok}


def _handle_jobs_release_worker(params: dict) -> dict:
    """Release all leases for a worker (disconnect/failover path)."""
    from imganalyzer.db.queue import JobQueue

    worker_id = str(params.get("workerId", "")).strip()
    if not worker_id:
        raise ValueError("workerId is required")

    conn = _get_db()
    queue = JobQueue(conn)
    released = queue.release_worker_leases(worker_id)
    return {"released": released}


def _handle_rebuild(params: dict) -> dict:
    """Re-enqueue module jobs."""
    from imganalyzer.pipeline.batch import BatchProcessor

    conn = _get_db()
    processor = BatchProcessor(conn)

    module = params["module"]
    image_path = params.get("imagePath")
    force = params.get("force", True)
    failed_only = params.get("failedOnly", False)

    count = processor.rebuild_module(
        module=module,
        image_path=image_path,
        force=force,
        failed_only=failed_only,
    )
    return {"enqueued": count}


def _handle_search(params: dict) -> dict:
    """Execute a search query and return results.

    Reuses the same logic as the search-json CLI command but without
    subprocess overhead and with proper CLIP model lifecycle.
    """
    import json as _json

    from imganalyzer.db.search import SearchEngine

    conn = _get_db()
    engine = SearchEngine(conn)

    query = params.get("query", "")
    mode = params.get("mode", "hybrid")
    semantic_weight = params.get("semanticWeight", 0.5)
    face = params.get("face")
    faces_raw = params.get("faces")
    face_match = params.get("faceMatch", "all")
    similar_to_image_id = params.get("similarToImageId")
    camera = params.get("camera")
    lens = params.get("lens")
    location = params.get("location")
    country = params.get("country")
    aesthetic_min = params.get("aestheticMin")
    aesthetic_max = params.get("aestheticMax")
    sharpness_min = params.get("sharpnessMin")
    sharpness_max = params.get("sharpnessMax")
    noise_max = params.get("noiseMax")
    iso_min = params.get("isoMin")
    iso_max = params.get("isoMax")
    faces_min = params.get("facesMin")
    faces_max = params.get("facesMax")
    date_from = params.get("dateFrom")
    date_to = params.get("dateTo")
    recurring_month_day = params.get("recurringMonthDay")
    time_of_day = params.get("timeOfDay")
    has_people = params.get("hasPeople")
    sort_by = params.get("sortBy", "relevance")
    expanded_terms_raw = params.get("expandedTerms", [])
    try:
        limit = int(params.get("limit", 200))
    except (TypeError, ValueError):
        raise ValueError("limit must be an integer")
    try:
        offset = int(params.get("offset", 0))
    except (TypeError, ValueError):
        raise ValueError("offset must be an integer")
    limit = max(1, min(limit, 500))
    offset = max(0, offset)

    if recurring_month_day is not None:
        recurring_month_day = str(recurring_month_day).strip()
        if recurring_month_day and not re.fullmatch(r"\d{2}-\d{2}", recurring_month_day):
            raise ValueError("recurringMonthDay must be in MM-DD format")
    valid_time_of_day = {"morning", "afternoon", "evening", "night"}
    if time_of_day is not None and time_of_day not in valid_time_of_day:
        raise ValueError("timeOfDay must be one of morning, afternoon, evening or night")
    valid_sort_by = {"relevance", "best", "aesthetic", "sharpness", "cleanest", "newest"}
    if sort_by not in valid_sort_by:
        raise ValueError("sortBy must be one of relevance, best, aesthetic, sharpness, cleanest or newest")
    if expanded_terms_raw is None:
        expanded_terms_raw = []
    if not isinstance(expanded_terms_raw, list):
        raise ValueError("expandedTerms must be an array")
    expanded_terms: list[str] = []
    seen_terms: set[str] = set()
    for term in expanded_terms_raw:
        if not isinstance(term, str):
            raise ValueError("expandedTerms entries must be strings")
        clean = term.strip()
        lowered = clean.casefold()
        if clean and lowered not in seen_terms:
            seen_terms.add(lowered)
            expanded_terms.append(clean)

    if faces_raw is None:
        faces_raw = []
    if not isinstance(faces_raw, list):
        raise ValueError("faces must be an array")
    if face_match not in {None, "any", "all"}:
        raise ValueError("faceMatch must be 'any' or 'all'")
    normalized_faces: list[str] = []
    seen_faces: set[str] = set()
    for raw_face in [*faces_raw, face]:
        if raw_face is None:
            continue
        if not isinstance(raw_face, str):
            raise ValueError("face and faces entries must be strings")
        clean = raw_face.strip()
        lowered = clean.casefold()
        if clean and lowered not in seen_faces:
            seen_faces.add(lowered)
            normalized_faces.append(clean)
    faces = normalized_faces
    face = faces[0] if faces else None
    face_match = "all" if face_match is None else face_match

    if not faces and query:
        resolve_face_queries = getattr(engine, "resolve_face_queries", None)
        if callable(resolve_face_queries):
            resolved_faces, remaining_query, resolved_match = resolve_face_queries(str(query))
            if resolved_faces:
                faces = resolved_faces
                face = faces[0]
                face_match = resolved_match
                query = remaining_query
        else:
            resolve_face_query = getattr(engine, "resolve_face_query", None)
            if callable(resolve_face_query):
                resolved_face, remaining_query = resolve_face_query(str(query))
                if resolved_face:
                    faces = [resolved_face]
                    face = resolved_face
                    query = remaining_query

    has_text_query = bool(query and query.strip())
    base_conditions: list[str] = []
    base_params: list[Any] = []
    datetime_expr = "replace(COALESCE(m.date_time_original, ''), ' ', 'T')"
    hour_expr = (
        "CASE "
        f"WHEN length({datetime_expr}) >= 13 "
        f"THEN CAST(substr({datetime_expr}, 12, 2) AS INTEGER) "
        "END"
    )

    if camera:
        base_conditions.append("(m.camera_make LIKE ? OR m.camera_model LIKE ?)")
        base_params.extend([f"%{camera}%", f"%{camera}%"])
    if lens:
        base_conditions.append("m.lens_model LIKE ?")
        base_params.append(f"%{lens}%")
    if country:
        base_conditions.append("m.location_country LIKE ?")
        base_params.append(f"%{country}%")
    if location:
        base_conditions.append(
            "(m.location_city LIKE ? OR m.location_state LIKE ? OR m.location_country LIKE ?)"
        )
        base_params.extend([f"%{location}%", f"%{location}%", f"%{location}%"])
    if date_from:
        base_conditions.append("m.date_time_original >= ?")
        base_params.append(date_from)
    if date_to:
        base_conditions.append("m.date_time_original <= ?")
        base_params.append(date_to + "T23:59:59")
    if recurring_month_day:
        base_conditions.append(f"substr({datetime_expr}, 6, 5) = ?")
        base_params.append(recurring_month_day)
    if time_of_day == "morning":
        base_conditions.append(f"{hour_expr} BETWEEN 5 AND 11")
    elif time_of_day == "afternoon":
        base_conditions.append(f"{hour_expr} BETWEEN 12 AND 16")
    elif time_of_day == "evening":
        base_conditions.append(f"{hour_expr} BETWEEN 17 AND 20")
    elif time_of_day == "night":
        base_conditions.append(f"({hour_expr} >= 21 OR {hour_expr} < 5)")
    if iso_min is not None:
        base_conditions.append("CAST(m.iso AS REAL) >= ?")
        base_params.append(iso_min)
    if iso_max is not None:
        base_conditions.append("CAST(m.iso AS REAL) <= ?")
        base_params.append(iso_max)
    if aesthetic_min is not None:
        base_conditions.append("COALESCE(ap.perception_iaa, ae.aesthetic_score) >= ?")
        base_params.append(aesthetic_min)
    if aesthetic_max is not None:
        base_conditions.append("COALESCE(ap.perception_iaa, ae.aesthetic_score) <= ?")
        base_params.append(aesthetic_max)
    if sharpness_min is not None:
        base_conditions.append("t.sharpness_score >= ?")
        base_params.append(sharpness_min)
    if sharpness_max is not None:
        base_conditions.append("t.sharpness_score <= ?")
        base_params.append(sharpness_max)
    if noise_max is not None:
        base_conditions.append("t.noise_level <= ?")
        base_params.append(noise_max)
    if faces_min is not None:
        base_conditions.append("COALESCE(la.face_count, af.face_count) >= ?")
        base_params.append(faces_min)
    if faces_max is not None:
        base_conditions.append("COALESCE(la.face_count, af.face_count) <= ?")
        base_params.append(faces_max)
    if has_people is True:
        base_conditions.append("COALESCE(la.has_people, ob.has_person) = 1")
    elif has_people is False:
        base_conditions.append(
            "(COALESCE(la.has_people, ob.has_person) = 0 OR "
            "COALESCE(la.has_people, ob.has_person) IS NULL)"
        )

    select_cols = """
        i.id AS image_id, i.file_path, i.width, i.height, i.file_size,
        m.camera_make, m.camera_model, m.lens_model, m.focal_length,
        m.f_number, m.exposure_time, m.iso, m.date_time_original,
        m.gps_latitude, m.gps_longitude, m.location_city, m.location_state,
        m.location_country,
        t.sharpness_score, t.sharpness_label, t.exposure_ev, t.exposure_label,
        t.noise_level, t.noise_label, t.snr_db, t.dynamic_range_stops,
        t.highlight_clipping_pct, t.shadow_clipping_pct, t.avg_saturation,
        t.dominant_colors,
        COALESCE(la.description, b2.description) AS description,
        COALESCE(la.scene_type, b2.scene_type) AS scene_type,
        COALESCE(la.main_subject, b2.main_subject) AS main_subject,
        COALESCE(la.lighting, b2.lighting) AS lighting,
        COALESCE(la.mood, b2.mood) AS mood,
        COALESCE(la.keywords, b2.keywords) AS keywords,
        COALESCE(la.detected_objects, ob.detected_objects) AS detected_objects,
        COALESCE(la.face_count, af.face_count) AS face_count,
        COALESCE(la.face_identities, af.face_identities) AS face_identities,
        COALESCE(la.has_people, ob.has_person) AS has_people,
        COALESCE(la.ocr_text, ocr.ocr_text) AS ocr_text,
        (
            SELECT ca.description
            FROM analysis_cloud_ai ca
            WHERE ca.image_id = i.id
              AND ca.description IS NOT NULL
              AND TRIM(ca.description) != ''
            ORDER BY ca.analyzed_at DESC, ca.id DESC
            LIMIT 1
        ) AS cloud_description,
        COALESCE(ap.perception_iaa, ae.aesthetic_score) AS aesthetic_score,
        COALESCE(ap.perception_iaa_label, ae.aesthetic_label) AS aesthetic_label,
        NULL AS aesthetic_reason,
        ap.perception_iaa, ap.perception_iaa_label,
        ap.perception_iqa, ap.perception_iqa_label,
        ap.perception_ista, ap.perception_ista_label
    """

    joins = """
        FROM images i
        LEFT JOIN analysis_metadata    m  ON m.image_id  = i.id
        LEFT JOIN analysis_technical   t  ON t.image_id  = i.id
        LEFT JOIN analysis_caption     la ON la.image_id = i.id
        LEFT JOIN analysis_blip2       b2 ON b2.image_id = i.id
        LEFT JOIN analysis_objects     ob ON ob.image_id = i.id
        LEFT JOIN analysis_ocr        ocr ON ocr.image_id = i.id
        LEFT JOIN analysis_faces       af ON af.image_id = i.id
        LEFT JOIN analysis_aesthetic   ae ON ae.image_id = i.id
        LEFT JOIN analysis_perception  ap ON ap.image_id = i.id
    """

    def _json_field(val: Any) -> Any:
        if val is None:
            return None
        try:
            return _json.loads(val)
        except Exception:
            return val

    def _build_where_clause(candidate_ids: list[int] | None = None) -> tuple[str, list[Any]]:
        conditions = list(base_conditions)
        query_params = list(base_params)
        if candidate_ids is not None:
            if not candidate_ids:
                return "WHERE 1 = 0", []
            placeholders = ",".join("?" * len(candidate_ids))
            conditions.insert(0, f"i.id IN ({placeholders})")
            query_params = [*candidate_ids, *query_params]
        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        return where_clause, query_params

    def _rows_to_records(
        rows: list[sqlite3.Row],
        score_lookup: dict[int, float] | None,
    ) -> list[dict[str, Any]]:
        records = []
        for row in rows:
            image_id = int(row["image_id"])
            records.append({
                "image_id": image_id,
                "file_path": row["file_path"],
                "score": score_lookup.get(image_id) if score_lookup is not None else None,
                "width": row["width"],
                "height": row["height"],
                "file_size": row["file_size"],
                "camera_make": row["camera_make"],
                "camera_model": row["camera_model"],
                "lens_model": row["lens_model"],
                "focal_length": row["focal_length"],
                "f_number": row["f_number"],
                "exposure_time": row["exposure_time"],
                "iso": row["iso"],
                "date_time_original": row["date_time_original"],
                "gps_latitude": row["gps_latitude"],
                "gps_longitude": row["gps_longitude"],
                "location_city": row["location_city"],
                "location_state": row["location_state"],
                "location_country": row["location_country"],
                "sharpness_score": row["sharpness_score"],
                "sharpness_label": row["sharpness_label"],
                "exposure_ev": row["exposure_ev"],
                "exposure_label": row["exposure_label"],
                "noise_level": row["noise_level"],
                "noise_label": row["noise_label"],
                "snr_db": row["snr_db"],
                "dynamic_range_stops": row["dynamic_range_stops"],
                "highlight_clipping_pct": row["highlight_clipping_pct"],
                "shadow_clipping_pct": row["shadow_clipping_pct"],
                "avg_saturation": row["avg_saturation"],
                "dominant_colors": _json_field(row["dominant_colors"]),
                "description": row["description"],
                "scene_type": row["scene_type"],
                "main_subject": row["main_subject"],
                "lighting": row["lighting"],
                "mood": row["mood"],
                "keywords": _json_field(row["keywords"]),
                "detected_objects": _json_field(row["detected_objects"]),
                "face_count": row["face_count"],
                "face_identities": _json_field(row["face_identities"]),
                "has_people": bool(row["has_people"]) if row["has_people"] is not None else None,
                "ocr_text": row["ocr_text"],
                "cloud_description": row["cloud_description"],
                "aesthetic_score": row["aesthetic_score"],
                "aesthetic_label": row["aesthetic_label"],
                "aesthetic_reason": row["aesthetic_reason"],
                "perception_iaa": row["perception_iaa"],
                "perception_iaa_label": row["perception_iaa_label"],
                "perception_iqa": row["perception_iqa"],
                "perception_iqa_label": row["perception_iqa_label"],
                "perception_ista": row["perception_ista"],
                "perception_ista_label": row["perception_ista_label"],
            })
        if score_lookup is not None and sort_by == "relevance":
            records.sort(key=lambda record: -(record["score"] or 0.0))
        return records

    def _fetch_records(
        candidate_ids: list[int] | None,
        score_lookup: dict[int, float] | None,
    ) -> list[dict[str, Any]]:
        where_clause, query_params = _build_where_clause(candidate_ids)
        rows = conn.execute(
            f"SELECT {select_cols} {joins} {where_clause}",
            query_params,
        ).fetchall()
        return _rows_to_records(rows, score_lookup)

    def _quality_score(record: dict[str, Any]) -> float:
        aesthetic = float(record["aesthetic_score"] or 0.0)
        sharpness = float(record["sharpness_score"] or 0.0)
        noise = float(record["noise_level"] or 0.0)
        return (aesthetic * 12.0) + (sharpness * 0.25) - (noise * 120.0)

    def _sort_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if sort_by == "relevance":
            return records
        if sort_by == "best":
            return sorted(
                records,
                key=lambda record: (
                    _quality_score(record),
                    float(record["aesthetic_score"] or 0.0),
                    float(record["sharpness_score"] or 0.0),
                ),
                reverse=True,
            )
        if sort_by == "aesthetic":
            return sorted(
                records,
                key=lambda record: (
                    float(record["aesthetic_score"] or 0.0),
                    float(record["sharpness_score"] or 0.0),
                ),
                reverse=True,
            )
        if sort_by == "sharpness":
            return sorted(
                records,
                key=lambda record: (
                    float(record["sharpness_score"] or 0.0),
                    float(record["aesthetic_score"] or 0.0),
                ),
                reverse=True,
            )
        if sort_by == "cleanest":
            return sorted(
                records,
                key=lambda record: (
                    record["noise_level"] is None,
                    float(record["noise_level"]) if record["noise_level"] is not None else float("inf"),
                    -float(record["sharpness_score"] or 0.0),
                ),
            )
        if sort_by == "newest":
            return sorted(
                records,
                key=lambda record: ((record["date_time_original"] or ""), int(record["image_id"])),
                reverse=True,
            )
        return records

    def _browse_order_clause() -> str:
        if sort_by == "best":
            return (
                " ORDER BY "
                "((COALESCE(ap.perception_iaa, ae.aesthetic_score, 0) * 12.0) + "
                "(COALESCE(t.sharpness_score, 0) * 0.25) - "
                "(COALESCE(t.noise_level, 0) * 120.0)) DESC, "
                "COALESCE(ap.perception_iaa, ae.aesthetic_score) DESC, "
                "t.sharpness_score DESC, i.id DESC"
            )
        if sort_by == "aesthetic":
            return (
                " ORDER BY COALESCE(ap.perception_iaa, ae.aesthetic_score) DESC, "
                "t.sharpness_score DESC, i.id DESC"
            )
        if sort_by == "sharpness":
            return (
                " ORDER BY t.sharpness_score DESC, "
                "COALESCE(ap.perception_iaa, ae.aesthetic_score) DESC, i.id DESC"
            )
        if sort_by == "cleanest":
            return " ORDER BY t.noise_level ASC, t.sharpness_score DESC, i.id DESC"
        if sort_by == "newest":
            return " ORDER BY m.date_time_original DESC, i.id DESC"
        return ""

    def _search_text_terms(candidate_limit: int) -> list[dict[str, Any]]:
        terms: list[str] = []
        if has_text_query:
            terms.append(query.strip())
        seen_local = {term.casefold() for term in terms}
        for term in expanded_terms:
            lowered = term.casefold()
            if lowered not in seen_local:
                seen_local.add(lowered)
                terms.append(term)
        if not terms:
            return []
        if len(terms) == 1:
            return engine.search(
                terms[0],
                limit=candidate_limit,
                semantic_weight=semantic_weight,
                mode=mode,
            )

        fused_scores: dict[int, float] = {}
        file_paths: dict[int, str] = {}
        for term in terms:
            term_results = engine.search(
                term,
                limit=candidate_limit,
                semantic_weight=semantic_weight,
                mode=mode,
            )
            for rank, result in enumerate(term_results):
                image_id = int(result["image_id"])
                fused_scores[image_id] = fused_scores.get(image_id, 0.0) + (1.0 / (61 + rank))
                if "file_path" in result:
                    file_paths.setdefault(image_id, str(result["file_path"]))

        return [
            {
                "image_id": image_id,
                "file_path": file_paths.get(image_id, ""),
                "score": score,
            }
            for image_id, score in sorted(
                fused_scores.items(),
                key=lambda item: -item[1],
            )[:candidate_limit]
        ]

    def _search_face_terms(candidate_limit: int) -> list[dict[str, Any]]:
        if not faces:
            return []
        if len(faces) == 1:
            return engine.search_face(faces[0], limit=candidate_limit)
        search_faces = getattr(engine, "search_faces", None)
        if callable(search_faces):
            return search_faces(faces, limit=candidate_limit, match_mode=face_match)

        per_face_results = [engine.search_face(name, limit=candidate_limit) for name in faces]
        if face_match != "any" and any(not rows for rows in per_face_results):
            return []

        aggregate: dict[int, dict[str, Any]] = {}
        for rows in per_face_results:
            for row in rows:
                image_id = int(row["image_id"])
                current = aggregate.setdefault(image_id, {
                    "image_id": image_id,
                    "file_path": row.get("file_path", ""),
                    "score": 0.0,
                    "matches": 0,
                })
                current["score"] = float(current["score"]) + float(row.get("score", 0.0))
                current["matches"] = int(current["matches"]) + 1

        required_matches = len(faces) if face_match == "all" else 1
        combined = [
            {
                "image_id": int(item["image_id"]),
                "file_path": item["file_path"],
                "score": float(item["score"]),
            }
            for item in aggregate.values()
            if int(item["matches"]) >= required_matches
        ]
        combined.sort(key=lambda item: (-float(item["score"]), int(item["image_id"])))
        return combined[:candidate_limit]

    def _combine_face_and_text_terms(candidate_limit: int) -> tuple[list[dict[str, Any]], bool]:
        face_results = _search_face_terms(candidate_limit)
        text_results = _search_text_terms(candidate_limit)
        if not face_results or not text_results:
            return [], len(face_results) < candidate_limit and len(text_results) < candidate_limit

        face_scores = {
            int(result["image_id"]): float(result["score"])
            for result in face_results
        }
        combined_results: list[dict[str, Any]] = []
        for result in text_results:
            image_id = int(result["image_id"])
            if image_id not in face_scores:
                continue
            combined_results.append({
                **result,
                "score": float(result["score"]) + face_scores[image_id],
            })

        search_exhausted = len(face_results) < candidate_limit and len(text_results) < candidate_limit
        return combined_results, search_exhausted

    if similar_to_image_id is not None:
        try:
            similar_to_image_id = int(similar_to_image_id)
        except (TypeError, ValueError):
            raise ValueError("similarToImageId must be an integer")

    if similar_to_image_id is not None or faces or ((has_text_query or expanded_terms) and mode != "browse"):
        page_end = offset + limit
        quality_sort_floor = 400 if sort_by in {"best", "aesthetic", "sharpness", "cleanest"} else 200
        candidate_limit = max((page_end + 1) * 4, quality_sort_floor)
        max_candidate_limit = max(candidate_limit, 5000)
        search_exhausted = True
        records: list[dict[str, Any]] = []

        while True:
            if similar_to_image_id is not None:
                search_results = engine.search_similar_image(similar_to_image_id, limit=candidate_limit)
                search_exhausted = len(search_results) < candidate_limit
            elif faces and (has_text_query or expanded_terms):
                search_results, search_exhausted = _combine_face_and_text_terms(candidate_limit)
            elif faces:
                search_results = _search_face_terms(candidate_limit)
                search_exhausted = len(search_results) < candidate_limit
            else:
                search_results = _search_text_terms(candidate_limit)
                search_exhausted = len(search_results) < candidate_limit

            candidate_ids = [int(result["image_id"]) for result in search_results]
            if not candidate_ids:
                if not search_exhausted and candidate_limit < max_candidate_limit:
                    candidate_limit = min(max_candidate_limit, candidate_limit * 2)
                    continue
                return {"results": [], "total": 0, "hasMore": False}

            score_map = {
                int(result["image_id"]): float(result["score"])
                for result in search_results
            }
            records = _fetch_records(candidate_ids, score_map)
            records = _sort_records(records)
            enough_for_page = len(records) > page_end
            if search_exhausted or enough_for_page or candidate_limit >= max_candidate_limit:
                break
            candidate_limit = min(max_candidate_limit, candidate_limit * 2)

        has_more = len(records) > page_end or not search_exhausted
        total: int | None = len(records) if search_exhausted else None
        return {
            "results": records[offset: offset + limit],
            "total": total,
            "hasMore": has_more,
        }

    where_clause, query_params = _build_where_clause()
    total_row = conn.execute(
        f"SELECT COUNT(*) AS cnt {joins} {where_clause}",
        query_params,
    ).fetchone()
    total = int(total_row["cnt"]) if total_row else 0
    rows = conn.execute(
        f"SELECT {select_cols} {joins} {where_clause}{_browse_order_clause()} LIMIT ? OFFSET ?",
        [*query_params, limit, offset],
    ).fetchall()
    records = _rows_to_records(rows, None)
    has_more = offset + len(records) < total
    return {"results": records, "total": total, "hasMore": has_more}


def _handle_search_resolve_face_query(params: dict) -> dict[str, Any]:
    from imganalyzer.db.search import SearchEngine

    query = params.get("query", "")
    if not isinstance(query, str):
        raise ValueError("query must be a string")

    conn = _get_db()
    engine = SearchEngine(conn)
    resolve_face_queries = getattr(engine, "resolve_face_queries", None)
    if callable(resolve_face_queries):
        faces, remaining_query, face_match = resolve_face_queries(query)
    else:
        face, remaining_query = engine.resolve_face_query(query)
        faces = [face] if face else []
        face_match = "all"
    return {
        "face": faces[0] if faces else None,
        "faces": faces,
        "faceMatch": face_match,
        "remainingQuery": remaining_query,
    }


def _processed_exists_clause(alias: str = "i") -> str:
    """SQL predicate: image has at least one processed analysis record."""
    return f"""(
        EXISTS(SELECT 1 FROM analysis_metadata m  WHERE m.image_id  = {alias}.id) OR
        EXISTS(SELECT 1 FROM analysis_technical t WHERE t.image_id  = {alias}.id) OR
        EXISTS(SELECT 1 FROM analysis_caption la WHERE la.image_id = {alias}.id) OR
        EXISTS(SELECT 1 FROM analysis_blip2 b2 WHERE b2.image_id    = {alias}.id) OR
        EXISTS(SELECT 1 FROM analysis_objects ob WHERE ob.image_id   = {alias}.id) OR
        EXISTS(SELECT 1 FROM analysis_ocr ocr WHERE ocr.image_id     = {alias}.id) OR
        EXISTS(SELECT 1 FROM analysis_faces af WHERE af.image_id     = {alias}.id) OR
        EXISTS(SELECT 1 FROM analysis_cloud_ai ca WHERE ca.image_id  = {alias}.id) OR
        EXISTS(SELECT 1 FROM analysis_perception ap WHERE ap.image_id = {alias}.id) OR
        EXISTS(SELECT 1 FROM analysis_aesthetic ae WHERE ae.image_id = {alias}.id) OR
        EXISTS(SELECT 1 FROM embeddings em WHERE em.image_id         = {alias}.id)
    )"""


def _escape_like(value: str) -> str:
    """Escape a value for SQLite LIKE with ESCAPE '\\'."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _encode_gallery_cursor(path: str, image_id: int) -> str:
    payload = json.dumps({"path": path, "id": image_id}, separators=(",", ":")).encode("utf-8")
    token = base64.urlsafe_b64encode(payload).decode("ascii")
    return token.rstrip("=")


def _decode_gallery_cursor(token: str) -> tuple[str, int]:
    padded = token + ("=" * (-len(token) % 4))
    payload = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise ValueError("Invalid gallery cursor payload")
    path = decoded.get("path")
    image_id = decoded.get("id")
    if not isinstance(path, str) or not isinstance(image_id, int):
        raise ValueError("Invalid gallery cursor fields")
    return path, image_id


def _handle_gallery_list_folders(params: dict) -> dict:
    """List processed-image folders for the gallery sidebar tree."""
    conn = _get_db()
    processed = _processed_exists_clause("i")
    rows = conn.execute(
        f"""
        SELECT i.file_path
        FROM images i
        WHERE {processed}
        ORDER BY i.file_path COLLATE NOCASE
        """
    ).fetchall()

    total_images = len(rows)
    folder_counts: dict[str, int] = {}
    parent_of: dict[str, str | None] = {}
    children_by_parent: dict[str | None, set[str]] = {}

    for row in rows:
        norm_path = str(row["file_path"]).replace("\\", "/")
        if "/" not in norm_path:
            continue
        folder = norm_path.rsplit("/", 1)[0].rstrip("/")
        if not folder:
            continue
        parts = [p for p in folder.split("/") if p]
        if not parts:
            continue

        parent: str | None = None
        current = ""
        for part in parts:
            current = part if not current else f"{current}/{part}"
            folder_counts[current] = folder_counts.get(current, 0) + 1
            if current not in parent_of:
                parent_of[current] = parent
            children_by_parent.setdefault(parent, set()).add(current)
            parent = current

    folders = []
    for path, image_count in folder_counts.items():
        folders.append({
            "path": path,
            "name": path.rsplit("/", 1)[-1],
            "parent_path": parent_of.get(path),
            "depth": path.count("/"),
            "image_count": image_count,
            "child_count": len(children_by_parent.get(path, set())),
        })

    folders.sort(key=lambda f: str(f["path"]).lower())
    return {"folders": folders, "totalImages": total_images}


def _handle_gallery_list_images_chunk(params: dict) -> dict:
    """Return a progressive chunk of processed images for gallery browsing."""
    import json as _json

    conn = _get_db()
    path_expr = "REPLACE(i.file_path, '\\', '/')"

    try:
        chunk_size = int(params.get("chunkSize", 300))
    except (TypeError, ValueError):
        raise ValueError("chunkSize must be an integer")
    chunk_size = max(50, min(chunk_size, 2000))

    recursive = bool(params.get("recursive", True))
    folder_path = params.get("folderPath")
    cursor = params.get("cursor")

    conditions: list[str] = [_processed_exists_clause("i")]
    sql_params: list[Any] = []

    if folder_path is not None:
        if not isinstance(folder_path, str):
            raise ValueError("folderPath must be a string")
        folder_norm = folder_path.replace("\\", "/").rstrip("/")
        if folder_norm:
            escaped = _escape_like(folder_norm)
            conditions.append(f"{path_expr} LIKE ? ESCAPE '\\'")
            sql_params.append(f"{escaped}/%")
            if not recursive:
                conditions.append(f"INSTR(SUBSTR({path_expr}, LENGTH(?) + 2), '/') = 0")
                sql_params.append(folder_norm)

    base_conditions = list(conditions)
    base_params = list(sql_params)

    if cursor is not None:
        if not isinstance(cursor, str):
            raise ValueError("cursor must be a string")
        cursor_path, cursor_id = _decode_gallery_cursor(cursor)
        conditions.append(
            f"({path_expr} COLLATE NOCASE > ? COLLATE NOCASE OR "
            f"({path_expr} COLLATE NOCASE = ? COLLATE NOCASE AND i.id > ?))"
        )
        sql_params.extend([cursor_path, cursor_path, cursor_id])

    where_clause = " AND ".join(conditions)
    base_where_clause = " AND ".join(base_conditions)

    select_cols = f"""
        i.id AS image_id, i.file_path, {path_expr} AS normalized_path, i.width, i.height, i.file_size,
        m.camera_make, m.camera_model, m.lens_model, m.focal_length,
        m.f_number, m.exposure_time, m.iso, m.date_time_original,
        m.gps_latitude, m.gps_longitude, m.location_city, m.location_state,
        m.location_country,
        t.sharpness_score, t.sharpness_label, t.exposure_ev, t.exposure_label,
        t.noise_level, t.noise_label, t.snr_db, t.dynamic_range_stops,
        t.highlight_clipping_pct, t.shadow_clipping_pct, t.avg_saturation,
        t.dominant_colors,
        COALESCE(la.description, b2.description) AS description,
        COALESCE(la.scene_type, b2.scene_type) AS scene_type,
        COALESCE(la.main_subject, b2.main_subject) AS main_subject,
        COALESCE(la.lighting, b2.lighting) AS lighting,
        COALESCE(la.mood, b2.mood) AS mood,
        COALESCE(la.keywords, b2.keywords) AS keywords,
        COALESCE(la.detected_objects, ob.detected_objects) AS detected_objects,
        COALESCE(la.face_count, af.face_count) AS face_count,
        COALESCE(la.face_identities, af.face_identities) AS face_identities,
        COALESCE(la.has_people, ob.has_person) AS has_people,
        COALESCE(la.ocr_text, ocr.ocr_text) AS ocr_text,
        (
            SELECT ca.description
            FROM analysis_cloud_ai ca
            WHERE ca.image_id = i.id
              AND ca.description IS NOT NULL
              AND TRIM(ca.description) != ''
            ORDER BY ca.analyzed_at DESC, ca.id DESC
            LIMIT 1
        ) AS cloud_description,
        COALESCE(ap.perception_iaa, ae.aesthetic_score) AS aesthetic_score,
        COALESCE(ap.perception_iaa_label, ae.aesthetic_label) AS aesthetic_label,
        NULL AS aesthetic_reason,
        ap.perception_iaa, ap.perception_iaa_label,
        ap.perception_iqa, ap.perception_iqa_label,
        ap.perception_ista, ap.perception_ista_label
    """

    joins = """
        FROM images i
        LEFT JOIN analysis_metadata    m  ON m.image_id  = i.id
        LEFT JOIN analysis_technical   t  ON t.image_id  = i.id
        LEFT JOIN analysis_caption     la ON la.image_id = i.id
        LEFT JOIN analysis_blip2       b2 ON b2.image_id = i.id
        LEFT JOIN analysis_objects     ob ON ob.image_id = i.id
        LEFT JOIN analysis_ocr        ocr ON ocr.image_id = i.id
        LEFT JOIN analysis_faces       af ON af.image_id = i.id
        LEFT JOIN analysis_aesthetic   ae ON ae.image_id = i.id
        LEFT JOIN analysis_perception  ap ON ap.image_id = i.id
    """

    rows = conn.execute(
        f"""
        SELECT {select_cols}
        {joins}
        WHERE {where_clause}
        ORDER BY {path_expr} COLLATE NOCASE, i.id
        LIMIT ?
        """,
        [*sql_params, chunk_size + 1],
    ).fetchall()

    total = None
    if cursor is None:
        total_row = conn.execute(
            f"""
            SELECT COUNT(*) AS cnt
            FROM images i
            WHERE {base_where_clause}
            """,
            base_params,
        ).fetchone()
        total = int(total_row["cnt"]) if total_row else 0

    has_more = len(rows) > chunk_size
    if has_more:
        rows = rows[:chunk_size]

    def _json_field(val: Any) -> Any:
        if val is None:
            return None
        try:
            return _json.loads(val)
        except (TypeError, ValueError):
            return val

    items = []
    for row in rows:
        items.append({
            "image_id": row["image_id"],
            "file_path": row["file_path"],
            "score": None,
            "width": row["width"],
            "height": row["height"],
            "file_size": row["file_size"],
            "camera_make": row["camera_make"],
            "camera_model": row["camera_model"],
            "lens_model": row["lens_model"],
            "focal_length": row["focal_length"],
            "f_number": row["f_number"],
            "exposure_time": row["exposure_time"],
            "iso": row["iso"],
            "date_time_original": row["date_time_original"],
            "gps_latitude": row["gps_latitude"],
            "gps_longitude": row["gps_longitude"],
            "location_city": row["location_city"],
            "location_state": row["location_state"],
            "location_country": row["location_country"],
            "sharpness_score": row["sharpness_score"],
            "sharpness_label": row["sharpness_label"],
            "exposure_ev": row["exposure_ev"],
            "exposure_label": row["exposure_label"],
            "noise_level": row["noise_level"],
            "noise_label": row["noise_label"],
            "snr_db": row["snr_db"],
            "dynamic_range_stops": row["dynamic_range_stops"],
            "highlight_clipping_pct": row["highlight_clipping_pct"],
            "shadow_clipping_pct": row["shadow_clipping_pct"],
            "avg_saturation": row["avg_saturation"],
            "dominant_colors": _json_field(row["dominant_colors"]),
            "description": row["description"],
            "scene_type": row["scene_type"],
            "main_subject": row["main_subject"],
            "lighting": row["lighting"],
            "mood": row["mood"],
            "keywords": _json_field(row["keywords"]),
            "detected_objects": _json_field(row["detected_objects"]),
            "face_count": row["face_count"],
            "face_identities": _json_field(row["face_identities"]),
            "has_people": bool(row["has_people"]) if row["has_people"] is not None else None,
            "ocr_text": row["ocr_text"],
            "cloud_description": row["cloud_description"],
            "aesthetic_score": row["aesthetic_score"],
            "aesthetic_label": row["aesthetic_label"],
            "aesthetic_reason": row["aesthetic_reason"],
            "perception_iaa": row["perception_iaa"],
            "perception_iaa_label": row["perception_iaa_label"],
            "perception_iqa": row["perception_iqa"],
            "perception_iqa_label": row["perception_iqa_label"],
            "perception_ista": row["perception_ista"],
            "perception_ista_label": row["perception_ista_label"],
            "_normalized_path": row["normalized_path"],
        })

    next_cursor = None
    if has_more and items:
        last = items[-1]
        next_cursor = _encode_gallery_cursor(last["_normalized_path"], int(last["image_id"]))

    for item in items:
        item.pop("_normalized_path", None)

    return {
        "items": items,
        "nextCursor": next_cursor,
        "hasMore": has_more,
        "total": total,
    }


def _handle_analyze(req_id: int | str, params: dict) -> None:
    """Analyze a single image — streaming progress, then result."""
    from imganalyzer.analyzer import AnalysisCancelled, Analyzer

    image_path = params["imagePath"]
    ai_backend = params.get("aiBackend", "local")
    overwrite = params.get("overwrite", True)
    verbose = params.get("verbose", True)

    cancel_event = threading.Event()
    _analyze_cancel[image_path] = cancel_event

    def _run():
        try:
            analyzer = Analyzer(
                ai_backend=ai_backend,
                run_technical=True,
                verbose=verbose,
            )

            def _emit_progress(stage: str) -> None:
                _send_notification("analyze/progress", {
                    "imagePath": image_path,
                    "stage": stage,
                })
            try:
                img_path = Path(image_path)
                result = analyzer.analyze(
                    img_path,
                    cancel_event=cancel_event,
                    progress_cb=_emit_progress,
                )

                if cancel_event.is_set():
                    raise AnalysisCancelled("Analysis cancelled")

                if overwrite:
                    xmp_path = img_path.with_suffix(".xmp")
                    result.write_xmp(xmp_path)
                    _emit_progress("XMP written")
                else:
                    _emit_progress("Done.")

                if cancel_event.is_set():
                    raise AnalysisCancelled("Analysis cancelled")

                _send_result(req_id, {
                    "ok": True,
                    "xmpPath": str(img_path.with_suffix(".xmp")),
                })
            except AnalysisCancelled:
                _send_result(req_id, {
                    "ok": False,
                    "cancelled": True,
                })
            except Exception as exc:
                _send_error(req_id, -1, f"Analysis failed: {exc}")
        except Exception as exc:
            _send_error(req_id, -1, f"Analysis failed: {exc}")
        finally:
            _analyze_cancel.pop(image_path, None)

    t = threading.Thread(target=_run, daemon=True, name=f"rpc-analyze-{Path(image_path).name}")
    t.start()


def _handle_cancel_analyze(params: dict) -> dict:
    """Cancel a running single-image analysis."""
    image_path = params.get("imagePath", "")
    evt = _analyze_cancel.get(image_path)
    if evt:
        evt.set()
        return {"cancelled": True}
    return {"cancelled": False}


def _handle_thumbnail(params: dict) -> dict:
    """Generate a thumbnail JPEG and return as base64.

    This runs in the persistent process, avoiding conda subprocess overhead
    (~1-3s) per thumbnail.
    """
    image_path = params["imagePath"]
    path = Path(image_path)
    ext = path.suffix.lower()

    RAW_EXTS = {
        ".arw", ".cr2", ".cr3", ".nef", ".nrw", ".orf", ".raf", ".rw2",
        ".dng", ".pef", ".srw", ".erf", ".kdc", ".mrw", ".3fr", ".fff",
        ".sr2", ".srf", ".x3f", ".iiq", ".mos", ".raw",
    }

    from PIL import Image
    from imganalyzer.readers.standard import pillow_decode_guard, register_optional_pillow_opener

    if ext in RAW_EXTS:
        import rawpy
        with _suppress_c_stderr():
            raw_ctx = rawpy.imread(str(path))
        with raw_ctx as raw:
            rgb = raw.postprocess(use_camera_wb=True, output_bps=8, half_size=True)
        img = Image.fromarray(rgb)
    else:
        register_optional_pillow_opener(path)
        with pillow_decode_guard(path):
            img = Image.open(path)
            img = img.convert("RGB")

    img.thumbnail((400, 300), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    jpeg_bytes = buf.getvalue()

    return {"data": base64.b64encode(jpeg_bytes).decode("ascii")}


def _handle_fullimage(params: dict) -> dict:
    """Generate a full-res JPEG for RAW/HEIC and return as base64.

    For native browser formats (jpg, png, webp, etc), returns a flag
    indicating the file should be read directly by Electron instead.
    """
    image_path = params["imagePath"]
    path = Path(image_path)
    ext = path.suffix.lower()

    NATIVE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
    RAW_EXTS = {
        ".arw", ".cr2", ".cr3", ".nef", ".nrw", ".orf", ".raf", ".rw2",
        ".dng", ".pef", ".srw", ".erf", ".kdc", ".mrw", ".3fr", ".fff",
        ".sr2", ".srf", ".x3f", ".iiq", ".mos", ".raw",
    }

    if ext in NATIVE_EXTS:
        # Electron can read these directly — no need to decode via Python
        return {"native": True, "path": image_path}

    from PIL import Image
    from imganalyzer.readers.standard import pillow_decode_guard, register_optional_pillow_opener

    if ext in RAW_EXTS:
        import rawpy
        with _suppress_c_stderr():
            raw_ctx = rawpy.imread(str(path))
        with raw_ctx as raw:
            rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
        img = Image.fromarray(rgb)
    else:
        register_optional_pillow_opener(path)
        with pillow_decode_guard(path):
            img = Image.open(path)
            img = img.convert("RGB")

    # Limit to 4K display size
    MAX_DIM = 3840
    if img.width > MAX_DIM or img.height > MAX_DIM:
        img.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    jpeg_bytes = buf.getvalue()

    return {"native": False, "data": base64.b64encode(jpeg_bytes).decode("ascii")}


def _handle_cachedimage(params: dict) -> dict:
    """Return a pre-decoded cached image (1024px) as base64 JPEG for lightbox.

    This is the middle tier of the three-tier lightbox:
    thumbnail (400×300) → cached (1024px) → full-res (3840px background).

    Accepts either ``imageId`` (int) or ``imagePath`` (str).  When only
    ``imagePath`` is provided the image ID is resolved via the DB.
    """
    image_id = params.get("imageId")
    image_path = params.get("imagePath")

    if image_id is None and image_path is not None:
        db = get_db()
        repo = Repository(db)
        row = repo.get_image_by_path(str(image_path))
        if row is None:
            return {"available": False}
        image_id = row["id"]

    if image_id is None:
        raise ValueError("imageId or imagePath is required")
    image_id = int(image_id)

    try:
        store = _get_decoded_store()
        result = store.get(image_id)
    except Exception:
        return {"available": False}

    if result is None:
        return {"available": False}

    pil_image, _meta = result
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=92)
    jpeg_bytes = buf.getvalue()

    return {
        "available": True,
        "data": base64.b64encode(jpeg_bytes).decode("ascii"),
        "width": pil_image.width,
        "height": pil_image.height,
    }


def _handle_decode_status(params: dict) -> dict:
    """Return pre-decode pipeline progress."""
    try:
        decoder = _get_pre_decoder()
        return decoder.progress()
    except Exception:
        return {"done": 0, "failed": 0, "total": 0, "running": False}


# ── Face management handlers ─────────────────────────────────────────────────

def _handle_faces_list(params: dict) -> dict:
    """List all face identities with image counts."""
    from imganalyzer.db.repository import Repository

    conn = _get_db()
    repo = Repository(conn)
    faces = repo.list_face_summary()
    return {"faces": faces}


def _handle_faces_images(params: dict) -> dict:
    """Get images containing a specific face identity."""
    from imganalyzer.db.repository import Repository

    conn = _get_db()
    repo = Repository(conn)
    name = params["name"]
    limit = params.get("limit", 100)
    images = repo.get_images_for_face(name, limit=limit)
    return {"images": images}


def _handle_faces_set_alias(params: dict) -> dict:
    """Set or update the display name (alias) for a face cluster or identity.

    When ``cluster_id`` is provided, stores a per-cluster label.
    Otherwise falls back to updating the ``face_identities`` registry.
    """
    from imganalyzer.db.repository import Repository

    conn = _get_db()
    repo = Repository(conn)
    display_name = params.get("display_name", "").strip()

    cluster_id = params.get("cluster_id")
    if cluster_id is not None:
        repo.set_cluster_label(int(cluster_id), display_name or None)
        return {"ok": True}

    canonical_name = params["canonical_name"]

    # If the identity doesn't exist yet, create it
    existing = repo.get_face_identity(canonical_name)
    if existing is None:
        repo.register_face_identity(canonical_name, display_name or None)
    else:
        repo.rename_face(canonical_name, display_name or None)

    return {"ok": True}


def _handle_faces_clusters(params: dict) -> dict:
    """List face clusters (or identity groups if not clustered yet)."""
    from imganalyzer.db.repository import Repository

    conn = _get_db()
    repo = Repository(conn)
    limit = params.get("limit", 0)
    offset = params.get("offset", 0)
    clusters, total_count = repo.list_face_clusters(limit=limit, offset=offset)
    has_occurrences = repo.get_face_occurrences_count() > 0
    deferred_ids = sorted(repo.get_deferred_cluster_ids())
    return {
        "clusters": clusters,
        "has_occurrences": has_occurrences,
        "total_count": total_count,
        "deferred_cluster_ids": deferred_ids,
    }


def _handle_faces_cluster_relink(params: dict) -> dict:
    """Relink a cluster to a label and optionally a person in one transaction."""
    from imganalyzer.db.repository import Repository

    conn = _get_db()
    repo = Repository(conn)
    display_name = params.get("display_name")
    if isinstance(display_name, str):
        display_name = display_name.strip() or None
    elif display_name is not None:
        raise ValueError("display_name must be a string or null")

    person_id = params.get("person_id")
    if person_id is not None:
        person_id = int(person_id)

    update_person = bool(params.get("update_person", False))
    updated = repo.relink_cluster(
        int(params["cluster_id"]),
        display_name,
        person_id,
        update_person=update_person,
    )
    _invalidate_person_link_suggestion_cache()
    return {"ok": True, "updated": updated}


def _handle_faces_cluster_defer(params: dict) -> dict:
    """Mark a cluster as deferred (parked for later review)."""
    from imganalyzer.db.repository import Repository

    conn = _get_db()
    repo = Repository(conn)
    repo.defer_cluster(int(params["cluster_id"]))
    return {"ok": True}


def _handle_faces_cluster_undefer(params: dict) -> dict:
    """Remove deferred status from a cluster."""
    from imganalyzer.db.repository import Repository

    conn = _get_db()
    repo = Repository(conn)
    repo.undefer_cluster(int(params["cluster_id"]))
    return {"ok": True}


def _handle_faces_cluster_undefer_all(params: dict) -> dict:
    """Remove deferred status from all clusters."""
    from imganalyzer.db.repository import Repository

    conn = _get_db()
    repo = Repository(conn)
    cleared = repo.undefer_all_clusters()
    return {"ok": True, "cleared": cleared}


def _handle_faces_split_cluster(params: dict) -> dict:
    """Split a mixed-identity cluster into sub-clusters."""
    from imganalyzer.db.repository import Repository

    conn = _get_db()
    repo = Repository(conn)
    cluster_id = int(params["cluster_id"])
    threshold = float(params.get("threshold", 0.65))
    result = repo.split_cluster(cluster_id, threshold=threshold)
    _invalidate_person_link_suggestion_cache()
    return result


def _handle_faces_cluster_purity(params: dict) -> dict:
    """Compute purity score for a cluster."""
    from imganalyzer.db.repository import Repository

    conn = _get_db()
    repo = Repository(conn)
    cluster_id = int(params["cluster_id"])
    sample_size = int(params.get("sample_size", 20))
    return repo.compute_cluster_purity(cluster_id, sample_size=sample_size)


def _handle_faces_impure_clusters(params: dict) -> dict:
    """List clusters with purity below threshold."""
    from imganalyzer.db.repository import Repository

    conn = _get_db()
    repo = Repository(conn)
    purity_threshold = float(params.get("purity_threshold", 0.75))
    min_faces = int(params.get("min_faces", 3))
    limit = int(params.get("limit", 50))
    clusters = repo.get_impure_clusters(
        purity_threshold=purity_threshold,
        min_faces=min_faces,
        limit=limit,
    )
    return {"clusters": clusters}


def _handle_faces_cluster_link_suggestions(params: dict) -> dict:
    """Suggest likely person/alias targets for a cluster relink action."""
    from imganalyzer.db.repository import Repository

    conn = _get_db()
    repo = Repository(conn)

    cluster_id = int(params["cluster_id"])
    limit = int(params.get("limit", 12))
    limit = max(1, min(limit, 100))

    suggestions = repo.suggest_cluster_link_targets(
        cluster_id,
        limit=limit,
        include_persons=bool(params.get("include_persons", True)),
        include_aliases=bool(params.get("include_aliases", True)),
    )
    return {"suggestions": suggestions}


# ── Person (cross-age identity grouping) ─────────────────────────────────


def _handle_faces_persons(_params: dict) -> dict:
    """List all persons with stats."""
    from imganalyzer.db.repository import Repository

    conn = _get_db()
    repo = Repository(conn)
    return {"persons": repo.list_persons()}


def _handle_faces_person_create(params: dict) -> dict:
    """Create a new person."""
    from imganalyzer.db.repository import Repository

    conn = _get_db()
    repo = Repository(conn)
    person_id = repo.create_person(params["name"], params.get("notes"))
    return {"id": person_id}


def _handle_faces_person_rename(params: dict) -> dict:
    """Rename a person."""
    from imganalyzer.db.repository import Repository

    conn = _get_db()
    repo = Repository(conn)
    repo.rename_person(int(params["person_id"]), params["name"])
    return {"ok": True}


def _handle_faces_person_delete(params: dict) -> dict:
    """Delete a person (clears links on occurrences)."""
    from imganalyzer.db.repository import Repository

    conn = _get_db()
    repo = Repository(conn)
    repo.delete_person(int(params["person_id"]))
    _invalidate_person_link_suggestion_cache()
    return {"ok": True}


def _handle_faces_person_link(params: dict) -> dict:
    """Link a cluster to a person."""
    from imganalyzer.db.repository import Repository

    conn = _get_db()
    repo = Repository(conn)
    updated = repo.link_cluster_to_person(
        int(params["cluster_id"]), int(params["person_id"])
    )
    _invalidate_person_link_suggestion_cache()
    return {"ok": True, "updated": updated}


def _handle_faces_person_unlink(params: dict) -> dict:
    """Unlink a cluster from its person."""
    from imganalyzer.db.repository import Repository

    conn = _get_db()
    repo = Repository(conn)
    updated = repo.unlink_cluster_from_person(int(params["cluster_id"]))
    _invalidate_person_link_suggestion_cache()
    return {"ok": True, "updated": updated}


def _handle_faces_person_clusters(params: dict) -> dict:
    """Get clusters belonging to a person."""
    from imganalyzer.db.repository import Repository

    conn = _get_db()
    repo = Repository(conn)
    clusters = repo.get_person_clusters(int(params["person_id"]))
    return {"clusters": clusters}


def _handle_faces_person_link_suggestions(params: dict) -> dict:
    """Suggest likely unlinked clusters for a person."""
    from imganalyzer.db.repository import Repository

    person_id = int(params["person_id"])
    limit = int(params.get("limit", 12))
    limit = max(1, min(limit, 100))

    cache_key = (person_id, limit)
    cached = _person_link_suggestion_cache.get(cache_key)
    now = time.time()
    if cached is not None:
        cached_at, cached_payload = cached
        if now - cached_at <= _PERSON_LINK_SUGGESTION_CACHE_TTL_SECONDS:
            return {"suggestions": cached_payload}

    conn = _get_db()
    repo = Repository(conn)
    suggestions = repo.suggest_person_link_clusters(person_id, limit=limit)
    _person_link_suggestion_cache[cache_key] = (now, suggestions)
    return {"suggestions": suggestions}


def _handle_faces_cluster_images(params: dict) -> dict:
    """Get face occurrences for a specific cluster or identity."""
    from imganalyzer.db.repository import Repository

    conn = _get_db()
    repo = Repository(conn)
    cluster_id = params.get("cluster_id")
    identity_name = params.get("identity_name")
    limit = params.get("limit", 50)
    occurrences = repo.get_cluster_occurrences(
        cluster_id=cluster_id,
        identity_name=identity_name,
        limit=limit,
    )
    return {"occurrences": occurrences}


def _handle_faces_crop(params: dict) -> dict:
    """Crop a face from a source image using stored bounding box coordinates.

    Returns a base64-encoded JPEG of the face crop.
    """
    from imganalyzer.db.repository import Repository

    conn = _get_db()
    repo = Repository(conn)
    occurrence_id = params["occurrence_id"]
    occ = repo.get_face_occurrence(occurrence_id)
    if occ is None:
        return {"error": "Occurrence not found"}

    # Serve pre-generated thumbnail if available (1ms DB read vs 50-2000ms re-crop)
    thumbnail = occ.get("thumbnail")
    if thumbnail is not None:
        return {"data": base64.b64encode(thumbnail).decode("ascii")}

    thumbnail = _generate_face_occurrence_thumbnail(occ)
    repo.set_face_occurrence_thumbnail(occurrence_id, thumbnail)
    conn.commit()
    return {"data": base64.b64encode(thumbnail).decode("ascii")}


_RAW_FACE_CROP_EXTS = {
    ".arw", ".cr2", ".cr3", ".nef", ".nrw", ".orf", ".raf", ".rw2",
    ".dng", ".pef", ".srw", ".erf", ".kdc", ".mrw", ".3fr", ".fff",
    ".sr2", ".srf", ".x3f", ".iiq", ".mos", ".raw",
}


def _open_face_crop_image(path: Path):
    from PIL import Image
    from imganalyzer.readers.standard import pillow_decode_guard, register_optional_pillow_opener
    from imganalyzer.analysis.ai.faces import _get_pil_exif_orientation

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    ext = path.suffix.lower()
    register_optional_pillow_opener(path)
    if ext in _RAW_FACE_CROP_EXTS:
        import rawpy

        with _suppress_c_stderr():
            raw_ctx = rawpy.imread(str(path))
        with raw_ctx as raw:
            rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
        return Image.fromarray(rgb), 1  # RAW files have no EXIF orientation

    with pillow_decode_guard(path):
        img = Image.open(path)
        orientation = _get_pil_exif_orientation(img)
        # Belt-and-suspenders: pillow-heif auto-rotates HEIC/HEIF/AVIF;
        # if _get_pil_exif_orientation didn't catch it via format, check ext.
        if ext in (".heic", ".heif", ".avif"):
            orientation = 1
        return img.convert("RGB"), orientation


def _render_face_occurrence_thumbnail(occ: dict[str, Any], img, exif_orientation: int = 1) -> bytes:
    from PIL import Image
    from imganalyzer.analysis.ai.faces import _apply_orientation
    from imganalyzer.pipeline.modules import _AI_MAX_LONG_EDGE

    # Crop face region with some padding. bbox coordinates were computed on a
    # pre-resized image (max _AI_MAX_LONG_EDGE on the long edge) — scale them
    # to the original resolution.
    w, h = img.size
    orig_long_edge = max(w, h)
    if orig_long_edge <= _AI_MAX_LONG_EDGE:
        # Image wasn't resized during detection — bbox coords are at original res
        scale = 1.0
    else:
        # Determine detection resolution by checking if the stored bbox fits
        # within the current _AI_MAX_LONG_EDGE detection frame (accounting for
        # aspect ratio).  Faces analyzed at the legacy 1920px long edge may
        # have bbox coords that exceed the current 1024px detection dimensions.
        det_ratio = _AI_MAX_LONG_EDGE / orig_long_edge
        det_w = w * det_ratio
        det_h = h * det_ratio
        if occ["bbox_x2"] <= det_w and occ["bbox_y2"] <= det_h:
            scale = orig_long_edge / _AI_MAX_LONG_EDGE
        else:
            # bbox exceeds current detection frame — legacy 1920 detection
            scale = orig_long_edge / 1920

    x1 = max(0, int(occ["bbox_x1"] * scale))
    y1 = max(0, int(occ["bbox_y1"] * scale))
    x2 = min(w, int(occ["bbox_x2"] * scale))
    y2 = min(h, int(occ["bbox_y2"] * scale))

    # Add 20% padding around the face
    fw = x2 - x1
    fh = y2 - y1
    pad_x = int(fw * 0.2)
    pad_y = int(fh * 0.2)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    crop = img.crop((x1, y1, x2, y2))

    # Apply EXIF orientation so the face appears upright
    if exif_orientation != 1:
        crop = _apply_orientation(crop, exif_orientation)

    # Resize to max 200px on the longest side
    max_dim = 200
    cw, ch = crop.size
    if cw > max_dim or ch > max_dim:
        scale = max_dim / max(cw, ch)
        crop = crop.resize(
            (int(cw * scale), int(ch * scale)), Image.LANCZOS
        )

    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _generate_face_occurrence_thumbnail(
    occ: dict[str, Any], img=None, exif_orientation: int = 1,
) -> bytes:
    file_path = occ["file_path"]
    path = Path(file_path)
    if img is not None:
        return _render_face_occurrence_thumbnail(occ, img, exif_orientation)

    source, orientation = _open_face_crop_image(path)
    try:
        return _render_face_occurrence_thumbnail(occ, source, orientation)
    finally:
        close = getattr(source, "close", None)
        if callable(close):
            close()


def _handle_faces_run_clustering(req_id: int | str, params: dict) -> None:
    """Run face clustering in a background thread to avoid blocking the RPC loop."""
    import sqlite3
    from imganalyzer.db.connection import get_db_path
    from imganalyzer.db.repository import Repository

    threshold = params.get("threshold", 0.55)

    _send_result(req_id, {"started": True})

    def _run_clustering() -> None:
        try:
            db_path = get_db_path()
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.row_factory = sqlite3.Row
            repo = Repository(conn)
            num_clusters = repo.cluster_faces(threshold=threshold)
            conn.commit()
            conn.close()
            _send_notification("faces/clustering-done", {"num_clusters": num_clusters})
        except Exception as exc:
            _send_notification("faces/clustering-done", {"error": str(exc)})

    t = threading.Thread(target=_run_clustering, daemon=True)
    t.start()


def _handle_faces_crop_batch(params: dict) -> dict:
    """Return thumbnails for multiple face occurrences in one round-trip.

    Accepts ``{"ids": [1, 2, 3, ...]}`` and returns
    ``{"thumbnails": {"1": "<base64>", "2": "<base64>", ...}}``.
    """
    from imganalyzer.db.repository import Repository

    conn = _get_db()
    repo = Repository(conn)
    ids = params.get("ids", [])
    if not ids:
        return {"thumbnails": {}}

    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"""SELECT fo.id, fo.thumbnail, fo.bbox_x1, fo.bbox_y1, fo.bbox_x2, fo.bbox_y2,
                   i.file_path
            FROM face_occurrences fo
            JOIN images i ON i.id = fo.image_id
            WHERE fo.id IN ({placeholders})""",
        ids,
    ).fetchall()

    thumbnails: dict[str, str] = {}
    missing_rows_by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row_d = dict(row)
        oid = str(row_d["id"])
        if row_d.get("thumbnail") is not None:
            thumbnails[oid] = base64.b64encode(row_d["thumbnail"]).decode("ascii")
        else:
            missing_rows_by_path[row_d["file_path"]].append(row_d)

    updated = False
    for file_path, group in missing_rows_by_path.items():
        try:
            img, orientation = _open_face_crop_image(Path(file_path))
        except Exception:
            sys.stderr.write(f"[faces/crop-batch] cannot open {file_path}\n")
            continue
        try:
            for row_d in group:
                try:
                    thumbnail = _generate_face_occurrence_thumbnail(
                        row_d, img=img, exif_orientation=orientation,
                    )
                    repo.set_face_occurrence_thumbnail(row_d["id"], thumbnail)
                    thumbnails[str(row_d["id"])] = base64.b64encode(thumbnail).decode("ascii")
                    updated = True
                except Exception:
                    sys.stderr.write(
                        f"[faces/crop-batch] crop failed for occ {row_d['id']}\n"
                    )
        finally:
            close = getattr(img, "close", None)
            if callable(close):
                close()

    if updated:
        conn.commit()

    return {"thumbnails": thumbnails}


# ── Image details (single-image metadata lookup) ────────────────────────────


def _handle_image_details(params: dict) -> dict:
    """Return full analysis metadata for a single image by image_id or file_path.

    Reuses the same column set as the search handler so the result can
    be rendered by the same AnalysisSidebar component on the frontend.
    """
    import json as _json

    conn = _get_db()
    image_id = params.get("image_id")
    file_path = params.get("file_path")
    if image_id is None and not file_path:
        return {"result": None, "error": "Must provide image_id or file_path"}

    select_cols = """
        i.id AS image_id, i.file_path, i.width, i.height, i.file_size,
        m.camera_make, m.camera_model, m.lens_model, m.focal_length,
        m.f_number, m.exposure_time, m.iso, m.date_time_original,
        m.gps_latitude, m.gps_longitude, m.location_city, m.location_state,
        m.location_country,
        t.sharpness_score, t.sharpness_label, t.exposure_ev, t.exposure_label,
        t.noise_level, t.noise_label, t.snr_db, t.dynamic_range_stops,
        t.highlight_clipping_pct, t.shadow_clipping_pct, t.avg_saturation,
        t.dominant_colors,
        COALESCE(la.description, b2.description) AS description,
        COALESCE(la.scene_type, b2.scene_type) AS scene_type,
        COALESCE(la.main_subject, b2.main_subject) AS main_subject,
        COALESCE(la.lighting, b2.lighting) AS lighting,
        COALESCE(la.mood, b2.mood) AS mood,
        COALESCE(la.keywords, b2.keywords) AS keywords,
        COALESCE(la.detected_objects, ob.detected_objects) AS detected_objects,
        COALESCE(la.face_count, af.face_count) AS face_count,
        COALESCE(la.face_identities, af.face_identities) AS face_identities,
        COALESCE(la.has_people, ob.has_person) AS has_people,
        COALESCE(la.ocr_text, ocr.ocr_text) AS ocr_text,
        (
            SELECT ca.description
            FROM analysis_cloud_ai ca
            WHERE ca.image_id = i.id
              AND ca.description IS NOT NULL
              AND TRIM(ca.description) != ''
            ORDER BY ca.analyzed_at DESC, ca.id DESC
            LIMIT 1
        ) AS cloud_description,
        COALESCE(ap.perception_iaa, ae.aesthetic_score) AS aesthetic_score,
        COALESCE(ap.perception_iaa_label, ae.aesthetic_label) AS aesthetic_label,
        NULL AS aesthetic_reason,
        ap.perception_iaa, ap.perception_iaa_label,
        ap.perception_iqa, ap.perception_iqa_label,
        ap.perception_ista, ap.perception_ista_label
    """
    joins = """
        FROM images i
        LEFT JOIN analysis_metadata    m  ON m.image_id  = i.id
        LEFT JOIN analysis_technical   t  ON t.image_id  = i.id
        LEFT JOIN analysis_caption     la ON la.image_id = i.id
        LEFT JOIN analysis_blip2       b2 ON b2.image_id = i.id
        LEFT JOIN analysis_objects     ob ON ob.image_id = i.id
        LEFT JOIN analysis_ocr        ocr ON ocr.image_id = i.id
        LEFT JOIN analysis_faces       af ON af.image_id = i.id
        LEFT JOIN analysis_aesthetic   ae ON ae.image_id = i.id
        LEFT JOIN analysis_perception  ap ON ap.image_id = i.id
    """

    if image_id is not None:
        where = "WHERE i.id = ?"
        sql_params: list[Any] = [int(image_id)]
    else:
        where = "WHERE i.file_path = ?"
        sql_params = [str(file_path)]

    row = conn.execute(
        f"SELECT {select_cols} {joins} {where} LIMIT 1",
        sql_params,
    ).fetchone()

    if row is None:
        return {"result": None}

    def _jf(val: Any) -> Any:
        if val is None:
            return None
        if isinstance(val, str):
            try:
                return _json.loads(val)
            except (ValueError, TypeError):
                return [val]
        return val

    record = {
        "image_id": int(row["image_id"]),
        "file_path": row["file_path"],
        "score": None,
        "width": row["width"],
        "height": row["height"],
        "file_size": row["file_size"],
        "camera_make": row["camera_make"],
        "camera_model": row["camera_model"],
        "lens_model": row["lens_model"],
        "focal_length": row["focal_length"],
        "f_number": row["f_number"],
        "exposure_time": row["exposure_time"],
        "iso": row["iso"],
        "date_time_original": row["date_time_original"],
        "gps_latitude": row["gps_latitude"],
        "gps_longitude": row["gps_longitude"],
        "location_city": row["location_city"],
        "location_state": row["location_state"],
        "location_country": row["location_country"],
        "sharpness_score": row["sharpness_score"],
        "sharpness_label": row["sharpness_label"],
        "exposure_ev": row["exposure_ev"],
        "exposure_label": row["exposure_label"],
        "noise_level": row["noise_level"],
        "noise_label": row["noise_label"],
        "snr_db": row["snr_db"],
        "dynamic_range_stops": row["dynamic_range_stops"],
        "highlight_clipping_pct": row["highlight_clipping_pct"],
        "shadow_clipping_pct": row["shadow_clipping_pct"],
        "avg_saturation": row["avg_saturation"],
        "dominant_colors": _jf(row["dominant_colors"]),
        "description": row["description"],
        "scene_type": row["scene_type"],
        "main_subject": row["main_subject"],
        "lighting": row["lighting"],
        "mood": row["mood"],
        "keywords": _jf(row["keywords"]),
        "detected_objects": _jf(row["detected_objects"]),
        "face_count": row["face_count"],
        "face_identities": _jf(row["face_identities"]),
        "has_people": bool(row["has_people"]) if row["has_people"] is not None else None,
        "ocr_text": row["ocr_text"],
        "cloud_description": row["cloud_description"],
        "aesthetic_score": row["aesthetic_score"],
        "aesthetic_label": row["aesthetic_label"],
        "aesthetic_reason": row["aesthetic_reason"],
        "perception_iaa": row["perception_iaa"],
        "perception_iaa_label": row["perception_iaa_label"],
        "perception_iqa": row["perception_iqa"],
        "perception_iqa_label": row["perception_iqa_label"],
        "perception_ista": row["perception_ista"],
        "perception_ista_label": row["perception_ista_label"],
    }
    return {"result": record}


# ── Method dispatch ──────────────────────────────────────────────────────────

# Methods that return a result synchronously (the response is sent
# from the main loop after the handler returns).
_SYNC_METHODS: dict[str, Any] = {
    "status": _handle_status,
    "queue_clear": _handle_queue_clear,
    "workers/register": _handle_workers_register,
    "workers/heartbeat": _handle_workers_heartbeat,
    "workers/list": _handle_workers_list,
    "workers/pause": _handle_workers_pause,
    "workers/resume": _handle_workers_resume,
    "workers/remove": _handle_workers_remove,
    "jobs/claim": _handle_jobs_claim,
    "jobs/release-expired": _handle_jobs_release_expired,
    "jobs/heartbeat": _handle_jobs_heartbeat,
    "jobs/release": _handle_jobs_release,
    "jobs/complete": _handle_jobs_complete,
    "jobs/fail": _handle_jobs_fail,
    "jobs/skip": _handle_jobs_skip,
    "jobs/release-worker": _handle_jobs_release_worker,
    "rebuild": _handle_rebuild,
    "search": _handle_search,
    "search/resolveFaceQuery": _handle_search_resolve_face_query,
    "image/details": _handle_image_details,
    "gallery/listFolders": _handle_gallery_list_folders,
    "gallery/listImagesChunk": _handle_gallery_list_images_chunk,
    "thumbnail": _handle_thumbnail,
    "fullimage": _handle_fullimage,
    "cachedimage": _handle_cachedimage,
    "decode/status": _handle_decode_status,
    "cancel_run": _handle_cancel_run,
    "cancel_analyze": _handle_cancel_analyze,
    "faces/list": _handle_faces_list,
    "faces/images": _handle_faces_images,
    "faces/set-alias": _handle_faces_set_alias,
    "faces/clusters": _handle_faces_clusters,
    "faces/cluster-relink": _handle_faces_cluster_relink,
    "faces/cluster-defer": _handle_faces_cluster_defer,
    "faces/cluster-undefer": _handle_faces_cluster_undefer,
    "faces/cluster-undefer-all": _handle_faces_cluster_undefer_all,
    "faces/split-cluster": _handle_faces_split_cluster,
    "faces/cluster-purity": _handle_faces_cluster_purity,
    "faces/impure-clusters": _handle_faces_impure_clusters,
    "faces/cluster-link-suggestions": _handle_faces_cluster_link_suggestions,
    "faces/cluster-images": _handle_faces_cluster_images,
    "faces/crop": _handle_faces_crop,
    "faces/crop-batch": _handle_faces_crop_batch,
    "faces/persons": _handle_faces_persons,
    "faces/person-create": _handle_faces_person_create,
    "faces/person-rename": _handle_faces_person_rename,
    "faces/person-delete": _handle_faces_person_delete,
    "faces/person-link-cluster": _handle_faces_person_link,
    "faces/person-unlink-cluster": _handle_faces_person_unlink,
    "faces/person-clusters": _handle_faces_person_clusters,
    "faces/person-link-suggestions": _handle_faces_person_link_suggestions,
}

# Methods that send their own result/error asynchronously (streaming).
# They receive (req_id, params) and are responsible for calling
# _send_result or _send_error themselves.
_ASYNC_METHODS: dict[str, Any] = {
    "ingest": _handle_ingest,
    "run": _handle_run,
    "analyze": _handle_analyze,
    "faces/run-clustering": _handle_faces_run_clustering,
}


def _graceful_shutdown() -> None:
    """Signal running workers to stop and wait briefly for shutdown."""
    _run_cancel.set()
    if _active_worker is not None:
        _active_worker._shutdown.set()
    if _run_thread is not None and _run_thread.is_alive():
        _run_thread.join(timeout=5)


def _dispatch_http_request(msg: Any) -> tuple[dict[str, Any], bool]:
    """Dispatch a JSON-RPC request for HTTP transport.

    Returns ``(response, should_shutdown_server)``.
    """
    req_id = msg.get("id") if isinstance(msg, dict) else None
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32600, "message": "Invalid request"},
        }, False

    method = msg.get("method", "")
    params = msg.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32602, "message": "Invalid params"},
        }, False

    if method == "shutdown":
        _graceful_shutdown()
        return {"jsonrpc": "2.0", "id": req_id, "result": {"ok": True}}, True

    if method in _SYNC_METHODS:
        try:
            result = _SYNC_METHODS[method](params)
            return {"jsonrpc": "2.0", "id": req_id, "result": result}, False
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -1, "message": str(exc)},
            }, False

    if method in _ASYNC_METHODS:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": -32601,
                "message": f"Method not available over HTTP transport: {method}",
            },
        }, False

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }, False


def _serve_http_jsonrpc(
    host: str,
    port: int,
    auth_token: str | None = None,
    rpc_path: str = "/jsonrpc",
) -> None:
    """Run the server in HTTP JSON-RPC mode for remote workers."""
    global _http_transport
    _http_transport = True

    _ensure_runtime_state_reconciled(
        context="http-startup",
        recover_master_jobs=_master_worker_runtime_status() != "online",
    )

    normalized_path = rpc_path.strip() or "/jsonrpc"
    if not normalized_path.startswith("/"):
        normalized_path = f"/{normalized_path}"

    class _JsonRpcHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _write_json(self, status_code: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload, default=str, separators=(",", ":")).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _check_auth(self) -> bool:
            if not auth_token:
                return True
            auth_header = self.headers.get("Authorization", "")
            expected = f"Bearer {auth_token}"
            if auth_header == expected:
                return True
            self.send_response(401)
            self.send_header("WWW-Authenticate", "Bearer")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return False

        def do_POST(self) -> None:
            req_path = self.path.split("?", 1)[0].rstrip("/") or "/"
            expected_path = normalized_path.rstrip("/") or "/"
            if req_path != expected_path:
                self.send_error(404, "Not Found")
                return
            if not self._check_auth():
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                content_length = 0
            body = self.rfile.read(max(0, content_length))
            try:
                message = json.loads(body.decode("utf-8"))
            except Exception as exc:
                self._write_json(
                    200,
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": f"Parse error: {exc}"},
                    },
                )
                return

            response, should_shutdown = _dispatch_http_request(message)
            self._write_json(200, response)
            if should_shutdown:
                threading.Thread(target=self.server.shutdown, daemon=True).start()

        def do_GET(self) -> None:
            if self.path in ("/health", "/healthz"):
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            # Decoded image serving: GET /images/decoded/{image_id}
            if self.path.startswith("/images/decoded/"):
                if not self._check_auth():
                    return
                self._serve_decoded_image()
                return

            self.send_error(405, "Method Not Allowed")

        def _serve_decoded_image(self) -> None:
            """Serve a pre-decoded image from the DecodedImageStore."""
            parts = self.path.split("/")
            # /images/decoded/{image_id} or /images/decoded/{image_id}/meta
            if len(parts) < 4:
                self.send_error(400, "Bad Request")
                return

            try:
                image_id = int(parts[3].split("?")[0])
            except (ValueError, IndexError):
                self.send_error(400, "Invalid image ID")
                return

            is_meta = len(parts) >= 5 and parts[4] == "meta"

            try:
                store = _get_decoded_store()
            except Exception:
                self.send_error(503, "Decoded image store not available")
                return

            if is_meta:
                meta = store.get_metadata(image_id)
                if meta is None:
                    self.send_error(404, "Not Found")
                    return
                from imganalyzer.cache.decoded_store import _encode_binary_fields
                body = json.dumps(
                    _encode_binary_fields(meta),
                    separators=(",", ":"),
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            img_bytes = store.get_image_bytes(image_id)
            if img_bytes is None:
                self.send_error(404, "Not Found")
                return

            meta = store.get_metadata(image_id) or {}
            content_type = {
                "webp": "image/webp",
                "jpeg": "image/jpeg",
                "png": "image/png",
            }.get(store._fmt, "application/octet-stream")

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(img_bytes)))
            self.send_header("X-Image-Width", str(meta.get("width", "")))
            self.send_header("X-Image-Height", str(meta.get("height", "")))
            self.send_header("X-Original-Format", str(meta.get("format", "")))
            self.send_header("X-Is-Raw", str(meta.get("is_raw", False)).lower())
            self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            self.wfile.write(img_bytes)

        def log_message(self, fmt: str, *args: Any) -> None:
            # Keep logs on stderr, stdout is reserved in stdio mode.
            sys.stderr.write(f"[server.http] {self.address_string()} - {fmt % args}\n")

    class _WatchdogHTTPServer(ThreadingHTTPServer):
        """ThreadingHTTPServer with handler error logging and stale-thread cleanup."""

        _cleanup_interval = 120  # seconds between thread-list cleanups
        _last_cleanup = 0.0

        def handle_error(self, request: Any, client_address: Any) -> None:
            sys.stderr.write(
                f"[server.http] handler error for {client_address}: "
                f"{traceback.format_exc()}\n"
            )

        def service_actions(self) -> None:
            now = time.monotonic()
            if now - self._last_cleanup < self._cleanup_interval:
                return
            self._last_cleanup = now
            threads = self._threads
            if not isinstance(threads, list) or len(threads) <= 50:
                return
            alive = [t for t in threads if t.is_alive()]
            self._threads = alive
            pruned = len(threads) - len(alive)
            if pruned:
                sys.stderr.write(
                    f"[server.http] pruned {pruned} finished handler threads "
                    f"({len(alive)} still alive)\n"
                )

    server = _WatchdogHTTPServer((host, port), _JsonRpcHandler)

    # Eagerly initialise the decoded image store so the cache gate is
    # active from the very first jobs/claim request.  Without this,
    # workers could receive jobs before any status poll triggers lazy init
    # and then fail because hasDecodedCache is not set.
    try:
        _get_decoded_store()
    except Exception as exc:
        sys.stderr.write(
            f"[server.http] decoded store init failed (will retry lazily): {exc}\n"
        )

    sys.stderr.write(
        f"[server.http] listening on http://{host}:{port}{normalized_path} "
        f"(auth={'on' if auth_token else 'off'})\n"
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        sys.stderr.write("[server.http] stopped\n")


def _dispatch(msg: dict) -> None:
    """Dispatch a JSON-RPC request."""
    req_id = msg.get("id")
    method = msg.get("method", "")
    params = msg.get("params", {})

    if method == "shutdown":
        _graceful_shutdown()
        _send_result(req_id, {"ok": True})
        sys.exit(0)

    if method in _SYNC_METHODS:
        try:
            result = _SYNC_METHODS[method](params)
            _send_result(req_id, result)
        except Exception as exc:
            _send_error(req_id, -1, str(exc))
    elif method in _ASYNC_METHODS:
        # Async methods run in their own thread and send the response themselves.
        # Run in a thread to keep the main loop responsive.
        def _run_async():
            try:
                _ASYNC_METHODS[method](req_id, params)
            except Exception as exc:
                _send_error(req_id, -1, str(exc))

        t = threading.Thread(target=_run_async, daemon=True, name=f"rpc-{method}")
        t.start()
    else:
        _send_error(req_id, -32601, f"Method not found: {method}")


# ── Main loop ────────────────────────────────────────────────────────────────

def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
        return value if value > 0 else default
    except ValueError:
        return default


def main() -> None:
    """Read JSON-RPC requests from stdin (default) or serve HTTP JSON-RPC."""
    # Load dotenv early so all handlers have access to env vars
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--transport",
        choices=("stdio", "http"),
        default=os.getenv("IMGANALYZER_SERVER_TRANSPORT", "stdio"),
    )
    parser.add_argument(
        "--host",
        default=os.getenv("IMGANALYZER_COORDINATOR_HOST", "127.0.0.1"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_env_int("IMGANALYZER_COORDINATOR_PORT", 8765),
    )
    parser.add_argument(
        "--auth-token",
        default=os.getenv("IMGANALYZER_COORDINATOR_TOKEN", ""),
    )
    parser.add_argument(
        "--rpc-path",
        default=os.getenv("IMGANALYZER_COORDINATOR_RPC_PATH", "/jsonrpc"),
    )
    parser.add_argument(
        "--decoded-cache-dir",
        default=os.getenv("IMGANALYZER_DECODED_CACHE_DIR", ""),
        help="Directory for the decoded image cache (default: ~/.cache/imganalyzer/decoded)",
    )
    parser.add_argument(
        "--decoded-cache-max-gb",
        type=float,
        default=float(os.getenv("IMGANALYZER_DECODED_CACHE_MAX_GB", "300")),
        help="Maximum decoded cache size in GB (default: 300)",
    )
    parser.add_argument(
        "--decoded-cache-resolution",
        type=int,
        default=int(os.getenv("IMGANALYZER_DECODED_CACHE_RESOLUTION", "1024")),
        help="Max long-edge resolution for cached images (default: 1024)",
    )
    parser.add_argument(
        "--decoded-cache-format",
        choices=("webp", "jpeg", "png"),
        default=os.getenv("IMGANALYZER_DECODED_CACHE_FORMAT", "webp"),
        help="Image format for decoded cache (default: webp)",
    )
    parser.add_argument(
        "--decoded-cache-quality",
        type=int,
        default=int(os.getenv("IMGANALYZER_DECODED_CACHE_QUALITY", "95")),
        help="Compression quality 1-100 for decoded cache (default: 95)",
    )
    parser.add_argument(
        "--pre-decode-workers",
        type=int,
        default=_env_int("IMGANALYZER_PRE_DECODE_WORKERS", 0),
        help="Number of pre-decode threads (default: cpu_count())",
    )
    args, _unknown = parser.parse_known_args(sys.argv[1:])

    # Propagate CLI args to env vars so lazy-init functions pick them up.
    if args.decoded_cache_dir:
        os.environ["IMGANALYZER_DECODED_CACHE_DIR"] = args.decoded_cache_dir
    if args.decoded_cache_max_gb != 300:
        os.environ["IMGANALYZER_DECODED_CACHE_MAX_GB"] = str(args.decoded_cache_max_gb)
    if args.decoded_cache_resolution != 1024:
        os.environ["IMGANALYZER_DECODED_CACHE_RESOLUTION"] = str(args.decoded_cache_resolution)
    if args.decoded_cache_format != "webp":
        os.environ["IMGANALYZER_DECODED_CACHE_FORMAT"] = args.decoded_cache_format
    if args.decoded_cache_quality != 95:
        os.environ["IMGANALYZER_DECODED_CACHE_QUALITY"] = str(args.decoded_cache_quality)
    if args.pre_decode_workers:
        os.environ["IMGANALYZER_PRE_DECODE_WORKERS"] = str(args.pre_decode_workers)

    _ensure_runtime_state_reconciled(
        context="startup",
        recover_master_jobs=_master_worker_runtime_status() != "online",
    )

    if args.transport == "http":
        host_value = str(args.host).strip()
        host_norm = host_value.lower()
        token_value = str(args.auth_token).strip()
        is_loopback = host_norm in {"127.0.0.1", "localhost", "::1"}
        if not is_loopback and not token_value:
            sys.stderr.write(
                "[server.http] --auth-token is required for non-loopback --host values\n"
            )
            sys.exit(2)
        _serve_http_jsonrpc(
            host=host_value,
            port=max(1, args.port),
            auth_token=token_value or None,
            rpc_path=args.rpc_path,
        )
        return

    # Signal readiness
    _send_notification("server/ready", {"pid": os.getpid(), "version": "1.0"})

    # Read line-delimited JSON from stdin
    stdin = sys.stdin
    # stdin was redirected to stderr above; we need the real stdin
    stdin = io.TextIOWrapper(sys.__stdin__.buffer, encoding="utf-8")

    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            _send_error(None, -32700, f"Parse error: {exc}")
            continue

        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
            _send_error(msg.get("id") if isinstance(msg, dict) else None, -32600, "Invalid request")
            continue

        _dispatch(msg)

    # stdin closed — Electron process died or pipe broken
    sys.exit(0)


if __name__ == "__main__":
    main()
