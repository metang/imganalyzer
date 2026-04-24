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
import hmac
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import inspect
import json
import os
import re
import contextlib
import sqlite3
import sys
import threading
import time
import traceback
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from imganalyzer.readers.raw import _suppress_c_stderr

warnings.filterwarnings(
    "ignore",
    message=r"urllib3 \([^)]+\) or chardet \([^)]+\)/charset_normalizer \([^)]+\) doesn't match a supported version!",
)

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
    from imganalyzer.db.connection import create_connection, get_db_path
    from imganalyzer.db.schema import ensure_schema

    db_path = get_db_path()
    db_key = str(db_path.resolve())
    if db_key not in _schema_ready_paths:
        with _schema_init_lock:
            if db_key not in _schema_ready_paths:
                bootstrap = create_connection(
                    busy_timeout_ms=_DB_BUSY_TIMEOUT_MS,
                )
                ensure_schema(bootstrap)
                bootstrap.close()
                _schema_ready_paths.add(db_key)
    conn = create_connection(busy_timeout_ms=_DB_BUSY_TIMEOUT_MS)
    return conn


def _get_db() -> sqlite3.Connection:
    """Return a thread-local SQLite connection for the current server thread."""
    conn = getattr(_db_local, "conn", None)
    if conn is None:
        conn = _open_server_db()
        _db_local.conn = conn
    return conn


# Cached SearchEngine — avoids rebuilding embedding matrices on every RPC call.
#
# IMPORTANT: cache is keyed by DB path, NOT by ``sqlite3.Connection`` identity.
# The JSON-RPC server uses thread-local connections via ``_get_db()``; keying
# on the connection object would defeat the cache whenever a different thread
# handled the request, forcing a ~1.5 GB embedding-matrix rebuild each time.
# See perf_plan.md item B1.
#
# ``engine.conn`` is swapped to the caller's current thread-local connection
# on every acquisition, guarded by ``_search_engine_lock`` so concurrent RPC
# threads don't race on the shared engine's mutable ``conn`` field.  Callers
# obtained via ``_get_search_engine(conn)`` MUST call
# ``_release_search_engine()`` in a ``finally`` block (or use the
# ``_search_engine_ctx`` context manager) when done.
_cached_search_engine: Any = None
_cached_search_engine_key: tuple[str, int] | None = None
_search_engine_lock = threading.RLock()


def _get_search_engine(conn: sqlite3.Connection) -> Any:
    """Return the process-wide cached SearchEngine, bound to *conn*.

    Acquires ``_search_engine_lock`` (caller must call
    ``_release_search_engine()`` when done) and swaps the engine's
    ``conn``/``repo.conn`` to the caller's thread-local connection so
    SQLite calls stay on the owning thread while the embedding-matrix
    caches on the engine persist across RPC calls and threads.
    """
    global _cached_search_engine, _cached_search_engine_key
    from imganalyzer.db import search as _search_module
    from imganalyzer.db.connection import get_db_path

    # Include the SearchEngine class ``id`` in the cache key so that tests
    # which monkeypatch ``imganalyzer.db.search.SearchEngine`` with a fake
    # get a fresh instance instead of the previously-cached real engine.
    db_key: tuple[str, int] = (str(get_db_path()), id(_search_module.SearchEngine))
    _search_engine_lock.acquire()
    try:
        if (
            _cached_search_engine is None
            or _cached_search_engine_key != db_key
        ):
            _cached_search_engine = _search_module.SearchEngine(conn)
            _cached_search_engine_key = db_key
        _cached_search_engine.conn = conn
        repo = getattr(_cached_search_engine, "repo", None)
        if repo is not None:
            repo.conn = conn
    except BaseException:
        _search_engine_lock.release()
        raise
    return _cached_search_engine


def _release_search_engine() -> None:
    """Release the lock held by ``_get_search_engine``."""
    try:
        _search_engine_lock.release()
    except RuntimeError:
        # Not held by this thread — ignore so callers can use finally blocks
        # without worrying about early-exit paths before acquisition.
        pass


@contextlib.contextmanager
def _search_engine_ctx(conn: sqlite3.Connection) -> Any:
    """Context-manager wrapper around ``_get_search_engine``."""
    engine = _get_search_engine(conn)
    try:
        yield engine
    finally:
        _release_search_engine()


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
        [table_name],
    ).fetchone()
    return row is not None


# SQLite's default SQLITE_MAX_VARIABLE_NUMBER is 999 on some builds; stay well under.
_FACE_CLUSTER_IN_CHUNK = 500


def _get_face_clusters_for_image_ids(
    conn: sqlite3.Connection,
    image_ids: list[int],
) -> dict[int, list[dict[str, Any]]]:
    """Return face cluster memberships keyed by image_id in a single pass.

    Issues one SQL query per chunk of up to ``_FACE_CLUSTER_IN_CHUNK`` ids
    (not one per image), so a page of N result images costs at most
    ceil(N / 500) queries — typically one. Replaces the previous per-image
    N+1 pattern.
    """
    if not image_ids or not _table_exists(conn, "face_occurrences"):
        return {}

    unique_image_ids: list[int] = list(dict.fromkeys(image_ids))

    has_cluster_labels = _table_exists(conn, "face_cluster_labels")
    has_persons = _table_exists(conn, "face_persons")
    cluster_name_expr = (
        "COALESCE(fcl.display_name, 'Cluster ' || fo.cluster_id)"
        if has_cluster_labels
        else "'Cluster ' || fo.cluster_id"
    )
    person_name_expr = "fp.name" if has_persons else "NULL"
    cluster_label_join = (
        "LEFT JOIN face_cluster_labels fcl ON fcl.cluster_id = fo.cluster_id"
        if has_cluster_labels
        else ""
    )
    person_join = (
        "LEFT JOIN face_persons fp ON fp.id = fo.person_id"
        if has_persons
        else ""
    )

    clusters_by_image: dict[int, list[dict[str, Any]]] = {}
    for chunk_start in range(0, len(unique_image_ids), _FACE_CLUSTER_IN_CHUNK):
        chunk = unique_image_ids[chunk_start : chunk_start + _FACE_CLUSTER_IN_CHUNK]
        placeholders = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"""
            SELECT
                fo.image_id AS image_id,
                fo.cluster_id AS cluster_id,
                MAX({cluster_name_expr}) AS cluster_label,
                MAX(fo.person_id) AS person_id,
                MAX({person_name_expr}) AS person_name,
                COUNT(*) AS face_count
            FROM face_occurrences fo
            {cluster_label_join}
            {person_join}
            WHERE fo.image_id IN ({placeholders})
              AND fo.cluster_id IS NOT NULL
            GROUP BY fo.image_id, fo.cluster_id
            ORDER BY fo.image_id, face_count DESC, fo.cluster_id
            """,
            chunk,
        ).fetchall()

        for row in rows:
            image_id = int(row["image_id"])
            clusters_by_image.setdefault(image_id, []).append(
                {
                    "cluster_id": int(row["cluster_id"]),
                    "cluster_label": row["cluster_label"],
                    "person_id": int(row["person_id"]) if row["person_id"] is not None else None,
                    "person_name": row["person_name"],
                    "face_count": int(row["face_count"]),
                }
            )
    return clusters_by_image


# Back-compat alias — existing call sites use this name.
_get_face_clusters_for_images = _get_face_clusters_for_image_ids


def _is_transient_db_lock_error(exc: Exception) -> bool:
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    text = str(exc).lower()
    return any(marker in text for marker in _TRANSIENT_DB_LOCK_MARKERS)


# ── State for cancellable operations ─────────────────────────────────────────

_run_cancel = threading.Event()
_run_lock = threading.Lock()
_run_thread: threading.Thread | None = None
_active_worker: Any = None  # Worker | None — set while a run is active
_ingest_lock = threading.Lock()

_analyze_cancel: dict[str, threading.Event] = {}  # imagePath -> cancel event

_MASTER_WORKER_ID = "master"
_MASTER_WORKER_LABEL = "Master device"
_DB_BUSY_TIMEOUT_MS = 30000
_LEGACY_QUEUE_MODULE_MAP: dict[str, str] = {
    "blip2": "caption",
    "cloud_ai": "caption",
    "local_ai": "caption",
    "aesthetic": "perception",
}
_PERSON_LINK_SUGGESTION_CACHE_TTL_SECONDS = 30.0
_person_link_suggestion_cache: dict[tuple[int, int], tuple[float, list[dict[str, Any]]]] = {}
_TRANSIENT_DB_LOCK_MARKERS = (
    "database is locked",
    "database table is locked",
    "database schema is locked",
)
_LOCK_RETRYABLE_METHODS = {
    "status",
    "workers/list",
    "workers/register",
    "workers/heartbeat",
    "faces/person-link-cluster",
    "faces/person-unlink-cluster",
    "faces/cluster-relink",
    "jobs/claim",
    "jobs/release-expired",
    "jobs/heartbeat",
    "jobs/release",
    "jobs/complete",
    "jobs/fail",
    "jobs/skip",
    "jobs/release-worker",
}
_LOCK_RETRY_ATTEMPTS = 4
_LOCK_RETRY_INITIAL_DELAY_S = 0.15
_STATUS_CACHE_DEFAULT_TTL_MS = 800
_STATUS_CACHE_MAX_TTL_MS = 5000
_status_cache_lock = threading.Lock()
_status_cache: dict[str, tuple[float, dict[str, Any]]] = {}

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
_DECODE_BUFFER_SIZE = int(os.getenv("IMGANALYZER_DECODE_BUFFER_SIZE", "10"))
_MIN_DECODE_BUFFER = 5  # never let in-flight drop below this
_last_replenish_time: float = 0.0
_REPLENISH_INTERVAL: float = 5.0  # seconds between buffer checks
_REMOTE_DECODE_PRIORITY_MODULES = ("caption", "objects", "embedding", "perception")


def _auto_trigger_pre_decode() -> None:
    """Kept for backward compatibility — calls _replenish_decode_buffer."""
    _replenish_decode_buffer()


def _replenish_decode_buffer() -> None:
    """Feed images to the pre-decoder based on CPU and disk availability.

    Called from ``_handle_jobs_claim`` on every request.  Uses a
    :class:`~imganalyzer.cache.pre_decode.ResourceSampler` to check
    system utilisation before deciding whether (and how much) to feed.

    * **High capacity** (CPU < 65 %, disk < 50 %): feed up to
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

        from imganalyzer.db.connection import create_connection
        conn2 = create_connection(busy_timeout_ms=_DB_BUSY_TIMEOUT_MS)
        try:
            # Prioritize images that can immediately unblock distributed
            # workers (caption / objects / embedding / perception) before
            # master-only backlog. Within that, older and higher-fanout jobs
            # go first.
            remote_ph = ",".join("?" * len(_REMOTE_DECODE_PRIORITY_MODULES))
            pending_rows = conn2.execute(
                f"""SELECT image_id,
                           COUNT(*) AS cnt,
                           SUM(CASE WHEN module IN ({remote_ph}) THEN 1 ELSE 0 END) AS remote_cnt,
                           MIN(queued_at) AS oldest_queued_at
                    FROM job_queue
                    WHERE status = 'pending'
                      AND attempts <= max_attempts
                    GROUP BY image_id
                    ORDER BY remote_cnt DESC, cnt DESC, oldest_queued_at ASC, image_id ASC"""
                ,
                list(_REMOTE_DECODE_PRIORITY_MODULES),
            ).fetchall()
            online_workers_row = conn2.execute(
                """SELECT COUNT(*) AS cnt
                   FROM worker_nodes
                   WHERE id != 'master'
                     AND status IN ('online', 'active')
                     AND last_heartbeat > datetime('now', '-5 minutes')"""
            ).fetchall()
            online_workers = int(online_workers_row[0]["cnt"]) if online_workers_row else 0
            pending_ids = [int(r["image_id"]) for r in pending_rows]

            if not pending_ids:
                return

            remote_ready = sum(
                1
                for r in pending_rows
                if int(r["remote_cnt"] or 0) > 0 and int(r["image_id"]) in cached_ids
            )
            if online_workers > 0 and remote_ready < max(_MIN_DECODE_BUFFER, online_workers * 2):
                batch_size = max(
                    batch_size,
                    min(max(_DECODE_BUFFER_SIZE * 4, online_workers * 25), 100),
                )
                need = batch_size - max(in_flight, 0)
                if need <= 0:
                    return

            uncached = [iid for iid in pending_ids if iid not in cached_ids][:need]

            if not uncached:
                return

            # Fetch file paths for the batch
            ph = ",".join("?" * len(uncached))
            rows = conn2.execute(
                f"SELECT id, file_path FROM images WHERE id IN ({ph})",
                uncached,
            ).fetchall()
        finally:
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


def _as_bool_param(value: Any, default: bool) -> bool:
    """Coerce a request parameter into bool."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _master_worker_runtime_status() -> str:
    if _active_worker is not None:
        return "online"
    if _run_thread is not None and _run_thread.is_alive():
        return "online"
    return "offline"


def _sync_master_worker_node(conn: sqlite3.Connection) -> None:
    _upsert_master_worker_node(conn, status=_master_worker_runtime_status())


def _ensure_master_worker_node(conn: sqlite3.Connection) -> None:
    """Ensure the master worker row exists without refreshing heartbeat every poll."""
    row = conn.execute(
        "SELECT 1 FROM worker_nodes WHERE id = ?",
        [_MASTER_WORKER_ID],
    ).fetchone()
    if row is None:
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

    lite = _as_bool_param(params.get("lite"), False)
    include_recent_results = _as_bool_param(
        params.get("include_recent_results"),
        default=not lite,
    )
    include_module_avg_ms = _as_bool_param(
        params.get("include_module_avg_ms"),
        default=not lite,
    )
    cache_enabled = lite and _as_bool_param(params.get("cache"), False)
    cache_ttl_ms_raw = params.get("cache_ttl_ms", _STATUS_CACHE_DEFAULT_TTL_MS)
    try:
        cache_ttl_ms = int(cache_ttl_ms_raw)
    except (TypeError, ValueError):
        cache_ttl_ms = _STATUS_CACHE_DEFAULT_TTL_MS
    cache_ttl_ms = max(100, min(cache_ttl_ms, _STATUS_CACHE_MAX_TTL_MS))
    cache_key = (
        "status:"
        f"lite={int(lite)}:"
        f"recent={int(include_recent_results)}:"
        f"module_avg={int(include_module_avg_ms)}"
    )

    if cache_enabled:
        with _status_cache_lock:
            cached = _status_cache.get(cache_key)
        if cached is not None:
            cached_at, payload = cached
            if (time.monotonic() - cached_at) * 1000.0 <= cache_ttl_ms:
                return payload

    conn = _get_db()
    _ensure_master_worker_node(conn)
    queue = JobQueue(conn)
    repo = Repository(conn)

    total_images = repo.count_images()
    module_stats = queue.stats()
    totals = queue.total_stats()
    remaining_images = queue.remaining_image_count()
    module_avg_ms: dict[str, float] = {}
    if include_module_avg_ms:
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
    w = _active_worker
    if w is not None and w.current_chunk_ids is not None:
        chunk_ids = w.current_chunk_ids
        for mod in modules_out:
            cp = queue.pending_count(module=mod, image_ids=chunk_ids)
            cr = queue.running_count(module=mod, image_ids=chunk_ids)
            if cp > 0 or cr > 0:
                chunk_modules[mod] = cp + cr
        chunk_info = {
            "size": len(chunk_ids),
            "index": w.current_chunk_index,
            "total": w.total_chunks,
            "modules": chunk_modules,
        }

    result = {
        "total_images": total_images,
        "modules": modules_out,
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
    }
    if include_module_avg_ms:
        result["module_avg_ms"] = module_avg_ms
    if include_recent_results:
        result["recent_results"] = _recent_queue_results(conn)

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

    # Drive the decode pipeline on every status poll (~1 s) so the
    # pre-decoder keeps running even when no distributed workers are
    # actively claiming.  The 5 s internal throttle prevents excessive
    # DB queries.
    try:
        _replenish_decode_buffer()
    except Exception:
        pass

    if cache_enabled:
        with _status_cache_lock:
            _status_cache[cache_key] = (time.monotonic(), result)

    return result


def _handle_ingest(req_id: int | str, params: dict) -> None:
    """Ingest folders — streaming progress notifications, then final result."""
    from imganalyzer.pipeline.batch import BatchProcessor
    from imganalyzer.db.connection import create_connection

    # Open a FRESH connection for this thread — _handle_ingest runs in a
    # daemon thread (async RPC method) and cannot reuse the main-thread
    # singleton returned by _get_db().
    conn = create_connection(busy_timeout_ms=_DB_BUSY_TIMEOUT_MS)
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
    # Serialised with _ingest_lock to prevent concurrent ingests from
    # racing on the global builtins.print binding.
    import builtins

    with _ingest_lock:
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

    with _run_lock:
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

        try:
            _replenish_decode_buffer()
        except Exception as exc:
            sys.stderr.write(f"[server] pre-decode warmup failed: {exc}\n")

        def _run_worker():
            global _active_worker
            try:
                from imganalyzer.pipeline.worker import Worker
                from imganalyzer.db.connection import create_connection

                # Open a FRESH connection for this thread — SQLite connections
                # cannot be shared across threads (check_same_thread=True by default).
                conn = create_connection(busy_timeout_ms=_DB_BUSY_TIMEOUT_MS)
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
    w = _active_worker
    if w is not None:
        w._shutdown.set()
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
    runtime_status = _master_worker_runtime_status()
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
            "status": runtime_status,
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
        "status": runtime_status if runtime_status == "online" else (row["status"] or runtime_status),
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
    _ensure_master_worker_node(conn)
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
    w = _active_worker
    if worker_id == _MASTER_WORKER_ID and w is not None:
        w._shutdown.set()

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
_DISTRIBUTED_MASTER_ONLY_MODULES = frozenset({"metadata", "technical", "faces"})
_DISTRIBUTED_CACHE_OPTIONAL_MODULES = frozenset({"embedding"})
_LEASE_TTL_FLOORS_SECONDS: dict[str, int] = {
    "caption": 300,
    "perception": 300,
}


def _claim_lease_ttl_seconds(
    requested_ttl_seconds: int,
    *,
    module_filter: str | None,
    modules_filter: list[str] | None,
    prefer_module: str | None,
) -> int:
    """Return a safe lease TTL for the current claim mix.

    Slow modules like caption/perception should tolerate a few transient
    heartbeat misses without immediate reclaim churn.
    """
    ttl = max(5, int(requested_ttl_seconds))
    candidates = [module_filter, prefer_module]
    if modules_filter:
        candidates.extend(modules_filter)
    for module_name in candidates:
        if not module_name:
            continue
        ttl = max(ttl, _LEASE_TTL_FLOORS_SECONDS.get(str(module_name), 0))
    return ttl


def _request_requires_decoded_cache(
    worker_id: str,
    *,
    module_filter: str | None,
    modules_filter: list[str] | None,
    worker_caps: dict[str, Any],
) -> bool:
    """Whether this distributed claim must be backed by coordinator decode cache."""
    if worker_id == _MASTER_WORKER_ID:
        return False
    if "requiresDecodedCache" in worker_caps:
        return bool(worker_caps.get("requiresDecodedCache"))
    if module_filter is not None:
        return module_filter not in _DISTRIBUTED_CACHE_OPTIONAL_MODULES
    if modules_filter is not None:
        return any(m not in _DISTRIBUTED_CACHE_OPTIONAL_MODULES for m in modules_filter)
    return True


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
        get_worker_capabilities,
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

    w = _active_worker
    active_chunk_ids: set[int] | None = None
    if w is not None:
        active_chunk_ids = w.current_chunk_ids
    policy = compute_claim_policy(
        conn,
        worker_id=worker_id,
        batch_size=batch_size,
        module=module,
        modules_list=modules_list,
        active_chunk_ids=active_chunk_ids,
        coordinator_run_active=w is not None,
    )
    if not policy.allow_claims:
        return {"jobs": [], "reason": policy.reason}

    requested = policy.requested
    scan_size = policy.scan_size
    module_filter = policy.module_filter
    modules_filter = policy.modules_filter
    prefer_module = policy.prefer_module
    prefer_image_ids = policy.prefer_image_ids
    worker_caps = get_worker_capabilities(conn, worker_id)
    requires_decoded_cache = _request_requires_decoded_cache(
        worker_id,
        module_filter=module_filter,
        modules_filter=modules_filter,
        worker_caps=worker_caps,
    )

    # Defense in depth: the scheduler should already exclude these for
    # distributed workers, but keep the coordinator-side guard here too.
    if module_filter and module_filter in _DISTRIBUTED_MASTER_ONLY_MODULES:
        return {"jobs": [], "reason": "master_only_modules"}
    if modules_filter:
        modules_filter = [
            m for m in modules_filter if m not in _DISTRIBUTED_MASTER_ONLY_MODULES
        ]
        if not modules_filter:
            return {"jobs": [], "reason": "master_only_modules"}

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
    total_defers = 0
    total_skips = 0
    # Some distributed workers are coordinator-cache backed and cannot reach
    # the source image path directly. Those workers must only receive jobs for
    # images already decoded on the coordinator.
    cache_gate_ids: set[int] | None = None
    try:
        _auto_trigger_pre_decode()
        cache_gate_ids = _get_decoded_store().cached_image_ids()
    except Exception as exc:
        if requires_decoded_cache:
            sys.stderr.write(f"[server] decoded-cache claim gating failed: {exc}\n")
            return {"jobs": [], "reason": "decoded_cache_unavailable"}

    if requires_decoded_cache and pending_eligible > 0 and not cache_gate_ids:
        return {"jobs": [], "reason": "waiting_for_decoded_cache"}
    # When a module yields only invalid jobs for several consecutive batches,
    # exclude it so lower-priority modules (e.g. embedding) can be reached.
    exhausted_modules: set[str] = set()
    module_miss_streak: dict[str, int] = {}
    _MISS_THRESHOLD = scan_size * 2  # skip module after this many consecutive misses
    safe_lease_ttl_seconds = _claim_lease_ttl_seconds(
        lease_ttl_seconds,
        module_filter=module_filter,
        modules_filter=modules_filter,
        prefer_module=prefer_module,
    )
    while len(jobs) < requested and scanned_candidates < max_scan_candidates:
        claim_size = min(scan_size, max_scan_candidates - scanned_candidates)
        excl = sorted(exhausted_modules) if exhausted_modules else None
        claim_kwargs = {
            "worker_id": worker_id,
            "lease_ttl_seconds": safe_lease_ttl_seconds,
            "batch_size": claim_size,
            "module": module_filter,
            "modules": modules_filter,
            "exclude_modules": excl,
            "prefer_module": prefer_module,
            "prefer_image_ids": prefer_image_ids,
            "master_reserve": policy.master_reserve,
        }
        claimed = queue.claim_leased(
            **claim_kwargs,
            restrict_image_ids=cache_gate_ids if requires_decoded_cache else None,
        )
        if not claimed:
            break
        scanned_candidates += len(claimed)

        batch_valid_modules: set[str] = set()
        pending_skips: list[tuple[int, str, str]] = []
        pending_releases: list[tuple[int, str]] = []
        pending_defers: list[tuple[int, str, int]] = []

        for job in claimed:
            if len(jobs) >= requested:
                pending_releases.append((job["id"], job["lease_token"]))
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
                pending_skips.append((job["id"], job["lease_token"], "already_analyzed"))
                continue

            prereq = _PREREQUISITES.get(module_name)
            if prereq and not repo.is_analyzed(image_id, prereq):
                prereq_status = queue.get_image_module_job_status(image_id, prereq)
                if prereq_status in ("failed", "skipped"):
                    pending_skips.append((
                        job["id"],
                        job["lease_token"],
                        f"prerequisite_{prereq}_{prereq_status}",
                    ))
                else:
                    delay_seconds = 30 if prereq_status == "running" else 15
                    pending_defers.append((job["id"], job["lease_token"], delay_seconds))
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
            if cache_gate_ids is not None and image_id in cache_gate_ids:
                job_entry["hasDecodedCache"] = True
                job_entry["requireDecodedCache"] = True
            else:
                job_entry["requireDecodedCache"] = requires_decoded_cache
            jobs.append(job_entry)

        # Batch all skip/release ops into a single transaction to reduce
        # write contention when multiple workers poll simultaneously.
        total_skips += len(pending_skips)
        total_defers += len(pending_defers)
        queue.batch_skip_release_leased(pending_skips, pending_releases, pending_defers)

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

    if not jobs:
        if requires_decoded_cache and pending_eligible > 0 and scanned_candidates == 0:
            reason = "waiting_for_decoded_cache"
        else:
            reason = "deferred_prerequisites" if total_defers else "no_eligible_jobs"
        return {"jobs": [], "reason": reason, "deferred": total_defers, "skipped": total_skips}

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
    delay_seconds = int(params.get("delaySeconds", 0))
    if job_id <= 0 or not lease_token:
        raise ValueError("jobId and leaseToken are required")

    conn = _get_db()
    queue = JobQueue(conn)
    ok = queue.release_leased(job_id, lease_token, delay_seconds=delay_seconds)
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
    image_id = 0
    module_name = ""

    from imganalyzer.db.connection import begin_immediate
    begin_immediate(conn)
    try:
        lease = conn.execute(
            "SELECT worker_id FROM job_leases WHERE job_id = ? AND lease_token = ?",
            [job_id, lease_token],
        ).fetchone()
        if lease is None:
            existing_job = conn.execute(
                "SELECT status FROM job_queue WHERE id = ?",
                [job_id],
            ).fetchone()
            conn.rollback()
            if existing_job is not None and str(existing_job["status"] or "").lower() == "done":
                # Idempotent completion path: previous attempt may have committed
                # successfully but response delivery timed out to the worker.
                return {"ok": True}
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

    if module_name in _DISTRIBUTED_SEARCH_MODULES:
        delay_s = _LOCK_RETRY_INITIAL_DELAY_S
        for attempt in range(1, _LOCK_RETRY_ATTEMPTS + 1):
            try:
                begin_immediate(conn)
                try:
                    repo.update_search_artifacts(image_id)
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                break
            except Exception as exc:
                conn.rollback()
                if not _is_transient_db_lock_error(exc) or attempt >= _LOCK_RETRY_ATTEMPTS:
                    sys.stderr.write(
                        f"[server] jobs/complete search update failed for image {image_id}: {exc}\n"
                    )
                    break
                time.sleep(delay_s)
                delay_s = min(delay_s * 2, 1.0)

        try:
            from imganalyzer.storyline.incremental import check_and_add_image

            check_and_add_image(conn, image_id)
        except Exception as exc:
            sys.stderr.write(
                f"[server] jobs/complete storyline refresh failed for image {image_id}: {exc}\n"
            )

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
    engine = _get_search_engine(conn)
    try:

        # Progress callback — sends JSON-RPC notifications during slow phases
        _progress_sent: set[str] = set()

        def _progress(phase: str, message: str, progress: float) -> None:
            if phase in _progress_sent:
                return
            _progress_sent.add(phase)
            _send_notification("search/progress", {
                "phase": phase,
                "message": message,
                "progress": progress,
            })

        query = params.get("query", "")
        mode = params.get("mode", "hybrid")
        semantic_weight = params.get("semanticWeight", 0.5)
        semantic_profile = params.get("semanticProfile", "balanced")
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
        map_bounds = params.get("mapBounds")  # {north, south, east, west}
        sort_by = params.get("sortBy", "relevance")
        rank_preference = params.get("rankPreference")
        expanded_terms_raw = params.get("expandedTerms", [])
        must_terms_raw = params.get("mustTerms", [])
        should_terms_raw = params.get("shouldTerms", [])
        debug_search = params.get("debugSearch", False)
        facet_request = params.get("facetRequest", False)
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
        valid_rank_preference = {"relevance", "quality", "recency", "aesthetic", "cleanest", "sharpest"}
        valid_semantic_profile = {"image_dominant", "balanced", "description_dominant"}
        if rank_preference is not None and rank_preference not in valid_rank_preference:
            raise ValueError("rankPreference must be one of relevance, quality, recency, aesthetic, cleanest or sharpest")
        if semantic_profile is None:
            semantic_profile = "balanced"
        if not isinstance(semantic_profile, str):
            raise ValueError("semanticProfile must be one of image_dominant, balanced or description_dominant")
        semantic_profile = semantic_profile.strip().lower()
        if semantic_profile not in valid_semantic_profile:
            raise ValueError("semanticProfile must be one of image_dominant, balanced or description_dominant")
        if sort_by == "relevance" and rank_preference:
            sort_by = {
                "relevance": "relevance",
                "quality": "best",
                "recency": "newest",
                "aesthetic": "aesthetic",
                "cleanest": "cleanest",
                "sharpest": "sharpness",
            }[rank_preference]
        if not isinstance(debug_search, bool):
            raise ValueError("debugSearch must be a boolean")
        if not isinstance(facet_request, bool):
            raise ValueError("facetRequest must be a boolean")
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
        if must_terms_raw is None:
            must_terms_raw = []
        if should_terms_raw is None:
            should_terms_raw = []
        if not isinstance(must_terms_raw, list):
            raise ValueError("mustTerms must be an array")
        if not isinstance(should_terms_raw, list):
            raise ValueError("shouldTerms must be an array")
        must_terms: list[str] = []
        should_terms: list[str] = []
        seen_must: set[str] = set()
        seen_should: set[str] = set()
        for term in must_terms_raw:
            if not isinstance(term, str):
                raise ValueError("mustTerms entries must be strings")
            clean = term.strip()
            lowered = clean.casefold()
            if clean and lowered not in seen_must:
                seen_must.add(lowered)
                must_terms.append(clean)
        for term in should_terms_raw:
            if not isinstance(term, str):
                raise ValueError("shouldTerms entries must be strings")
            clean = term.strip()
            lowered = clean.casefold()
            if clean and lowered not in seen_should:
                seen_should.add(lowered)
                should_terms.append(clean)

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
            _progress("resolving", "Resolving names…", 0.0)
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

        if not faces:
            resolve_face_queries = getattr(engine, "resolve_face_queries", None)
            if callable(resolve_face_queries):
                normalized_faces: list[str] = []
                seen_resolved_faces: set[str] = set()

                def _consume_face_terms(terms: list[str]) -> list[str]:
                    nonlocal face_match
                    remaining_terms: list[str] = []
                    for raw_term in terms:
                        resolved_from_term, remaining_term, resolved_match = resolve_face_queries(raw_term)
                        if resolved_from_term:
                            for resolved_face in resolved_from_term:
                                lowered = resolved_face.casefold()
                                if lowered in seen_resolved_faces:
                                    continue
                                seen_resolved_faces.add(lowered)
                                normalized_faces.append(resolved_face)
                            if len(normalized_faces) > 1:
                                face_match = resolved_match if resolved_match in {"any", "all"} else "all"
                        cleaned_remaining = remaining_term.strip()
                        if cleaned_remaining:
                            remaining_terms.append(cleaned_remaining)
                    return remaining_terms

                must_terms = _consume_face_terms(must_terms)
                should_terms = _consume_face_terms(should_terms)
                if normalized_faces:
                    faces = normalized_faces
                    face = faces[0]
                    if len(faces) > 1 and face_match not in {"any", "all"}:
                        face_match = "all"

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
            cam_esc = _escape_like(camera)
            base_conditions.append(
                "(m.camera_make LIKE ? ESCAPE '\\' OR m.camera_model LIKE ? ESCAPE '\\')"
            )
            base_params.extend([f"%{cam_esc}%", f"%{cam_esc}%"])
        if lens:
            base_conditions.append("m.lens_model LIKE ? ESCAPE '\\'")
            base_params.append(f"%{_escape_like(lens)}%")
        if country:
            base_conditions.append("m.location_country LIKE ? ESCAPE '\\'")
            base_params.append(f"%{_escape_like(country)}%")
        if location:
            loc_esc = _escape_like(location)
            base_conditions.append(
                "(m.location_city LIKE ? ESCAPE '\\'"
                " OR m.location_state LIKE ? ESCAPE '\\'"
                " OR m.location_country LIKE ? ESCAPE '\\')"
            )
            base_params.extend([f"%{loc_esc}%", f"%{loc_esc}%", f"%{loc_esc}%"])
        if map_bounds and isinstance(map_bounds, dict) and _table_exists(conn, "geo_rtree"):
            mb_south = float(map_bounds.get("south", -90))
            mb_north = float(map_bounds.get("north", 90))
            mb_west = float(map_bounds.get("west", -180))
            mb_east = float(map_bounds.get("east", 180))
            base_conditions.append(
                "m.image_id IN ("
                "  SELECT r.id FROM geo_rtree r"
                "  WHERE r.min_lat >= ? AND r.max_lat <= ?"
                "  AND r.min_lng >= ? AND r.max_lng <= ?)"
            )
            base_params.extend([mb_south, mb_north, mb_west, mb_east])
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
            *,
            include_face_clusters: bool = True,
        ) -> list[dict[str, Any]]:
            has_face_occurrences = _table_exists(conn, "face_occurrences")
            if include_face_clusters:
                image_ids = [int(row["image_id"]) for row in rows]
                face_clusters_by_image = (
                    _get_face_clusters_for_image_ids(conn, image_ids)
                    if has_face_occurrences
                    else {}
                )
            else:
                face_clusters_by_image = {}
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
                    "face_clusters": (
                        face_clusters_by_image.get(image_id, [])
                        if has_face_occurrences
                        else None
                    ),
                })
            if score_lookup is not None and sort_by == "relevance":
                records.sort(key=lambda record: -(record["score"] or 0.0))
            return records

        def _fetch_records(
            candidate_ids: list[int] | None,
            score_lookup: dict[int, float] | None,
            *,
            include_face_clusters: bool = True,
        ) -> list[dict[str, Any]]:
            where_clause, query_params = _build_where_clause(candidate_ids)
            rows = conn.execute(
                f"SELECT {select_cols} {joins} {where_clause}",
                query_params,
            ).fetchall()
            return _rows_to_records(
                rows, score_lookup, include_face_clusters=include_face_clusters
            )

        def _fetch_records_page(
            candidate_ids: list[int] | None,
            score_lookup: dict[int, float] | None,
            order_clause: str,
            page_limit: int,
            page_offset: int,
        ) -> tuple[list[dict[str, Any]], int]:
            """Fetch a single page with SQL-level ORDER BY + LIMIT/OFFSET.

            Returns (page_records, total_count) where total_count is the total
            number of images matching the filters (possibly restricted to
            candidate_ids). Face clusters are NOT attached here — call
            ``_attach_face_clusters`` on the returned page slice.
            """
            where_clause, query_params = _build_where_clause(candidate_ids)
            total_row = conn.execute(
                f"SELECT COUNT(*) AS cnt {joins} {where_clause}",
                query_params,
            ).fetchone()
            total_count = int(total_row["cnt"]) if total_row else 0
            rows = conn.execute(
                f"SELECT {select_cols} {joins} {where_clause}{order_clause}"
                " LIMIT ? OFFSET ?",
                [*query_params, page_limit, page_offset],
            ).fetchall()
            return (
                _rows_to_records(rows, score_lookup, include_face_clusters=False),
                total_count,
            )

        def _attach_face_clusters(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
            """Populate ``face_clusters`` on the provided records in one SQL pass."""
            if not records:
                return records
            if not _table_exists(conn, "face_occurrences"):
                for record in records:
                    record["face_clusters"] = None
                return records
            image_ids = [int(record["image_id"]) for record in records]
            clusters_by_image = _get_face_clusters_for_image_ids(conn, image_ids)
            for record in records:
                record["face_clusters"] = clusters_by_image.get(
                    int(record["image_id"]), []
                )
            return records

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

        search_params = inspect.signature(engine.search).parameters
        search_supports_profile = "semantic_profile" in search_params
        search_supports_progress = "progress_cb" in search_params
        search_supports_candidate_ids = "candidate_ids" in search_params
        face_candidate_ids_cache: set[int] | None = None

        def _search_text_terms(
            candidate_limit: int,
            *,
            candidate_ids: set[int] | None = None,
        ) -> list[dict[str, Any]]:
            if candidate_ids is not None and not candidate_ids:
                return []

            def _run_search(term: str) -> list[dict[str, Any]]:
                kwargs: dict[str, Any] = {
                    "limit": candidate_limit,
                    "semantic_weight": semantic_weight,
                    "mode": mode,
                }
                if search_supports_profile:
                    kwargs["semantic_profile"] = semantic_profile
                if search_supports_progress:
                    kwargs["progress_cb"] = _progress
                if search_supports_candidate_ids and candidate_ids is not None:
                    kwargs["candidate_ids"] = candidate_ids
                return engine.search(term, **kwargs)

            terms: list[str] = []
            terms.extend(must_terms)
            if has_text_query:
                terms.append(query.strip())
            terms.extend(should_terms)
            seen_local = {term.casefold() for term in terms}
            for term in expanded_terms:
                lowered = term.casefold()
                if lowered not in seen_local:
                    seen_local.add(lowered)
                    terms.append(term)
            if not terms:
                return []
            if len(terms) == 1:
                return _run_search(terms[0])

            fused_scores: dict[int, float] = {}
            file_paths: dict[int, str] = {}
            for term in terms:
                term_results = _run_search(term)
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

        def _get_face_candidate_ids() -> set[int]:
            nonlocal face_candidate_ids_cache
            if face_candidate_ids_cache is not None:
                return face_candidate_ids_cache
            if not faces:
                face_candidate_ids_cache = set()
                return face_candidate_ids_cache
            if len(faces) == 1:
                face_rows = engine.search_face(faces[0], limit=None)
            else:
                search_faces = getattr(engine, "search_faces", None)
                if callable(search_faces):
                    face_rows = search_faces(faces, limit=None, match_mode=face_match)
                else:
                    face_rows = _search_face_terms(0)
            face_candidate_ids_cache = {int(row["image_id"]) for row in face_rows}
            return face_candidate_ids_cache

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
            if search_supports_candidate_ids:
                face_candidate_ids = _get_face_candidate_ids()
                text_results = _search_text_terms(candidate_limit, candidate_ids=face_candidate_ids)
                return text_results, len(text_results) < candidate_limit

            face_results = _search_face_terms(candidate_limit)
            text_results = _search_text_terms(candidate_limit)
            face_exhausted = len(face_results) < candidate_limit
            text_exhausted = len(text_results) < candidate_limit
            if not face_results or not text_results:
                # If face set is exhausted and intersection is empty, retrying
                # with a larger candidate_limit will never produce results.
                return [], face_exhausted or (face_exhausted and text_exhausted)

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

            # When the face set is fully known (exhausted), the intersection
            # can't grow by fetching more text candidates — report exhausted.
            search_exhausted = face_exhausted or (face_exhausted and text_exhausted)
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
            _search_deadline = time.monotonic() + 30.0  # wall-clock timeout

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
                    if (
                        not search_exhausted
                        and candidate_limit < max_candidate_limit
                        and time.monotonic() < _search_deadline
                    ):
                        candidate_limit = min(max_candidate_limit, candidate_limit * 2)
                        continue
                    return {"results": [], "total": 0, "hasMore": False}

                score_map = {
                    int(result["image_id"]): float(result["score"])
                    for result in search_results
                }
                _progress("filtering", "Filtering results…", 0.85)

                # B3: When the final sort is not score-based, push ORDER BY +
                # LIMIT/OFFSET to SQL so we never materialize the full
                # candidate pool (up to ``max_candidate_limit`` rows, currently
                # 5000) in Python. The ``relevance`` branch still needs the
                # whole filtered set in memory to preserve engine-supplied
                # fusion scores — that pool stays capped at 5000 candidates
                # and is sorted/paginated in Python.
                if sort_by != "relevance":
                    order_clause = _browse_order_clause()
                    page_records, matched_total = _fetch_records_page(
                        candidate_ids,
                        score_map,
                        order_clause,
                        limit,
                        offset,
                    )
                    enough_for_page = matched_total > page_end
                    if (
                        search_exhausted
                        or enough_for_page
                        or candidate_limit >= max_candidate_limit
                        or time.monotonic() >= _search_deadline
                    ):
                        _attach_face_clusters(page_records)
                        has_more = matched_total > page_end or not search_exhausted
                        total_out: int | None = matched_total if search_exhausted else None
                        return {
                            "results": page_records,
                            "total": total_out,
                            "hasMore": has_more,
                        }
                    candidate_limit = min(max_candidate_limit, candidate_limit * 2)
                    continue

                # sort_by == "relevance": fetch full candidate set (cluster
                # lookup deferred) and re-rank in Python by fused score.
                records = _fetch_records(
                    candidate_ids, score_map, include_face_clusters=False
                )
                records = _sort_records(records)
                enough_for_page = len(records) > page_end
                if (
                    search_exhausted
                    or enough_for_page
                    or candidate_limit >= max_candidate_limit
                    or time.monotonic() >= _search_deadline
                ):
                    break
                candidate_limit = min(max_candidate_limit, candidate_limit * 2)

            # B4: fetch face clusters only for the paginated slice (not the
            # entire candidate pool).
            page_records = records[offset: offset + limit]
            _attach_face_clusters(page_records)
            has_more = len(records) > page_end or not search_exhausted
            total: int | None = len(records) if search_exhausted else None
            return {
                "results": page_records,
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
    finally:
        _release_search_engine()


def _handle_search_resolve_face_query(params: dict) -> dict[str, Any]:
    from imganalyzer.db.search import SearchEngine

    query = params.get("query", "")
    if not isinstance(query, str):
        raise ValueError("query must be a string")

    conn = _get_db()
    with _search_engine_ctx(conn) as engine:
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


def _handle_search_warmup(_params: dict) -> dict:
    """Pre-load search caches to speed up the first search query.

    Loads embedding matrices and rich-description ID set eagerly so the
    first user-initiated search doesn't pay the cold-start cost.
    """
    import sys

    conn = _get_db()
    with _search_engine_ctx(conn) as engine:
        try:
            engine._image_clip_cache.get(conn, "image_clip")
            engine._desc_clip_cache.get(conn, "description_clip")
            engine._get_rich_desc_image_ids()
        except Exception as exc:
            print(f"[search/warmup] partial failure: {exc}", file=sys.stderr)
    return {"ok": True}


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
    # Use file_path natively (no REPLACE) so SQLite can use idx_images_file_path
    # for both the LIKE prefix scan and the ORDER BY.
    path_expr = "i.file_path"

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
            # Build LIKE patterns for both separators using '!' as the escape
            # character (so literal '\' in the backslash pattern is preserved).
            def _esc(s: str) -> str:
                return s.replace("!", "!!").replace("%", "!%").replace("_", "!_")
            fwd = _esc(folder_norm) + "/%"
            bwd = _esc(folder_norm.replace("/", "\\")) + "\\%"
            conditions.append(
                "(i.file_path LIKE ? ESCAPE '!' OR i.file_path LIKE ? ESCAPE '!')"
            )
            sql_params.extend([fwd, bwd])
            if not recursive:
                conditions.append(
                    "(INSTR(SUBSTR(i.file_path, LENGTH(?) + 2), '/') = 0 AND "
                    "INSTR(SUBSTR(i.file_path, LENGTH(?) + 2), '\\') = 0)"
                )
                sql_params.extend([folder_norm, folder_norm])

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

    has_face_occurrences = _table_exists(conn, "face_occurrences")
    face_clusters_by_image = (
        _get_face_clusters_for_images(conn, [int(row["image_id"]) for row in rows])
        if has_face_occurrences
        else {}
    )

    items = []
    for row in rows:
        image_id = int(row["image_id"])
        items.append({
            "image_id": image_id,
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
            "face_clusters": (
                face_clusters_by_image.get(image_id, [])
                if has_face_occurrences
                else None
            ),
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
    from PIL import Image
    from imganalyzer.readers import open_as_pil

    img = open_as_pil(path)

    img.thumbnail((400, 300), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    jpeg_bytes = buf.getvalue()

    return {"data": base64.b64encode(jpeg_bytes).decode("ascii")}


def _handle_thumbnails_batch(params: dict) -> dict:
    """Generate thumbnails for multiple images in a single RPC call.

    Checks the pre-decoded image store (1024px WebP cache) first and only
    falls back to full source decode for uncached images.  This makes
    cluster preview nearly instant for already-analysed images.

    Params:
        items (list[{image_id, file_path}]): images to thumbnail
              OR paths (list[str]): legacy file-path-only form
    Returns:
        thumbnails: {file_path: base64_data_url, ...}
        errors: {file_path: error_message, ...}
    """
    from PIL import Image as PILImage

    # Accept either [{image_id, file_path}, ...] or [path, ...]
    raw_items: list = params.get("items", [])
    if not raw_items:
        raw_paths = params.get("paths", [])
        raw_items = [{"file_path": p} for p in raw_paths]
    if not raw_items:
        return {"thumbnails": {}, "errors": {}}

    thumbnails: dict[str, str] = {}
    errors: dict[str, str] = {}
    store = _get_decoded_store()
    conn = _get_db()

    cached_jobs: list[tuple[str, int]] = []   # (cache_key, image_id) — fast path
    slow_items: list[tuple[str, str]] = []    # (cache_key, file_path) — slow path

    # Cap per-batch work so a huge call doesn't monopolize the worker pool.
    for item in raw_items[:200]:
        file_path = item.get("file_path", "") if isinstance(item, dict) else str(item)
        image_id = item.get("image_id") if isinstance(item, dict) else None
        cache_key = file_path or (str(image_id) if image_id is not None else "")
        try:
            if not file_path and image_id is not None:
                row = conn.execute(
                    "SELECT file_path FROM images WHERE id = ?",
                    [image_id],
                ).fetchone()
                if row is not None:
                    file_path = str(row["file_path"] or "")

            if image_id is not None and store.has(int(image_id)):
                if cache_key:
                    cached_jobs.append((cache_key, int(image_id)))
                continue

            if not file_path:
                raise ValueError("file_path or image_id is required")
            if cache_key:
                slow_items.append((cache_key, file_path))
        except Exception as exc:
            if cache_key:
                errors[cache_key] = str(exc)

    # Fast path: parallelize the cache-bytes-read + resize + JPEG encode + b64.
    # Each one is ~5-15ms but doing them sequentially blocks the RPC thread.
    def _resize_cached(cache_key: str, image_id: int) -> tuple[str, str | None, str | None]:
        try:
            img_bytes = store.get_image_bytes(image_id)
            if not img_bytes:
                return cache_key, None, "cache miss"
            img = PILImage.open(io.BytesIO(img_bytes))
            img.thumbnail((400, 300), PILImage.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return cache_key, f"data:image/jpeg;base64,{b64}", None
        except Exception as exc:
            return cache_key, None, str(exc)

    if cached_jobs:
        max_workers = min(8, len(cached_jobs))
        with ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="thumb-cached"
        ) as pool:
            for cache_key, dataurl, err in pool.map(
                lambda job: _resize_cached(*job), cached_jobs
            ):
                if dataurl is not None:
                    thumbnails[cache_key] = dataurl
                elif err and cache_key not in thumbnails:
                    errors[cache_key] = err

    if slow_items:
        # Opportunistically feed every missing item to the background decoder.
        # The slow-path full decode below will render *this* batch; the
        # PreDecoder warms the cache for the rest of the folder so the user's
        # next scroll is served from the fast path.
        try:
            decoder = _get_pre_decoder()
            feed_pairs: list[tuple[int, str]] = []
            seen_ids: set[int] = set()
            for raw in raw_items[:200]:
                if not isinstance(raw, dict):
                    continue
                rid = raw.get("image_id")
                rfp = raw.get("file_path")
                if rid is None or not rfp:
                    continue
                rid_int = int(rid)
                if rid_int in seen_ids or store.has(rid_int):
                    continue
                seen_ids.add(rid_int)
                feed_pairs.append((rid_int, str(rfp)))
            if feed_pairs:
                decoder.feed(feed_pairs)
        except Exception as exc:
            sys.stderr.write(f"[server] thumb prefetch feed failed: {exc}\n")

        max_workers = min(4, len(slow_items))
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="thumb-batch") as pool:
            future_map = {
                pool.submit(_handle_thumbnail, {"imagePath": file_path}): cache_key
                for cache_key, file_path in slow_items
            }
            for future in as_completed(future_map):
                cache_key = future_map[future]
                try:
                    result = future.result()
                    thumbnails[cache_key] = f"data:image/jpeg;base64,{result['data']}"
                except Exception as exc:
                    errors[cache_key] = str(exc)

    return {"thumbnails": thumbnails, "errors": errors}


def _handle_fullimage(params: dict) -> dict:
    """Generate a full-res JPEG for RAW/HEIC and return as base64.

    For native browser formats (jpg, png, webp, etc), returns a flag
    indicating the file should be read directly by Electron instead.
    """
    image_path = params["imagePath"]
    path = Path(image_path)
    ext = path.suffix.lower()

    NATIVE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
    if ext in NATIVE_EXTS:
        # Electron can read these directly — no need to decode via Python
        return {"native": True, "path": image_path}

    from PIL import Image
    from imganalyzer.readers import open_as_pil

    img = open_as_pil(path)

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
        db = _get_db()
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


def _handle_decode_enqueue_missing(params: dict) -> dict:
    """Enqueue decode jobs for every image not present in the decoded cache.

    Params (all optional):
        limit (int): cap the number of images enqueued in this call.
        raw_first (bool): prioritise RAW formats first (default True).

    Returns counters describing how many images were missing, fed, and
    already cached / queued.
    """
    limit = params.get("limit")
    raw_first = bool(params.get("raw_first", True))
    try:
        limit_int = int(limit) if limit is not None else None
    except (TypeError, ValueError):
        limit_int = None

    store = _get_decoded_store()
    cached_ids = store.cached_image_ids()

    from imganalyzer.db.connection import create_connection
    conn = create_connection(busy_timeout_ms=_DB_BUSY_TIMEOUT_MS)
    try:
        rows = conn.execute(
            "SELECT id, file_path FROM images ORDER BY id"
        ).fetchall()
    finally:
        conn.close()

    total_images = len(rows)
    missing: list[tuple[int, str]] = [
        (int(r["id"]), str(r["file_path"]))
        for r in rows
        if int(r["id"]) not in cached_ids
    ]
    missing_count = len(missing)
    if limit_int is not None and limit_int >= 0:
        missing = missing[:limit_int]

    fed = 0
    if missing:
        decoder = _get_pre_decoder()
        fed = decoder.feed(missing, raw_first=raw_first)

    sys.stderr.write(
        f"[server] decode/enqueue_missing: total={total_images}"
        f" cached={len(cached_ids)} missing={missing_count}"
        f" requested={len(missing)} fed={fed}\n"
    )
    return {
        "total_images": total_images,
        "cached": len(cached_ids),
        "missing": missing_count,
        "requested": len(missing),
        "fed": fed,
    }


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
    has_occurrences = repo.has_face_occurrences()
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


def _handle_faces_person_similar_images(params: dict) -> dict:
    """Find images containing faces similar to a person's identity."""
    from imganalyzer.db.repository import Repository

    person_id = int(params["person_id"])
    limit = int(params.get("limit", 100))
    limit = max(1, min(limit, 500))
    min_similarity_raw = params.get("min_similarity", 0.35)
    try:
        min_similarity = float(min_similarity_raw)
    except (TypeError, ValueError):
        min_similarity = 0.35
    min_similarity = max(0.0, min(min_similarity, 1.0))

    conn = _get_db()
    repo = Repository(conn)
    images = repo.find_similar_images_for_person(
        person_id,
        limit=limit,
        min_similarity=min_similarity,
    )
    return {"images": images}


def _handle_faces_person_link_occurrences(params: dict) -> dict:
    """Link specific face occurrences to a person (direct link, no cluster required)."""
    from imganalyzer.db.repository import Repository

    person_id = int(params["person_id"])
    occurrence_ids = [int(x) for x in params.get("occurrence_ids", [])]
    if not occurrence_ids:
        return {"ok": False, "updated": 0, "error": "No occurrence IDs provided"}

    conn = _get_db()
    repo = Repository(conn)
    updated = repo.link_occurrences_to_person(occurrence_ids, person_id)
    return {"ok": True, "updated": updated}


def _handle_faces_person_unlink_occurrence(params: dict) -> dict:
    """Unlink a single face occurrence from its person (clear person_id)."""
    from imganalyzer.db.repository import Repository

    occurrence_id = int(params["occurrence_id"])
    conn = _get_db()
    repo = Repository(conn)
    updated = repo.unlink_occurrence_from_person(occurrence_id)
    return {"ok": True, "updated": updated}


def _handle_faces_person_direct_links(params: dict) -> dict:
    """Get face occurrences directly linked to a person (not via cluster)."""
    from imganalyzer.db.repository import Repository

    person_id = int(params["person_id"])
    conn = _get_db()
    repo = Repository(conn)
    links = repo.get_person_direct_links(person_id)
    return {"links": links}


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
    from imganalyzer.readers import open_as_pil
    from imganalyzer.readers.standard import pillow_decode_guard, register_optional_pillow_opener
    from imganalyzer.analysis.ai.faces import _get_pil_exif_orientation

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    ext = path.suffix.lower()
    register_optional_pillow_opener(path)
    if ext in _RAW_FACE_CROP_EXTS:
        return open_as_pil(path), 1  # RAW files have no EXIF orientation

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
    from imganalyzer.db.connection import create_connection
    from imganalyzer.db.repository import Repository

    threshold = params.get("threshold", 0.55)

    _send_result(req_id, {"started": True})

    def _run_clustering() -> None:
        try:
            conn = create_connection(busy_timeout_ms=_DB_BUSY_TIMEOUT_MS)
            repo = Repository(conn)

            def _progress(phase: str, fraction: float, n_clusters: int) -> None:
                _send_notification("faces/clustering-progress", {
                    "phase": phase,
                    "fraction": fraction,
                    "numClusters": n_clusters,
                })

            num_clusters = repo.cluster_faces(
                threshold=threshold, progress_cb=_progress,
            )
            conn.close()
            _send_notification("faces/clustering-done", {"num_clusters": num_clusters})
        except Exception as exc:
            _send_notification("faces/clustering-done", {"error": str(exc)})

    t = threading.Thread(target=_run_clustering, daemon=True)
    t.start()


def _handle_faces_crop_batch_async(req_id: int | str, params: dict) -> None:
    """Async wrapper — runs crop-batch in a thread to avoid blocking the main loop.

    Opening original images from NAS/network paths can take seconds per file.
    With many missing thumbnails this would block the stdin loop for 60+ seconds,
    causing status-poll timeouts.
    """
    try:
        result = _handle_faces_crop_batch(params)
        _send_result(req_id, result)
    except Exception as exc:
        _send_error(req_id, -1, str(exc))


def _run_async_one_shot(
    req_id: int | str,
    params: dict,
    handler: Callable[[dict], Any],
) -> None:
    try:
        _send_result(req_id, handler(params))
    except Exception as exc:
        _send_error(req_id, -1, str(exc))


def _handle_thumbnail_async(req_id: int | str, params: dict) -> None:
    _run_async_one_shot(req_id, params, _handle_thumbnail)


def _handle_thumbnails_batch_async(req_id: int | str, params: dict) -> None:
    _run_async_one_shot(req_id, params, _handle_thumbnails_batch)


def _handle_fullimage_async(req_id: int | str, params: dict) -> None:
    _run_async_one_shot(req_id, params, _handle_fullimage)


def _handle_gallery_list_images_chunk_async(req_id: int | str, params: dict) -> None:
    _run_async_one_shot(req_id, params, _handle_gallery_list_images_chunk)


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
                    thumbnails[str(row_d["id"])] = base64.b64encode(thumbnail).decode("ascii")
                    # Persist thumbnail cache opportunistically; if DB is briefly
                    # locked we still return thumbnail to caller and retry shortly.
                    delay_s = _LOCK_RETRY_INITIAL_DELAY_S
                    for attempt in range(1, _LOCK_RETRY_ATTEMPTS + 1):
                        try:
                            repo.set_face_occurrence_thumbnail(row_d["id"], thumbnail)
                            updated = True
                            break
                        except Exception as exc:
                            if not _is_transient_db_lock_error(exc) or attempt >= _LOCK_RETRY_ATTEMPTS:
                                raise
                            time.sleep(delay_s)
                            delay_s = min(delay_s * 2, 1.0)
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

    image_id_value = int(row["image_id"])
    has_face_occurrences = _table_exists(conn, "face_occurrences")
    face_clusters = (
        _get_face_clusters_for_images(conn, [image_id_value]).get(image_id_value, [])
        if has_face_occurrences
        else None
    )

    record = {
        "image_id": image_id_value,
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
        "face_clusters": face_clusters,
    }
    return {"result": record}


# ── Geo/map handlers ────────────────────────────────────────────────────────


def _zoom_to_geohash_precision(zoom: int, detail: int = 0) -> int:
    """Map a Leaflet zoom level to a geohash prefix length for clustering."""
    if zoom <= 2:
        precision = 1
    elif zoom <= 5:
        precision = 2
    elif zoom <= 9:
        precision = 3
    elif zoom <= 13:
        precision = 4
    elif zoom <= 16:
        precision = 5
    elif zoom <= 18:
        precision = 6
    else:
        precision = 7
    return max(1, min(8, precision + detail))


def _handle_geo_clusters(params: dict) -> dict:
    """Return server-side clustered image markers for a map viewport.

    Params:
        north, south, east, west (float): bounding box
        zoom (int): Leaflet zoom level (1-20)
        detail (int): clustering detail boost (-2 to +4, default 0)
        limit (int): max clusters to return (default 500)
    Returns:
        clusters: [{cell, center_lat, center_lng, count, sample_ids}]
        total: total geotagged images in bounds
    """
    conn = _get_db()
    north = float(params.get("north", 90))
    south = float(params.get("south", -90))
    east = float(params.get("east", 180))
    west = float(params.get("west", -180))
    zoom = int(params.get("zoom", 2))
    detail = max(-2, min(4, int(params.get("detail", 0))))
    limit = min(int(params.get("limit", 500)), 2000)

    precision = _zoom_to_geohash_precision(zoom, detail)

    # Use R*tree for fast bounding-box lookup, then cluster by geohash prefix
    try:
        rows = conn.execute(
            """
            SELECT substr(m.geohash, 1, ?) AS cell,
                   AVG(m.gps_latitude) AS center_lat,
                   AVG(m.gps_longitude) AS center_lng,
                   COUNT(*) AS count,
                   MIN(m.image_id) AS sample1
            FROM geo_rtree r
            JOIN analysis_metadata m ON m.image_id = r.id
            WHERE r.min_lat >= ? AND r.max_lat <= ?
              AND r.min_lng >= ? AND r.max_lng <= ?
              AND m.geohash IS NOT NULL
            GROUP BY cell
            ORDER BY count DESC
            LIMIT ?
            """,
            [precision, south, north, west, east, limit],
        ).fetchall()
    except Exception:
        # geo_rtree may not exist (pre-v30)
        return {"clusters": [], "total": 0}

    clusters = []
    total = 0
    for row in rows:
        count = row["count"]
        total += count
        clusters.append({
            "cell": row["cell"],
            "center_lat": round(row["center_lat"], 6),
            "center_lng": round(row["center_lng"], 6),
            "count": count,
            "sample_ids": [row["sample1"]],
        })

    return {"clusters": clusters, "total": total}


def _spread_by_date(items: list, limit: int, random_mod) -> list:
    """Pick up to *limit* items spread evenly across date_time_original."""
    if not items or limit <= 0:
        return []
    if len(items) <= limit:
        return list(items)

    dated = [c for c in items if c["date"]]
    undated = [c for c in items if not c["date"]]
    random_mod.shuffle(undated)

    if not dated:
        return random_mod.sample(items, limit)

    random_mod.shuffle(dated)  # randomize within-date ties
    dated.sort(key=lambda r: r["date"])

    if len(dated) <= limit:
        picked = dated
        remaining = limit - len(picked)
        if remaining > 0 and undated:
            picked.extend(undated[:remaining])
        return picked

    step = len(dated) / limit
    return [dated[int(i * step)] for i in range(limit)]


def _handle_geo_cluster_preview(params: dict) -> dict:
    """Return representative preview images for a geohash cluster cell.

    Selection algorithm:
      1. High-aesthetic images (score > 7.5) are prioritised.
      2. If fewer than `limit` high-quality images exist, remaining slots
         are filled from other images, spread evenly across the date range.
      3. Always returns min(total, limit) images.

    Performance: uses index-friendly LIKE prefix and lightweight ID+date query
    to avoid fetching all rows (e.g. 10K) with full JOINs.

    Params:
        cell (str): geohash cell prefix (from geo/clusters response)
        limit (int): max images to return (default 10)
    Returns:
        images: [{image_id, file_path, date, aesthetic_score}]
        total: total images in the cell
    """
    import random

    conn = _get_db()
    cell = str(params.get("cell", "")).strip()
    if not cell:
        return {"images": [], "total": 0, "error": "cell is required"}
    limit = min(int(params.get("limit", 10)), 50)
    like_prefix = cell + "%"

    try:
        # Lightweight query: only image_id, date, aesthetic score (no file_path JOIN)
        rows = conn.execute(
            """
            SELECT m.image_id,
                   m.date_time_original AS date,
                   COALESCE(ap.perception_iaa, ae.aesthetic_score) AS score
            FROM analysis_metadata m
            LEFT JOIN analysis_perception ap ON ap.image_id = m.image_id
            LEFT JOIN analysis_aesthetic ae ON ae.image_id = m.image_id
            WHERE m.geohash LIKE ?
            """,
            [like_prefix],
        ).fetchall()
    except Exception as exc:
        return {"images": [], "total": 0, "error": str(exc)}

    total = len(rows)
    if total == 0:
        return {"images": [], "total": 0}

    # Step 1: prioritise high-aesthetic images, but always fill to `limit`
    high_quality = [r for r in rows if r["score"] is not None and r["score"] > 7.5]
    hq_set = set(id(r) for r in high_quality)
    others = [r for r in rows if id(r) not in hq_set]

    if len(high_quality) >= limit:
        # Enough high-quality — spread those evenly across dates
        candidates = high_quality
    else:
        # Take all high-quality, fill remaining from others
        candidates = list(high_quality)
        needed = limit - len(candidates)
        candidates.extend(_spread_by_date(others, needed, random))
    # Now pick `limit` from candidates, spread by date
    picked = _spread_by_date(candidates, limit, random)

    # Fetch full details (file_path) only for the selected IDs
    selected_ids = [r["image_id"] for r in picked]
    score_map = {r["image_id"]: r["score"] for r in picked}
    date_map = {r["image_id"]: r["date"] for r in picked}
    placeholders = ",".join("?" * len(selected_ids))
    full_rows = conn.execute(
        f"SELECT id, file_path FROM images WHERE id IN ({placeholders})",
        selected_ids,
    ).fetchall()

    images = [
        {
            "image_id": r["id"],
            "file_path": r["file_path"],
            "date": date_map.get(r["id"]),
            "aesthetic_score": round(score_map[r["id"]], 2)
            if score_map.get(r["id"]) is not None
            else None,
        }
        for r in full_rows
    ]

    return {"images": images, "total": total}


def _handle_geo_nearby(params: dict) -> dict:
    """Return images near a given coordinate.

    Params:
        lat (float): center latitude
        lng (float): center longitude
        radiusKm (float): search radius in km (default 1.0)
        limit (int): max results (default 50)
        excludeId (int|None): image_id to exclude (the current photo)
    Returns:
        images: [{image_id, gps_latitude, gps_longitude, file_path}]
        total: count of all images in radius
    """
    conn = _get_db()
    lat = float(params["lat"])
    lng = float(params["lng"])
    radius_km = float(params.get("radiusKm", 1.0))
    limit = min(int(params.get("limit", 50)), 200)
    exclude_id = params.get("excludeId")

    # Approximate bounding box from radius (1 degree ≈ 111 km)
    import math

    dlat = radius_km / 111.0
    cos_lat = max(abs(math.cos(math.radians(lat))), 0.01)
    dlng = radius_km / (111.0 * cos_lat)

    north = lat + dlat
    south = lat - dlat
    east = lng + dlng
    west = lng - dlng

    try:
        query = """
            SELECT m.image_id, m.gps_latitude, m.gps_longitude, i.file_path
            FROM geo_rtree r
            JOIN analysis_metadata m ON m.image_id = r.id
            JOIN images i ON i.id = m.image_id
            WHERE r.min_lat >= ? AND r.max_lat <= ?
              AND r.min_lng >= ? AND r.max_lng <= ?
        """
        query_params: list[Any] = [south, north, west, east]

        if exclude_id is not None:
            query += " AND m.image_id != ?"
            query_params.append(int(exclude_id))

        count_row = conn.execute(
            f"SELECT COUNT(*) AS cnt FROM ({query})", query_params
        ).fetchone()
        total = int(count_row["cnt"]) if count_row else 0

        query += " LIMIT ?"
        query_params.append(limit)
        rows = conn.execute(query, query_params).fetchall()
    except Exception:
        return {"images": [], "total": 0}

    images = [
        {
            "image_id": row["image_id"],
            "gps_latitude": round(row["gps_latitude"], 6),
            "gps_longitude": round(row["gps_longitude"], 6),
            "file_path": row["file_path"],
        }
        for row in rows
    ]

    return {"images": images, "total": total}


def _handle_geo_stats(params: dict) -> dict:
    """Return aggregate GPS/location statistics.

    Returns:
        total_images: total image count
        geotagged: count with GPS data
        countries: [{country, count}]
        top_cities: [{city, state, country, count}]
    """
    conn = _get_db()
    try:
        total_row = conn.execute("SELECT COUNT(*) AS cnt FROM images").fetchone()
        total_images = int(total_row["cnt"]) if total_row else 0

        geo_row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM analysis_metadata "
            "WHERE gps_latitude IS NOT NULL AND gps_longitude IS NOT NULL"
        ).fetchone()
        geotagged = int(geo_row["cnt"]) if geo_row else 0

        country_rows = conn.execute(
            "SELECT location_country AS country, COUNT(*) AS count "
            "FROM analysis_metadata "
            "WHERE location_country IS NOT NULL AND location_country != '' "
            "GROUP BY location_country ORDER BY count DESC LIMIT 50"
        ).fetchall()
        countries = [{"country": r["country"], "count": r["count"]} for r in country_rows]

        city_rows = conn.execute(
            "SELECT location_city AS city, location_state AS state, "
            "       location_country AS country, COUNT(*) AS count "
            "FROM analysis_metadata "
            "WHERE location_city IS NOT NULL AND location_city != '' "
            "GROUP BY location_city, location_state, location_country "
            "ORDER BY count DESC LIMIT 20"
        ).fetchall()
        top_cities = [
            {"city": r["city"], "state": r["state"], "country": r["country"], "count": r["count"]}
            for r in city_rows
        ]
    except Exception:
        return {"total_images": 0, "geotagged": 0, "countries": [], "top_cities": []}

    return {
        "total_images": total_images,
        "geotagged": geotagged,
        "countries": countries,
        "top_cities": top_cities,
    }


def _handle_geo_heatmap(params: dict) -> dict:
    """Return density grid for heatmap visualization.

    Params:
        north, south, east, west (float): bounding box
        zoom (int): Leaflet zoom level
    Returns:
        points: [{lat, lng, weight}] — grid cells with aggregated counts
    """
    conn = _get_db()
    north = float(params.get("north", 90))
    south = float(params.get("south", -90))
    east = float(params.get("east", 180))
    west = float(params.get("west", -180))
    zoom = int(params.get("zoom", 2))

    precision = _zoom_to_geohash_precision(zoom)

    try:
        rows = conn.execute(
            """
            SELECT substr(m.geohash, 1, ?) AS cell,
                   AVG(m.gps_latitude) AS lat,
                   AVG(m.gps_longitude) AS lng,
                   COUNT(*) AS weight
            FROM geo_rtree r
            JOIN analysis_metadata m ON m.image_id = r.id
            WHERE r.min_lat >= ? AND r.max_lat <= ?
              AND r.min_lng >= ? AND r.max_lng <= ?
              AND m.geohash IS NOT NULL
            GROUP BY cell
            LIMIT 2000
            """,
            [precision, south, north, west, east],
        ).fetchall()
    except Exception:
        return {"points": []}

    points = [
        {"lat": round(r["lat"], 6), "lng": round(r["lng"], 6), "weight": r["weight"]}
        for r in rows
    ]
    return {"points": points}


def _handle_geo_stats_extended(params: dict) -> dict:
    """Return rich location statistics for the stats dashboard.

    Extends geo/stats with monthly activity, location diversity, camera
    distribution by country, furthest-from-home, and GPS source breakdown.

    Params (all optional):
        home_lat, home_lng (float): user's home coordinates for distance calc
    """
    import math

    conn = _get_db()
    result: dict[str, Any] = {}

    try:
        # ── Basic counts (same as geo/stats) ─────────────────────────────
        total_row = conn.execute("SELECT COUNT(*) AS cnt FROM images").fetchone()
        result["total_images"] = int(total_row["cnt"]) if total_row else 0

        geo_row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM analysis_metadata "
            "WHERE gps_latitude IS NOT NULL AND gps_longitude IS NOT NULL"
        ).fetchone()
        result["geotagged"] = int(geo_row["cnt"]) if geo_row else 0

        # ── GPS source breakdown ─────────────────────────────────────────
        try:
            source_rows = conn.execute(
                "SELECT COALESCE(gps_source, 'exif') AS source, COUNT(*) AS count "
                "FROM analysis_metadata "
                "WHERE gps_latitude IS NOT NULL "
                "GROUP BY source ORDER BY count DESC"
            ).fetchall()
            result["gps_sources"] = [
                {"source": r["source"], "count": r["count"]} for r in source_rows
            ]
        except Exception:
            # gps_source column may not exist yet (pre-v31)
            result["gps_sources"] = [{"source": "exif", "count": result["geotagged"]}]

        # ── Countries ────────────────────────────────────────────────────
        country_rows = conn.execute(
            "SELECT location_country AS country, COUNT(*) AS count "
            "FROM analysis_metadata "
            "WHERE location_country IS NOT NULL AND location_country != '' "
            "GROUP BY location_country ORDER BY count DESC LIMIT 50"
        ).fetchall()
        result["countries"] = [
            {"country": r["country"], "count": r["count"]} for r in country_rows
        ]

        # ── Top cities ───────────────────────────────────────────────────
        city_rows = conn.execute(
            "SELECT location_city AS city, location_state AS state, "
            "       location_country AS country, COUNT(*) AS count "
            "FROM analysis_metadata "
            "WHERE location_city IS NOT NULL AND location_city != '' "
            "GROUP BY location_city, location_state, location_country "
            "ORDER BY count DESC LIMIT 20"
        ).fetchall()
        result["top_cities"] = [
            {"city": r["city"], "state": r["state"],
             "country": r["country"], "count": r["count"]}
            for r in city_rows
        ]

        # ── Monthly activity (images per month) ──────────────────────────
        month_rows = conn.execute(
            "SELECT substr(m.date_time_original, 1, 7) AS month, COUNT(*) AS count "
            "FROM analysis_metadata m "
            "WHERE m.date_time_original IS NOT NULL "
            "  AND m.gps_latitude IS NOT NULL "
            "GROUP BY month ORDER BY month"
        ).fetchall()
        result["monthly_activity"] = [
            {"month": r["month"], "count": r["count"]}
            for r in month_rows if r["month"]
        ]

        # ── Location diversity (unique geohash-4 cells per month) ────────
        diversity_rows = conn.execute(
            "SELECT substr(m.date_time_original, 1, 7) AS month, "
            "       COUNT(DISTINCT substr(m.geohash, 1, 4)) AS unique_places "
            "FROM analysis_metadata m "
            "WHERE m.date_time_original IS NOT NULL "
            "  AND m.geohash IS NOT NULL "
            "GROUP BY month ORDER BY month"
        ).fetchall()
        result["location_diversity"] = [
            {"month": r["month"], "unique_places": r["unique_places"]}
            for r in diversity_rows if r["month"]
        ]

        # ── Camera × country distribution ────────────────────────────────
        cam_rows = conn.execute(
            "SELECT m.location_country AS country, m.camera_model AS camera, "
            "       COUNT(*) AS count "
            "FROM analysis_metadata m "
            "WHERE m.location_country IS NOT NULL AND m.location_country != '' "
            "  AND m.camera_model IS NOT NULL AND m.camera_model != '' "
            "GROUP BY m.location_country, m.camera_model "
            "ORDER BY count DESC LIMIT 100"
        ).fetchall()
        result["camera_by_country"] = [
            {"country": r["country"], "camera": r["camera"], "count": r["count"]}
            for r in cam_rows
        ]

        # ── Top 10 locations (geohash-5 ~1.2km clusters) ────────────────
        top_loc_rows = conn.execute(
            "SELECT substr(m.geohash, 1, 5) AS cell, "
            "       AVG(m.gps_latitude) AS lat, AVG(m.gps_longitude) AS lng, "
            "       COUNT(*) AS count, "
            "       MIN(m.location_city) AS city, "
            "       MIN(m.location_state) AS state, "
            "       MIN(m.location_country) AS country "
            "FROM analysis_metadata m "
            "WHERE m.geohash IS NOT NULL "
            "GROUP BY cell ORDER BY count DESC LIMIT 10"
        ).fetchall()
        result["top_locations"] = [
            {"cell": r["cell"], "lat": round(r["lat"], 6), "lng": round(r["lng"], 6),
             "count": r["count"], "city": r["city"], "state": r["state"],
             "country": r["country"]}
            for r in top_loc_rows
        ]

        # ── Furthest from home ───────────────────────────────────────────
        home_lat = params.get("home_lat")
        home_lng = params.get("home_lng")
        if home_lat is not None and home_lng is not None:
            home_lat, home_lng = float(home_lat), float(home_lng)

            def _haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
                R = 6371.0  # km
                dlat = math.radians(lat2 - lat1)
                dlng = math.radians(lng2 - lng1)
                a = (math.sin(dlat / 2) ** 2 +
                     math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
                     math.sin(dlng / 2) ** 2)
                return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

            # Sample up to 10k distant candidates from rtree extremes
            far_rows = conn.execute(
                "SELECT m.image_id, m.gps_latitude AS lat, m.gps_longitude AS lng, "
                "       i.file_path "
                "FROM analysis_metadata m "
                "JOIN images i ON i.id = m.image_id "
                "WHERE m.gps_latitude IS NOT NULL "
                "ORDER BY (ABS(m.gps_latitude - ?) + ABS(m.gps_longitude - ?)) DESC "
                "LIMIT 200",
                [home_lat, home_lng],
            ).fetchall()
            best = None
            best_dist = 0.0
            for r in far_rows:
                d = _haversine(home_lat, home_lng, r["lat"], r["lng"])
                if d > best_dist:
                    best_dist = d
                    best = r
            if best:
                result["furthest_from_home"] = {
                    "image_id": best["image_id"],
                    "file_path": best["file_path"],
                    "lat": round(best["lat"], 6),
                    "lng": round(best["lng"], 6),
                    "distance_km": round(best_dist, 1),
                }
            else:
                result["furthest_from_home"] = None
        else:
            result["furthest_from_home"] = None

    except Exception as exc:
        import traceback
        sys.stderr.write(f"[geo/stats-extended] error: {traceback.format_exc()}\n")
        return {"error": str(exc)}

    return result


def _handle_geo_gap_filler_preview(params: dict) -> dict:
    """Scan for images missing GPS and estimate locations from temporal neighbors.

    Params:
        max_gap_minutes (int): max time gap for interpolation (default 60)
        preview_limit (int): max preview rows to return (default 100)
    Returns:
        fillable, total_missing, previews[...]
    """
    import math
    from imganalyzer.db.geohash import encode as geohash_encode

    conn = _get_db()
    max_gap = int(params.get("max_gap_minutes", 60))
    preview_limit = int(params.get("preview_limit", 100))

    try:
        # Images without GPS that have a timestamp
        missing_rows = conn.execute(
            "SELECT m.image_id, m.date_time_original AS dto, i.file_path "
            "FROM analysis_metadata m "
            "JOIN images i ON i.id = m.image_id "
            "WHERE (m.gps_latitude IS NULL OR m.gps_longitude IS NULL) "
            "  AND m.date_time_original IS NOT NULL "
            "ORDER BY m.date_time_original"
        ).fetchall()
        total_missing = len(missing_rows)

        # All geotagged images ordered by time (for temporal neighbor search)
        geo_rows = conn.execute(
            "SELECT m.image_id, m.date_time_original AS dto, "
            "       m.gps_latitude AS lat, m.gps_longitude AS lng "
            "FROM analysis_metadata m "
            "WHERE m.gps_latitude IS NOT NULL AND m.gps_longitude IS NOT NULL "
            "  AND m.date_time_original IS NOT NULL "
            "ORDER BY m.date_time_original"
        ).fetchall()

        if not geo_rows:
            return {"fillable": 0, "total_missing": total_missing, "previews": []}

        geo_times = []
        for r in geo_rows:
            try:
                from datetime import datetime as _dt
                t = _dt.fromisoformat(r["dto"].replace("Z", "+00:00"))
                geo_times.append((t, r["lat"], r["lng"], r["image_id"]))
            except (ValueError, TypeError):
                continue

        # Check existing overrides
        override_ids = set()
        try:
            ov_rows = conn.execute(
                "SELECT DISTINCT image_id FROM overrides "
                "WHERE table_name = 'analysis_metadata' "
                "  AND field_name IN ('gps_latitude', 'gps_longitude')"
            ).fetchall()
            override_ids = {r["image_id"] for r in ov_rows}
        except Exception:
            pass

        fillable = 0
        previews: list[dict] = []

        import bisect
        from datetime import datetime as _dt

        geo_timestamps = [gt[0] for gt in geo_times]

        for row in missing_rows:
            if row["image_id"] in override_ids:
                continue
            try:
                t = _dt.fromisoformat(row["dto"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                continue

            # Binary search for nearest neighbors
            idx = bisect.bisect_left(geo_timestamps, t)
            before = geo_times[idx - 1] if idx > 0 else None
            after = geo_times[idx] if idx < len(geo_times) else None

            before_gap = (t - before[0]).total_seconds() / 60 if before else float("inf")
            after_gap = (after[0] - t).total_seconds() / 60 if after else float("inf")

            if before_gap > max_gap and after_gap > max_gap:
                continue

            # Interpolate
            if before_gap <= max_gap and after_gap <= max_gap:
                total_gap = before_gap + after_gap
                if total_gap < 0.01:
                    lat = before[1]
                    lng = before[2]
                else:
                    w_after = before_gap / total_gap
                    lat = before[1] + w_after * (after[1] - before[1])
                    lng = before[2] + w_after * (after[2] - before[2])
                gap_min = total_gap
                confidence = max(0.1, 1.0 - (gap_min / max_gap))
            elif before_gap <= max_gap:
                lat, lng = before[1], before[2]
                gap_min = before_gap
                confidence = max(0.1, 0.7 * (1.0 - (gap_min / max_gap)))
            else:
                lat, lng = after[1], after[2]
                gap_min = after_gap
                confidence = max(0.1, 0.7 * (1.0 - (gap_min / max_gap)))

            fillable += 1
            if len(previews) < preview_limit:
                preview: dict[str, Any] = {
                    "image_id": row["image_id"],
                    "file_path": row["file_path"],
                    "inferred_lat": round(lat, 6),
                    "inferred_lng": round(lng, 6),
                    "confidence": round(confidence, 3),
                }
                if before and before_gap <= max_gap:
                    preview["nearest_before"] = {
                        "image_id": before[3],
                        "gap_minutes": round(before_gap, 1),
                    }
                if after and after_gap <= max_gap:
                    preview["nearest_after"] = {
                        "image_id": after[3],
                        "gap_minutes": round(after_gap, 1),
                    }
                previews.append(preview)

    except Exception as exc:
        import traceback
        sys.stderr.write(f"[geo/gap-filler-preview] error: {traceback.format_exc()}\n")
        return {"error": str(exc)}

    return {"fillable": fillable, "total_missing": total_missing, "previews": previews}


def _handle_geo_gap_filler_apply(params: dict) -> dict:
    """Apply GPS gap filling — write inferred coordinates to the database.

    Params:
        max_gap_minutes (int): max time gap (default 60)
        min_confidence (float): minimum confidence threshold (default 0.5)
    Returns:
        filled, skipped_override, skipped_low_confidence
    """
    from imganalyzer.db.geohash import encode as geohash_encode
    from imganalyzer.db.connection import begin_immediate

    conn = _get_db()
    max_gap = int(params.get("max_gap_minutes", 60))
    min_confidence = float(params.get("min_confidence", 0.5))

    try:
        # Re-run the same analysis as preview
        preview_result = _handle_geo_gap_filler_preview({
            "max_gap_minutes": max_gap,
            "preview_limit": 999_999_999,  # get all fillable
        })
        if "error" in preview_result:
            return preview_result

        # Check overrides
        override_ids = set()
        try:
            ov_rows = conn.execute(
                "SELECT DISTINCT image_id FROM overrides "
                "WHERE table_name = 'analysis_metadata' "
                "  AND field_name IN ('gps_latitude', 'gps_longitude')"
            ).fetchall()
            override_ids = {r["image_id"] for r in ov_rows}
        except Exception:
            pass

        filled = 0
        skipped_override = 0
        skipped_low_confidence = 0

        begin_immediate(conn)
        try:
            for p in preview_result["previews"]:
                if p["image_id"] in override_ids:
                    skipped_override += 1
                    continue
                if p["confidence"] < min_confidence:
                    skipped_low_confidence += 1
                    continue

                lat = p["inferred_lat"]
                lng = p["inferred_lng"]
                gh = geohash_encode(lat, lng, precision=8)

                conn.execute(
                    "UPDATE analysis_metadata "
                    "SET gps_latitude = ?, gps_longitude = ?, geohash = ?, "
                    "    gps_source = 'inferred' "
                    "WHERE image_id = ?",
                    [lat, lng, gh, p["image_id"]],
                )

                # Update R*tree
                conn.execute(
                    "INSERT OR REPLACE INTO geo_rtree (id, min_lat, max_lat, min_lng, max_lng) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [p["image_id"], lat, lat, lng, lng],
                )

                filled += 1
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    except Exception as exc:
        import traceback
        sys.stderr.write(f"[geo/gap-filler-apply] error: {traceback.format_exc()}\n")
        return {"error": str(exc)}

    return {
        "filled": filled,
        "skipped_override": skipped_override,
        "skipped_low_confidence": skipped_low_confidence,
    }


def _rdp_simplify(
    points: list[tuple[float, float]], epsilon: float
) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker line simplification (pure Python)."""
    if len(points) <= 2:
        return points

    # Find the point farthest from the line between first and last
    start, end = points[0], points[-1]
    max_dist = 0.0
    max_idx = 0
    dx, dy = end[0] - start[0], end[1] - start[1]
    line_len_sq = dx * dx + dy * dy

    for i in range(1, len(points) - 1):
        px, py = points[i][0] - start[0], points[i][1] - start[1]
        if line_len_sq > 0:
            # perpendicular distance
            dist = abs(px * dy - py * dx) / (line_len_sq**0.5)
        else:
            dist = (px * px + py * py) ** 0.5
        if dist > max_dist:
            max_dist = dist
            max_idx = i

    if max_dist > epsilon:
        left = _rdp_simplify(points[: max_idx + 1], epsilon)
        right = _rdp_simplify(points[max_idx:], epsilon)
        return left[:-1] + right
    else:
        return [start, end]


def _handle_geo_trip_detect(params: dict) -> dict:
    """Auto-detect trips from geotagged photo sequences.

    A "trip" is a sequence of photos where GPS moves >10km from the starting
    area, with gaps >4h between trip segments.

    Params:
        min_images (int): minimum photos per trip (default 5)
    Returns:
        trips: [{start_date, end_date, start_location, end_location,
                 image_count, distance_km}]
    """
    import math
    from datetime import datetime as _dt, timedelta

    conn = _get_db()
    min_images = int(params.get("min_images", 5))

    try:
        rows = conn.execute(
            "SELECT m.image_id, m.date_time_original AS dto, "
            "       m.gps_latitude AS lat, m.gps_longitude AS lng, "
            "       m.location_city AS city, m.location_country AS country "
            "FROM analysis_metadata m "
            "WHERE m.gps_latitude IS NOT NULL AND m.gps_longitude IS NOT NULL "
            "  AND m.date_time_original IS NOT NULL "
            "ORDER BY m.date_time_original"
        ).fetchall()

        if not rows:
            return {"trips": []}

        def _haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
            R = 6371.0
            dlat = math.radians(lat2 - lat1)
            dlng = math.radians(lng2 - lng1)
            a = (math.sin(dlat / 2) ** 2 +
                 math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
                 math.sin(dlng / 2) ** 2)
            return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        # Parse timestamps and build image records
        images: list[dict] = []
        for r in rows:
            try:
                t = _dt.fromisoformat(r["dto"].replace("Z", "+00:00"))
                images.append({
                    "id": r["image_id"], "t": t,
                    "lat": r["lat"], "lng": r["lng"],
                    "city": r["city"], "country": r["country"],
                })
            except (ValueError, TypeError):
                continue

        if len(images) < min_images:
            return {"trips": []}

        # Segment into trips: split on time gaps > 4h
        gap_threshold = timedelta(hours=4)
        segments: list[list[dict]] = []
        current_seg: list[dict] = [images[0]]
        for i in range(1, len(images)):
            gap = images[i]["t"] - images[i - 1]["t"]
            if gap > gap_threshold:
                segments.append(current_seg)
                current_seg = []
            current_seg.append(images[i])
        if current_seg:
            segments.append(current_seg)

        # Filter: keep segments with enough images AND significant movement (>10km)
        trips: list[dict] = []
        for seg in segments:
            if len(seg) < min_images:
                continue
            # Compute total distance
            total_dist = 0.0
            for i in range(1, len(seg)):
                total_dist += _haversine(
                    seg[i - 1]["lat"], seg[i - 1]["lng"],
                    seg[i]["lat"], seg[i]["lng"],
                )
            max_dist = _haversine(
                seg[0]["lat"], seg[0]["lng"],
                seg[-1]["lat"], seg[-1]["lng"],
            )
            # Only include if there's meaningful movement (>10km total displacement)
            # or significant path distance (>20km)
            if max_dist < 10 and total_dist < 20:
                continue

            start_loc = seg[0]["city"] or seg[0]["country"] or (
                f"{seg[0]['lat']:.2f}, {seg[0]['lng']:.2f}"
            )
            end_loc = seg[-1]["city"] or seg[-1]["country"] or (
                f"{seg[-1]['lat']:.2f}, {seg[-1]['lng']:.2f}"
            )
            trips.append({
                "start_date": seg[0]["t"].isoformat(),
                "end_date": seg[-1]["t"].isoformat(),
                "start_location": start_loc,
                "end_location": end_loc,
                "image_count": len(seg),
                "distance_km": round(total_dist, 1),
            })

    except Exception as exc:
        import traceback
        sys.stderr.write(f"[geo/trip-detect] error: {traceback.format_exc()}\n")
        return {"error": str(exc)}

    return {"trips": trips}


def _handle_geo_trip_timeline(params: dict) -> dict:
    """Return a trip route with stops for timeline visualization.

    Params:
        start_date (str): ISO date start (inclusive)
        end_date (str): ISO date end (inclusive)
        simplify (bool): apply RDP simplification (default True)
    Returns:
        stops: [{lat, lng, start_time, end_time, count, cover_image_id, cover_file_path}]
        route_points: [{lat, lng}]
        total_images: int
    """
    import math
    from datetime import datetime as _dt, timedelta

    conn = _get_db()
    start_date = params.get("start_date", "")
    end_date = params.get("end_date", "")
    simplify = params.get("simplify", True)

    if not start_date or not end_date:
        return {"error": "start_date and end_date are required"}

    try:
        rows = conn.execute(
            "SELECT m.image_id, m.date_time_original AS dto, "
            "       m.gps_latitude AS lat, m.gps_longitude AS lng, "
            "       m.geohash, i.file_path "
            "FROM analysis_metadata m "
            "JOIN images i ON i.id = m.image_id "
            "LEFT JOIN analysis_aesthetic a ON a.image_id = m.image_id "
            "WHERE m.gps_latitude IS NOT NULL AND m.gps_longitude IS NOT NULL "
            "  AND m.date_time_original >= ? AND m.date_time_original <= ? "
            "ORDER BY m.date_time_original",
            [start_date, end_date],
        ).fetchall()

        if not rows:
            return {"stops": [], "route_points": [], "total_images": 0}

        # Parse into records
        images: list[dict] = []
        for r in rows:
            try:
                t = _dt.fromisoformat(r["dto"].replace("Z", "+00:00"))
                images.append({
                    "id": r["image_id"], "t": t,
                    "lat": r["lat"], "lng": r["lng"],
                    "gh6": (r["geohash"] or "")[:6],
                    "file_path": r["file_path"],
                })
            except (ValueError, TypeError):
                continue

        total_images = len(images)
        if not images:
            return {"stops": [], "route_points": [], "total_images": 0}

        # Group into stops: same geohash-6 (~150m) AND within 30 min
        stop_gap = timedelta(minutes=30)
        stops_raw: list[list[dict]] = []
        current_stop: list[dict] = [images[0]]

        for i in range(1, len(images)):
            same_area = images[i]["gh6"] == images[i - 1]["gh6"]
            close_time = (images[i]["t"] - images[i - 1]["t"]) <= stop_gap
            if same_area and close_time:
                current_stop.append(images[i])
            else:
                stops_raw.append(current_stop)
                current_stop = [images[i]]
        if current_stop:
            stops_raw.append(current_stop)

        # Build stop summaries
        stops: list[dict] = []
        for group in stops_raw:
            avg_lat = sum(im["lat"] for im in group) / len(group)
            avg_lng = sum(im["lng"] for im in group) / len(group)
            # Cover image: first image in the stop (could use aesthetic score later)
            cover = group[0]
            stops.append({
                "lat": round(avg_lat, 6),
                "lng": round(avg_lng, 6),
                "start_time": group[0]["t"].isoformat(),
                "end_time": group[-1]["t"].isoformat(),
                "count": len(group),
                "cover_image_id": cover["id"],
                "cover_file_path": cover["file_path"],
            })

        # Build route polyline from all image positions
        route_points = [(im["lat"], im["lng"]) for im in images]

        # Simplify route if requested
        if simplify and len(route_points) > 100:
            # Adaptive epsilon based on route extent
            lats = [p[0] for p in route_points]
            lngs = [p[1] for p in route_points]
            extent = max(max(lats) - min(lats), max(lngs) - min(lngs))
            epsilon = extent * 0.001  # ~0.1% of extent
            route_points = _rdp_simplify(route_points, epsilon)
            # Cap at 1000 points
            if len(route_points) > 1000:
                step = len(route_points) // 1000
                route_points = route_points[::step] + [route_points[-1]]

        route_out = [
            {"lat": round(p[0], 6), "lng": round(p[1], 6)} for p in route_points
        ]

    except Exception as exc:
        import traceback
        sys.stderr.write(f"[geo/trip-timeline] error: {traceback.format_exc()}\n")
        return {"error": str(exc)}

    return {"stops": stops, "route_points": route_out, "total_images": total_images}


def _handle_geo_geocode(params: dict) -> dict:
    """Resolve a location name to coordinates using the image database.

    Computes the centroid of all geotagged images whose city, state, or
    country matches the query text.  No external geocoding API needed.

    Params:
        location (str): location text to resolve (e.g. "Beijing", "California")
    Returns:
        lat, lng (float): centroid of matching images
        count (int): number of images that matched
    """
    conn = _get_db()
    location = str(params.get("location", "")).strip()
    if not location:
        return {"error": "location parameter required"}

    pattern = f"%{location}%"
    try:
        row = conn.execute(
            """
            SELECT AVG(gps_latitude) AS lat, AVG(gps_longitude) AS lng,
                   COUNT(*) AS cnt
            FROM analysis_metadata
            WHERE (location_city LIKE ? OR location_state LIKE ? OR location_country LIKE ?)
              AND gps_latitude IS NOT NULL AND gps_longitude IS NOT NULL
            """,
            [pattern, pattern, pattern],
        ).fetchone()
    except Exception as exc:
        return {"error": str(exc)}

    if not row or not row["cnt"]:
        return {"lat": None, "lng": None, "count": 0}

    return {
        "lat": round(row["lat"], 6),
        "lng": round(row["lng"], 6),
        "count": row["cnt"],
    }


# ── Albums / Storyline handlers ──────────────────────────────────────────────


def _handle_albums_list(_params: dict) -> dict:
    """List all smart albums."""
    from imganalyzer.storyline.albums import list_albums

    conn = _get_db()
    albums = list_albums(conn)
    return {
        "albums": [
            {
                "id": a.id,
                "name": a.name,
                "description": a.description,
                "cover_image_id": a.cover_image_id,
                "story_enabled": a.story_enabled,
                "sort_order": a.sort_order,
                "item_count": a.item_count,
                "chapter_count": a.chapter_count,
                "created_at": a.created_at,
                "updated_at": a.updated_at,
            }
            for a in albums
        ]
    }


def _handle_albums_create(params: dict) -> dict:
    """Create a smart album and materialize membership."""
    from imganalyzer.storyline.albums import create_album

    conn = _get_db()
    album = create_album(
        conn,
        name=params["name"],
        rules=params["rules"],
        description=params.get("description"),
        story_enabled=params.get("story_enabled", True),
        sort_order=params.get("sort_order", "chronological"),
    )
    return {"id": album.id, "item_count": album.item_count}


def _handle_albums_get(params: dict) -> dict:
    """Get a smart album by ID."""
    from imganalyzer.storyline.albums import get_album

    conn = _get_db()
    album = get_album(conn, params["album_id"])
    if album is None:
        return {"error": "Album not found"}
    return {
        "id": album.id,
        "name": album.name,
        "description": album.description,
        "cover_image_id": album.cover_image_id,
        "rules": album.rules,
        "story_enabled": album.story_enabled,
        "sort_order": album.sort_order,
        "item_count": album.item_count,
        "chapter_count": album.chapter_count,
        "created_at": album.created_at,
        "updated_at": album.updated_at,
    }


def _handle_albums_update(params: dict) -> dict:
    """Update a smart album."""
    from imganalyzer.storyline.albums import update_album

    conn = _get_db()
    kwargs: dict = {}
    if "name" in params:
        kwargs["name"] = params["name"]
    if "description" in params:
        kwargs["description"] = params["description"]
    if "rules" in params:
        kwargs["rules"] = params["rules"]
    if "story_enabled" in params:
        kwargs["story_enabled"] = params["story_enabled"]
    if "sort_order" in params:
        kwargs["sort_order"] = params["sort_order"]

    album = update_album(conn, params["album_id"], **kwargs)
    if album is None:
        return {"error": "Album not found"}
    return {"id": album.id, "item_count": album.item_count}


def _handle_albums_delete(params: dict) -> dict:
    """Delete a smart album."""
    from imganalyzer.storyline.albums import delete_album

    conn = _get_db()
    ok = delete_album(conn, params["album_id"])
    return {"deleted": ok}


def _handle_albums_refresh(params: dict) -> dict:
    """Re-evaluate album rules and refresh membership."""
    from imganalyzer.storyline.albums import refresh_membership

    conn = _get_db()
    count = refresh_membership(conn, params["album_id"])
    return {"item_count": count}


def _handle_albums_story(params: dict) -> dict:
    """Get the story structure (chapters) for an album."""
    from imganalyzer.storyline.generator import get_story_chapters

    conn = _get_db()
    chapters = get_story_chapters(conn, params["album_id"])
    return {"chapters": chapters}


def _handle_albums_story_generate(params: dict) -> dict:
    """Generate (or regenerate) story structure for an album."""
    import time as _time

    from imganalyzer.storyline.evaluator import evaluate_story
    from imganalyzer.storyline.generator import generate_story

    conn = _get_db()
    album_id = params["album_id"]

    t0 = _time.monotonic()
    result = generate_story(
        conn,
        album_id,
        time_window_minutes=params.get("time_window_minutes", 30),
        chapter_gap_hours=params.get("chapter_gap_hours", 4),
        chapter_distance_km=params.get("chapter_distance_km", 50.0),
        force_year_breaks=params.get("force_year_breaks", True),
    )
    gen_time = _time.monotonic() - t0

    report = evaluate_story(conn, album_id, generation_time_s=gen_time)

    return {
        "images": result.image_count,
        "moments": result.moment_count,
        "chapters": result.chapter_count,
        "generation_time_s": round(gen_time, 2),
        "evaluation": report.to_dict(),
    }


def _handle_albums_chapter_moments(params: dict) -> dict:
    """Get moments for a story chapter."""
    from imganalyzer.storyline.generator import get_chapter_moments

    conn = _get_db()
    moments = get_chapter_moments(conn, params["chapter_id"])
    return {"moments": moments}


def _handle_albums_moment_images(params: dict) -> dict:
    """Get images for a story moment."""
    from imganalyzer.storyline.generator import get_moment_images

    conn = _get_db()
    images = get_moment_images(conn, params["moment_id"])
    return {"images": images}


def _handle_albums_check_new(params: dict) -> dict:
    """Check a newly analyzed image against all smart album rules."""
    from imganalyzer.storyline.incremental import check_and_add_image

    conn = _get_db()
    added = check_and_add_image(conn, params["image_id"])
    return {"added_to_albums": added}


def _handle_albums_generate_narrative(params: dict) -> dict:
    """Generate AI narratives for all chapters in an album."""
    from imganalyzer.storyline.narrative import generate_all_chapter_narratives

    conn = _get_db()
    updated = generate_all_chapter_narratives(
        conn,
        params["album_id"],
        use_ai=params.get("use_ai", True),
    )
    return {"chapters_updated": updated}


def _handle_albums_export(params: dict) -> dict:
    """Export a story album as a standalone HTML file."""
    from imganalyzer.storyline.export import export_story_html
    import os

    output_path = params["output_path"]
    # Validate the output path is absolute and within a writeable directory
    if not os.path.isabs(output_path):
        return {"error": "output_path must be an absolute path"}
    parent = os.path.dirname(output_path)
    if not os.path.isdir(parent):
        return {"error": f"Parent directory does not exist: {parent}"}

    conn = _get_db()
    output = export_story_html(
        conn,
        params["album_id"],
        output_path,
        include_thumbnails=params.get("include_thumbnails", True),
        max_heroes_per_chapter=params.get("max_heroes_per_chapter", 6),
    )
    return {"path": str(output)}


def _handle_albums_presets(_params: dict) -> dict:
    """List available album presets."""
    from imganalyzer.storyline.presets import PRESET_REGISTRY

    return {"presets": PRESET_REGISTRY}


def _handle_albums_create_preset(params: dict) -> dict:
    """Create a smart album from a preset."""
    from imganalyzer.storyline.presets import (
        create_growth_story,
        create_location_story,
        create_on_this_day,
        create_person_timeline,
        create_together_album,
        create_year_in_review,
    )

    conn = _get_db()
    preset = params["preset"]

    if preset == "year_in_review":
        album = create_year_in_review(conn, year=params.get("year"))
    elif preset == "on_this_day":
        album = create_on_this_day(conn, month=params.get("month"), day=params.get("day"))
    elif preset == "person_timeline":
        album = create_person_timeline(
            conn, params["person_id"], person_name=params.get("person_name"),
        )
    elif preset == "growth_story":
        album = create_growth_story(
            conn, params["person_id"], person_name=params.get("person_name"),
        )
    elif preset == "together":
        album = create_together_album(
            conn, params["person_ids"], person_names=params.get("person_names"),
        )
    elif preset == "location":
        album = create_location_story(
            conn, params["country"], city=params.get("city"),
        )
    else:
        return {"error": f"Unknown preset: {preset}"}

    return {"id": album.id, "name": album.name, "item_count": album.item_count}


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
    "search/warmup": _handle_search_warmup,
    "search/resolveFaceQuery": _handle_search_resolve_face_query,
    "image/details": _handle_image_details,
    "gallery/listFolders": _handle_gallery_list_folders,
    "gallery/listImagesChunk": _handle_gallery_list_images_chunk,
    "thumbnail": _handle_thumbnail,
    "thumbnails/batch": _handle_thumbnails_batch,
    "fullimage": _handle_fullimage,
    "cachedimage": _handle_cachedimage,
    "decode/status": _handle_decode_status,
    "decode/enqueue_missing": _handle_decode_enqueue_missing,
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
    "faces/persons": _handle_faces_persons,
    "faces/person-create": _handle_faces_person_create,
    "faces/person-rename": _handle_faces_person_rename,
    "faces/person-delete": _handle_faces_person_delete,
    "faces/person-link-cluster": _handle_faces_person_link,
    "faces/person-unlink-cluster": _handle_faces_person_unlink,
    "faces/person-clusters": _handle_faces_person_clusters,
    "faces/person-link-suggestions": _handle_faces_person_link_suggestions,
    "faces/person-similar-images": _handle_faces_person_similar_images,
    "faces/person-link-occurrences": _handle_faces_person_link_occurrences,
    "faces/person-unlink-occurrence": _handle_faces_person_unlink_occurrence,
    "faces/person-direct-links": _handle_faces_person_direct_links,
    "geo/clusters": _handle_geo_clusters,
    "geo/cluster-preview": _handle_geo_cluster_preview,
    "geo/nearby": _handle_geo_nearby,
    "geo/stats": _handle_geo_stats,
    "geo/heatmap": _handle_geo_heatmap,
    "geo/stats-extended": _handle_geo_stats_extended,
    "geo/gap-filler-preview": _handle_geo_gap_filler_preview,
    "geo/gap-filler-apply": _handle_geo_gap_filler_apply,
    "geo/trip-detect": _handle_geo_trip_detect,
    "geo/trip-timeline": _handle_geo_trip_timeline,
    "geo/geocode": _handle_geo_geocode,
    # ── Albums / Storyline ───────────────────────────────────
    "albums/list": _handle_albums_list,
    "albums/create": _handle_albums_create,
    "albums/get": _handle_albums_get,
    "albums/update": _handle_albums_update,
    "albums/delete": _handle_albums_delete,
    "albums/refresh": _handle_albums_refresh,
    "albums/story": _handle_albums_story,
    "albums/story/generate": _handle_albums_story_generate,
    "albums/story/generate-narrative": _handle_albums_generate_narrative,
    "albums/chapter/moments": _handle_albums_chapter_moments,
    "albums/moment/images": _handle_albums_moment_images,
    "albums/check-new": _handle_albums_check_new,
    "albums/export": _handle_albums_export,
    "albums/presets": _handle_albums_presets,
    "albums/create-preset": _handle_albums_create_preset,
}

# Methods that send their own result/error asynchronously (streaming).
# They receive (req_id, params) and are responsible for calling
# _send_result or _send_error themselves.
_ASYNC_METHODS: dict[str, Any] = {
    "ingest": _handle_ingest,
    "run": _handle_run,
    "analyze": _handle_analyze,
    "faces/run-clustering": _handle_faces_run_clustering,
    "faces/crop-batch": _handle_faces_crop_batch_async,
}

# StdIO-only async wrappers for expensive one-shot methods. They remain sync for
# HTTP transport so the HTTP coordinator can still serve them directly.
_STDIO_ASYNC_METHODS: dict[str, Any] = {
    "gallery/listImagesChunk": _handle_gallery_list_images_chunk_async,
    "thumbnail": _handle_thumbnail_async,
    "thumbnails/batch": _handle_thumbnails_batch_async,
    "fullimage": _handle_fullimage_async,
}


def _call_sync_handler(method: str, params: dict[str, Any]) -> Any:
    """Invoke a sync JSON-RPC handler with transient DB lock retries when needed."""
    handler = _SYNC_METHODS[method]
    if method not in _LOCK_RETRYABLE_METHODS:
        return handler(params)

    delay_s = _LOCK_RETRY_INITIAL_DELAY_S
    for attempt in range(1, _LOCK_RETRY_ATTEMPTS + 1):
        try:
            return handler(params)
        except Exception as exc:
            if not _is_transient_db_lock_error(exc) or attempt >= _LOCK_RETRY_ATTEMPTS:
                raise
            time.sleep(delay_s)
            delay_s = min(delay_s * 2, 1.0)


def _graceful_shutdown() -> None:
    """Signal running workers to stop and wait briefly for shutdown."""
    _run_cancel.set()
    w = _active_worker
    if w is not None:
        w._shutdown.set()
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
            result = _call_sync_handler(method, params)
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
            try:
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, OSError):
                # Worker closed its connection before we finished writing
                # the response — harmless; the worker will retry on its
                # next poll cycle.
                pass

        def _check_auth(self) -> bool:
            if not auth_token:
                return True
            auth_header = self.headers.get("Authorization", "")
            expected = f"Bearer {auth_token}"
            if hmac.compare_digest(auth_header, expected):
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
            # Suppress noisy connection-abort errors from workers that
            # close the socket before we finish writing the response.
            exc_text = traceback.format_exc()
            if "ConnectionAbortedError" in exc_text or "ConnectionResetError" in exc_text:
                return
            sys.stderr.write(
                f"[server.http] handler error for {client_address}: "
                f"{exc_text}\n"
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

    # Emit the ready signal as soon as the socket is bound so the
    # Electron coordinator doesn't time out waiting for it while we do
    # slower one-time init (decoded-store scan, etc.).
    sys.stderr.write(
        f"[server.http] listening on http://{host}:{port}{normalized_path} "
        f"(auth={'on' if auth_token else 'off'})\n"
    )
    sys.stderr.flush()

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

    if method in _STDIO_ASYNC_METHODS:
        def _run_async():
            try:
                _STDIO_ASYNC_METHODS[method](req_id, params)
            except Exception as exc:
                _send_error(req_id, -1, str(exc))

        t = threading.Thread(target=_run_async, daemon=True, name=f"rpc-{method}")
        t.start()
    elif method in _SYNC_METHODS:
        try:
            result = _call_sync_handler(method, params)
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
