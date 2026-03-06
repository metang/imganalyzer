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
import io
import json
import os
import re
import sqlite3
import sys
import threading
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


def _send(obj: dict[str, Any]) -> None:
    """Write a JSON-RPC message to the real stdout (one line, thread-safe)."""
    line = json.dumps(obj, default=str, separators=(",", ":"))
    with _send_lock:
        _real_stdout.write(line + "\n")
        _real_stdout.flush()


def _send_result(req_id: int | str, result: Any) -> None:
    _send({"jsonrpc": "2.0", "id": req_id, "result": result})


def _send_error(req_id: int | str | None, code: int, message: str) -> None:
    _send({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})


def _send_notification(method: str, params: Any) -> None:
    _send({"jsonrpc": "2.0", "method": method, "params": params})


# ── Lazy singletons ──────────────────────────────────────────────────────────

_db_conn = None
_db_lock = threading.Lock()


def _get_db():
    """Get or create the shared DB connection (thread-safe)."""
    global _db_conn
    with _db_lock:
        if _db_conn is None:
            from imganalyzer.db.connection import get_db
            _db_conn = get_db()
        return _db_conn


# ── State for cancellable operations ─────────────────────────────────────────

_run_cancel = threading.Event()
_run_thread: threading.Thread | None = None
_active_worker: Any = None  # Worker | None — set while a run is active

_analyze_cancel: dict[str, threading.Event] = {}  # imagePath -> cancel event


# ── Method handlers ──────────────────────────────────────────────────────────

def _handle_status(params: dict) -> dict:
    """Return queue stats as JSON."""
    from imganalyzer.db.queue import JobQueue
    from imganalyzer.db.repository import Repository

    conn = _get_db()
    queue = JobQueue(conn)
    repo = Repository(conn)

    total_images = repo.count_images()
    module_stats = queue.stats()
    totals = queue.total_stats()

    from imganalyzer.db.repository import ALL_MODULES
    queue_modules = list(module_stats.keys())
    ordered = [m for m in ALL_MODULES if m in module_stats or True]
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

    return {
        "total_images": total_images,
        "modules": modules_out,
        "totals": {
            "pending": totals.get("pending", 0),
            "running": totals.get("running", 0),
            "done": totals.get("done", 0),
            "failed": totals.get("failed", 0),
            "skipped": totals.get("skipped", 0),
        },
    }


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
    _send_result(req_id, {
        "ok": True,
        "registered": ingest_stats.get("registered", 0),
        "enqueued": ingest_stats.get("enqueued", 0),
        "skipped": ingest_stats.get("skipped", 0),
    })


def _handle_run(req_id: int | str, params: dict) -> None:
    """Start processing the queue — streaming [RESULT] notifications."""
    global _run_thread

    if _run_thread is not None and _run_thread.is_alive():
        # The previous worker may still be winding down after cancel_run.
        # Wait briefly for it to finish before rejecting.
        _run_thread.join(timeout=10)
        if _run_thread.is_alive():
            _send_error(req_id, -2, "A run is already in progress")
            return

    _run_cancel.clear()

    workers = params.get("workers", 1)
    cloud_workers = params.get("cloudWorkers", 4)
    cloud_provider = params.get("cloudProvider", "copilot")
    verbose = params.get("verbose", True)
    write_xmp = not params.get("noXmp", True)
    batch_size = params.get("batchSize", 10)
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
            worker_kwargs = dict(
                conn=conn,
                workers=workers,
                cloud_workers=cloud_workers,
                force=False,
                cloud_provider=cloud_provider,
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
            try:
                result = worker.run(batch_size=batch_size)
                _send_notification("run/done", result)
            except Exception as exc:
                _send_notification("run/error", {"error": str(exc)})
            finally:
                worker_mod._result_notify = None
                _active_worker = None
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

    if not face and query:
        resolve_face_query = getattr(engine, "resolve_face_query", None)
        if callable(resolve_face_query):
            resolved_face, remaining_query = resolve_face_query(str(query))
            if resolved_face:
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
        base_conditions.append("ae.aesthetic_score >= ?")
        base_params.append(aesthetic_min)
    if aesthetic_max is not None:
        base_conditions.append("ae.aesthetic_score <= ?")
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
        ae.aesthetic_score, ae.aesthetic_label, ae.aesthetic_reason
    """

    joins = """
        FROM images i
        LEFT JOIN analysis_metadata  m  ON m.image_id  = i.id
        LEFT JOIN analysis_technical t  ON t.image_id  = i.id
        LEFT JOIN analysis_local_ai  la ON la.image_id = i.id
        LEFT JOIN analysis_blip2     b2 ON b2.image_id = i.id
        LEFT JOIN analysis_objects   ob ON ob.image_id = i.id
        LEFT JOIN analysis_ocr      ocr ON ocr.image_id = i.id
        LEFT JOIN analysis_faces     af ON af.image_id = i.id
        LEFT JOIN analysis_aesthetic ae ON ae.image_id = i.id
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
                "((COALESCE(ae.aesthetic_score, 0) * 12.0) + "
                "(COALESCE(t.sharpness_score, 0) * 0.25) - "
                "(COALESCE(t.noise_level, 0) * 120.0)) DESC, "
                "ae.aesthetic_score DESC, t.sharpness_score DESC, i.id DESC"
            )
        if sort_by == "aesthetic":
            return " ORDER BY ae.aesthetic_score DESC, t.sharpness_score DESC, i.id DESC"
        if sort_by == "sharpness":
            return " ORDER BY t.sharpness_score DESC, ae.aesthetic_score DESC, i.id DESC"
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

    def _combine_face_and_text_terms(candidate_limit: int) -> tuple[list[dict[str, Any]], bool]:
        face_results = engine.search_face(face, limit=candidate_limit)
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

    if similar_to_image_id is not None or face or ((has_text_query or expanded_terms) and mode != "browse"):
        page_end = offset + limit
        candidate_limit = max((page_end + 1) * 4, 200)
        max_candidate_limit = max(candidate_limit, 5000)
        search_exhausted = True
        records: list[dict[str, Any]] = []

        while True:
            if similar_to_image_id is not None:
                search_results = engine.search_similar_image(similar_to_image_id, limit=candidate_limit)
                search_exhausted = len(search_results) < candidate_limit
            elif face and (has_text_query or expanded_terms):
                search_results, search_exhausted = _combine_face_and_text_terms(candidate_limit)
            elif face:
                search_results = engine.search_face(face, limit=candidate_limit)
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
    face, remaining_query = engine.resolve_face_query(query)
    return {"face": face, "remainingQuery": remaining_query}


def _processed_exists_clause(alias: str = "i") -> str:
    """SQL predicate: image has at least one processed analysis record."""
    return f"""(
        EXISTS(SELECT 1 FROM analysis_metadata m  WHERE m.image_id  = {alias}.id) OR
        EXISTS(SELECT 1 FROM analysis_technical t WHERE t.image_id  = {alias}.id) OR
        EXISTS(SELECT 1 FROM analysis_local_ai la WHERE la.image_id = {alias}.id) OR
        EXISTS(SELECT 1 FROM analysis_blip2 b2 WHERE b2.image_id    = {alias}.id) OR
        EXISTS(SELECT 1 FROM analysis_objects ob WHERE ob.image_id   = {alias}.id) OR
        EXISTS(SELECT 1 FROM analysis_ocr ocr WHERE ocr.image_id     = {alias}.id) OR
        EXISTS(SELECT 1 FROM analysis_faces af WHERE af.image_id     = {alias}.id) OR
        EXISTS(SELECT 1 FROM analysis_cloud_ai ca WHERE ca.image_id  = {alias}.id) OR
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
        ae.aesthetic_score, ae.aesthetic_label, ae.aesthetic_reason
    """

    joins = """
        FROM images i
        LEFT JOIN analysis_metadata  m  ON m.image_id  = i.id
        LEFT JOIN analysis_technical t  ON t.image_id  = i.id
        LEFT JOIN analysis_local_ai  la ON la.image_id = i.id
        LEFT JOIN analysis_blip2     b2 ON b2.image_id = i.id
        LEFT JOIN analysis_objects   ob ON ob.image_id = i.id
        LEFT JOIN analysis_ocr      ocr ON ocr.image_id = i.id
        LEFT JOIN analysis_faces     af ON af.image_id = i.id
        LEFT JOIN analysis_aesthetic ae ON ae.image_id = i.id
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

    if ext in (".heic", ".heif"):
        from pillow_heif import register_heif_opener
        register_heif_opener()

    from PIL import Image

    if ext in RAW_EXTS:
        import rawpy
        with _suppress_c_stderr():
            raw_ctx = rawpy.imread(str(path))
        with raw_ctx as raw:
            rgb = raw.postprocess(use_camera_wb=True, output_bps=8, half_size=True)
        img = Image.fromarray(rgb)
    else:
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

    # RAW or HEIC — decode to JPEG
    if ext in (".heic", ".heif"):
        from pillow_heif import register_heif_opener
        register_heif_opener()

    from PIL import Image

    if ext in RAW_EXTS:
        import rawpy
        with _suppress_c_stderr():
            raw_ctx = rawpy.imread(str(path))
        with raw_ctx as raw:
            rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
        img = Image.fromarray(rgb)
    else:
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
    return {
        "clusters": clusters,
        "has_occurrences": has_occurrences,
        "total_count": total_count,
    }


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
    return {"ok": True}


def _handle_faces_person_link(params: dict) -> dict:
    """Link a cluster to a person."""
    from imganalyzer.db.repository import Repository

    conn = _get_db()
    repo = Repository(conn)
    updated = repo.link_cluster_to_person(
        int(params["cluster_id"]), int(params["person_id"])
    )
    return {"ok": True, "updated": updated}


def _handle_faces_person_unlink(params: dict) -> dict:
    """Unlink a cluster from its person."""
    from imganalyzer.db.repository import Repository

    conn = _get_db()
    repo = Repository(conn)
    updated = repo.unlink_cluster_from_person(int(params["cluster_id"]))
    return {"ok": True, "updated": updated}


def _handle_faces_person_clusters(params: dict) -> dict:
    """Get clusters belonging to a person."""
    from imganalyzer.db.repository import Repository

    conn = _get_db()
    repo = Repository(conn)
    clusters = repo.get_person_clusters(int(params["person_id"]))
    return {"clusters": clusters}


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

    file_path = occ["file_path"]
    path = Path(file_path)
    if not path.exists():
        return {"error": f"Image file not found: {file_path}"}

    from PIL import Image

    ext = path.suffix.lower()
    if ext in (".heic", ".heif"):
        from pillow_heif import register_heif_opener
        register_heif_opener()

    RAW_EXTS = {
        ".arw", ".cr2", ".cr3", ".nef", ".nrw", ".orf", ".raf", ".rw2",
        ".dng", ".pef", ".srw", ".erf", ".kdc", ".mrw", ".3fr", ".fff",
        ".sr2", ".srf", ".x3f", ".iiq", ".mos", ".raw",
    }

    if ext in RAW_EXTS:
        import rawpy
        with _suppress_c_stderr():
            raw_ctx = rawpy.imread(str(path))
        with raw_ctx as raw:
            rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
        img = Image.fromarray(rgb)
    else:
        img = Image.open(path)
        img = img.convert("RGB")

    # Crop face region with some padding.
    # IMPORTANT: bbox coordinates were computed on a pre-resized image
    # (max 1920px long edge) — scale them to the original resolution.
    w, h = img.size
    det_long_edge = 1920  # _AI_MAX_LONG_EDGE in modules.py
    orig_long_edge = max(w, h)
    if orig_long_edge > det_long_edge:
        scale = orig_long_edge / det_long_edge
    else:
        scale = 1.0

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
    return {"data": base64.b64encode(buf.getvalue()).decode("ascii")}


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
    conn = _get_db()
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
    for row in rows:
        row_d = dict(row)
        oid = str(row_d["id"])
        if row_d.get("thumbnail") is not None:
            thumbnails[oid] = base64.b64encode(row_d["thumbnail"]).decode("ascii")
        else:
            # Fallback: generate on-the-fly for legacy rows
            result = _handle_faces_crop({"occurrence_id": row_d["id"]})
            if "data" in result:
                thumbnails[oid] = result["data"]

    return {"thumbnails": thumbnails}


# ── Method dispatch ──────────────────────────────────────────────────────────

# Methods that return a result synchronously (the response is sent
# from the main loop after the handler returns).
_SYNC_METHODS: dict[str, Any] = {
    "status": _handle_status,
    "queue_clear": _handle_queue_clear,
    "rebuild": _handle_rebuild,
    "search": _handle_search,
    "search/resolveFaceQuery": _handle_search_resolve_face_query,
    "gallery/listFolders": _handle_gallery_list_folders,
    "gallery/listImagesChunk": _handle_gallery_list_images_chunk,
    "thumbnail": _handle_thumbnail,
    "fullimage": _handle_fullimage,
    "cancel_run": _handle_cancel_run,
    "cancel_analyze": _handle_cancel_analyze,
    "faces/list": _handle_faces_list,
    "faces/images": _handle_faces_images,
    "faces/set-alias": _handle_faces_set_alias,
    "faces/clusters": _handle_faces_clusters,
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
            self.send_error(405, "Method Not Allowed")

        def log_message(self, fmt: str, *args: Any) -> None:
            # Keep logs on stderr, stdout is reserved in stdio mode.
            sys.stderr.write(f"[server.http] {self.address_string()} - {fmt % args}\n")

    server = ThreadingHTTPServer((host, port), _JsonRpcHandler)
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
    args, _unknown = parser.parse_known_args(sys.argv[1:])

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
