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
"""
from __future__ import annotations

import base64
import io
import json
import os
import signal
import sqlite3
import sys
import threading
import traceback
from pathlib import Path
from typing import Any

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
    stale_timeout = params.get("staleTimeout")  # None = use Worker default (10 min)

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
    camera = params.get("camera")
    lens = params.get("lens")
    location = params.get("location")
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
    has_people = params.get("hasPeople")
    limit = params.get("limit", 200)
    offset = params.get("offset", 0)

    # Step 1: Get candidate set from text/semantic/browse search
    has_text_query = bool(query and query.strip())
    candidate_ids: list[int] | None = None
    score_map: dict[int, float] = {}

    if face:
        text_results = engine.search_face(face, limit=limit * 4)
        candidate_ids = [r["image_id"] for r in text_results]
        score_map = {r["image_id"]: r["score"] for r in text_results}
    elif has_text_query and mode != "browse":
        text_results = engine.search(
            query, limit=limit * 4, semantic_weight=semantic_weight, mode=mode,
        )
        candidate_ids = [r["image_id"] for r in text_results]
        score_map = {r["image_id"]: r["score"] for r in text_results}

    # Step 2: Build SQL with metric filters
    conditions: list[str] = []
    sql_params: list = []

    if candidate_ids is not None:
        if not candidate_ids:
            return {"results": [], "total": 0}
        id_ph = ",".join("?" * len(candidate_ids))
        conditions.append(f"i.id IN ({id_ph})")
        sql_params.extend(candidate_ids)

    if camera:
        conditions.append("(m.camera_make LIKE ? OR m.camera_model LIKE ?)")
        sql_params.extend([f"%{camera}%", f"%{camera}%"])
    if lens:
        conditions.append("m.lens_model LIKE ?")
        sql_params.append(f"%{lens}%")
    if location:
        conditions.append(
            "(m.location_city LIKE ? OR m.location_state LIKE ? OR m.location_country LIKE ?)"
        )
        sql_params.extend([f"%{location}%", f"%{location}%", f"%{location}%"])
    if date_from:
        conditions.append("m.date_time_original >= ?")
        sql_params.append(date_from)
    if date_to:
        conditions.append("m.date_time_original <= ?")
        sql_params.append(date_to + "T23:59:59")
    if iso_min is not None:
        conditions.append("CAST(m.iso AS REAL) >= ?")
        sql_params.append(iso_min)
    if iso_max is not None:
        conditions.append("CAST(m.iso AS REAL) <= ?")
        sql_params.append(iso_max)
    if aesthetic_min is not None:
        conditions.append("ae.aesthetic_score >= ?")
        sql_params.append(aesthetic_min)
    if aesthetic_max is not None:
        conditions.append("ae.aesthetic_score <= ?")
        sql_params.append(aesthetic_max)
    if sharpness_min is not None:
        conditions.append("t.sharpness_score >= ?")
        sql_params.append(sharpness_min)
    if sharpness_max is not None:
        conditions.append("t.sharpness_score <= ?")
        sql_params.append(sharpness_max)
    if noise_max is not None:
        conditions.append("t.noise_level <= ?")
        sql_params.append(noise_max)
    if faces_min is not None:
        conditions.append("la.face_count >= ?")
        sql_params.append(faces_min)
    if faces_max is not None:
        conditions.append("la.face_count <= ?")
        sql_params.append(faces_max)
    if has_people is True:
        conditions.append("la.has_people = 1")
    elif has_people is False:
        conditions.append("(la.has_people = 0 OR la.has_people IS NULL)")

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

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
        la.description, la.scene_type, la.main_subject, la.lighting, la.mood,
        la.keywords, la.detected_objects, la.face_count, la.face_identities,
        la.has_people, la.ocr_text,
        ae.aesthetic_score, ae.aesthetic_label, ae.aesthetic_reason
    """

    joins = """
        FROM images i
        LEFT JOIN analysis_metadata  m  ON m.image_id  = i.id
        LEFT JOIN analysis_technical t  ON t.image_id  = i.id
        LEFT JOIN analysis_local_ai  la ON la.image_id = i.id
        LEFT JOIN analysis_aesthetic ae ON ae.image_id = i.id
    """

    if score_map:
        sql = f"SELECT {select_cols} {joins} {where_clause}"
    else:
        sql = f"SELECT {select_cols} {joins} {where_clause} LIMIT ? OFFSET ?"
        sql_params.extend([limit, offset])

    rows = conn.execute(sql, sql_params).fetchall()

    def _json_field(val):
        if val is None:
            return None
        try:
            return _json.loads(val)
        except Exception:
            return val

    records = []
    for row in rows:
        iid = row["image_id"]
        score = score_map.get(iid, 0.0) if score_map else None
        records.append({
            "image_id": iid,
            "file_path": row["file_path"],
            "score": score,
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
            "aesthetic_score": row["aesthetic_score"],
            "aesthetic_label": row["aesthetic_label"],
            "aesthetic_reason": row["aesthetic_reason"],
        })

    if score_map:
        records.sort(key=lambda r: -(r["score"] or 0.0))
        total = len(records)
        records = records[offset: offset + limit]
    else:
        total = len(records)

    return {"results": records, "total": total}


def _handle_analyze(req_id: int | str, params: dict) -> None:
    """Analyze a single image — streaming progress, then result."""
    from imganalyzer.analyzer import Analyzer

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

            # Intercept print output for progress stages
            import builtins
            _orig_print = builtins.print

            def _patched_print(*args, **kwargs):
                text = " ".join(str(a) for a in args)
                # Forward stage progress to Electron
                _send_notification("analyze/progress", {
                    "imagePath": image_path,
                    "stage": text.strip(),
                })
                _orig_print(*args, **kwargs)

            builtins.print = _patched_print
            try:
                img_path = Path(image_path)
                result = analyzer.analyze(img_path)

                if overwrite:
                    xmp_path = img_path.with_suffix(".xmp")
                    result.write_xmp(xmp_path)

                _send_result(req_id, {
                    "ok": True,
                    "xmpPath": str(img_path.with_suffix(".xmp")),
                })
            except Exception as exc:
                _send_error(req_id, -1, f"Analysis failed: {exc}")
            finally:
                builtins.print = _orig_print
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
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(use_camera_wb=True, output_bps=8, half_size=True)
        import numpy as np
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
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
        import numpy as np
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
    clusters = repo.list_face_clusters()
    has_occurrences = repo.get_face_occurrences_count() > 0
    return {"clusters": clusters, "has_occurrences": has_occurrences}


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
        import numpy as np
        with rawpy.imread(str(path)) as raw:
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


def _handle_faces_run_clustering(params: dict) -> dict:
    """Run face clustering on all stored occurrences."""
    from imganalyzer.db.repository import Repository

    conn = _get_db()
    repo = Repository(conn)
    threshold = params.get("threshold", 0.55)
    num_clusters = repo.cluster_faces(threshold=threshold)
    conn.commit()
    return {"num_clusters": num_clusters}


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
    "faces/run-clustering": _handle_faces_run_clustering,
}

# Methods that send their own result/error asynchronously (streaming).
# They receive (req_id, params) and are responsible for calling
# _send_result or _send_error themselves.
_ASYNC_METHODS: dict[str, Any] = {
    "ingest": _handle_ingest,
    "run": _handle_run,
    "analyze": _handle_analyze,
}


def _dispatch(msg: dict) -> None:
    """Dispatch a JSON-RPC request."""
    req_id = msg.get("id")
    method = msg.get("method", "")
    params = msg.get("params", {})

    if method == "shutdown":
        # Signal the active worker to stop and wait briefly for it to exit
        _run_cancel.set()
        if _active_worker is not None:
            _active_worker._shutdown.set()
        if _run_thread is not None and _run_thread.is_alive():
            _run_thread.join(timeout=5)
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

def main() -> None:
    """Read JSON-RPC requests from stdin, dispatch, write responses to stdout."""
    # Load dotenv early so all handlers have access to env vars
    from dotenv import load_dotenv
    load_dotenv()

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
