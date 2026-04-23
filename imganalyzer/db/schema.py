"""Database schema — CREATE TABLE statements and migration runner.

The migration system uses a simple version counter stored in a ``schema_version``
table.  Each migration is a function ``_migrate_vN(conn)`` that runs the DDL for
version *N*.  ``ensure_schema`` applies all pending migrations in order.
"""
from __future__ import annotations

import json
import sqlite3

# ── Current schema version ────────────────────────────────────────────────────
SCHEMA_VERSION = 33


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create or upgrade the database schema to the latest version."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER NOT NULL
        )
    """)
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    current = row["version"] if row else 0

    migrations = {
        1: _migrate_v1,
        2: _migrate_v2,
        3: _migrate_v3,
        4: _migrate_v4,
        5: _migrate_v5,
        6: _migrate_v6,
        7: _migrate_v7,
        8: _migrate_v8,
        9: _migrate_v9,
        10: _migrate_v10,
        11: _migrate_v11,
        12: _migrate_v12,
        13: _migrate_v13,
        14: _migrate_v14,
        15: _migrate_v15,
        16: _migrate_v16,
        17: _migrate_v17,
        18: _migrate_v18,
        19: _migrate_v19,
        20: _migrate_v20,
        21: _migrate_v21,
        22: _migrate_v22,
        23: _migrate_v23,
        24: _migrate_v24,
        25: _migrate_v25,
        26: _migrate_v26,
        27: _migrate_v27,
        28: _migrate_v28,
        29: _migrate_v29,
        30: _migrate_v30,
        31: _migrate_v31,
        32: _migrate_v32,
        33: _migrate_v33,
    }

    for v in range(current + 1, SCHEMA_VERSION + 1):
        fn = migrations.get(v)
        if fn is None:
            raise RuntimeError(f"Missing migration function for schema version {v}")
        fn(conn)
        if v == 1 and current == 0:
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", [v])
        else:
            conn.execute("UPDATE schema_version SET version = ?", [v])
        conn.commit()


# ── Migration v1: Initial schema ─────────────────────────────────────────────

def _migrate_v1(conn: sqlite3.Connection) -> None:
    """Initial schema — all core tables."""

    # ── images ─────────────────────────────────────────────────────────────
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS images (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path       TEXT    NOT NULL UNIQUE,
            file_hash       TEXT,
            file_size       INTEGER,
            width           INTEGER,
            height          INTEGER,
            format          TEXT,
            date_added      TEXT    NOT NULL DEFAULT (datetime('now')),
            date_modified   TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_images_file_path ON images(file_path);
        CREATE INDEX IF NOT EXISTS idx_images_file_hash ON images(file_hash);

        -- ── analysis_metadata ─────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS analysis_metadata (
            image_id            INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
            camera_make         TEXT,
            camera_model        TEXT,
            lens_model          TEXT,
            focal_length        REAL,
            focal_length_35mm   REAL,
            f_number            REAL,
            exposure_time       TEXT,
            iso                 INTEGER,
            exposure_bias       REAL,
            white_balance       TEXT,
            metering_mode       TEXT,
            flash               TEXT,
            color_space         TEXT,
            orientation         TEXT,
            date_time_original  TEXT,
            gps_latitude        REAL,
            gps_longitude       REAL,
            gps_altitude        REAL,
            location_city       TEXT,
            location_state      TEXT,
            location_country    TEXT,
            location_country_code TEXT,
            raw_white_level     INTEGER,
            analyzed_at         TEXT
        );

        -- ── analysis_technical ────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS analysis_technical (
            image_id                INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
            sharpness_score         REAL,
            sharpness_label         TEXT,
            exposure_ev             REAL,
            exposure_label          TEXT,
            noise_level             REAL,
            noise_label             TEXT,
            snr_db                  REAL,
            dynamic_range_stops     REAL,
            highlight_clipping_pct  REAL,
            shadow_clipping_pct     REAL,
            avg_saturation          REAL,
            warm_cool_ratio         REAL,
            dominant_colors         TEXT,  -- JSON array
            analyzed_at             TEXT
        );

        -- ── analysis_caption ──────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS analysis_caption (
            image_id            INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
            description         TEXT,
            scene_type          TEXT,
            main_subject        TEXT,
            lighting            TEXT,
            mood                TEXT,
            keywords            TEXT,  -- JSON array
            detected_objects    TEXT,  -- JSON array
            face_count          INTEGER,
            face_identities     TEXT,  -- JSON array
            face_details        TEXT,  -- JSON (list of dicts)
            has_people          INTEGER DEFAULT 0,  -- boolean
            ocr_text            TEXT,
            technical_notes     TEXT,
            analyzed_at         TEXT
        );

        -- ── analysis_cloud_ai ─────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS analysis_cloud_ai (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id            INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            provider            TEXT    NOT NULL,  -- 'openai'|'anthropic'|'google'|'copilot'
            description         TEXT,
            scene_type          TEXT,
            main_subject        TEXT,
            lighting            TEXT,
            mood                TEXT,
            keywords            TEXT,  -- JSON array
            detected_objects    TEXT,  -- JSON array
            landmark            TEXT,
            dominant_colors_ai  TEXT,  -- JSON array
            technical_notes     TEXT,
            raw_response        TEXT,  -- full JSON response for reference
            analyzed_at         TEXT,
            UNIQUE(image_id, provider)
        );
        CREATE INDEX IF NOT EXISTS idx_cloud_ai_image_provider
            ON analysis_cloud_ai(image_id, provider);

        -- ── analysis_aesthetic ─────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS analysis_aesthetic (
            image_id            INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
            aesthetic_score     REAL,
            aesthetic_label     TEXT,
            aesthetic_reason    TEXT,
            provider            TEXT,  -- which cloud model gave this score
            analyzed_at         TEXT
        );

        -- ── face_identities ───────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS face_identities (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_name  TEXT    NOT NULL UNIQUE,
            display_name    TEXT,
            aliases         TEXT    DEFAULT '[]',  -- JSON array
            notes           TEXT,
            created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
            updated_at      TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_face_identities_name
            ON face_identities(canonical_name);

        -- ── face_embeddings ───────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            identity_id     INTEGER NOT NULL REFERENCES face_identities(id) ON DELETE CASCADE,
            embedding       BLOB    NOT NULL,  -- 512-d float32
            source_image    TEXT,
            registered_at   TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        -- ── overrides ─────────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS overrides (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id        INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            table_name      TEXT    NOT NULL,
            field_name      TEXT    NOT NULL,
            value           TEXT,  -- JSON-encoded value
            overridden_by   TEXT    DEFAULT 'user',
            overridden_at   TEXT    NOT NULL DEFAULT (datetime('now')),
            note            TEXT,
            UNIQUE(image_id, table_name, field_name)
        );
        CREATE INDEX IF NOT EXISTS idx_overrides_image
            ON overrides(image_id, table_name, field_name);

        -- ── job_queue ─────────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS job_queue (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id        INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            module          TEXT    NOT NULL,
            priority        INTEGER DEFAULT 0,
            status          TEXT    NOT NULL DEFAULT 'pending',
            attempts        INTEGER DEFAULT 0,
            max_attempts    INTEGER DEFAULT 3,
            error_message   TEXT,
            queued_at       TEXT    NOT NULL DEFAULT (datetime('now')),
            started_at      TEXT,
            completed_at    TEXT,
            skip_reason     TEXT,
            UNIQUE(image_id, module)
        );
        CREATE INDEX IF NOT EXISTS idx_job_queue_status ON job_queue(status, priority DESC);
        CREATE INDEX IF NOT EXISTS idx_job_queue_image_module
            ON job_queue(image_id, module);

        -- ── embeddings (CLIP vectors) ─────────────────────────────────────
        CREATE TABLE IF NOT EXISTS embeddings (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id        INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            embedding_type  TEXT    NOT NULL,  -- 'image_clip'|'description_clip'
            vector          BLOB    NOT NULL,  -- float32 numpy array as bytes
            model_version   TEXT,
            computed_at     TEXT    NOT NULL DEFAULT (datetime('now')),
            UNIQUE(image_id, embedding_type)
        );
        CREATE INDEX IF NOT EXISTS idx_embeddings_image_type
            ON embeddings(image_id, embedding_type);

        -- ── FTS5 search index ─────────────────────────────────────────────
        CREATE VIRTUAL TABLE IF NOT EXISTS search_index USING fts5(
            image_id,
            description_text,
            subjects_text,
            keywords_text,
            faces_text,
            exif_text,
            content='',
            tokenize='porter unicode61'
        );
    """)


# ── Migration v2: Add missing metadata columns ────────────────────────────────

def _migrate_v2(conn: sqlite3.Connection) -> None:
    """Add camera_serial and any other metadata columns discovered post-v1."""
    # Use ADD COLUMN IF NOT EXISTS (SQLite 3.37+). Fall back to try/except for older.
    for col_def in (
        "camera_serial TEXT",
        "software TEXT",
        "copyright TEXT",
        "artist TEXT",
    ):
        col_name = col_def.split()[0]
        try:
            conn.execute(
                f"ALTER TABLE analysis_metadata ADD COLUMN {col_def}"
            )
        except Exception:
            pass  # column already exists


# ── Migration v3: Individual AI pass tables ───────────────────────────────────

def _migrate_v3(conn: sqlite3.Connection) -> None:
    """Add split tables for the individual AI pipeline passes (blip2, objects, ocr, faces)."""
    conn.executescript("""
        -- ── analysis_blip2 ────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS analysis_blip2 (
            image_id     INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
            description  TEXT,
            scene_type   TEXT,
            main_subject TEXT,
            lighting     TEXT,
            mood         TEXT,
            keywords     TEXT,   -- JSON array
            analyzed_at  TEXT
        );

        -- ── analysis_objects ──────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS analysis_objects (
            image_id          INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
            detected_objects  TEXT,   -- JSON array of "label:pct%" strings
            has_person        INTEGER DEFAULT 0,
            has_text          INTEGER DEFAULT 0,
            text_boxes        TEXT,   -- JSON array of [x0,y0,x1,y1] lists
            analyzed_at       TEXT
        );

        -- ── analysis_ocr ──────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS analysis_ocr (
            image_id    INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
            ocr_text    TEXT,
            analyzed_at TEXT
        );

        -- ── analysis_faces ────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS analysis_faces (
            image_id        INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
            face_count      INTEGER DEFAULT 0,
            face_identities TEXT,   -- JSON array of name strings
            face_details    TEXT,   -- JSON array of "name:age:gender" strings
            analyzed_at     TEXT
        );
    """)


# ── Migration v4: Performance indexes ────────────────────────────────────────

def _migrate_v4(conn: sqlite3.Connection) -> None:
    """Add composite and covering indexes for performance-critical queries."""
    conn.executescript("""
        -- Composite index for queue claim(): status + module + priority + queued_at
        CREATE INDEX IF NOT EXISTS idx_job_queue_claim
            ON job_queue(status, module, priority DESC, queued_at ASC);

        -- Metadata query indexes
        CREATE INDEX IF NOT EXISTS idx_metadata_date
            ON analysis_metadata(date_time_original);
        CREATE INDEX IF NOT EXISTS idx_metadata_iso
            ON analysis_metadata(iso);
        CREATE INDEX IF NOT EXISTS idx_metadata_camera
            ON analysis_metadata(camera_make, camera_model);

        -- Face identity lookup by display_name
        CREATE INDEX IF NOT EXISTS idx_face_identities_display
            ON face_identities(display_name);
    """)


# ── Migration v5: Face aliases table ─────────────────────────────────────────

def _migrate_v5(conn: sqlite3.Connection) -> None:
    """Create an indexed ``face_aliases`` table and migrate existing JSON aliases."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS face_aliases (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            identity_id  INTEGER NOT NULL REFERENCES face_identities(id) ON DELETE CASCADE,
            alias        TEXT    NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_face_aliases_alias
            ON face_aliases(alias);
        CREATE INDEX IF NOT EXISTS idx_face_aliases_identity
            ON face_aliases(identity_id);
    """)

    # Migrate existing JSON aliases from face_identities.aliases column
    rows = conn.execute(
        "SELECT id, aliases FROM face_identities WHERE aliases IS NOT NULL AND aliases != '[]'"
    ).fetchall()
    for row in rows:
        try:
            aliases = json.loads(row["aliases"])
        except (json.JSONDecodeError, TypeError):
            continue
        for alias in aliases:
            if alias:
                conn.execute(
                    "INSERT INTO face_aliases (identity_id, alias) VALUES (?, ?)",
                    [row["id"], alias],
                )


# ── Migration v6: UNIQUE constraint on job_queue(image_id, module) ───────────

def _migrate_v6(conn: sqlite3.Connection) -> None:
    """Add UNIQUE(image_id, module) to job_queue to prevent duplicate jobs.

    SQLite cannot ALTER TABLE to add a constraint, so we recreate the table.
    Duplicate rows (same image_id + module) are deduplicated by keeping the
    row with the highest id (most recent).
    """
    conn.executescript("""
        -- Deduplicate: keep only the latest row per (image_id, module)
        DELETE FROM job_queue
        WHERE id NOT IN (
            SELECT MAX(id) FROM job_queue GROUP BY image_id, module
        );

        -- Recreate with UNIQUE constraint
        CREATE TABLE job_queue_new (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id        INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            module          TEXT    NOT NULL,
            priority        INTEGER DEFAULT 0,
            status          TEXT    NOT NULL DEFAULT 'pending',
            attempts        INTEGER DEFAULT 0,
            max_attempts    INTEGER DEFAULT 3,
            error_message   TEXT,
            queued_at       TEXT    NOT NULL DEFAULT (datetime('now')),
            started_at      TEXT,
            completed_at    TEXT,
            skip_reason     TEXT,
            UNIQUE(image_id, module)
        );

        INSERT INTO job_queue_new
            SELECT * FROM job_queue;

        DROP TABLE job_queue;
        ALTER TABLE job_queue_new RENAME TO job_queue;

        CREATE INDEX IF NOT EXISTS idx_job_queue_status
            ON job_queue(status, priority DESC);
        CREATE INDEX IF NOT EXISTS idx_job_queue_image_module
            ON job_queue(image_id, module);
    """)


# ── Migration v7: Per-face occurrence table for clustering & crops ───────────

def _migrate_v7(conn: sqlite3.Connection) -> None:
    """Add ``face_occurrences`` table to store per-face bounding boxes and
    embeddings for each detected face in each image.

    This enables:
    - Face crop thumbnails (bbox coordinates)
    - Embedding-based clustering of unknown faces
    - Per-face linking to cluster / identity
    """
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS face_occurrences (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id    INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            face_idx    INTEGER NOT NULL,          -- 0-based index within the image
            bbox_x1     REAL    NOT NULL,           -- bounding box (pixel coords)
            bbox_y1     REAL    NOT NULL,
            bbox_x2     REAL    NOT NULL,
            bbox_y2     REAL    NOT NULL,
            embedding   BLOB,                       -- 512-d float32 (2048 bytes)
            age         INTEGER,
            gender      TEXT,
            identity_name TEXT,                     -- matched name or 'Unknown'
            cluster_id  INTEGER,                    -- assigned by clustering algorithm
            created_at  TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(image_id, face_idx)
        );

        CREATE INDEX IF NOT EXISTS idx_face_occ_image
            ON face_occurrences(image_id);
        CREATE INDEX IF NOT EXISTS idx_face_occ_cluster
            ON face_occurrences(cluster_id);
        CREATE INDEX IF NOT EXISTS idx_face_occ_identity
            ON face_occurrences(identity_name);
    """)


def _migrate_v8(conn: sqlite3.Connection) -> None:
    """Add ``det_score`` column to ``face_occurrences`` for detection confidence filtering."""
    conn.execute("""
        ALTER TABLE face_occurrences ADD COLUMN det_score REAL
    """)


def _migrate_v9(conn: sqlite3.Connection) -> None:
    """Add ``thumbnail`` BLOB column to ``face_occurrences`` for pre-generated crops."""
    conn.execute("""
        ALTER TABLE face_occurrences ADD COLUMN thumbnail BLOB
    """)


def _migrate_v10(conn: sqlite3.Connection) -> None:
    """Add ``face_cluster_labels`` table for per-cluster display names."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS face_cluster_labels (
            cluster_id INTEGER PRIMARY KEY,
            display_name TEXT NOT NULL
        )
    """)


def _migrate_v11(conn: sqlite3.Connection) -> None:
    """Add ``face_persons`` table and ``person_id`` column on ``face_occurrences``."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS face_persons (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT NOT NULL,
            notes      TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        ALTER TABLE face_occurrences ADD COLUMN person_id INTEGER REFERENCES face_persons(id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_face_occurrences_person_id
        ON face_occurrences(person_id)
    """)


# ── Migration v12: Profiler tables ───────────────────────────────────────────

def _migrate_v12(conn: sqlite3.Connection) -> None:
    """Add profiler tables for batch processing performance analysis."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS profiler_runs (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at   TEXT NOT NULL,
            ended_at     TEXT,
            total_images INTEGER DEFAULT 0,
            gpu_name     TEXT,
            gpu_vram_gb  REAL,
            cpu_count    INTEGER,
            ram_gb       REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS profiler_events (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          INTEGER NOT NULL REFERENCES profiler_runs(id) ON DELETE CASCADE,
            image_id        INTEGER,
            module          TEXT,
            phase           INTEGER,
            event_type      TEXT NOT NULL,
            start_ts        REAL NOT NULL,
            duration_ms     REAL NOT NULL,
            thread_name     TEXT,
            batch_size      INTEGER DEFAULT 1,
            image_file_size INTEGER,
            image_format    TEXT,
            image_width     INTEGER,
            image_height    INTEGER
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_profiler_events_run
        ON profiler_events(run_id, event_type)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_profiler_events_module
        ON profiler_events(run_id, module)
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS profiler_snapshots (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id               INTEGER NOT NULL REFERENCES profiler_runs(id) ON DELETE CASCADE,
            ts                   REAL NOT NULL,
            gpu_util_pct         REAL,
            gpu_mem_used_mb      REAL,
            gpu_mem_total_mb     REAL,
            cpu_pct              REAL,
            ram_used_mb          REAL,
            prefetch_queue_depth INTEGER
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_profiler_snapshots_run
        ON profiler_snapshots(run_id)
    """)


# ── Migration v13: Corrupt files table ───────────────────────────────────────

def _migrate_v13(conn: sqlite3.Connection) -> None:
    """Add ``corrupt_files`` table to persist paths of unreadable/corrupt files."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS corrupt_files (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id    INTEGER REFERENCES images(id) ON DELETE CASCADE,
            file_path   TEXT    NOT NULL,
            error_msg   TEXT,
            detected_at TEXT    NOT NULL DEFAULT (datetime('now')),
            UNIQUE(file_path)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_corrupt_files_image_id
        ON corrupt_files(image_id)
    """)


# ── Migration v14: Covering indexes for face cluster queries ─────────────────

def _migrate_v14(conn: sqlite3.Connection) -> None:
    """Add indexes to speed up face cluster aggregation queries."""
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_face_occ_cluster_identity
        ON face_occurrences(cluster_id, identity_name)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_face_occ_cluster_person
        ON face_occurrences(cluster_id, person_id)
    """)


# ── Migration v15: Distributed worker lease tables ────────────────────────────

def _migrate_v15(conn: sqlite3.Connection) -> None:
    """Add worker registry + lease tables for distributed execution."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS worker_nodes (
            id              TEXT PRIMARY KEY,
            display_name    TEXT NOT NULL,
            platform        TEXT,
            capabilities    TEXT,  -- JSON object
            status          TEXT NOT NULL DEFAULT 'offline',
            last_heartbeat  TEXT,
            created_at      TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS job_leases (
            job_id            INTEGER PRIMARY KEY REFERENCES job_queue(id) ON DELETE CASCADE,
            worker_id         TEXT NOT NULL REFERENCES worker_nodes(id) ON DELETE CASCADE,
            lease_token       TEXT NOT NULL UNIQUE,
            leased_at         TEXT NOT NULL DEFAULT (datetime('now')),
            heartbeat_at      TEXT NOT NULL DEFAULT (datetime('now')),
            lease_expires_at  TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_job_leases_worker
        ON job_leases(worker_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_job_leases_expiry
        ON job_leases(lease_expires_at)
    """)


# ── Migration v16: Track node ownership on queue rows ─────────────────────────

def _migrate_v16(conn: sqlite3.Connection) -> None:
    """Persist the last node that touched a queue row for live progress UI."""
    for col_def in (
        "last_node_id TEXT",
        "last_node_role TEXT",
    ):
        try:
            conn.execute(f"ALTER TABLE job_queue ADD COLUMN {col_def}")
        except Exception:
            pass

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_job_queue_node_status
        ON job_queue(last_node_role, last_node_id, status, completed_at)
    """)


# ── Migration v17: Perception analysis table ──────────────────────────────────

def _migrate_v17(conn: sqlite3.Connection) -> None:
    """Add perception analysis table (UniPercept IAA / IQA / ISTA scores)."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analysis_perception (
            image_id              INTEGER PRIMARY KEY
                                  REFERENCES images(id) ON DELETE CASCADE,
            perception_iaa        REAL,
            perception_iaa_label  TEXT,
            perception_iqa        REAL,
            perception_iqa_label  TEXT,
            perception_ista       REAL,
            perception_ista_label TEXT,
            analyzed_at           TEXT
        )
    """)


# ── Migration v18: Worker module affinity tracking ────────────────────────────

def _migrate_v18(conn: sqlite3.Connection) -> None:
    """Add last_module column to worker_nodes for module-affinity scheduling."""
    try:
        conn.execute("ALTER TABLE worker_nodes ADD COLUMN last_module TEXT")
    except sqlite3.OperationalError:
        pass  # column already exists


def _migrate_v19(conn: sqlite3.Connection) -> None:
    """Rename analysis_local_ai → analysis_caption.

    The ``local_ai`` module has been renamed to ``caption`` — all modules now
    use local AI models, so the old name was meaningless.  The table keeps the
    same columns; only the name changes.
    """
    # Rename the table (safe for existing data and indexes)
    try:
        conn.execute("ALTER TABLE analysis_local_ai RENAME TO analysis_caption")
    except sqlite3.OperationalError:
        # Table may already be named analysis_caption (fresh install) or missing
        pass


def _migrate_v20(conn: sqlite3.Connection) -> None:
    """Add unified scheduler state columns/tables.

    - Worker control state (pause/resume) and affinity epoch fields on worker_nodes
    - Per-worker per-module runtime statistics for ETA-aware balancing
    """
    for col_def in (
        "desired_state TEXT NOT NULL DEFAULT 'active'",
        "state_reason TEXT",
        "state_updated_at TEXT",
        "epoch_module TEXT",
        "epoch_expires_at TEXT",
    ):
        try:
            conn.execute(f"ALTER TABLE worker_nodes ADD COLUMN {col_def}")
        except sqlite3.OperationalError:
            pass  # column already exists

    conn.execute(
        """UPDATE worker_nodes
           SET desired_state = COALESCE(desired_state, 'active')"""
    )

    conn.execute(
        """CREATE TABLE IF NOT EXISTS worker_module_stats (
               worker_id      TEXT NOT NULL REFERENCES worker_nodes(id) ON DELETE CASCADE,
               module         TEXT NOT NULL,
               avg_ms         REAL NOT NULL,
               samples        INTEGER NOT NULL DEFAULT 1,
               updated_at     TEXT NOT NULL DEFAULT (datetime('now')),
               PRIMARY KEY(worker_id, module)
           )"""
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_worker_module_stats_updated
           ON worker_module_stats(updated_at)"""
    )


# ── Migration v21: Add processing_ms to job_queue ────────────────────────────

def _migrate_v21(conn: sqlite3.Connection) -> None:
    """Add processing_ms column to job_queue for accurate per-job timing.

    Previously, module_avg_processing_ms() used ``completed_at − started_at``
    which is inflated for batch-claimed jobs because ``started_at`` is set at
    claim time for the entire batch, not when each job actually begins
    processing.  Storing the actual processing duration fixes this.
    """
    try:
        conn.execute("ALTER TABLE job_queue ADD COLUMN processing_ms INTEGER")
    except sqlite3.OperationalError:
        pass  # column already exists


def _migrate_v22(conn: sqlite3.Connection) -> None:
    """Clear pre-generated face thumbnails so they regenerate with EXIF orientation.

    Previously, face crop thumbnails were generated from un-rotated pixel data,
    causing upside-down or sideways crops for images with EXIF orientation tags.
    Clearing the cached thumbnails forces on-the-fly regeneration with the fix.
    """
    conn.execute("UPDATE face_occurrences SET thumbnail = NULL WHERE thumbnail IS NOT NULL")


def _migrate_v23(conn: sqlite3.Connection) -> None:
    """Re-clear face thumbnails after fixing detection-resolution scale factor.

    Migration v22 cleared thumbnails, but on-the-fly regeneration used a
    hardcoded 1920px detection long-edge while the actual value was 1024px.
    This caused crops from wrong image regions.  Also fixes HEIF/AVIF
    double-rotation (pillow-heif auto-applies EXIF orientation).
    """
    conn.execute("UPDATE face_occurrences SET thumbnail = NULL WHERE thumbnail IS NOT NULL")


def _migrate_v24(conn: sqlite3.Connection) -> None:
    """Re-clear face thumbnails after improving detection-resolution heuristic.

    The simple max(bbox) > 1024 heuristic failed for faces near the origin of
    large images analyzed at legacy 1920px.  Now uses aspect-ratio-aware check:
    if bbox exceeds the 1024-detection-frame dimensions (accounting for the
    image's aspect ratio), we know it was analyzed at 1920px.
    """
    conn.execute("UPDATE face_occurrences SET thumbnail = NULL WHERE thumbnail IS NOT NULL")


def _migrate_v25(conn: sqlite3.Connection) -> None:
    """Add face_cluster_deferred table for parking clusters for later review."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS face_cluster_deferred (
            cluster_id INTEGER PRIMARY KEY
        )
    """)


def _migrate_v26(conn: sqlite3.Connection) -> None:
    """Add denormalized search_features table for modular search/reranking."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS search_features (
            image_id             INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
            desc_lex             TEXT,
            desc_summary         TEXT,
            desc_quality         REAL,
            keywords_text        TEXT,
            objects_text         TEXT,
            ocr_text             TEXT,
            faces_text           TEXT,
            camera_make          TEXT,
            camera_model         TEXT,
            lens_model           TEXT,
            date_time_original   TEXT,
            location_city        TEXT,
            location_state       TEXT,
            location_country     TEXT,
            sharpness_score      REAL,
            noise_level          REAL,
            snr_db               REAL,
            dynamic_range_stops  REAL,
            perception_iaa       REAL,
            perception_iqa       REAL,
            perception_ista      REAL,
            aesthetic_score      REAL,
            face_count           INTEGER,
            has_people           INTEGER,
            updated_at           TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_search_features_date
            ON search_features(date_time_original);
        CREATE INDEX IF NOT EXISTS idx_search_features_country
            ON search_features(location_country);
        CREATE INDEX IF NOT EXISTS idx_search_features_face_count
            ON search_features(face_count);
        CREATE INDEX IF NOT EXISTS idx_search_features_quality
            ON search_features(perception_iaa, sharpness_score, noise_level);
    """)


def _migrate_v27(conn: sqlite3.Connection) -> None:
    """Enable safer FTS delete semantics where supported.

    Existing databases may have large contentless FTS indexes; rebuilding them
    in migration could be very expensive. We only recreate the table when it's
    empty. Otherwise, runtime update logic handles legacy contentless deletes.
    """
    exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='search_index'"
    ).fetchone()
    if exists is None:
        return

    row = conn.execute("SELECT COUNT(*) AS cnt FROM search_index").fetchone()
    if row is None or int(row["cnt"]) != 0:
        return

    conn.execute("DROP TABLE IF EXISTS search_index")
    try:
        conn.execute(
            """
            CREATE VIRTUAL TABLE search_index USING fts5(
                image_id,
                description_text,
                subjects_text,
                keywords_text,
                faces_text,
                exif_text,
                content='',
                contentless_delete=1,
                tokenize='porter unicode61'
            )
            """
        )
    except sqlite3.OperationalError:
        conn.execute(
            """
            CREATE VIRTUAL TABLE search_index USING fts5(
                image_id,
                description_text,
                subjects_text,
                keywords_text,
                faces_text,
                exif_text,
                content='',
                tokenize='porter unicode61'
            )
            """
        )


# ── Migration v28: Force-rebuild FTS with contentless_delete=1 ─────────────────

def _migrate_v28(conn: sqlite3.Connection) -> None:
    """Rebuild search_index FTS table with ``contentless_delete=1``.

    Migration v27 only upgraded empty tables.  Databases with existing rows
    kept the legacy contentless FTS layout where every DELETE requires a full
    ``fts5vocab`` scan (~3.4 s per image on a 1M-row index).  This migration
    drops the table unconditionally, recreates it with ``contentless_delete=1``
    (SQLite ≥ 3.43.0), and repopulates from analysis data.

    If the SQLite version is too old for ``contentless_delete=1`` the table is
    recreated without it — the same as before, so no regression.
    """
    import sys

    exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='search_index'"
    ).fetchone()

    # Check if the table already has contentless_delete=1
    if exists:
        ddl = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='search_index'"
        ).fetchone()
        if ddl and ddl[0] and "contentless_delete=1" in ddl[0]:
            return  # already upgraded, nothing to do

    # Drop old table and helper vocab table
    conn.execute("DROP TABLE IF EXISTS search_index_vocab")
    conn.execute("DROP TABLE IF EXISTS search_index")

    # Recreate with contentless_delete=1 if supported
    try:
        conn.execute("""
            CREATE VIRTUAL TABLE search_index USING fts5(
                image_id,
                description_text,
                subjects_text,
                keywords_text,
                faces_text,
                exif_text,
                content='',
                contentless_delete=1,
                tokenize='porter unicode61'
            )
        """)
    except sqlite3.OperationalError:
        # SQLite too old for contentless_delete — fall back to plain contentless
        conn.execute("""
            CREATE VIRTUAL TABLE search_index USING fts5(
                image_id,
                description_text,
                subjects_text,
                keywords_text,
                faces_text,
                exif_text,
                content='',
                tokenize='porter unicode61'
            )
        """)

    # Repopulate from existing analysis data.
    # We do a lightweight direct-insert here rather than going through
    # Repository.update_search_index to avoid importing the full module
    # and to keep the migration self-contained.
    _repopulate_fts_v28(conn)


def _repopulate_fts_v28(conn: sqlite3.Connection) -> None:
    """Bulk-insert FTS rows from analysis tables (migration helper)."""
    import sys

    image_ids = [
        r[0] for r in conn.execute("SELECT id FROM images ORDER BY id").fetchall()
    ]
    total = len(image_ids)
    if total == 0:
        return

    batch_size = 500
    inserted = 0

    def _table_exists(name: str) -> bool:
        return conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", [name]
        ).fetchone() is not None

    has_blip2 = _table_exists("analysis_blip2")
    has_faces = _table_exists("analysis_faces")
    has_face_occ = _table_exists("face_occurrences")
    has_face_persons = _table_exists("face_persons")
    has_face_labels = _table_exists("face_cluster_labels")

    for start in range(0, total, batch_size):
        batch = image_ids[start : start + batch_size]
        placeholders = ",".join("?" for _ in batch)

        # Pre-fetch all analysis rows for this batch keyed by image_id
        caption_map: dict[int, sqlite3.Row] = {}
        for r in conn.execute(
            f"SELECT * FROM analysis_caption WHERE image_id IN ({placeholders})", batch
        ).fetchall():
            caption_map[r["image_id"]] = r

        blip2_map: dict[int, sqlite3.Row] = {}
        if has_blip2:
            for r in conn.execute(
                f"SELECT * FROM analysis_blip2 WHERE image_id IN ({placeholders})", batch
            ).fetchall():
                blip2_map[r["image_id"]] = r

        cloud_map: dict[int, list[sqlite3.Row]] = {}
        for r in conn.execute(
            f"SELECT * FROM analysis_cloud_ai WHERE image_id IN ({placeholders})", batch
        ).fetchall():
            cloud_map.setdefault(r["image_id"], []).append(r)

        meta_map: dict[int, sqlite3.Row] = {}
        for r in conn.execute(
            f"SELECT * FROM analysis_metadata WHERE image_id IN ({placeholders})", batch
        ).fetchall():
            meta_map[r["image_id"]] = r

        faces_map: dict[int, sqlite3.Row] = {}
        if has_faces:
            for r in conn.execute(
                f"SELECT * FROM analysis_faces WHERE image_id IN ({placeholders})", batch
            ).fetchall():
                faces_map[r["image_id"]] = r

        occ_map: dict[int, list[tuple[str | None, str | None]]] = {}
        if has_face_occ:
            person_sel = "fp.name" if has_face_persons else "NULL"
            person_join = "LEFT JOIN face_persons fp ON fp.id = fo.person_id" if has_face_persons else ""
            label_sel = "fcl.display_name" if has_face_labels else "NULL"
            label_join = (
                "LEFT JOIN face_cluster_labels fcl ON fcl.cluster_id = fo.cluster_id"
                if has_face_labels else ""
            )
            for r in conn.execute(
                f"""SELECT DISTINCT fo.image_id, {person_sel} AS pname, {label_sel} AS lbl
                    FROM face_occurrences fo {person_join} {label_join}
                    WHERE fo.image_id IN ({placeholders})""",
                batch,
            ).fetchall():
                occ_map.setdefault(r["image_id"], []).append((r["pname"], r["lbl"]))

        rows_to_insert: list[tuple] = []
        for image_id in batch:
            desc_parts: list[str] = []
            subjects_parts: list[str] = []
            kw_parts: list[str] = []
            faces_parts: list[str] = []
            exif_parts: list[str] = []

            local = caption_map.get(image_id)
            if local:
                if local["description"]:
                    desc_parts.append(local["description"])
                if local["main_subject"]:
                    subjects_parts.append(local["main_subject"])
                if local["scene_type"]:
                    subjects_parts.append(local["scene_type"])
                if local["keywords"]:
                    try:
                        kw_parts.extend(json.loads(local["keywords"]))
                    except (json.JSONDecodeError, TypeError):
                        pass
                if local["face_identities"]:
                    try:
                        faces_parts.extend(json.loads(local["face_identities"]))
                    except (json.JSONDecodeError, TypeError):
                        pass
                if local["mood"]:
                    kw_parts.append(local["mood"])
                if local["lighting"]:
                    kw_parts.append(local["lighting"])

            blip2 = blip2_map.get(image_id)
            if blip2:
                if blip2["description"] and blip2["description"] not in desc_parts:
                    desc_parts.append(blip2["description"])
                if blip2["main_subject"] and blip2["main_subject"] not in subjects_parts:
                    subjects_parts.append(blip2["main_subject"])
                if blip2["scene_type"] and blip2["scene_type"] not in subjects_parts:
                    subjects_parts.append(blip2["scene_type"])
                if blip2["keywords"]:
                    try:
                        for kw in json.loads(blip2["keywords"]):
                            if kw not in kw_parts:
                                kw_parts.append(kw)
                    except (json.JSONDecodeError, TypeError):
                        pass
                if blip2["mood"] and blip2["mood"] not in kw_parts:
                    kw_parts.append(blip2["mood"])
                if blip2["lighting"] and blip2["lighting"] not in kw_parts:
                    kw_parts.append(blip2["lighting"])

            faces_row = faces_map.get(image_id)
            if faces_row and faces_row["face_identities"]:
                try:
                    for name in json.loads(faces_row["face_identities"]):
                        if name not in faces_parts:
                            faces_parts.append(name)
                except (json.JSONDecodeError, TypeError):
                    pass

            for pname, lbl in occ_map.get(image_id, []):
                if pname and pname not in faces_parts:
                    faces_parts.append(pname)
                if lbl and lbl not in faces_parts:
                    faces_parts.append(lbl)

            for cloud in cloud_map.get(image_id, []):
                if cloud["description"]:
                    desc_parts.append(cloud["description"])
                if cloud["main_subject"]:
                    subjects_parts.append(cloud["main_subject"])
                if cloud["scene_type"]:
                    subjects_parts.append(cloud["scene_type"])
                if cloud["keywords"]:
                    try:
                        kw_parts.extend(json.loads(cloud["keywords"]))
                    except (json.JSONDecodeError, TypeError):
                        pass

            meta = meta_map.get(image_id)
            if meta:
                for field in (
                    "camera_make", "camera_model", "lens_model",
                    "location_city", "location_state", "location_country",
                ):
                    if meta[field]:
                        exif_parts.append(meta[field])

            kw_parts = list(dict.fromkeys(kw_parts))
            faces_parts = list(dict.fromkeys(faces_parts))

            rows_to_insert.append((
                image_id,
                str(image_id),
                " ".join(desc_parts),
                " ".join(subjects_parts),
                " ".join(kw_parts),
                " ".join(faces_parts),
                " ".join(exif_parts),
            ))

        conn.executemany(
            """INSERT INTO search_index
               (rowid, image_id, description_text, subjects_text,
                keywords_text, faces_text, exif_text)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            rows_to_insert,
        )
        inserted += len(rows_to_insert)
        sys.stderr.write(
            f"\r  FTS rebuild: {inserted}/{total} images indexed"
        )

    if total:
        sys.stderr.write(
            f"\r  FTS rebuild: {inserted}/{total} images indexed — done\n"
        )


# ── Migration v29: Covering index for face_occurrences cluster stats ───────────

def _migrate_v29(conn: sqlite3.Connection) -> None:
    """Add covering indexes for fast face cluster/person aggregation.

    ``list_face_clusters`` uses ``GROUP BY cluster_id`` with
    ``COUNT(DISTINCT image_id)`` — the covering index lets SQLite answer
    from the index alone.

    ``list_persons`` uses ``GROUP BY person_id`` with ``COUNT(DISTINCT
    cluster_id)`` and ``COUNT(DISTINCT image_id)`` — same principle.
    """
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_face_occ_cluster_image "
        "ON face_occurrences(cluster_id, image_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_face_occ_person_cluster_image "
        "ON face_occurrences(person_id, cluster_id, image_id)"
    )


def _migrate_v30(conn: sqlite3.Connection) -> None:
    """Add R*tree spatial index and geohash column for map feature.

    1. ``geo_rtree`` — R*tree virtual table for O(log N) bounding-box queries.
    2. ``geohash`` column on ``analysis_metadata`` — precomputed 8-char geohash
       for fast ``GROUP BY`` clustering at various zoom levels.
    3. Backfill both from existing GPS data.
    """
    from imganalyzer.db.geohash import encode as geohash_encode

    # 1. R*tree spatial index
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS geo_rtree USING rtree(
            id,
            min_lat, max_lat,
            min_lng, max_lng
        )
    """)

    # 2. Geohash column
    try:
        conn.execute("ALTER TABLE analysis_metadata ADD COLUMN geohash TEXT")
    except sqlite3.OperationalError:
        pass  # column already exists

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_metadata_geohash "
        "ON analysis_metadata(geohash)"
    )

    # 3. Backfill R*tree and geohash from existing GPS data (cursor-based
    #    iteration to avoid loading all rows into memory at once).
    cursor = conn.execute(
        "SELECT image_id, gps_latitude, gps_longitude "
        "FROM analysis_metadata "
        "WHERE gps_latitude IS NOT NULL AND gps_longitude IS NOT NULL"
    )
    batch: list[tuple] = []
    for row in cursor:
        image_id, lat, lng = row[0], row[1], row[2]
        gh = geohash_encode(lat, lng, precision=8)
        conn.execute(
            "INSERT OR REPLACE INTO geo_rtree (id, min_lat, max_lat, min_lng, max_lng) "
            "VALUES (?, ?, ?, ?, ?)",
            [image_id, lat, lat, lng, lng],
        )
        batch.append((gh, image_id))
        if len(batch) >= 5000:
            conn.executemany(
                "UPDATE analysis_metadata SET geohash = ? WHERE image_id = ?",
                batch,
            )
            batch.clear()
    if batch:
        conn.executemany(
            "UPDATE analysis_metadata SET geohash = ? WHERE image_id = ?",
            batch,
        )


def _migrate_v31(conn: sqlite3.Connection) -> None:
    """Add gps_source column to track origin of GPS coordinates.

    Values: 'exif' (original EXIF data), 'inferred' (gap filler),
            'manual' (user override).
    """
    try:
        conn.execute(
            "ALTER TABLE analysis_metadata ADD COLUMN gps_source TEXT DEFAULT 'exif'"
        )
    except sqlite3.OperationalError:
        pass  # column already exists

    # Backfill: set gps_source='exif' for all rows that already have GPS
    conn.execute(
        "UPDATE analysis_metadata SET gps_source = 'exif' "
        "WHERE gps_latitude IS NOT NULL AND gps_longitude IS NOT NULL "
        "AND gps_source IS NULL"
    )


def _migrate_v32(conn: sqlite3.Connection) -> None:
    """Add smart albums, story chapters, moments, and related tables.

    Smart albums store rule-based image collections with materialized
    membership for fast lookups.  Story chapters and moments provide a
    hierarchical narrative structure on top of album images.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS smart_albums (
            id              TEXT PRIMARY KEY,
            name            TEXT NOT NULL,
            description     TEXT,
            cover_image_id  INTEGER REFERENCES images(id) ON DELETE SET NULL,
            rules           TEXT NOT NULL,
            story_enabled   INTEGER DEFAULT 1,
            sort_order      TEXT DEFAULT 'chronological',
            item_count      INTEGER DEFAULT 0,
            chapter_count   INTEGER DEFAULT 0,
            created_at      TEXT DEFAULT (datetime('now')),
            updated_at      TEXT DEFAULT (datetime('now'))
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS album_items (
            album_id    TEXT NOT NULL REFERENCES smart_albums(id) ON DELETE CASCADE,
            image_id    INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            added_at    TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (album_id, image_id)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_album_items_image "
        "ON album_items(image_id)"
    )

    conn.execute("""
        CREATE TABLE IF NOT EXISTS story_chapters (
            id              TEXT PRIMARY KEY,
            album_id        TEXT NOT NULL REFERENCES smart_albums(id) ON DELETE CASCADE,
            title           TEXT,
            summary         TEXT,
            sort_order      INTEGER NOT NULL,
            start_date      TEXT,
            end_date        TEXT,
            location        TEXT,
            cover_image_id  INTEGER REFERENCES images(id) ON DELETE SET NULL,
            image_count     INTEGER DEFAULT 0,
            moment_count    INTEGER DEFAULT 0
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chapters_album "
        "ON story_chapters(album_id, sort_order)"
    )

    conn.execute("""
        CREATE TABLE IF NOT EXISTS story_moments (
            id              TEXT PRIMARY KEY,
            chapter_id      TEXT NOT NULL REFERENCES story_chapters(id) ON DELETE CASCADE,
            title           TEXT,
            sort_order      INTEGER NOT NULL,
            start_time      TEXT,
            end_time        TEXT,
            lat             REAL,
            lng             REAL,
            hero_image_id   INTEGER REFERENCES images(id) ON DELETE SET NULL,
            image_count     INTEGER DEFAULT 0
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_moments_chapter "
        "ON story_moments(chapter_id, sort_order)"
    )

    conn.execute("""
        CREATE TABLE IF NOT EXISTS moment_images (
            moment_id   TEXT NOT NULL REFERENCES story_moments(id) ON DELETE CASCADE,
            image_id    INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            sort_order  INTEGER NOT NULL,
            is_hero     INTEGER DEFAULT 0,
            PRIMARY KEY (moment_id, image_id)
        )
    """)


def _migrate_v33(conn: sqlite3.Connection) -> None:
    """Add geocode_cache table for shared, persistent reverse-geocoding cache.

    Reverse geocoding is moved out of the synchronous metadata ingest path
    (see imganalyzer.analysis.geocode_resolver).  Results are keyed by
    (lat, lon) rounded to 4 decimal places (~11 m) so that images taken
    at the same location reuse a single HTTP request.  The cache is
    shared across all workers via SQLite.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS geocode_cache (
            lat_key         REAL    NOT NULL,
            lon_key         REAL    NOT NULL,
            location_json   TEXT    NOT NULL,
            fetched_at      INTEGER NOT NULL,
            PRIMARY KEY (lat_key, lon_key)
        )
    """)
