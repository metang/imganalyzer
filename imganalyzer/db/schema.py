"""Database schema — CREATE TABLE statements and migration runner.

The migration system uses a simple version counter stored in a ``schema_version``
table.  Each migration is a function ``_migrate_vN(conn)`` that runs the DDL for
version *N*.  ``ensure_schema`` applies all pending migrations in order.
"""
from __future__ import annotations

import json
import sqlite3

# ── Current schema version ────────────────────────────────────────────────────
SCHEMA_VERSION = 5


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

        -- ── analysis_local_ai ─────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS analysis_local_ai (
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
            skip_reason     TEXT
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

