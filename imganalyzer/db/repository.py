"""Repository — CRUD operations for all analysis tables.

All write operations that touch analysis data use the atomic-write pattern:
the caller builds the full result dict in memory, then calls a single
``upsert_*`` method which writes everything inside one transaction.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── Module → table mapping ────────────────────────────────────────────────────
MODULE_TABLE_MAP: dict[str, str] = {
    "metadata":   "analysis_metadata",
    "technical":  "analysis_technical",
    "local_ai":   "analysis_local_ai",
    "blip2":      "analysis_blip2",
    "objects":    "analysis_objects",
    "ocr":        "analysis_ocr",
    "faces":      "analysis_faces",
    "cloud_ai":   "analysis_cloud_ai",
    "aesthetic":  "analysis_aesthetic",
    "embedding":  "embeddings",
}

ALL_MODULES = list(MODULE_TABLE_MAP.keys())


class Repository:
    """Data access layer for the imganalyzer database."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self._column_cache: dict[str, set[str]] = {}
        # Negative cache: set of image IDs that have ANY overrides.
        # Loaded lazily on first _apply_override_mask call.  At 500K images
        # with ~0.01% override rate, this avoids ~4.5M empty SELECT queries.
        self._override_ids: set[int] | None = None

    def _known_columns(self, table: str) -> set[str]:
        """Return the set of column names for *table* (cached per instance)."""
        if table not in self._column_cache:
            rows = self.conn.execute(f"PRAGMA table_info({table})").fetchall()
            self._column_cache[table] = {r["name"] for r in rows}
        return self._column_cache[table]

    def _filter_to_known_columns(self, table: str, data: dict[str, Any]) -> dict[str, Any]:
        """Drop keys from *data* that don't correspond to columns in *table*.

        This prevents INSERT failures when the analyser returns fields that were
        added after the schema was created (e.g. camera_serial, sharpness_raw).
        Unknown fields are silently discarded rather than crashing the whole
        transaction.
        """
        known = self._known_columns(table)
        return {k: v for k, v in data.items() if k in known}

    # ── images ─────────────────────────────────────────────────────────────

    def register_image(
        self,
        file_path: str,
        file_hash: str | None = None,
        file_size: int | None = None,
        width: int | None = None,
        height: int | None = None,
        fmt: str | None = None,
    ) -> int:
        """Insert or return existing image row.  Returns image_id."""
        row = self.conn.execute(
            "SELECT id FROM images WHERE file_path = ?", [file_path]
        ).fetchone()
        if row:
            return row["id"]
        cur = self.conn.execute(
            """INSERT INTO images (file_path, file_hash, file_size, width, height, format)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [file_path, file_hash, file_size, width, height, fmt],
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_image(self, image_id: int) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM images WHERE id = ?", [image_id]).fetchone()
        return dict(row) if row else None

    def get_image_by_path(self, file_path: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM images WHERE file_path = ?", [file_path]
        ).fetchone()
        return dict(row) if row else None

    def update_image(self, image_id: int, **fields: Any) -> None:
        if not fields:
            return
        sets = ", ".join(f"{k} = ?" for k in fields)
        vals = list(fields.values()) + [image_id]
        self.conn.execute(f"UPDATE images SET {sets} WHERE id = ?", vals)
        self.conn.commit()

    def count_images(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM images").fetchone()
        return row["cnt"]

    def iter_image_ids(self) -> list[int]:
        """Return all image IDs (for bulk operations)."""
        rows = self.conn.execute("SELECT id FROM images ORDER BY id").fetchall()
        return [r["id"] for r in rows]

    # ── Generic analysis CRUD ──────────────────────────────────────────────

    def get_analysis(self, image_id: int, module: str) -> dict[str, Any] | None:
        """Return the analysis row for *module*, or None if not yet analyzed."""
        table = MODULE_TABLE_MAP.get(module)
        if table is None:
            raise ValueError(f"Unknown module: {module}")

        if module == "cloud_ai":
            # cloud_ai can have multiple rows (one per provider)
            rows = self.conn.execute(
                f"SELECT * FROM {table} WHERE image_id = ?", [image_id]
            ).fetchall()
            if not rows:
                return None
            return {"providers": [dict(r) for r in rows]}

        row = self.conn.execute(
            f"SELECT * FROM {table} WHERE image_id = ?", [image_id]
        ).fetchone()
        return dict(row) if row else None

    def is_analyzed(self, image_id: int, module: str) -> bool:
        """Check if a module has completed analysis (has analyzed_at set)."""
        table = MODULE_TABLE_MAP.get(module)
        if table is None:
            raise ValueError(f"Unknown module: {module}")

        if module == "embedding":
            row = self.conn.execute(
                "SELECT 1 FROM embeddings WHERE image_id = ? LIMIT 1", [image_id]
            ).fetchone()
            return row is not None

        if module == "cloud_ai":
            row = self.conn.execute(
                "SELECT 1 FROM analysis_cloud_ai WHERE image_id = ? AND analyzed_at IS NOT NULL LIMIT 1",
                [image_id],
            ).fetchone()
            return row is not None

        row = self.conn.execute(
            f"SELECT analyzed_at FROM {table} WHERE image_id = ?", [image_id]
        ).fetchone()
        return row is not None and row["analyzed_at"] is not None

    def upsert_metadata(self, image_id: int, data: dict[str, Any]) -> None:
        """Atomic write of the full metadata analysis result."""
        data = self._filter_to_known_columns("analysis_metadata", data)
        data = self._apply_override_mask(image_id, "analysis_metadata", data)
        self.conn.execute("DELETE FROM analysis_metadata WHERE image_id = ?", [image_id])
        cols = ["image_id"] + list(data.keys()) + ["analyzed_at"]
        placeholders = ", ".join(["?"] * len(cols))
        col_str = ", ".join(cols)
        vals = [image_id] + list(data.values()) + [_now()]
        self.conn.execute(
            f"INSERT INTO analysis_metadata ({col_str}) VALUES ({placeholders})", vals
        )

    def upsert_technical(self, image_id: int, data: dict[str, Any]) -> None:
        """Atomic write of the full technical analysis result."""
        # JSON-encode list fields
        if "dominant_colors" in data and isinstance(data["dominant_colors"], list):
            data["dominant_colors"] = json.dumps(data["dominant_colors"])
        data = self._filter_to_known_columns("analysis_technical", data)
        data = self._apply_override_mask(image_id, "analysis_technical", data)
        self.conn.execute("DELETE FROM analysis_technical WHERE image_id = ?", [image_id])
        cols = ["image_id"] + list(data.keys()) + ["analyzed_at"]
        placeholders = ", ".join(["?"] * len(cols))
        col_str = ", ".join(cols)
        vals = [image_id] + list(data.values()) + [_now()]
        self.conn.execute(
            f"INSERT INTO analysis_technical ({col_str}) VALUES ({placeholders})", vals
        )

    def upsert_local_ai(self, image_id: int, data: dict[str, Any]) -> None:
        """Atomic write of the full local AI analysis result."""
        for key in ("keywords", "detected_objects", "face_identities"):
            if key in data and isinstance(data[key], list):
                data[key] = json.dumps(data[key])
        if "face_details" in data and isinstance(data["face_details"], (list, dict)):
            data["face_details"] = json.dumps(data["face_details"])
        # Coerce has_people to int
        if "has_people" in data:
            data["has_people"] = 1 if data["has_people"] else 0
        data = self._filter_to_known_columns("analysis_local_ai", data)
        data = self._apply_override_mask(image_id, "analysis_local_ai", data)
        self.conn.execute("DELETE FROM analysis_local_ai WHERE image_id = ?", [image_id])
        cols = ["image_id"] + list(data.keys()) + ["analyzed_at"]
        placeholders = ", ".join(["?"] * len(cols))
        col_str = ", ".join(cols)
        vals = [image_id] + list(data.values()) + [_now()]
        self.conn.execute(
            f"INSERT INTO analysis_local_ai ({col_str}) VALUES ({placeholders})", vals
        )

    def upsert_cloud_ai(
        self, image_id: int, provider: str, data: dict[str, Any]
    ) -> None:
        """Atomic write of cloud AI result for a specific provider."""
        for key in ("keywords", "detected_objects", "dominant_colors_ai"):
            if key in data and isinstance(data[key], list):
                data[key] = json.dumps(data[key])
        if "raw_response" in data and not isinstance(data["raw_response"], str):
            data["raw_response"] = json.dumps(data["raw_response"])
        data = self._filter_to_known_columns("analysis_cloud_ai", data)
        data = self._apply_override_mask(image_id, "analysis_cloud_ai", data)
        self.conn.execute(
            "DELETE FROM analysis_cloud_ai WHERE image_id = ? AND provider = ?",
            [image_id, provider],
        )
        data["provider"] = provider
        cols = ["image_id"] + list(data.keys()) + ["analyzed_at"]
        placeholders = ", ".join(["?"] * len(cols))
        col_str = ", ".join(cols)
        vals = [image_id] + list(data.values()) + [_now()]
        self.conn.execute(
            f"INSERT INTO analysis_cloud_ai ({col_str}) VALUES ({placeholders})", vals
        )

    def upsert_aesthetic(self, image_id: int, data: dict[str, Any]) -> None:
        """Atomic write of the aesthetic analysis result."""
        data = self._apply_override_mask(image_id, "analysis_aesthetic", data)
        self.conn.execute("DELETE FROM analysis_aesthetic WHERE image_id = ?", [image_id])
        cols = ["image_id"] + list(data.keys()) + ["analyzed_at"]
        placeholders = ", ".join(["?"] * len(cols))
        col_str = ", ".join(cols)
        vals = [image_id] + list(data.values()) + [_now()]
        self.conn.execute(
            f"INSERT INTO analysis_aesthetic ({col_str}) VALUES ({placeholders})", vals
        )

    def upsert_blip2(self, image_id: int, data: dict[str, Any]) -> None:
        """Atomic write of the BLIP-2 captioning result."""
        if "keywords" in data and isinstance(data["keywords"], list):
            data["keywords"] = json.dumps(data["keywords"])
        data = self._filter_to_known_columns("analysis_blip2", data)
        data = self._apply_override_mask(image_id, "analysis_blip2", data)
        self.conn.execute("DELETE FROM analysis_blip2 WHERE image_id = ?", [image_id])
        cols = ["image_id"] + list(data.keys()) + ["analyzed_at"]
        placeholders = ", ".join(["?"] * len(cols))
        col_str = ", ".join(cols)
        vals = [image_id] + list(data.values()) + [_now()]
        self.conn.execute(
            f"INSERT INTO analysis_blip2 ({col_str}) VALUES ({placeholders})", vals
        )

    def upsert_objects(self, image_id: int, data: dict[str, Any]) -> None:
        """Atomic write of the GroundingDINO object detection result."""
        if "detected_objects" in data and isinstance(data["detected_objects"], list):
            data["detected_objects"] = json.dumps(data["detected_objects"])
        if "text_boxes" in data and isinstance(data["text_boxes"], list):
            data["text_boxes"] = json.dumps(data["text_boxes"])
        if "has_person" in data:
            data["has_person"] = 1 if data["has_person"] else 0
        if "has_text" in data:
            data["has_text"] = 1 if data["has_text"] else 0
        data = self._filter_to_known_columns("analysis_objects", data)
        data = self._apply_override_mask(image_id, "analysis_objects", data)
        self.conn.execute("DELETE FROM analysis_objects WHERE image_id = ?", [image_id])
        cols = ["image_id"] + list(data.keys()) + ["analyzed_at"]
        placeholders = ", ".join(["?"] * len(cols))
        col_str = ", ".join(cols)
        vals = [image_id] + list(data.values()) + [_now()]
        self.conn.execute(
            f"INSERT INTO analysis_objects ({col_str}) VALUES ({placeholders})", vals
        )

    def upsert_ocr(self, image_id: int, data: dict[str, Any]) -> None:
        """Atomic write of the TrOCR result."""
        data = self._filter_to_known_columns("analysis_ocr", data)
        data = self._apply_override_mask(image_id, "analysis_ocr", data)
        self.conn.execute("DELETE FROM analysis_ocr WHERE image_id = ?", [image_id])
        cols = ["image_id"] + list(data.keys()) + ["analyzed_at"]
        placeholders = ", ".join(["?"] * len(cols))
        col_str = ", ".join(cols)
        vals = [image_id] + list(data.values()) + [_now()]
        self.conn.execute(
            f"INSERT INTO analysis_ocr ({col_str}) VALUES ({placeholders})", vals
        )

    def upsert_faces(self, image_id: int, data: dict[str, Any]) -> None:
        """Atomic write of the InsightFace face analysis result."""
        if "face_identities" in data and isinstance(data["face_identities"], list):
            data["face_identities"] = json.dumps(data["face_identities"])
        if "face_details" in data and isinstance(data["face_details"], (list, dict)):
            data["face_details"] = json.dumps(data["face_details"])
        data = self._filter_to_known_columns("analysis_faces", data)
        data = self._apply_override_mask(image_id, "analysis_faces", data)
        self.conn.execute("DELETE FROM analysis_faces WHERE image_id = ?", [image_id])
        cols = ["image_id"] + list(data.keys()) + ["analyzed_at"]
        placeholders = ", ".join(["?"] * len(cols))
        col_str = ", ".join(cols)
        vals = [image_id] + list(data.values()) + [_now()]
        self.conn.execute(
            f"INSERT INTO analysis_faces ({col_str}) VALUES ({placeholders})", vals
        )

    def clear_analysis(self, image_id: int, module: str, *, commit: bool = True) -> None:
        """Delete analysis data for a module (preserves overrides)."""
        table = MODULE_TABLE_MAP.get(module)
        if table is None:
            raise ValueError(f"Unknown module: {module}")
        self.conn.execute(f"DELETE FROM {table} WHERE image_id = ?", [image_id])
        if module == "faces":
            self.conn.execute(
                "DELETE FROM face_occurrences WHERE image_id = ?", [image_id]
            )
        if commit:
            self.conn.commit()

    # ── Overrides ──────────────────────────────────────────────────────────

    def set_override(
        self,
        image_id: int,
        table_name: str,
        field_name: str,
        value: Any,
        note: str | None = None,
    ) -> None:
        """Set a manual override for a specific field."""
        json_value = json.dumps(value) if not isinstance(value, str) else value
        self.conn.execute(
            """INSERT INTO overrides (image_id, table_name, field_name, value, note)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(image_id, table_name, field_name)
               DO UPDATE SET value = excluded.value,
                             overridden_at = datetime('now'),
                             note = excluded.note""",
            [image_id, table_name, field_name, json_value, note],
        )
        self.conn.commit()
        # Invalidate override negative cache
        if self._override_ids is not None:
            self._override_ids.add(image_id)

    def get_overrides(self, image_id: int, table_name: str) -> dict[str, str]:
        """Return {field_name: value} for all overrides on this image+table."""
        rows = self.conn.execute(
            "SELECT field_name, value FROM overrides WHERE image_id = ? AND table_name = ?",
            [image_id, table_name],
        ).fetchall()
        return {r["field_name"]: r["value"] for r in rows}

    def remove_override(self, image_id: int, table_name: str, field_name: str) -> bool:
        cur = self.conn.execute(
            "DELETE FROM overrides WHERE image_id = ? AND table_name = ? AND field_name = ?",
            [image_id, table_name, field_name],
        )
        self.conn.commit()
        # Invalidate override negative cache (may no longer have overrides)
        self._override_ids = None
        return cur.rowcount > 0

    def _apply_override_mask(
        self, image_id: int, table_name: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Remove fields from *data* that have active overrides (protected).

        Uses a negative cache: image IDs with overrides are loaded once into
        a set.  The ~99.99% of images with no overrides skip the per-row
        SELECT entirely.
        """
        if self._override_ids is None:
            rows = self.conn.execute(
                "SELECT DISTINCT image_id FROM overrides"
            ).fetchall()
            self._override_ids = {r["image_id"] for r in rows}
        if image_id not in self._override_ids:
            return data
        overrides = self.get_overrides(image_id, table_name)
        if not overrides:
            return data
        return {k: v for k, v in data.items() if k not in overrides}

    # ── Face identities ───────────────────────────────────────────────────

    def register_face_identity(
        self,
        canonical_name: str,
        display_name: str | None = None,
    ) -> int:
        """Create a face identity.  Returns identity_id."""
        row = self.conn.execute(
            "SELECT id FROM face_identities WHERE canonical_name = ?", [canonical_name]
        ).fetchone()
        if row:
            return row["id"]
        cur = self.conn.execute(
            """INSERT INTO face_identities (canonical_name, display_name, aliases)
               VALUES (?, ?, '[]')""",
            [canonical_name, display_name or canonical_name],
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def add_face_embedding(
        self, identity_id: int, embedding_blob: bytes, source_image: str | None = None
    ) -> int:
        cur = self.conn.execute(
            """INSERT INTO face_embeddings (identity_id, embedding, source_image)
               VALUES (?, ?, ?)""",
            [identity_id, embedding_blob, source_image],
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def add_face_alias(self, canonical_name: str, alias: str) -> None:
        row = self.conn.execute(
            "SELECT id, aliases FROM face_identities WHERE canonical_name = ?",
            [canonical_name],
        ).fetchone()
        if row is None:
            raise ValueError(f"Face identity '{canonical_name}' not found")
        aliases = json.loads(row["aliases"] or "[]")
        if alias not in aliases:
            aliases.append(alias)
        self.conn.execute(
            "UPDATE face_identities SET aliases = ?, updated_at = ? WHERE id = ?",
            [json.dumps(aliases), _now(), row["id"]],
        )
        # Also insert into indexed face_aliases table (idempotent)
        if self._table_exists("face_aliases"):
            existing = self.conn.execute(
                "SELECT 1 FROM face_aliases WHERE identity_id = ? AND alias = ?",
                [row["id"], alias],
            ).fetchone()
            if not existing:
                self.conn.execute(
                    "INSERT INTO face_aliases (identity_id, alias) VALUES (?, ?)",
                    [row["id"], alias],
                )
        self.conn.commit()

    def remove_face_alias(self, canonical_name: str, alias: str) -> bool:
        row = self.conn.execute(
            "SELECT id, aliases FROM face_identities WHERE canonical_name = ?",
            [canonical_name],
        ).fetchone()
        if row is None:
            raise ValueError(f"Face identity '{canonical_name}' not found")
        aliases = json.loads(row["aliases"] or "[]")
        if alias not in aliases:
            return False
        aliases.remove(alias)
        self.conn.execute(
            "UPDATE face_identities SET aliases = ?, updated_at = ? WHERE id = ?",
            [json.dumps(aliases), _now(), row["id"]],
        )
        # Also remove from indexed face_aliases table
        if self._table_exists("face_aliases"):
            self.conn.execute(
                "DELETE FROM face_aliases WHERE identity_id = ? AND alias = ?",
                [row["id"], alias],
            )
        self.conn.commit()
        return True

    def rename_face(self, canonical_name: str, display_name: str) -> None:
        self.conn.execute(
            "UPDATE face_identities SET display_name = ?, updated_at = ? WHERE canonical_name = ?",
            [display_name, _now(), canonical_name],
        )
        self.conn.commit()

    def merge_faces(self, keep_name: str, merge_name: str) -> None:
        """Merge *merge_name* into *keep_name* — moves all embeddings."""
        keep = self.conn.execute(
            "SELECT id, aliases FROM face_identities WHERE canonical_name = ?",
            [keep_name],
        ).fetchone()
        merge = self.conn.execute(
            "SELECT id, aliases FROM face_identities WHERE canonical_name = ?",
            [merge_name],
        ).fetchone()
        if keep is None:
            raise ValueError(f"Face identity '{keep_name}' not found")
        if merge is None:
            raise ValueError(f"Face identity '{merge_name}' not found")

        # Move embeddings
        self.conn.execute(
            "UPDATE face_embeddings SET identity_id = ? WHERE identity_id = ?",
            [keep["id"], merge["id"]],
        )
        # Merge aliases (JSON column — backward compat)
        keep_aliases = json.loads(keep["aliases"] or "[]")
        merge_aliases = json.loads(merge["aliases"] or "[]")
        all_aliases = list(set(keep_aliases + merge_aliases + [merge_name]))
        if keep_name in all_aliases:
            all_aliases.remove(keep_name)
        self.conn.execute(
            "UPDATE face_identities SET aliases = ?, updated_at = ? WHERE id = ?",
            [json.dumps(all_aliases), _now(), keep["id"]],
        )
        # Sync face_aliases table: move merge's aliases to keep, add merge_name
        if self._table_exists("face_aliases"):
            self.conn.execute(
                "UPDATE face_aliases SET identity_id = ? WHERE identity_id = ?",
                [keep["id"], merge["id"]],
            )
            # Add merge_name as alias of keep (if not already present)
            existing = self.conn.execute(
                "SELECT 1 FROM face_aliases WHERE identity_id = ? AND alias = ?",
                [keep["id"], merge_name],
            ).fetchone()
            if not existing:
                self.conn.execute(
                    "INSERT INTO face_aliases (identity_id, alias) VALUES (?, ?)",
                    [keep["id"], merge_name],
                )
            # Remove keep_name from aliases (it's the canonical, not an alias)
            self.conn.execute(
                "DELETE FROM face_aliases WHERE identity_id = ? AND alias = ?",
                [keep["id"], keep_name],
            )
        # Delete merged identity (CASCADE deletes its face_aliases rows too)
        self.conn.execute("DELETE FROM face_identities WHERE id = ?", [merge["id"]])
        self.conn.commit()

    def list_face_identities(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT fi.*, COUNT(fe.id) as embedding_count
               FROM face_identities fi
               LEFT JOIN face_embeddings fe ON fi.id = fe.identity_id
               GROUP BY fi.id
               ORDER BY fi.canonical_name"""
        ).fetchall()
        return [dict(r) for r in rows]

    def get_face_identity(self, canonical_name: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM face_identities WHERE canonical_name = ?", [canonical_name]
        ).fetchone()
        return dict(row) if row else None

    def find_face_by_alias(self, name: str) -> dict[str, Any] | None:
        """Find a face identity by canonical name, display name, or alias.

        Uses the indexed ``face_aliases`` table (added in schema v5) instead of
        scanning every row and parsing JSON.  Falls back to the legacy JSON
        scan when the table has not been created yet.
        """
        # Check canonical_name / display_name first (indexed)
        row = self.conn.execute(
            "SELECT * FROM face_identities WHERE canonical_name = ? OR display_name = ?",
            [name, name],
        ).fetchone()
        if row:
            return dict(row)
        # Query indexed face_aliases table (O(log N) via idx_face_aliases_alias)
        if self._table_exists("face_aliases"):
            row = self.conn.execute(
                """SELECT fi.* FROM face_identities fi
                   JOIN face_aliases fa ON fi.id = fa.identity_id
                   WHERE fa.alias = ?""",
                [name],
            ).fetchone()
            if row:
                return dict(row)
        else:
            # Legacy fallback: full table scan + JSON parsing
            rows = self.conn.execute("SELECT * FROM face_identities").fetchall()
            for r in rows:
                aliases = json.loads(r["aliases"] or "[]")
                if name in aliases:
                    return dict(r)
        return None

    def remove_face_identity(self, canonical_name: str) -> bool:
        cur = self.conn.execute(
            "DELETE FROM face_identities WHERE canonical_name = ?", [canonical_name]
        )
        self.conn.commit()
        return cur.rowcount > 0

    def get_face_embeddings(self, identity_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM face_embeddings WHERE identity_id = ?", [identity_id]
        ).fetchall()
        return [dict(r) for r in rows]

    def list_face_summary(self) -> list[dict[str, Any]]:
        """Return all unique face identity names across both analysis tables,
        with image counts and display_name from the face_identities registry.

        Results are sorted by image_count DESC (most-seen faces first).
        """
        rows = self.conn.execute(
            """
            WITH all_faces AS (
                SELECT image_id, face_identities
                FROM analysis_faces
                WHERE face_identities IS NOT NULL AND face_identities != '[]'
                UNION
                SELECT image_id, face_identities
                FROM analysis_local_ai
                WHERE face_identities IS NOT NULL AND face_identities != '[]'
            )
            SELECT
                je.value       AS canonical_name,
                COUNT(DISTINCT af.image_id) AS image_count,
                fi.display_name,
                fi.id          AS identity_id
            FROM all_faces af, json_each(af.face_identities) je
            LEFT JOIN face_identities fi ON fi.canonical_name = je.value
            GROUP BY je.value
            ORDER BY image_count DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]

    def get_images_for_face(
        self, name: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Return images containing a specific face identity name.

        Queries both ``analysis_faces`` and ``analysis_local_ai`` using
        ``json_each()`` for reliable JSON-array matching.
        """
        rows = self.conn.execute(
            """
            WITH matched AS (
                SELECT DISTINCT af.image_id
                FROM analysis_faces af, json_each(af.face_identities) je
                WHERE je.value = ?
                UNION
                SELECT DISTINCT la.image_id
                FROM analysis_local_ai la, json_each(la.face_identities) je
                WHERE je.value = ?
            )
            SELECT
                m.image_id,
                i.file_path,
                COALESCE(af2.face_count, la2.face_count, 0) AS face_count
            FROM matched m
            JOIN images i ON i.id = m.image_id
            LEFT JOIN analysis_faces    af2 ON af2.image_id = m.image_id
            LEFT JOIN analysis_local_ai la2 ON la2.image_id = m.image_id
            ORDER BY i.file_path
            LIMIT ?
            """,
            [name, name, limit],
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Face occurrences & clustering ──────────────────────────────────────

    def upsert_face_occurrences(
        self, image_id: int, occurrences: list[dict[str, Any]]
    ) -> None:
        """Write per-face occurrence rows (bbox, embedding, age, gender).

        Replaces any existing occurrences for this image.
        """
        self.conn.execute(
            "DELETE FROM face_occurrences WHERE image_id = ?", [image_id]
        )
        for occ in occurrences:
            self.conn.execute(
                """INSERT INTO face_occurrences
                   (image_id, face_idx, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                    embedding, age, gender, identity_name, det_score, thumbnail)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    image_id,
                    occ["face_idx"],
                    occ["bbox_x1"],
                    occ["bbox_y1"],
                    occ["bbox_x2"],
                    occ["bbox_y2"],
                    occ.get("embedding"),
                    occ.get("age"),
                    occ.get("gender"),
                    occ.get("identity_name", "Unknown"),
                    occ.get("det_score"),
                    occ.get("thumbnail"),
                ],
            )

    def list_face_clusters(self) -> list[dict[str, Any]]:
        """Return cluster summaries sorted by face count descending.

        If clustering hasn't been run yet (all cluster_id are NULL),
        falls back to grouping by identity_name.

        Each row has: cluster_id, identity_name, display_name, image_count,
        face_count, representative_id (a face_occurrences.id for thumbnail).
        """
        # Check if any clustering has been done
        has_clusters = self.conn.execute(
            "SELECT 1 FROM face_occurrences WHERE cluster_id IS NOT NULL LIMIT 1"
        ).fetchone()

        if has_clusters:
            rows = self.conn.execute(
                """
                SELECT
                    fo.cluster_id,
                    -- Use the most common identity_name in the cluster
                    (SELECT fo2.identity_name
                     FROM face_occurrences fo2
                     WHERE fo2.cluster_id = fo.cluster_id
                     GROUP BY fo2.identity_name
                     ORDER BY COUNT(*) DESC LIMIT 1) AS identity_name,
                    fi.display_name,
                    fi.id AS identity_id,
                    COUNT(DISTINCT fo.image_id) AS image_count,
                    COUNT(fo.id)                AS face_count,
                    -- Pick the occurrence with the largest face area as representative
                    (SELECT fo3.id FROM face_occurrences fo3
                     WHERE fo3.cluster_id = fo.cluster_id
                     ORDER BY (fo3.bbox_x2 - fo3.bbox_x1) * (fo3.bbox_y2 - fo3.bbox_y1) DESC
                     LIMIT 1) AS representative_id
                FROM face_occurrences fo
                LEFT JOIN face_identities fi
                    ON fi.canonical_name = (
                        SELECT fo4.identity_name
                        FROM face_occurrences fo4
                        WHERE fo4.cluster_id = fo.cluster_id
                        GROUP BY fo4.identity_name
                        ORDER BY COUNT(*) DESC LIMIT 1
                    )
                WHERE fo.cluster_id IS NOT NULL
                GROUP BY fo.cluster_id
                ORDER BY face_count DESC
                """
            ).fetchall()
        else:
            # No clustering yet — group by identity_name
            rows = self.conn.execute(
                """
                SELECT
                    NULL AS cluster_id,
                    fo.identity_name,
                    fi.display_name,
                    fi.id AS identity_id,
                    COUNT(DISTINCT fo.image_id) AS image_count,
                    COUNT(fo.id)                AS face_count,
                    (SELECT fo2.id FROM face_occurrences fo2
                     WHERE fo2.identity_name = fo.identity_name
                     ORDER BY (fo2.bbox_x2 - fo2.bbox_x1) * (fo2.bbox_y2 - fo2.bbox_y1) DESC
                     LIMIT 1) AS representative_id
                FROM face_occurrences fo
                LEFT JOIN face_identities fi ON fi.canonical_name = fo.identity_name
                GROUP BY fo.identity_name
                ORDER BY face_count DESC
                """
            ).fetchall()

        return [dict(r) for r in rows]

    def get_cluster_occurrences(
        self, cluster_id: int | None = None, identity_name: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return face occurrences for a cluster (or by identity_name).

        Each row includes: id, image_id, file_path, bbox coords, age, gender,
        identity_name.
        """
        if cluster_id is not None:
            rows = self.conn.execute(
                """
                SELECT fo.id, fo.image_id, i.file_path,
                       fo.face_idx, fo.bbox_x1, fo.bbox_y1, fo.bbox_x2, fo.bbox_y2,
                       fo.age, fo.gender, fo.identity_name
                FROM face_occurrences fo
                JOIN images i ON i.id = fo.image_id
                WHERE fo.cluster_id = ?
                ORDER BY (fo.bbox_x2 - fo.bbox_x1) * (fo.bbox_y2 - fo.bbox_y1) DESC
                LIMIT ?
                """,
                [cluster_id, limit],
            ).fetchall()
        elif identity_name is not None:
            rows = self.conn.execute(
                """
                SELECT fo.id, fo.image_id, i.file_path,
                       fo.face_idx, fo.bbox_x1, fo.bbox_y1, fo.bbox_x2, fo.bbox_y2,
                       fo.age, fo.gender, fo.identity_name
                FROM face_occurrences fo
                JOIN images i ON i.id = fo.image_id
                WHERE fo.identity_name = ?
                ORDER BY (fo.bbox_x2 - fo.bbox_x1) * (fo.bbox_y2 - fo.bbox_y1) DESC
                LIMIT ?
                """,
                [identity_name, limit],
            ).fetchall()
        else:
            return []

        return [dict(r) for r in rows]

    def get_face_occurrence(self, occurrence_id: int) -> dict[str, Any] | None:
        """Return a single face occurrence with its image path."""
        row = self.conn.execute(
            """
            SELECT fo.*, i.file_path
            FROM face_occurrences fo
            JOIN images i ON i.id = fo.image_id
            WHERE fo.id = ?
            """,
            [occurrence_id],
        ).fetchone()
        return dict(row) if row else None

    def cluster_faces(self, threshold: float = 0.55) -> int:
        """Cluster all face occurrences by cosine similarity of embeddings.

        Uses greedy agglomerative clustering: iterate through occurrences
        sorted by id, assign each to the closest existing cluster within
        *threshold*, or create a new cluster.

        Returns the total number of clusters created.
        """
        import struct

        rows = self.conn.execute(
            """SELECT id, embedding, identity_name
               FROM face_occurrences
               WHERE embedding IS NOT NULL
               ORDER BY id"""
        ).fetchall()

        if not rows:
            return 0

        import numpy as _np

        # Parse embeddings
        items: list[tuple[int, _np.ndarray, str]] = []
        for r in rows:
            blob: bytes = r["embedding"]
            n_floats = len(blob) // 4
            vec = _np.array(struct.unpack(f"{n_floats}f", blob), dtype=_np.float32)
            norm = _np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            items.append((r["id"], vec, r["identity_name"]))

        # Build centroid matrix for vectorized similarity (O(n·k) instead of O(n²·k))
        centroids: list[_np.ndarray] = []
        cluster_ids: list[int] = []
        cluster_counts: list[int] = []
        cluster_identity: list[str] = []  # identity of first member
        assignments: list[tuple[int, int]] = []
        next_cluster_id = 1

        for occ_id, vec, identity in items:
            matched_idx = -1

            if centroids:
                # Vectorized cosine similarity against all centroids
                centroid_mat = _np.stack(centroids)
                sims = centroid_mat @ vec  # (k,) dot products

                if identity and identity != "Unknown":
                    # Named: match only clusters with same identity
                    for i in _np.argsort(-sims):
                        if sims[i] < threshold:
                            break
                        if cluster_identity[i] == identity:
                            matched_idx = i
                            break
                else:
                    # Unknown: match best cluster above threshold
                    best_idx = int(_np.argmax(sims))
                    if sims[best_idx] >= threshold:
                        matched_idx = best_idx

            if matched_idx >= 0:
                cid = cluster_ids[matched_idx]
                assignments.append((occ_id, cid))
                # Update centroid as running average
                n = cluster_counts[matched_idx] + 1
                centroids[matched_idx] = (
                    centroids[matched_idx] * (n - 1) / n + vec / n
                )
                cluster_counts[matched_idx] = n
            else:
                cid = next_cluster_id
                next_cluster_id += 1
                centroids.append(vec.copy())
                cluster_ids.append(cid)
                cluster_counts.append(1)
                cluster_identity.append(identity or "Unknown")
                assignments.append((occ_id, cid))

        # Write cluster assignments back to DB
        self.conn.execute("UPDATE face_occurrences SET cluster_id = NULL")
        self.conn.executemany(
            "UPDATE face_occurrences SET cluster_id = ? WHERE id = ?",
            [(cid, occ_id) for occ_id, cid in assignments],
        )

        return len(cluster_ids)

    def get_face_occurrences_count(self) -> int:
        """Return total number of face occurrences stored."""
        row = self.conn.execute("SELECT COUNT(*) AS cnt FROM face_occurrences").fetchone()
        return row["cnt"] if row else 0

    # ── Embeddings (CLIP) ──────────────────────────────────────────────────

    def upsert_embedding(
        self,
        image_id: int,
        embedding_type: str,
        vector_blob: bytes,
        model_version: str = "",
    ) -> None:
        self.conn.execute(
            """INSERT INTO embeddings (image_id, embedding_type, vector, model_version)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(image_id, embedding_type)
               DO UPDATE SET vector = excluded.vector,
                             model_version = excluded.model_version,
                             computed_at = datetime('now')""",
            [image_id, embedding_type, vector_blob, model_version],
        )

    def get_all_embeddings(self, embedding_type: str) -> list[tuple[int, bytes]]:
        """Return [(image_id, vector_blob), ...] for bulk similarity search."""
        rows = self.conn.execute(
            "SELECT image_id, vector FROM embeddings WHERE embedding_type = ?",
            [embedding_type],
        ).fetchall()
        return [(r["image_id"], r["vector"]) for r in rows]

    # ── Search index ───────────────────────────────────────────────────────

    def update_search_index(self, image_id: int) -> None:
        """Rebuild FTS5 entry for a single image by aggregating all analysis data."""
        # Gather text from all sources
        desc_parts: list[str] = []
        subjects_parts: list[str] = []
        kw_parts: list[str] = []
        faces_parts: list[str] = []
        exif_parts: list[str] = []

        # Local AI (legacy full-pipeline table)
        local = self.conn.execute(
            "SELECT * FROM analysis_local_ai WHERE image_id = ?", [image_id]
        ).fetchone()
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

        # BLIP-2 individual pass (supplement local_ai or used alone)
        blip2 = self.conn.execute(
            "SELECT * FROM analysis_blip2 WHERE image_id = ?", [image_id]
        ).fetchone() if self._table_exists("analysis_blip2") else None
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

        # Faces individual pass
        faces_row = self.conn.execute(
            "SELECT * FROM analysis_faces WHERE image_id = ?", [image_id]
        ).fetchone() if self._table_exists("analysis_faces") else None
        if faces_row and faces_row["face_identities"]:
            try:
                for name in json.loads(faces_row["face_identities"]):
                    if name not in faces_parts:
                        faces_parts.append(name)
            except (json.JSONDecodeError, TypeError):
                pass

        # Cloud AI (all providers)
        cloud_rows = self.conn.execute(
            "SELECT * FROM analysis_cloud_ai WHERE image_id = ?", [image_id]
        ).fetchall()
        for cloud in cloud_rows:
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

        # EXIF / metadata
        meta = self.conn.execute(
            "SELECT * FROM analysis_metadata WHERE image_id = ?", [image_id]
        ).fetchone()
        if meta:
            for field in ("camera_make", "camera_model", "lens_model",
                          "location_city", "location_state", "location_country"):
                if meta[field]:
                    exif_parts.append(meta[field])

        # Face aliases
        for face_name in faces_parts[:]:
            identity = self.find_face_by_alias(face_name)
            if identity:
                if identity.get("display_name"):
                    faces_parts.append(identity["display_name"])
                aliases = json.loads(identity.get("aliases") or "[]")
                faces_parts.extend(aliases)

        # Deduplicate
        kw_parts = list(dict.fromkeys(kw_parts))
        faces_parts = list(dict.fromkeys(faces_parts))

        # Delete old entry then insert
        self.conn.execute(
            "DELETE FROM search_index WHERE image_id = ?", [str(image_id)]
        )
        self.conn.execute(
            """INSERT INTO search_index
               (image_id, description_text, subjects_text, keywords_text, faces_text, exif_text)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [
                str(image_id),
                " ".join(desc_parts),
                " ".join(subjects_parts),
                " ".join(kw_parts),
                " ".join(faces_parts),
                " ".join(exif_parts),
            ],
        )

    def _table_exists(self, table: str) -> bool:
        """Return True if a table exists in the database (cached via column cache)."""
        row = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", [table]
        ).fetchone()
        return row is not None

    # ── Helpers ────────────────────────────────────────────────────────────

    def get_full_result(self, image_id: int) -> dict[str, Any]:
        """Return all analysis data for an image, applying overrides."""
        result: dict[str, Any] = {}
        img = self.get_image(image_id)
        if img:
            result["image"] = img

        for module in ("metadata", "technical", "local_ai", "blip2", "objects", "ocr", "faces", "aesthetic"):
            data = self.get_analysis(image_id, module)
            if data:
                # Apply overrides on top
                overrides = self.get_overrides(
                    image_id, MODULE_TABLE_MAP[module]
                )
                for field, val in overrides.items():
                    try:
                        data[field] = json.loads(val)
                    except (json.JSONDecodeError, TypeError):
                        data[field] = val
                result[module] = data

        cloud = self.get_analysis(image_id, "cloud_ai")
        if cloud:
            overrides = self.get_overrides(image_id, "analysis_cloud_ai")
            for prov_data in cloud.get("providers", []):
                for field, val in overrides.items():
                    try:
                        prov_data[field] = json.loads(val)
                    except (json.JSONDecodeError, TypeError):
                        prov_data[field] = val
            result["cloud_ai"] = cloud

        return result
