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
    "cloud_ai":   "analysis_cloud_ai",
    "aesthetic":  "analysis_aesthetic",
    "embedding":  "embeddings",
}

ALL_MODULES = list(MODULE_TABLE_MAP.keys())


class Repository:
    """Data access layer for the imganalyzer database."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

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

    def clear_analysis(self, image_id: int, module: str) -> None:
        """Delete analysis data for a module (preserves overrides)."""
        table = MODULE_TABLE_MAP.get(module)
        if table is None:
            raise ValueError(f"Unknown module: {module}")
        self.conn.execute(f"DELETE FROM {table} WHERE image_id = ?", [image_id])
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
        return cur.rowcount > 0

    def _apply_override_mask(
        self, image_id: int, table_name: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Remove fields from *data* that have active overrides (protected)."""
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
        # Merge aliases
        keep_aliases = json.loads(keep["aliases"] or "[]")
        merge_aliases = json.loads(merge["aliases"] or "[]")
        all_aliases = list(set(keep_aliases + merge_aliases + [merge_name]))
        if keep_name in all_aliases:
            all_aliases.remove(keep_name)
        self.conn.execute(
            "UPDATE face_identities SET aliases = ?, updated_at = ? WHERE id = ?",
            [json.dumps(all_aliases), _now(), keep["id"]],
        )
        # Delete merged identity
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
        """Find a face identity by canonical name, display name, or alias."""
        # Check canonical_name first
        row = self.conn.execute(
            "SELECT * FROM face_identities WHERE canonical_name = ? OR display_name = ?",
            [name, name],
        ).fetchone()
        if row:
            return dict(row)
        # Search in aliases JSON
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

        # Local AI
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

    # ── Helpers ────────────────────────────────────────────────────────────

    def get_full_result(self, image_id: int) -> dict[str, Any]:
        """Return all analysis data for an image, applying overrides."""
        result: dict[str, Any] = {}
        img = self.get_image(image_id)
        if img:
            result["image"] = img

        for module in ("metadata", "technical", "local_ai", "aesthetic"):
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
