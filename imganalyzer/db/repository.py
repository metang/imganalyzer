"""Repository — CRUD operations for all analysis tables.

All write operations that touch analysis data use the atomic-write pattern:
the caller builds the full result dict in memory, then calls a single
``upsert_*`` method which writes everything inside one transaction.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── Module → table mapping ────────────────────────────────────────────────────
MODULE_TABLE_MAP: dict[str, str] = {
    "metadata":   "analysis_metadata",
    "technical":  "analysis_technical",
    "caption":    "analysis_caption",
    "local_ai":   "analysis_caption",   # legacy alias
    "blip2":      "analysis_blip2",
    "objects":    "analysis_objects",
    "ocr":        "analysis_ocr",
    "faces":      "analysis_faces",
    "cloud_ai":   "analysis_cloud_ai",
    "aesthetic":  "analysis_aesthetic",
    "perception": "analysis_perception",
    "embedding":  "embeddings",
}

# Active pipeline modules.  Legacy modules (blip2, cloud_ai, aesthetic, local_ai, ocr)
# stay in MODULE_TABLE_MAP for backward-compatible reads of existing analysis data
# but are excluded here so they are never enqueued or processed.
_LEGACY_MODULES = frozenset({"blip2", "cloud_ai", "aesthetic", "local_ai", "ocr"})
ALL_MODULES = [m for m in MODULE_TABLE_MAP if m not in _LEGACY_MODULES]

# ── Suggestion algorithm tuning ───────────────────────────────────────────────
_MAX_REPRESENTATIVES = 150          # cap on person representative centroids
_EMBEDDINGS_PER_CLUSTER_PERSON = 5  # stratified sample size for person clusters
_EMBEDDINGS_PER_CLUSTER_CANDIDATE = 5  # stratified sample size for candidates
_CANDIDATE_POOL_MULTIPLIER = 50     # candidate_pool = max(limit * this, 500)


def _farthest_point_sample(
    matrix: "numpy.ndarray",  # noqa: F821 — lazy import
    k: int,
) -> list[int]:
    """Select *k* maximally-spread rows from an (N, D) L2-normalised matrix.

    Uses the greedy farthest-point sampling (FPS) algorithm:
    1. Start from row 0.
    2. At each step pick the row whose *minimum* cosine similarity to all
       previously selected rows is the *lowest* (i.e., the farthest point).

    Returns a list of *k* row indices.  If ``k >= N`` returns all indices.

    Complexity: O(k × N) dot products — dominated by a single (N,) update per
    iteration, fully vectorised with numpy.
    """
    import numpy as _np

    n = matrix.shape[0]
    if k >= n:
        return list(range(n))

    selected: list[int] = [0]
    # Track the *maximum* similarity each row has to any already-selected row.
    # (Higher similarity = closer = less interesting.)
    max_sim = matrix @ matrix[0]  # (N,) — similarity to seed

    for _ in range(1, k):
        # The next representative is the point least similar to its
        # closest already-selected representative.
        max_sim[selected[-1]] = 2.0  # exclude already-selected
        idx = int(_np.argmin(max_sim))
        selected.append(idx)
        # Update max_sim: for each row, keep the larger of old max and new sim
        new_sim = matrix @ matrix[idx]
        _np.maximum(max_sim, new_sim, out=max_sim)

    return selected


class _FaceEmbeddingCache:
    """Cached (N, 512) float32 matrix for face_occurrences embeddings.

    Loads all face occurrence embedding vectors into a contiguous numpy array
    with aligned ``occurrence_ids`` and ``image_ids`` lists.  A row-count
    check detects when new face analysis has run so the matrix is rebuilt.
    """

    def __init__(self) -> None:
        self.matrix: Any = None  # np.ndarray (N, 512) float32, L2-normalised
        self.occurrence_ids: list[int] = []
        self.image_ids: list[int] = []
        self.person_ids: list[int | None] = []
        self._row_count: int = 0

    def get(
        self, conn: sqlite3.Connection
    ) -> tuple[Any, list[int], list[int], list[int | None]]:
        """Return ``(matrix, occurrence_ids, image_ids, person_ids)``.

        Rebuilds automatically when new embeddings appear (row-count check).
        """
        import numpy as _np

        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM face_occurrences WHERE embedding IS NOT NULL"
        ).fetchone()
        current_count: int = row["cnt"] if row else 0

        if self.matrix is not None and current_count == self._row_count:
            return self.matrix, self.occurrence_ids, self.image_ids, self.person_ids

        rows = conn.execute(
            "SELECT id, image_id, person_id, embedding "
            "FROM face_occurrences WHERE embedding IS NOT NULL ORDER BY id"
        ).fetchall()

        if not rows:
            empty = _np.empty((0, 0), dtype=_np.float32)
            self.matrix = empty
            self.occurrence_ids = []
            self.image_ids = []
            self.person_ids = []
            self._row_count = 0
            return empty, [], [], []

        occ_ids: list[int] = []
        img_ids: list[int] = []
        p_ids: list[int | None] = []
        vecs: list[Any] = []
        for r in rows:
            occ_ids.append(r["id"])
            img_ids.append(r["image_id"])
            p_ids.append(r["person_id"])
            vec = _np.frombuffer(r["embedding"], dtype=_np.float32).copy()
            norm = float(_np.linalg.norm(vec))
            if norm > 0:
                vec /= norm
            vecs.append(vec)

        self.matrix = _np.vstack(vecs)  # (N, 512)
        self.occurrence_ids = occ_ids
        self.image_ids = img_ids
        self.person_ids = p_ids
        self._row_count = current_count
        return self.matrix, self.occurrence_ids, self.image_ids, self.person_ids


class Repository:
    """Data access layer for the imganalyzer database."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self._column_cache: dict[str, set[str]] = {}
        # Negative cache: set of image IDs that have ANY overrides.
        # Loaded lazily on first _apply_override_mask call.  At 500K images
        # with ~0.01% override rate, this avoids ~4.5M empty SELECT queries.
        self._override_ids: set[int] | None = None
        self._face_emb_cache = _FaceEmbeddingCache()

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

    def upsert_caption(self, image_id: int, data: dict[str, Any]) -> None:
        """Atomic write of the caption analysis result (description, keywords, etc.)."""
        for key in ("keywords", "detected_objects", "face_identities"):
            if key in data and isinstance(data[key], list):
                data[key] = json.dumps(data[key])
        if "face_details" in data and isinstance(data["face_details"], (list, dict)):
            data["face_details"] = json.dumps(data["face_details"])
        # Coerce has_people to int
        if "has_people" in data:
            data["has_people"] = 1 if data["has_people"] else 0
        data = self._filter_to_known_columns("analysis_caption", data)
        data = self._apply_override_mask(image_id, "analysis_caption", data)
        self.conn.execute("DELETE FROM analysis_caption WHERE image_id = ?", [image_id])
        cols = ["image_id"] + list(data.keys()) + ["analyzed_at"]
        placeholders = ", ".join(["?"] * len(cols))
        col_str = ", ".join(cols)
        vals = [image_id] + list(data.values()) + [_now()]
        self.conn.execute(
            f"INSERT INTO analysis_caption ({col_str}) VALUES ({placeholders})", vals
        )

    # Legacy alias for backward compatibility
    upsert_local_ai = upsert_caption

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

    def upsert_perception(self, image_id: int, data: dict[str, Any]) -> None:
        """Atomic write of the perception analysis result (IAA/IQA/ISTA)."""
        data = self._apply_override_mask(image_id, "analysis_perception", data)
        self.conn.execute("DELETE FROM analysis_perception WHERE image_id = ?", [image_id])
        cols = ["image_id"] + list(data.keys()) + ["analyzed_at"]
        placeholders = ", ".join(["?"] * len(cols))
        col_str = ", ".join(cols)
        vals = [image_id] + list(data.values()) + [_now()]
        self.conn.execute(
            f"INSERT INTO analysis_perception ({col_str}) VALUES ({placeholders})", vals
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

    def set_cluster_label(self, cluster_id: int, display_name: str | None) -> None:
        """Set or clear a per-cluster display name."""
        if display_name:
            self.conn.execute(
                "INSERT INTO face_cluster_labels (cluster_id, display_name) VALUES (?, ?)"
                " ON CONFLICT(cluster_id) DO UPDATE SET display_name = excluded.display_name",
                [cluster_id, display_name],
            )
        else:
            self.conn.execute(
                "DELETE FROM face_cluster_labels WHERE cluster_id = ?",
                [cluster_id],
            )
        self.conn.commit()

    def relink_cluster(
        self,
        cluster_id: int,
        display_name: str | None,
        person_id: int | None = None,
        *,
        update_person: bool = False,
    ) -> int:
        """Update a cluster label and optionally its linked person in one transaction."""
        if display_name:
            self.conn.execute(
                "INSERT INTO face_cluster_labels (cluster_id, display_name) VALUES (?, ?)"
                " ON CONFLICT(cluster_id) DO UPDATE SET display_name = excluded.display_name",
                [cluster_id, display_name],
            )
        else:
            self.conn.execute(
                "DELETE FROM face_cluster_labels WHERE cluster_id = ?",
                [cluster_id],
            )

        updated = 0
        if update_person:
            if person_id is not None:
                if not self._table_exists("face_persons"):
                    raise ValueError("face_persons table does not exist")
                person_row = self.conn.execute(
                    "SELECT 1 FROM face_persons WHERE id = ?",
                    [person_id],
                ).fetchone()
                if person_row is None:
                    raise ValueError(f"Person {person_id} not found")
            cur = self.conn.execute(
                "UPDATE face_occurrences SET person_id = ? WHERE cluster_id = ?",
                [person_id, cluster_id],
            )
            updated = cur.rowcount

        self.conn.commit()
        return updated

    # ── Person (cross-age identity grouping) ─────────────────────────────

    def create_person(self, name: str, notes: str | None = None) -> int:
        """Create a person and return its id."""
        cur = self.conn.execute(
            "INSERT INTO face_persons (name, notes) VALUES (?, ?)",
            [name, notes],
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def rename_person(self, person_id: int, name: str) -> None:
        """Rename a person."""
        self.conn.execute(
            "UPDATE face_persons SET name = ? WHERE id = ?",
            [name, person_id],
        )
        self.conn.commit()

    def delete_person(self, person_id: int) -> None:
        """Delete a person and clear person_id on all their occurrences."""
        self.conn.execute(
            "UPDATE face_occurrences SET person_id = NULL WHERE person_id = ?",
            [person_id],
        )
        self.conn.execute("DELETE FROM face_persons WHERE id = ?", [person_id])
        self.conn.commit()

    def list_persons(self) -> list[dict]:
        """Return persons with cluster count, face count, and a representative thumbnail.

        Uses a window function (``ROW_NUMBER``) to batch-compute representatives
        instead of a correlated subquery per person, avoiding O(N*M) scans.
        Includes cached ``representative_thumbnail`` (base64) when available.
        """
        import base64 as _b64

        # Step 1: aggregate stats (fast – uses person_id index)
        rows = self.conn.execute(
            """
            SELECT
                fp.id,
                fp.name,
                fp.notes,
                COUNT(DISTINCT fo.cluster_id) AS cluster_count,
                COUNT(fo.id)                  AS face_count,
                COUNT(DISTINCT fo.image_id)   AS image_count
            FROM face_persons fp
            LEFT JOIN face_occurrences fo ON fo.person_id = fp.id
            GROUP BY fp.id
            ORDER BY face_count DESC
            """
        ).fetchall()
        if not rows:
            return []

        # Step 2: batch-fetch representative IDs via window function
        person_ids = [r["id"] for r in rows]
        ph = ",".join("?" for _ in person_ids)
        reps = self.conn.execute(
            f"""
            SELECT person_id, id AS representative_id
            FROM (
                SELECT person_id, id,
                       ROW_NUMBER() OVER (
                           PARTITION BY person_id
                           ORDER BY (bbox_x2 - bbox_x1) * (bbox_y2 - bbox_y1) DESC
                       ) AS rn
                FROM face_occurrences
                WHERE person_id IN ({ph})
            )
            WHERE rn = 1
            """,
            person_ids,
        ).fetchall()
        rep_map: dict[int, int] = {r["person_id"]: r["representative_id"] for r in reps}

        # Step 3: fetch cached thumbnails for representatives only
        rep_ids = list(rep_map.values())
        thumb_map: dict[int, str | None] = {}
        if rep_ids:
            ph2 = ",".join("?" for _ in rep_ids)
            thumb_rows = self.conn.execute(
                f"SELECT id, thumbnail FROM face_occurrences WHERE id IN ({ph2})",
                rep_ids,
            ).fetchall()
            for tr in thumb_rows:
                blob = tr["thumbnail"]
                thumb_map[tr["id"]] = _b64.b64encode(blob).decode("ascii") if blob else None

        result: list[dict] = []
        for row in rows:
            d = dict(row)
            rep_id = rep_map.get(d["id"])
            d["representative_id"] = rep_id
            d["representative_thumbnail"] = thumb_map.get(rep_id) if rep_id else None
            result.append(d)
        return result

    def link_cluster_to_person(self, cluster_id: int, person_id: int) -> int:
        """Set person_id on all occurrences in *cluster_id*. Returns rows updated."""
        cur = self.conn.execute(
            "UPDATE face_occurrences SET person_id = ? WHERE cluster_id = ?",
            [person_id, cluster_id],
        )
        self.conn.commit()
        return cur.rowcount

    def unlink_cluster_from_person(self, cluster_id: int) -> int:
        """Clear person_id on all occurrences in *cluster_id*. Returns rows updated."""
        cur = self.conn.execute(
            "UPDATE face_occurrences SET person_id = NULL WHERE cluster_id = ?",
            [cluster_id],
        )
        self.conn.commit()
        return cur.rowcount

    # ── Cluster defer (park for later) ───────────────────────────────────────

    def defer_cluster(self, cluster_id: int) -> None:
        """Mark a cluster as deferred (parked for later review)."""
        self.conn.execute(
            "INSERT OR IGNORE INTO face_cluster_deferred (cluster_id) VALUES (?)",
            [cluster_id],
        )
        self.conn.commit()

    def undefer_cluster(self, cluster_id: int) -> None:
        """Remove deferred status from a cluster."""
        self.conn.execute(
            "DELETE FROM face_cluster_deferred WHERE cluster_id = ?",
            [cluster_id],
        )
        self.conn.commit()

    def undefer_all_clusters(self) -> int:
        """Remove deferred status from all clusters. Returns count cleared."""
        cur = self.conn.execute("DELETE FROM face_cluster_deferred")
        self.conn.commit()
        return cur.rowcount

    def get_deferred_cluster_ids(self) -> set[int]:
        """Return the set of cluster IDs currently deferred."""
        rows = self.conn.execute("SELECT cluster_id FROM face_cluster_deferred").fetchall()
        return {row[0] for row in rows}

    # ── Cluster purity & splitting ───────────────────────────────────────

    def compute_cluster_purity(
        self,
        cluster_id: int,
        *,
        sample_size: int = 20,
    ) -> dict[str, Any]:
        """Return a purity score for *cluster_id*.

        Purity = minimum cosine similarity between any sampled member
        embedding and the cluster centroid.  Lower values indicate likely
        mixed-identity clusters.
        """
        import numpy as _np

        rows = self.conn.execute(
            """
            SELECT embedding FROM (
                SELECT embedding,
                       ROW_NUMBER() OVER (ORDER BY id) AS rn
                FROM face_occurrences
                WHERE cluster_id = ?
                  AND embedding IS NOT NULL
            ) WHERE rn <= ?
            """,
            [cluster_id, sample_size],
        ).fetchall()

        if not rows:
            return {"purity_score": 1.0, "member_count": 0}

        vectors: list[_np.ndarray] = []
        for row in rows:
            vec = _np.frombuffer(row["embedding"], dtype=_np.float32).copy()
            norm = float(_np.linalg.norm(vec))
            if norm > 0:
                vec /= norm
            vectors.append(vec)

        if len(vectors) < 2:
            return {"purity_score": 1.0, "member_count": len(vectors)}

        mat = _np.vstack(vectors)
        centroid = mat.mean(axis=0)
        c_norm = float(_np.linalg.norm(centroid))
        if c_norm > 0:
            centroid /= c_norm

        sims = mat @ centroid
        return {
            "purity_score": round(float(sims.min()), 4),
            "member_count": len(vectors),
        }

    def get_impure_clusters(
        self,
        *,
        purity_threshold: float = 0.75,
        min_faces: int = 3,
        limit: int = 50,
        sample_size: int = 20,
    ) -> list[dict[str, Any]]:
        """Return clusters whose purity score is below *purity_threshold*.

        Only considers clusters with at least *min_faces* occurrences.
        Results sorted by purity ascending (worst first).
        """
        cluster_rows = self.conn.execute(
            """
            SELECT cluster_id, COUNT(*) AS face_count
            FROM face_occurrences
            WHERE cluster_id IS NOT NULL
              AND embedding IS NOT NULL
            GROUP BY cluster_id
            HAVING COUNT(*) >= ?
            ORDER BY face_count DESC
            """,
            [min_faces],
        ).fetchall()

        impure: list[dict[str, Any]] = []
        for row in cluster_rows:
            cid = int(row["cluster_id"])
            info = self.compute_cluster_purity(cid, sample_size=sample_size)
            if info["purity_score"] < purity_threshold:
                impure.append({
                    "cluster_id": cid,
                    "face_count": int(row["face_count"]),
                    "purity_score": info["purity_score"],
                })

        impure.sort(key=lambda x: x["purity_score"])
        return impure[:limit]

    def split_cluster(
        self,
        cluster_id: int,
        *,
        threshold: float = 0.65,
    ) -> dict[str, Any]:
        """Split a mixed-identity cluster into sub-clusters.

        Re-runs greedy agglomerative clustering on the embeddings within
        *cluster_id* at a tighter *threshold*.  The largest sub-cluster
        keeps the original ``cluster_id`` (and its ``person_id``); smaller
        sub-clusters receive new IDs with ``person_id`` cleared.

        Returns ``{"split_count": int, "new_cluster_ids": list[int]}``.
        """
        import numpy as _np

        rows = self.conn.execute(
            """
            SELECT id, embedding
            FROM face_occurrences
            WHERE cluster_id = ?
              AND embedding IS NOT NULL
            ORDER BY id
            """,
            [cluster_id],
        ).fetchall()

        if len(rows) < 2:
            return {"split_count": 0 if rows else 0, "new_cluster_ids": []}

        occ_ids: list[int] = []
        vectors: list[_np.ndarray] = []
        for row in rows:
            vec = _np.frombuffer(row["embedding"], dtype=_np.float32).copy()
            norm = float(_np.linalg.norm(vec))
            if norm > 0:
                vec /= norm
            vectors.append(vec)
            occ_ids.append(int(row["id"]))

        mat = _np.vstack(vectors)
        n = mat.shape[0]
        dim = mat.shape[1]

        # Greedy agglomerative within this cluster
        max_sub = min(n, 64)
        centroid_mat = _np.empty((max_sub, dim), dtype=_np.float32)
        sub_ids: list[list[int]] = []  # sub_cluster_idx -> list of occ_ids
        sub_counts: list[int] = []
        n_sub = 0

        for i in range(n):
            vec = mat[i]
            matched_idx = -1

            if n_sub > 0:
                sims = centroid_mat[:n_sub] @ vec
                best_idx = int(_np.argmax(sims))
                if sims[best_idx] >= threshold:
                    matched_idx = best_idx

            if matched_idx >= 0:
                sub_ids[matched_idx].append(occ_ids[i])
                cnt = sub_counts[matched_idx] + 1
                centroid_mat[matched_idx] = (
                    centroid_mat[matched_idx] * ((cnt - 1) / cnt) + vec * (1.0 / cnt)
                )
                sub_counts[matched_idx] = cnt
            else:
                if n_sub >= max_sub:
                    max_sub *= 2
                    new_mat = _np.empty((max_sub, dim), dtype=_np.float32)
                    new_mat[:n_sub] = centroid_mat[:n_sub]
                    centroid_mat = new_mat
                centroid_mat[n_sub] = vec
                sub_ids.append([occ_ids[i]])
                sub_counts.append(1)
                n_sub += 1

        if n_sub <= 1:
            return {"split_count": 1, "new_cluster_ids": []}

        # Sort sub-clusters by size descending; largest keeps original cluster_id
        order = sorted(range(n_sub), key=lambda j: sub_counts[j], reverse=True)

        # Get next available cluster_id
        max_existing = self.conn.execute(
            "SELECT COALESCE(MAX(cluster_id), 0) FROM face_occurrences"
        ).fetchone()[0]

        new_cluster_ids: list[int] = []
        for rank, idx in enumerate(order):
            if rank == 0:
                # Largest sub-cluster keeps original cluster_id + person_id
                continue
            new_id = max_existing + rank
            new_cluster_ids.append(new_id)
            placeholders = ",".join("?" for _ in sub_ids[idx])
            self.conn.execute(
                f"UPDATE face_occurrences SET cluster_id = ?, person_id = NULL "
                f"WHERE id IN ({placeholders})",
                [new_id, *sub_ids[idx]],
            )

        self.conn.commit()
        return {
            "split_count": n_sub,
            "new_cluster_ids": new_cluster_ids,
        }

    def get_person_clusters(self, person_id: int) -> list[dict]:
        """Return clusters belonging to a person, with face counts.

        Uses a window function (``ROW_NUMBER``) to batch-compute representatives
        instead of a correlated subquery per cluster.
        Includes cached ``representative_thumbnail`` (base64) when available.
        """
        import base64 as _b64

        # Step 1: aggregate stats (fast – uses person_id + cluster_id index)
        rows = self.conn.execute(
            """
            SELECT
                fo.cluster_id,
                COUNT(fo.id) AS face_count,
                COUNT(DISTINCT fo.image_id) AS image_count,
                COALESCE(fcl.display_name, 'Cluster ' || fo.cluster_id) AS label
            FROM face_occurrences fo
            LEFT JOIN face_cluster_labels fcl ON fcl.cluster_id = fo.cluster_id
            WHERE fo.person_id = ? AND fo.cluster_id IS NOT NULL
            GROUP BY fo.cluster_id
            ORDER BY face_count DESC
            """,
            [person_id],
        ).fetchall()
        if not rows:
            return []

        # Step 2: batch-fetch representative IDs via window function
        cluster_ids = [r["cluster_id"] for r in rows]
        ph = ",".join("?" for _ in cluster_ids)
        reps = self.conn.execute(
            f"""
            SELECT cluster_id, id AS representative_id
            FROM (
                SELECT cluster_id, id,
                       ROW_NUMBER() OVER (
                           PARTITION BY cluster_id
                           ORDER BY (bbox_x2 - bbox_x1) * (bbox_y2 - bbox_y1) DESC
                       ) AS rn
                FROM face_occurrences
                WHERE cluster_id IN ({ph})
            )
            WHERE rn = 1
            """,
            cluster_ids,
        ).fetchall()
        rep_map: dict[int, int] = {r["cluster_id"]: r["representative_id"] for r in reps}

        # Step 3: fetch cached thumbnails for representatives only
        rep_ids = list(rep_map.values())
        thumb_map: dict[int, str | None] = {}
        if rep_ids:
            ph2 = ",".join("?" for _ in rep_ids)
            thumb_rows = self.conn.execute(
                f"SELECT id, thumbnail FROM face_occurrences WHERE id IN ({ph2})",
                rep_ids,
            ).fetchall()
            for tr in thumb_rows:
                blob = tr["thumbnail"]
                thumb_map[tr["id"]] = _b64.b64encode(blob).decode("ascii") if blob else None

        result: list[dict] = []
        for row in rows:
            d = dict(row)
            rep_id = rep_map.get(d["cluster_id"])
            d["representative_id"] = rep_id
            d["representative_thumbnail"] = thumb_map.get(rep_id) if rep_id else None
            result.append(d)
        return result

    def suggest_person_link_clusters(
        self,
        person_id: int,
        *,
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        """Rank unlinked clusters by likely match against ``person_id`` embeddings.

        Uses **multi-representative matching**: instead of a single centroid,
        builds up to ``_MAX_REPRESENTATIVES`` diverse centroids (one per linked
        cluster, downsampled via farthest-point sampling when the person has
        many clusters).  Each candidate is scored against the *closest*
        representative, so clusters similar to the person at *any* age or
        appearance are surfaced.
        """
        if limit <= 0 or not self._table_exists("face_occurrences"):
            return []

        import numpy as _np

        def _centroid(blobs: list[bytes]) -> _np.ndarray | None:
            vectors: list[_np.ndarray] = []
            for blob in blobs:
                vec = _np.frombuffer(blob, dtype=_np.float32)
                if vec.size == 0:
                    continue
                norm = float(_np.linalg.norm(vec))
                if norm <= 0.0:
                    continue
                vectors.append((vec / norm).astype(_np.float32))
            if not vectors:
                return None
            center = _np.vstack(vectors).mean(axis=0).astype(_np.float32)
            center_norm = float(_np.linalg.norm(center))
            if center_norm <= 0.0:
                return None
            return center / center_norm

        # ── Phase 1: build person representative centroids ────────────────
        person_rows = self.conn.execute(
            f"""
            SELECT cluster_id, embedding
            FROM (
                SELECT
                    cluster_id,
                    embedding,
                    ROW_NUMBER() OVER (
                        PARTITION BY cluster_id ORDER BY id DESC
                    ) AS rn
                FROM face_occurrences
                WHERE person_id = ?
                  AND cluster_id IS NOT NULL
                  AND embedding IS NOT NULL
            )
            WHERE rn <= ?
            """,
            [person_id, _EMBEDDINGS_PER_CLUSTER_PERSON],
        ).fetchall()

        if not person_rows:
            # Fallback: person has embeddings but no cluster_id set —
            # compute a single centroid from all available embeddings.
            fallback_rows = self.conn.execute(
                """
                SELECT embedding
                FROM face_occurrences
                WHERE person_id = ?
                  AND embedding IS NOT NULL
                """,
                [person_id],
            ).fetchall()
            fallback_centroid = _centroid([r["embedding"] for r in fallback_rows])
            if fallback_centroid is None:
                return []
            rep_matrix = fallback_centroid.reshape(1, -1)
        else:
            # Group embeddings by cluster and compute per-cluster centroids
            cluster_blobs: dict[int, list[bytes]] = {}
            for row in person_rows:
                cid = int(row["cluster_id"])
                cluster_blobs.setdefault(cid, []).append(row["embedding"])

            centroids: list[_np.ndarray] = []
            for blobs in cluster_blobs.values():
                c = _centroid(blobs)
                if c is not None:
                    centroids.append(c)

            if not centroids:
                return []

            rep_matrix = _np.vstack(centroids).astype(_np.float32)  # (N, D)

            # Downsample via FPS if too many clusters
            if rep_matrix.shape[0] > _MAX_REPRESENTATIVES:
                indices = _farthest_point_sample(rep_matrix, _MAX_REPRESENTATIVES)
                rep_matrix = rep_matrix[indices]

        # ── Phase 2: build candidate pool ─────────────────────────────────
        candidate_pool = max(limit * _CANDIDATE_POOL_MULTIPLIER, 500)
        candidate_rows = self.conn.execute(
            """
            SELECT
                fo.cluster_id AS cluster_id,
                COUNT(fo.id) AS face_count,
                COUNT(DISTINCT fo.image_id) AS image_count,
                COALESCE(fcl.display_name, 'Unknown') AS label,
                (
                    SELECT fo2.id
                    FROM face_occurrences fo2
                    WHERE fo2.cluster_id = fo.cluster_id
                    ORDER BY (fo2.bbox_x2 - fo2.bbox_x1) * (fo2.bbox_y2 - fo2.bbox_y1) DESC
                    LIMIT 1
                ) AS representative_id
            FROM face_occurrences fo
            LEFT JOIN face_cluster_labels fcl ON fcl.cluster_id = fo.cluster_id
            WHERE fo.cluster_id IS NOT NULL
            GROUP BY fo.cluster_id
            HAVING SUM(CASE WHEN fo.person_id IS NOT NULL THEN 1 ELSE 0 END) = 0
            ORDER BY face_count DESC
            LIMIT ?
            """,
            [candidate_pool],
        ).fetchall()

        candidate_meta: dict[int, dict[str, Any]] = {}
        for row in candidate_rows:
            cluster_id = int(row["cluster_id"])
            candidate_meta[cluster_id] = {
                "cluster_id": cluster_id,
                "display_name": row["label"],
                "face_count": int(row["face_count"] or 0),
                "image_count": int(row["image_count"] or 0),
                "representative_id": (
                    int(row["representative_id"])
                    if row["representative_id"] is not None
                    else None
                ),
            }
        if not candidate_meta:
            return []

        candidate_cluster_ids = sorted(candidate_meta.keys())
        placeholders = ",".join("?" for _ in candidate_cluster_ids)
        candidate_embeddings_rows = self.conn.execute(
            f"""
            SELECT cluster_id, embedding
            FROM (
                SELECT
                    cluster_id,
                    embedding,
                    ROW_NUMBER() OVER (
                        PARTITION BY cluster_id ORDER BY id DESC
                    ) AS rn
                FROM face_occurrences
                WHERE cluster_id IN ({placeholders})
                  AND person_id IS NULL
                  AND embedding IS NOT NULL
            )
            WHERE rn <= ?
            """,
            [*candidate_cluster_ids, _EMBEDDINGS_PER_CLUSTER_CANDIDATE],
        ).fetchall()

        # Compute per-candidate centroids
        cand_blobs: dict[int, list[bytes]] = {}
        for row in candidate_embeddings_rows:
            cid = int(row["cluster_id"])
            cand_blobs.setdefault(cid, []).append(row["embedding"])

        cand_ids: list[int] = []
        cand_centroids: list[_np.ndarray] = []
        for cid in candidate_cluster_ids:
            blobs = cand_blobs.get(cid)
            if not blobs:
                continue
            c = _centroid(blobs)
            if c is not None:
                cand_ids.append(cid)
                cand_centroids.append(c)

        if not cand_centroids:
            return []

        cand_matrix = _np.vstack(cand_centroids).astype(_np.float32)  # (M, D)

        # ── Phase 3: vectorised scoring ───────────────────────────────────
        # sim_matrix shape: (K, M) — similarity of each rep to each candidate
        sim_matrix = rep_matrix @ cand_matrix.T
        # For each candidate, take the best match across all representatives
        max_scores = sim_matrix.max(axis=0)  # (M,)

        suggestions: list[dict[str, Any]] = []
        for i, cid in enumerate(cand_ids):
            meta = candidate_meta.get(cid)
            if meta is None:
                continue
            score = float(max_scores[i])
            display_name = meta.get("display_name")
            label = str(display_name) if display_name else "Unknown"
            suggestions.append(
                {
                    "cluster_id": cid,
                    "label": label,
                    "score": score,
                    "representative_id": (
                        int(meta["representative_id"])
                        if meta.get("representative_id") is not None
                        else None
                    ),
                    "face_count": int(meta.get("face_count") or 0),
                    "image_count": int(meta.get("image_count") or 0),
                    "reason": "Similarity to person-linked face embeddings",
                }
            )

        suggestions.sort(
            key=lambda item: (float(item["score"]), int(item["face_count"])),
            reverse=True,
        )
        return suggestions[:limit]

    def suggest_cluster_link_targets(
        self,
        cluster_id: int,
        *,
        limit: int = 12,
        include_persons: bool = True,
        include_aliases: bool = True,
    ) -> list[dict[str, Any]]:
        """Rank likely person/alias targets for ``cluster_id`` by embedding similarity."""
        if limit <= 0 or not self._table_exists("face_occurrences"):
            return []

        import numpy as _np

        def _centroid(blobs: list[bytes]) -> _np.ndarray | None:
            vectors: list[_np.ndarray] = []
            for blob in blobs:
                vec = _np.frombuffer(blob, dtype=_np.float32)
                if vec.size == 0:
                    continue
                norm = float(_np.linalg.norm(vec))
                if norm <= 0.0:
                    continue
                vectors.append((vec / norm).astype(_np.float32))
            if not vectors:
                return None
            center = _np.vstack(vectors).mean(axis=0).astype(_np.float32)
            center_norm = float(_np.linalg.norm(center))
            if center_norm <= 0.0:
                return None
            return center / center_norm

        source_rows = self.conn.execute(
            """
            SELECT embedding
            FROM face_occurrences
            WHERE cluster_id = ?
              AND embedding IS NOT NULL
            """,
            [cluster_id],
        ).fetchall()
        source_blobs = [row["embedding"] for row in source_rows]
        source_centroid = _centroid(source_blobs)
        if source_centroid is None:
            return []

        suggestions: list[dict[str, Any]] = []

        if include_persons and self._table_exists("face_persons"):
            person_meta = {int(row["id"]): row for row in self.list_persons()}
            person_rows = self.conn.execute(
                """
                SELECT person_id, embedding
                FROM face_occurrences
                WHERE person_id IS NOT NULL
                  AND embedding IS NOT NULL
                  AND (cluster_id IS NULL OR cluster_id != ?)
                """,
                [cluster_id],
            ).fetchall()

            person_embeddings: dict[int, list[bytes]] = {}
            for row in person_rows:
                person_id = int(row["person_id"])
                person_embeddings.setdefault(person_id, []).append(row["embedding"])

            for person_id, blobs in person_embeddings.items():
                centroid = _centroid(blobs)
                meta = person_meta.get(person_id)
                if centroid is None or meta is None:
                    continue
                score = float(_np.dot(source_centroid, centroid))
                suggestions.append(
                    {
                        "target_type": "person",
                        "label": str(meta["name"]),
                        "person_id": person_id,
                        "cluster_id": None,
                        "score": score,
                        "representative_id": (
                            int(meta["representative_id"])
                            if meta.get("representative_id") is not None
                            else None
                        ),
                        "face_count": int(meta.get("face_count") or 0),
                        "reason": "Similarity to person-linked face embeddings",
                    }
                )

        if include_aliases:
            clusters, _ = self.list_face_clusters(limit=0, offset=0)
            alias_meta: dict[int, dict[str, Any]] = {}
            for cluster in clusters:
                candidate_cluster_id = cluster.get("cluster_id")
                display_name = cluster.get("display_name")
                if (
                    candidate_cluster_id is None
                    or int(candidate_cluster_id) == cluster_id
                    or not display_name
                ):
                    continue
                alias_meta[int(candidate_cluster_id)] = cluster

            if alias_meta:
                candidate_cluster_ids = sorted(alias_meta.keys())
                placeholders = ",".join("?" for _ in candidate_cluster_ids)
                alias_rows = self.conn.execute(
                    f"""
                    SELECT cluster_id, embedding
                    FROM face_occurrences
                    WHERE cluster_id IN ({placeholders})
                      AND embedding IS NOT NULL
                    """,
                    candidate_cluster_ids,
                ).fetchall()

                alias_embeddings: dict[int, list[bytes]] = {}
                for row in alias_rows:
                    candidate_cluster_id = int(row["cluster_id"])
                    alias_embeddings.setdefault(candidate_cluster_id, []).append(
                        row["embedding"]
                    )

                for candidate_cluster_id, blobs in alias_embeddings.items():
                    centroid = _centroid(blobs)
                    meta = alias_meta.get(candidate_cluster_id)
                    if centroid is None or meta is None:
                        continue
                    score = float(_np.dot(source_centroid, centroid))
                    suggestions.append(
                        {
                            "target_type": "alias",
                            "label": str(meta["display_name"]),
                            "person_id": (
                                int(meta["person_id"])
                                if meta.get("person_id") is not None
                                else None
                            ),
                            "cluster_id": candidate_cluster_id,
                            "score": score,
                            "representative_id": (
                                int(meta["representative_id"])
                                if meta.get("representative_id") is not None
                                else None
                            ),
                            "face_count": int(meta.get("face_count") or 0),
                            "reason": "Similarity to labeled cluster embeddings",
                        }
                    )

        suggestions.sort(
            key=lambda item: (float(item["score"]), int(item["face_count"])),
            reverse=True,
        )
        return suggestions[:limit]

    def auto_assign_persons_after_recluster(self) -> int:
        """Propagate person_id within new clusters by majority vote.

        For each cluster that has *some* occurrences with person_id set,
        find the majority person_id (>50% of tagged occurrences) and apply
        it to all occurrences in that cluster.

        Returns the number of occurrences updated.
        """
        # Find clusters with mixed person assignments
        rows = self.conn.execute(
            """
            SELECT cluster_id, person_id, COUNT(*) AS cnt
            FROM face_occurrences
            WHERE cluster_id IS NOT NULL AND person_id IS NOT NULL
            GROUP BY cluster_id, person_id
            """
        ).fetchall()

        # Build per-cluster vote tallies
        cluster_votes: dict[int, list[tuple[int, int]]] = {}
        for r in rows:
            cid = r["cluster_id"]
            cluster_votes.setdefault(cid, []).append((r["person_id"], r["cnt"]))

        updated = 0
        for cid, votes in cluster_votes.items():
            # Total tagged occurrences in this cluster
            total_tagged = sum(cnt for _, cnt in votes)
            # Best person
            best_person, best_cnt = max(votes, key=lambda x: x[1])
            # Require majority (>50% of tagged)
            if best_cnt > total_tagged / 2:
                cur = self.conn.execute(
                    "UPDATE face_occurrences SET person_id = ? "
                    "WHERE cluster_id = ? AND (person_id IS NULL OR person_id != ?)",
                    [best_person, cid, best_person],
                )
                updated += cur.rowcount

        if updated:
            self.conn.commit()
        return updated

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

    def find_face_identities_by_alias(self, name: str) -> list[dict[str, Any]]:
        """Return all face identities matching *name* by canonical, display, or alias."""
        if not name:
            return []

        if self._table_exists("face_aliases"):
            rows = self.conn.execute(
                """
                SELECT DISTINCT fi.*
                FROM face_identities fi
                LEFT JOIN face_aliases fa ON fi.id = fa.identity_id
                WHERE fi.canonical_name = ? COLLATE NOCASE
                   OR fi.display_name = ? COLLATE NOCASE
                   OR fa.alias = ? COLLATE NOCASE
                ORDER BY fi.id
                """,
                [name, name, name],
            ).fetchall()
            return [dict(row) for row in rows]

        rows = self.conn.execute("SELECT * FROM face_identities ORDER BY id").fetchall()
        matches: list[dict[str, Any]] = []
        for row in rows:
            aliases = json.loads(row["aliases"] or "[]")
            if (
                row["canonical_name"].casefold() == name.casefold()
                or (row["display_name"] and row["display_name"].casefold() == name.casefold())
                or any(name.casefold() == alias.casefold() for alias in aliases)
            ):
                matches.append(dict(row))
        return matches

    def find_persons_by_name(self, name: str) -> list[dict[str, Any]]:
        """Return all person groups matching *name* case-insensitively."""
        if not name or not self._table_exists("face_persons"):
            return []
        rows = self.conn.execute(
            "SELECT * FROM face_persons WHERE name = ? COLLATE NOCASE ORDER BY id",
            [name],
        ).fetchall()
        return [dict(row) for row in rows]

    def find_clusters_by_label(self, name: str) -> list[dict[str, Any]]:
        """Return all face clusters whose display label matches *name*."""
        if not name or not self._table_exists("face_cluster_labels"):
            return []
        rows = self.conn.execute(
            """
            SELECT cluster_id, display_name
            FROM face_cluster_labels
            WHERE display_name = ? COLLATE NOCASE
            ORDER BY cluster_id
            """,
            [name],
        ).fetchall()
        return [dict(row) for row in rows]

    def find_face_by_alias(self, name: str) -> dict[str, Any] | None:
        """Find a face identity by canonical name, display name, or alias.

        Uses the indexed ``face_aliases`` table (added in schema v5) instead of
        scanning every row and parsing JSON.  Falls back to the legacy JSON
        scan when the table has not been created yet.
        """
        matches = self.find_face_identities_by_alias(name)
        return matches[0] if matches else None

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
                FROM analysis_caption                WHERE face_identities IS NOT NULL AND face_identities != '[]'
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
        self, name: str, limit: int | None = 100
    ) -> list[dict[str, Any]]:
        """Return images containing a specific face identity name.

        Queries both ``analysis_faces`` and ``analysis_caption`` using
        ``json_each()`` for reliable JSON-array matching.
        """
        limit_clause = "LIMIT ?" if limit is not None and limit > 0 else ""
        params: list[Any] = [name, name]
        if limit_clause:
            params.append(limit)
        rows = self.conn.execute(
            f"""
            WITH matched AS (
                SELECT DISTINCT af.image_id
                FROM analysis_faces af, json_each(af.face_identities) je
                WHERE je.value = ?
                UNION
                SELECT DISTINCT la.image_id
                FROM analysis_caption la, json_each(la.face_identities) je
                WHERE je.value = ?
            )
            SELECT
                m.image_id,
                i.file_path,
                COALESCE(af2.face_count, la2.face_count, 0) AS face_count
            FROM matched m
            JOIN images i ON i.id = m.image_id
            LEFT JOIN analysis_faces    af2 ON af2.image_id = m.image_id
            LEFT JOIN analysis_caption la2 ON la2.image_id = m.image_id
            ORDER BY i.file_path
            {limit_clause}
            """,
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def get_images_for_person(self, person_id: int, limit: int | None = 100) -> list[dict[str, Any]]:
        """Return images containing any face occurrence linked to *person_id*."""
        if not self._table_exists("face_occurrences"):
            return []
        limit_clause = "LIMIT ?" if limit is not None and limit > 0 else ""
        params: list[Any] = [person_id]
        if limit_clause:
            params.append(limit)
        rows = self.conn.execute(
            f"""
            WITH matched AS (
                SELECT DISTINCT fo.image_id
                FROM face_occurrences fo
                WHERE fo.person_id = ?
            )
            SELECT
                m.image_id,
                i.file_path,
                COALESCE(af.face_count, la.face_count, 0) AS face_count
            FROM matched m
            JOIN images i ON i.id = m.image_id
            LEFT JOIN analysis_faces af ON af.image_id = m.image_id
            LEFT JOIN analysis_caption la ON la.image_id = m.image_id
            ORDER BY i.file_path
            {limit_clause}
            """,
            params,
        ).fetchall()
        return [dict(row) for row in rows]

    def get_images_for_cluster(self, cluster_id: int, limit: int | None = 100) -> list[dict[str, Any]]:
        """Return images containing any face occurrence belonging to *cluster_id*."""
        if not self._table_exists("face_occurrences"):
            return []
        limit_clause = "LIMIT ?" if limit is not None and limit > 0 else ""
        params: list[Any] = [cluster_id]
        if limit_clause:
            params.append(limit)
        rows = self.conn.execute(
            f"""
            WITH matched AS (
                SELECT DISTINCT fo.image_id
                FROM face_occurrences fo
                WHERE fo.cluster_id = ?
            )
            SELECT
                m.image_id,
                i.file_path,
                COALESCE(af.face_count, la.face_count, 0) AS face_count
            FROM matched m
            JOIN images i ON i.id = m.image_id
            LEFT JOIN analysis_faces af ON af.image_id = m.image_id
            LEFT JOIN analysis_caption la ON la.image_id = m.image_id
            ORDER BY i.file_path
            {limit_clause}
            """,
            params,
        ).fetchall()
        return [dict(row) for row in rows]

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

    def list_face_clusters(
        self,
        limit: int = 0,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """Return cluster summaries sorted by face count descending.

        If clustering hasn't been run yet (all cluster_id are NULL),
        falls back to grouping by identity_name.

        Args:
            limit: Max rows to return. 0 means all (no limit).
            offset: Number of rows to skip.

        Returns:
            (clusters, total_count) — list of cluster dicts and the total
            number of clusters (for pagination).
        """
        # Check if any clustering has been done
        has_clusters = self.conn.execute(
            "SELECT 1 FROM face_occurrences WHERE cluster_id IS NOT NULL LIMIT 1"
        ).fetchone()

        pagination = ""
        params: list[int] = []
        if limit > 0:
            pagination = " LIMIT ? OFFSET ?"
            params = [limit, offset]

        if has_clusters:
            clustered_row = self.conn.execute(
                "SELECT COUNT(DISTINCT cluster_id) AS cnt FROM face_occurrences WHERE cluster_id IS NOT NULL"
            ).fetchone()
            unclustered_row = self.conn.execute(
                """
                SELECT COUNT(DISTINCT COALESCE(identity_name, 'Unknown')) AS cnt
                FROM face_occurrences
                WHERE cluster_id IS NULL
                """
            ).fetchone()
            total_count = (
                (clustered_row["cnt"] if clustered_row else 0)
                + (unclustered_row["cnt"] if unclustered_row else 0)
            )

            rows = self.conn.execute(
                f"""
                WITH cluster_agg AS (
                    SELECT
                        cluster_id,
                        COUNT(DISTINCT image_id) AS image_count,
                        COUNT(id)                AS face_count
                    FROM face_occurrences
                    WHERE cluster_id IS NOT NULL
                    GROUP BY cluster_id
                ),
                cluster_identity AS (
                    SELECT cluster_id, identity_name,
                           ROW_NUMBER() OVER (
                               PARTITION BY cluster_id
                               ORDER BY COUNT(*) DESC
                           ) AS rn
                    FROM face_occurrences
                    WHERE cluster_id IS NOT NULL
                    GROUP BY cluster_id, identity_name
                ),
                cluster_person AS (
                    SELECT cluster_id, person_id,
                           ROW_NUMBER() OVER (
                               PARTITION BY cluster_id
                               ORDER BY COUNT(*) DESC
                           ) AS rn
                    FROM face_occurrences
                    WHERE cluster_id IS NOT NULL AND person_id IS NOT NULL
                    GROUP BY cluster_id, person_id
                ),
                cluster_rep AS (
                    SELECT cluster_id, id AS representative_id,
                           ROW_NUMBER() OVER (
                               PARTITION BY cluster_id
                               ORDER BY (bbox_x2 - bbox_x1) * (bbox_y2 - bbox_y1) DESC
                           ) AS rn
                    FROM face_occurrences
                    WHERE cluster_id IS NOT NULL
                ),
                cluster_rows AS (
                    SELECT
                        ca.cluster_id,
                        ci.identity_name,
                        COALESCE(fcl.display_name, fp.name, fi.display_name) AS display_name,
                        fi.id AS identity_id,
                        ca.image_count,
                        ca.face_count,
                        cr.representative_id,
                        cp.person_id
                    FROM cluster_agg ca
                    LEFT JOIN cluster_identity ci
                        ON ci.cluster_id = ca.cluster_id AND ci.rn = 1
                    LEFT JOIN cluster_person cp
                        ON cp.cluster_id = ca.cluster_id AND cp.rn = 1
                    LEFT JOIN cluster_rep cr
                        ON cr.cluster_id = ca.cluster_id AND cr.rn = 1
                    LEFT JOIN face_identities fi
                        ON fi.canonical_name = ci.identity_name
                    LEFT JOIN face_cluster_labels fcl
                        ON fcl.cluster_id = ca.cluster_id
                    LEFT JOIN face_persons fp
                        ON fp.id = cp.person_id
                ),
                unclustered_agg AS (
                    SELECT
                        COALESCE(identity_name, 'Unknown') AS identity_name,
                        COUNT(DISTINCT image_id) AS image_count,
                        COUNT(id)                AS face_count
                    FROM face_occurrences
                    WHERE cluster_id IS NULL
                    GROUP BY COALESCE(identity_name, 'Unknown')
                ),
                unclustered_rep AS (
                    SELECT
                        COALESCE(identity_name, 'Unknown') AS identity_name,
                        id AS representative_id,
                        ROW_NUMBER() OVER (
                            PARTITION BY COALESCE(identity_name, 'Unknown')
                            ORDER BY (bbox_x2 - bbox_x1) * (bbox_y2 - bbox_y1) DESC
                        ) AS rn
                    FROM face_occurrences
                    WHERE cluster_id IS NULL
                ),
                unclustered_rows AS (
                    SELECT
                        NULL AS cluster_id,
                        ua.identity_name,
                        fi.display_name,
                        fi.id AS identity_id,
                        ua.image_count,
                        ua.face_count,
                        ur.representative_id,
                        NULL AS person_id
                    FROM unclustered_agg ua
                    LEFT JOIN unclustered_rep ur
                        ON ur.identity_name = ua.identity_name AND ur.rn = 1
                    LEFT JOIN face_identities fi
                        ON fi.canonical_name = ua.identity_name
                ),
                combined AS (
                    SELECT * FROM cluster_rows
                    UNION ALL
                    SELECT * FROM unclustered_rows
                )
                SELECT
                    cluster_id,
                    identity_name,
                    display_name,
                    identity_id,
                    image_count,
                    face_count,
                    representative_id,
                    person_id
                FROM combined
                ORDER BY face_count DESC
                {pagination}
                """,
                params,
            ).fetchall()
        else:
            total_row = self.conn.execute(
                """
                SELECT COUNT(DISTINCT COALESCE(identity_name, 'Unknown')) AS cnt
                FROM face_occurrences
                """
            ).fetchone()
            total_count = total_row["cnt"] if total_row else 0

            rows = self.conn.execute(
                f"""
                WITH name_agg AS (
                    SELECT
                        COALESCE(identity_name, 'Unknown') AS identity_name,
                        COUNT(DISTINCT image_id) AS image_count,
                        COUNT(id)                AS face_count
                    FROM face_occurrences
                    GROUP BY COALESCE(identity_name, 'Unknown')
                ),
                name_rep AS (
                    SELECT COALESCE(identity_name, 'Unknown') AS identity_name,
                           id AS representative_id,
                           ROW_NUMBER() OVER (
                                PARTITION BY COALESCE(identity_name, 'Unknown')
                                ORDER BY (bbox_x2 - bbox_x1) * (bbox_y2 - bbox_y1) DESC
                            ) AS rn
                    FROM face_occurrences
                )
                SELECT
                    NULL AS cluster_id,
                    na.identity_name,
                    fi.display_name,
                    fi.id AS identity_id,
                    na.image_count,
                    na.face_count,
                    nr.representative_id,
                    NULL AS person_id
                FROM name_agg na
                LEFT JOIN name_rep nr
                    ON nr.identity_name = na.identity_name AND nr.rn = 1
                LEFT JOIN face_identities fi
                    ON fi.canonical_name = na.identity_name
                ORDER BY na.face_count DESC
                {pagination}
                """,
                params,
            ).fetchall()

        return [dict(r) for r in rows], total_count

    def get_cluster_occurrences(
        self, cluster_id: int | None = None, identity_name: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return face occurrences for a cluster (or by unclustered identity_name).

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
                WHERE COALESCE(fo.identity_name, 'Unknown') = ?
                  AND fo.cluster_id IS NULL
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

    def set_face_occurrence_thumbnail(self, occurrence_id: int, thumbnail: bytes) -> None:
        """Persist a generated thumbnail for an existing face occurrence."""
        self.conn.execute(
            "UPDATE face_occurrences SET thumbnail = ? WHERE id = ?",
            [thumbnail, occurrence_id],
        )

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

        # Parse all embeddings into a contiguous matrix up front
        first_blob: bytes = rows[0]["embedding"]
        dim = len(first_blob) // 4
        n_items = len(rows)

        all_vecs = _np.empty((n_items, dim), dtype=_np.float32)
        occ_ids: list[int] = []
        identities: list[str] = []

        for idx, r in enumerate(rows):
            blob: bytes = r["embedding"]
            n_floats = len(blob) // 4
            vec = _np.array(struct.unpack(f"{n_floats}f", blob), dtype=_np.float32)
            norm = _np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            all_vecs[idx] = vec
            occ_ids.append(r["id"])
            identities.append(r["identity_name"] or "Unknown")

        # Pre-allocated centroid matrix — grows by doubling when full
        max_clusters = min(n_items, 1024)
        centroid_mat = _np.empty((max_clusters, dim), dtype=_np.float32)
        cluster_ids: list[int] = []
        cluster_counts: list[int] = []
        cluster_identity: list[str] = []
        assignments = _np.empty(n_items, dtype=_np.int32)
        n_clusters = 0
        next_cluster_id = 1

        for i in range(n_items):
            vec = all_vecs[i]
            identity = identities[i]
            matched_idx = -1

            if n_clusters > 0:
                # Single matrix-vector multiply against active centroids
                sims = centroid_mat[:n_clusters] @ vec

                if identity != "Unknown":
                    # Named: match only clusters with same identity
                    for j in _np.argsort(-sims):
                        if sims[j] < threshold:
                            break
                        if cluster_identity[j] == identity:
                            matched_idx = int(j)
                            break
                else:
                    best_idx = int(_np.argmax(sims))
                    if sims[best_idx] >= threshold:
                        matched_idx = best_idx

            if matched_idx >= 0:
                cid = cluster_ids[matched_idx]
                assignments[i] = cid
                # In-place centroid update (running average)
                n = cluster_counts[matched_idx] + 1
                centroid_mat[matched_idx] = (
                    centroid_mat[matched_idx] * ((n - 1) / n) + vec * (1.0 / n)
                )
                cluster_counts[matched_idx] = n
            else:
                # Grow centroid matrix if needed
                if n_clusters >= max_clusters:
                    max_clusters *= 2
                    new_mat = _np.empty((max_clusters, dim), dtype=_np.float32)
                    new_mat[:n_clusters] = centroid_mat[:n_clusters]
                    centroid_mat = new_mat

                cid = next_cluster_id
                next_cluster_id += 1
                centroid_mat[n_clusters] = vec
                cluster_ids.append(cid)
                cluster_counts.append(1)
                cluster_identity.append(identity)
                assignments[i] = cid
                n_clusters += 1

        # Write cluster assignments back to DB
        self.conn.execute("UPDATE face_occurrences SET cluster_id = NULL")
        self.conn.executemany(
            "UPDATE face_occurrences SET cluster_id = ? WHERE id = ?",
            [(int(assignments[i]), occ_ids[i]) for i in range(n_items)],
        )

        # Auto-recover person links from previous clustering
        self.auto_assign_persons_after_recluster()

        return n_clusters

    def get_face_occurrences_count(self) -> int:
        """Return total number of face occurrences stored."""
        row = self.conn.execute("SELECT COUNT(*) AS cnt FROM face_occurrences").fetchone()
        return row["cnt"] if row else 0

    def find_similar_images_for_person(
        self,
        person_id: int,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Find images containing faces most similar to a person's identity.

        Uses the cached face embedding matrix for vectorised scoring.
        Returns up to *limit* images ranked by best face similarity,
        excluding images already linked to this person's clusters.
        """
        import numpy as _np

        if not self._table_exists("face_occurrences") or not self._table_exists("face_persons"):
            return []

        # 1. Collect person's linked cluster IDs and their image IDs to exclude.
        person_occ_rows = self.conn.execute(
            "SELECT DISTINCT image_id FROM face_occurrences WHERE person_id = ?",
            [person_id],
        ).fetchall()
        exclude_image_ids: set[int] = {r["image_id"] for r in person_occ_rows}

        # 2. Compute person centroid from linked face embeddings.
        centroid_rows = self.conn.execute(
            "SELECT embedding FROM face_occurrences "
            "WHERE person_id = ? AND embedding IS NOT NULL",
            [person_id],
        ).fetchall()
        if not centroid_rows:
            return []

        vectors: list[Any] = []
        for row in centroid_rows:
            vec = _np.frombuffer(row["embedding"], dtype=_np.float32).copy()
            norm = float(_np.linalg.norm(vec))
            if norm > 0:
                vectors.append((vec / norm).astype(_np.float32))
        if not vectors:
            return []

        centroid = _np.vstack(vectors).mean(axis=0).astype(_np.float32)
        c_norm = float(_np.linalg.norm(centroid))
        if c_norm <= 0:
            return []
        centroid /= c_norm  # (512,)

        # 3. Get cached embedding matrix (auto-rebuilds if stale).
        matrix, occ_ids, img_ids, _p_ids = self._face_emb_cache.get(self.conn)
        if matrix is None or len(occ_ids) == 0:
            return []

        # 4. Vectorised scoring: single BLAS call.
        scores = matrix @ centroid  # (N,)

        # 5. Group by image_id, take max similarity per image.
        #    Exclude images already linked to this person.
        best_per_image: dict[int, tuple[float, int]] = {}  # image_id → (score, occ_id)
        for i in range(len(occ_ids)):
            iid = img_ids[i]
            if iid in exclude_image_ids:
                continue
            s = float(scores[i])
            cur = best_per_image.get(iid)
            if cur is None or s > cur[0]:
                best_per_image[iid] = (s, occ_ids[i])

        if not best_per_image:
            return []

        # 6. Sort by similarity descending, take top N.
        ranked = sorted(best_per_image.items(), key=lambda x: x[1][0], reverse=True)[:limit]
        result_image_ids = [r[0] for r in ranked]

        # 7. Fetch file paths.
        placeholders = ",".join("?" * len(result_image_ids))
        path_rows = self.conn.execute(
            f"SELECT id, file_path FROM images WHERE id IN ({placeholders})",
            result_image_ids,
        ).fetchall()
        path_map = {r["id"]: r["file_path"] for r in path_rows}

        return [
            {
                "image_id": iid,
                "file_path": path_map.get(iid, ""),
                "similarity": round(sim, 4),
                "best_occurrence_id": occ_id,
            }
            for iid, (sim, occ_id) in ranked
            if iid in path_map
        ]

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

        # Caption analysis (description, keywords, scene, etc.)
        local = self.conn.execute(
            "SELECT * FROM analysis_caption WHERE image_id = ?", [image_id]
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

        # BLIP-2 individual pass (supplement caption or used alone)
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

        if self._table_exists("face_occurrences"):
            person_name_select = "fp.name AS person_name" if self._table_exists("face_persons") else "NULL AS person_name"
            person_join = "LEFT JOIN face_persons fp ON fp.id = fo.person_id" if self._table_exists("face_persons") else ""
            cluster_label_select = (
                "fcl.display_name AS cluster_label"
                if self._table_exists("face_cluster_labels")
                else "NULL AS cluster_label"
            )
            cluster_join = (
                "LEFT JOIN face_cluster_labels fcl ON fcl.cluster_id = fo.cluster_id"
                if self._table_exists("face_cluster_labels")
                else ""
            )
            occurrence_rows = self.conn.execute(
                f"""
                SELECT DISTINCT
                    {person_name_select},
                    {cluster_label_select}
                FROM face_occurrences fo
                {person_join}
                {cluster_join}
                WHERE fo.image_id = ?
                """,
                [image_id],
            ).fetchall()
            for occurrence in occurrence_rows:
                if occurrence["person_name"]:
                    faces_parts.append(occurrence["person_name"])
                if occurrence["cluster_label"]:
                    faces_parts.append(occurrence["cluster_label"])

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
            for identity in self.find_face_identities_by_alias(face_name):
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

        for module in ("metadata", "technical", "caption", "blip2", "objects", "ocr", "faces", "aesthetic", "perception"):
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
