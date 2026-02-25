"""Search engine — FTS5 text search + CLIP cosine similarity for hybrid search."""
from __future__ import annotations

import json
import sqlite3
from typing import Any

import numpy as np

from imganalyzer.db.repository import Repository


class SearchEngine:
    """Hybrid search: FTS5 for text matching + CLIP embeddings for semantic."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.repo = Repository(conn)

    def search(
        self,
        query: str,
        limit: int = 20,
        semantic_weight: float = 0.5,
        mode: str = "hybrid",
    ) -> list[dict[str, Any]]:
        """Search images by text query.

        *mode* can be:
        - ``text``: FTS5 only (BM25 ranking)
        - ``semantic``: CLIP embedding similarity only
        - ``hybrid``: weighted combination of both

        Returns list of {image_id, file_path, score, match_type, snippet}.
        """
        if mode == "text":
            return self._fts_search(query, limit)
        elif mode == "semantic":
            return self._semantic_search(query, limit)
        else:
            return self._hybrid_search(query, limit, semantic_weight)

    def search_face(self, name: str, limit: int = 50) -> list[dict[str, Any]]:
        """Search for images containing a face identity (by name or alias)."""
        # Resolve identity (canonical, display, or alias)
        identity = self.repo.find_face_by_alias(name)
        if identity is None:
            # Fall back to FTS text search
            return self._fts_search(name, limit)

        # Search all name variants in face_identities JSON
        search_names = [identity["canonical_name"]]
        if identity.get("display_name"):
            search_names.append(identity["display_name"])
        aliases = json.loads(identity.get("aliases") or "[]")
        search_names.extend(aliases)

        results: list[dict[str, Any]] = []
        seen: set[int] = set()

        for search_name in search_names:
            rows = self.conn.execute(
                """SELECT la.image_id, i.file_path, la.face_identities
                   FROM analysis_local_ai la
                   JOIN images i ON i.id = la.image_id
                   WHERE la.face_identities LIKE ?
                   LIMIT ?""",
                [f"%{search_name}%", limit],
            ).fetchall()
            for r in rows:
                if r["image_id"] not in seen:
                    seen.add(r["image_id"])
                    results.append({
                        "image_id": r["image_id"],
                        "file_path": r["file_path"],
                        "score": 1.0,
                        "match_type": "face",
                        "snippet": f"Face: {search_name}",
                    })

        return results[:limit]

    def search_exif(
        self,
        camera: str | None = None,
        lens: str | None = None,
        location: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        iso_min: int | None = None,
        iso_max: int | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Search by EXIF metadata fields."""
        conditions: list[str] = []
        params: list[Any] = []

        if camera:
            conditions.append(
                "(m.camera_make LIKE ? OR m.camera_model LIKE ?)"
            )
            params.extend([f"%{camera}%", f"%{camera}%"])
        if lens:
            conditions.append("m.lens_model LIKE ?")
            params.append(f"%{lens}%")
        if location:
            conditions.append(
                "(m.location_city LIKE ? OR m.location_state LIKE ? OR m.location_country LIKE ?)"
            )
            params.extend([f"%{location}%", f"%{location}%", f"%{location}%"])
        if date_from:
            conditions.append("m.date_time_original >= ?")
            params.append(date_from)
        if date_to:
            conditions.append("m.date_time_original <= ?")
            params.append(date_to)
        if iso_min is not None:
            conditions.append("m.iso >= ?")
            params.append(iso_min)
        if iso_max is not None:
            conditions.append("m.iso <= ?")
            params.append(iso_max)

        if not conditions:
            return []

        where = " AND ".join(conditions)
        params.append(limit)

        rows = self.conn.execute(
            f"""SELECT m.image_id, i.file_path,
                       m.camera_make, m.camera_model, m.lens_model,
                       m.location_city, m.location_country
                FROM analysis_metadata m
                JOIN images i ON i.id = m.image_id
                WHERE {where}
                LIMIT ?""",
            params,
        ).fetchall()

        results = []
        for r in rows:
            parts = [
                p for p in [
                    r["camera_make"], r["camera_model"], r["lens_model"],
                    r["location_city"], r["location_country"]
                ] if p
            ]
            results.append({
                "image_id": r["image_id"],
                "file_path": r["file_path"],
                "score": 1.0,
                "match_type": "exif",
                "snippet": " | ".join(parts),
            })
        return results

    # ── Internal search methods ────────────────────────────────────────────

    def _fts_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Full-text search via FTS5 with BM25 ranking."""
        rows = self.conn.execute(
            """SELECT si.image_id,
                      i.file_path,
                      rank AS score,
                      snippet(search_index, 1, '<b>', '</b>', '...', 32) as snippet
               FROM search_index si
               JOIN images i ON i.id = CAST(si.image_id AS INTEGER)
               WHERE search_index MATCH ?
               ORDER BY rank
               LIMIT ?""",
            [query, limit],
        ).fetchall()

        return [
            {
                "image_id": int(r["image_id"]),
                "file_path": r["file_path"],
                "score": -r["score"],  # FTS5 rank is negative (lower=better)
                "match_type": "text",
                "snippet": r["snippet"],
            }
            for r in rows
        ]

    def _semantic_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        """CLIP embedding cosine similarity search."""
        from imganalyzer.embeddings.clip_embedder import (
            CLIPEmbedder, vector_from_bytes, cosine_similarity,
        )

        embedder = CLIPEmbedder()
        query_vec_bytes = embedder.embed_text(query)
        query_vec = vector_from_bytes(query_vec_bytes)

        all_embeddings = self.repo.get_all_embeddings("image_clip")
        if not all_embeddings:
            return []

        scored: list[tuple[int, float]] = []
        for image_id, vec_bytes in all_embeddings:
            vec = vector_from_bytes(vec_bytes)
            sim = cosine_similarity(query_vec, vec)
            scored.append((image_id, sim))

        scored.sort(key=lambda x: -x[1])
        top = scored[:limit]

        results = []
        for image_id, sim in top:
            img = self.repo.get_image(image_id)
            results.append({
                "image_id": image_id,
                "file_path": img["file_path"] if img else "?",
                "score": sim,
                "match_type": "semantic",
                "snippet": f"CLIP similarity: {sim:.3f}",
            })
        return results

    def _hybrid_search(
        self, query: str, limit: int, semantic_weight: float
    ) -> list[dict[str, Any]]:
        """Combine FTS5 and CLIP scores with weighted sum."""
        text_results = self._fts_search(query, limit * 2)
        semantic_results = self._semantic_search(query, limit * 2)

        # Normalize scores to [0, 1]
        text_scores: dict[int, float] = {}
        if text_results:
            max_text = max(r["score"] for r in text_results) or 1.0
            for r in text_results:
                text_scores[r["image_id"]] = r["score"] / max_text

        sem_scores: dict[int, float] = {}
        if semantic_results:
            max_sem = max(r["score"] for r in semantic_results) or 1.0
            for r in semantic_results:
                sem_scores[r["image_id"]] = r["score"] / max_sem

        # Merge
        all_ids = set(text_scores.keys()) | set(sem_scores.keys())
        combined: list[tuple[int, float]] = []
        for iid in all_ids:
            t = text_scores.get(iid, 0.0)
            s = sem_scores.get(iid, 0.0)
            score = (1 - semantic_weight) * t + semantic_weight * s
            combined.append((iid, score))

        combined.sort(key=lambda x: -x[1])
        top = combined[:limit]

        # Build result dicts
        # Cache path lookups
        path_cache: dict[int, str] = {}
        for r in text_results + semantic_results:
            path_cache[r["image_id"]] = r["file_path"]

        results = []
        for image_id, score in top:
            file_path = path_cache.get(image_id) or "?"
            if file_path == "?":
                img = self.repo.get_image(image_id)
                if img:
                    file_path = img["file_path"]
            results.append({
                "image_id": image_id,
                "file_path": file_path,
                "score": score,
                "match_type": "hybrid",
                "snippet": f"Combined score: {score:.3f}",
            })
        return results
