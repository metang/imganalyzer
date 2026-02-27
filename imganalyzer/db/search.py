"""Search engine — FTS5 text search + CLIP cosine similarity for hybrid search.

Semantic search algorithm notes
--------------------------------
CLIP embeddings live on a unit hypersphere (both encoders L2-normalise their
output).  Raw cosine similarity values are compressed into a narrow band
(roughly 0.15–0.45 even for highly relevant pairs), so absolute thresholds
and score/max normalisation are fragile.

We use **Reciprocal Rank Fusion (RRF)** throughout:

    rrf(rank, k=60) = 1 / (rank + k)

RRF converts any ranked list into a score that depends only on *position*, not
on the raw similarity magnitude.  This eliminates three problems at once:

1. Cross-modal score inflation — ``description_clip`` (text→text) cosine
   scores are systematically higher than ``image_clip`` (text→image) scores
   for the same query, so ``max()`` or direct averaging always favours
   described images.  RRF is scale-independent.

2. score/max normalisation fragility — if the top result happens to be a weak
   match, all scores get artificially inflated.

3. Heterogeneous source fusion — FTS5 BM25 scores and CLIP cosine scores live
   in entirely different numerical ranges; RRF merges them by rank only.

Additionally, the text query sent to the ``image_clip`` encoder is prefixed
with ``"a photo of "`` — a technique from the CLIP paper that substantially
improves text→image retrieval accuracy by matching the captioning register the
model was trained on.  Queries against ``description_clip`` are kept verbatim
because those embeddings were produced from plain AI-generated prose.

Description quality gating
---------------------------
Local-only BLIP-2 AI descriptions are very short (18–62 chars) and use
repetitive sentence structure (e.g. "a blue sky a beach scene a boat"), causing
their CLIP text embeddings to cluster near the centroid and produce inflated,
meaningless cosine scores against any query.  Cloud AI descriptions are rich
paragraphs (200-500+ chars) that produce discriminative embeddings.

``description_clip`` embeddings are therefore only used in RRF when the
underlying description text is rich (>= ``_DESC_QUALITY_THRESHOLD`` chars).
Images whose stored description is short fall back to ``image_clip`` only.
The threshold is calibrated empirically: local-AI max is ~62 chars; cloud-AI
min is well above 100 chars.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any

import numpy as np

from imganalyzer.db.repository import Repository

# RRF constant — standard default from the Cormack & Clarke (2009) paper.
# Higher k reduces the influence of rank-1 results; 60 is widely used.
_RRF_K = 60

# Candidate pool multiplier: fetch this many times the requested limit before
# RRF re-ranking so that low-ranked-but-relevant items aren't cut off early.
_POOL_FACTOR = 4

# Minimum description length (chars) for description_clip to be trusted.
# Local-AI (BLIP-2) descriptions top out at ~62 chars and produce noisy,
# centroid-clustered embeddings.  Cloud-AI descriptions are 200-500+ chars.
# 100 chars cleanly separates the two populations.
_DESC_QUALITY_THRESHOLD = 100

# Weight of description_clip's RRF contribution relative to image_clip.
# image_clip is the primary, reliable visual signal.  description_clip can
# corroborate a high image_clip score but must not overrule it — its cosine
# values cluster in a narrow band (~0.7–0.8) for virtually all images and are
# therefore a weak discriminator.  A weight of 0.25 lets description_clip
# serve as a tiebreaker boost without dragging irrelevant images to the top.
_DESC_WEIGHT = 0.25

# Z-score threshold for description_clip-only fallback candidates.
# Images that have no image_clip are ranked solely by description_clip cosine.
# We filter these to only those whose cosine is at least this many standard
# deviations above the population mean, to suppress noisy low-signal results.
_DESC_ONLY_ZSCORE_MIN = 1.5


def _rrf_score(rank: int, k: int = _RRF_K) -> float:
    """Reciprocal Rank Fusion score for a result at zero-based *rank*."""
    return 1.0 / (rank + 1 + k)  # +1 so rank-0 → 1/(1+k), not 1/k


def _rank_results(
    scored: list[tuple[int, float]]
) -> dict[int, int]:
    """Return {image_id: zero_based_rank} for a list sorted best-first."""
    scored_sorted = sorted(scored, key=lambda x: -x[1])
    return {image_id: rank for rank, (image_id, _) in enumerate(scored_sorted)}


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

    def _get_rich_desc_image_ids(self) -> frozenset[int]:
        """Return the set of image IDs whose stored description text is rich.

        ``description_clip`` embeddings built from short local-AI captions
        (BLIP-2 outputs of 18–62 chars) cluster near the embedding centroid and
        produce inflated, meaningless cosine scores for any query.  Only trust
        ``description_clip`` when the concatenated description text that was
        used to build the embedding is long enough to be discriminative.

        We approximate this by checking the *current* stored descriptions.
        Cloud-AI descriptions are typically 200-500+ chars; local-only
        descriptions are always < 100 chars.
        """
        # Combine local AI description + cloud AI description lengths per image.
        # An image is "rich" if any stored description exceeds the threshold.
        rows = self.conn.execute(
            """
            SELECT image_id,
                   MAX(COALESCE(length(local_desc), 0) + COALESCE(length(cloud_desc), 0)) AS total_len
            FROM (
                SELECT la.image_id,
                       la.description AS local_desc,
                       ca.description AS cloud_desc
                FROM analysis_local_ai la
                LEFT JOIN analysis_cloud_ai ca ON ca.image_id = la.image_id
                UNION ALL
                SELECT ca.image_id,
                       NULL AS local_desc,
                       ca.description AS cloud_desc
                FROM analysis_cloud_ai ca
                WHERE ca.image_id NOT IN (SELECT image_id FROM analysis_local_ai)
            )
            GROUP BY image_id
            HAVING total_len >= ?
            """,
            [_DESC_QUALITY_THRESHOLD],
        ).fetchall()
        return frozenset(r[0] for r in rows)

    def _semantic_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        """CLIP embedding semantic search — two-tier strategy.

        **Tier 1 — images with ``image_clip``** (primary visual signal):
            Ranked by ``image_clip`` cosine (query: ``"a photo of {query}"``).
            For images that *also* have a rich ``description_clip``, a small
            weighted boost is added as a tiebreaker:

                fused = 1.0 * rrf(image_rank) + _DESC_WEIGHT * rrf(desc_rank)

            ``description_clip`` cosines cluster in a narrow band (~0.7–0.8)
            for virtually all images regardless of relevance, so its RRF
            contribution is capped at ``_DESC_WEIGHT`` = 0.25 to prevent it
            from overruling the visual signal.

        **Tier 2 — images with only ``description_clip``, no ``image_clip``**:
            These have no visual signal.  They are only included when their
            ``description_clip`` cosine is statistically significant:
            z-score ≥ ``_DESC_ONLY_ZSCORE_MIN`` = 1.5 above the population
            mean.  Tier-2 results are appended *after* all Tier-1 results so
            they can never displace a visually-grounded result.
        """
        from imganalyzer.embeddings.clip_embedder import (
            CLIPEmbedder, vector_from_bytes, cosine_similarity,
        )

        embedder = CLIPEmbedder()

        # Fetch a larger pool so RRF has more candidates to re-rank.
        pool = max(limit * _POOL_FACTOR, 100)

        # Encode two query variants:
        # - visual query: prefixed for better text→image retrieval
        # - text query: verbatim for text→text description matching
        visual_query = query if query.lower().startswith("a photo of") else f"a photo of {query}"
        visual_query_vec = vector_from_bytes(embedder.embed_text(visual_query))
        text_query_vec   = vector_from_bytes(embedder.embed_text(query))

        image_embeddings = self.repo.get_all_embeddings("image_clip")
        desc_embeddings  = self.repo.get_all_embeddings("description_clip")

        if not image_embeddings and not desc_embeddings:
            return []

        # Determine which images have rich-enough descriptions to trust.
        rich_desc_ids = self._get_rich_desc_image_ids() if desc_embeddings else frozenset()

        # ── Score all available embeddings ────────────────────────────────

        image_scored: list[tuple[int, float]] = []
        image_id_set: set[int] = set()
        for image_id, vec_bytes in image_embeddings:
            vec = vector_from_bytes(vec_bytes)
            image_scored.append((image_id, cosine_similarity(visual_query_vec, vec)))
            image_id_set.add(image_id)

        # Only score rich descriptions; collect ALL of them for z-score stats.
        desc_cosines: dict[int, float] = {}
        for image_id, vec_bytes in desc_embeddings:
            if image_id not in rich_desc_ids:
                continue
            vec = vector_from_bytes(vec_bytes)
            desc_cosines[image_id] = cosine_similarity(text_query_vec, vec)

        # ── Tier 1: images that have image_clip ───────────────────────────

        image_scored.sort(key=lambda x: -x[1])
        image_pool = image_scored[:pool]
        image_ranks = {iid: rank for rank, (iid, _) in enumerate(image_pool)}

        # Build rank map for description_clip (only for Tier-1 boost)
        tier1_desc = [(iid, s) for iid, s in desc_cosines.items() if iid in image_id_set]
        tier1_desc.sort(key=lambda x: -x[1])
        tier1_desc_pool = tier1_desc[:pool]
        desc_ranks_t1 = {iid: rank for rank, (iid, _) in enumerate(tier1_desc_pool)}

        tier1_fused: list[tuple[int, float]] = []
        for image_id, rank in image_ranks.items():
            score = _rrf_score(rank)  # primary: image_clip weight = 1.0
            if image_id in desc_ranks_t1:
                score += _DESC_WEIGHT * _rrf_score(desc_ranks_t1[image_id])
            tier1_fused.append((image_id, score))

        tier1_fused.sort(key=lambda x: -x[1])
        tier1_top = tier1_fused[:limit]

        # ── Tier 2: description_clip-only fallback ────────────────────────

        # Images that have NO image_clip at all.
        desc_only_cosines = {
            iid: s for iid, s in desc_cosines.items() if iid not in image_id_set
        }

        tier2_top: list[tuple[int, float]] = []
        if desc_only_cosines:
            vals = np.array(list(desc_only_cosines.values()), dtype=np.float32)
            mean_c = float(vals.mean())
            std_c  = float(vals.std())
            if std_c > 0:
                # Keep only statistically significant outliers
                tier2_candidates = [
                    (iid, s) for iid, s in desc_only_cosines.items()
                    if (s - mean_c) / std_c >= _DESC_ONLY_ZSCORE_MIN
                ]
            else:
                tier2_candidates = []
            tier2_candidates.sort(key=lambda x: -x[1])
            tier2_top = tier2_candidates[:limit]

        # Combine: Tier 1 first, then Tier 2 (Tier 2 never displaces Tier 1)
        # Deduplicate in case an image somehow appears in both (shouldn't happen).
        seen_ids: set[int] = set()
        combined: list[tuple[int, float]] = []
        for image_id, score in tier1_top + tier2_top:
            if image_id not in seen_ids:
                seen_ids.add(image_id)
                combined.append((image_id, score))

        top = combined[:limit]

        # ── Build results ─────────────────────────────────────────────────

        path_cache: dict[int, str] = {}
        for image_id, _ in top:
            if image_id not in path_cache:
                img = self.repo.get_image(image_id)
                if img:
                    path_cache[image_id] = img["file_path"]

        results = []
        for image_id, score in top:
            file_path = path_cache.get(image_id, "?")
            results.append({
                "image_id": image_id,
                "file_path": file_path,
                "score": score,
                "match_type": "semantic",
                "snippet": f"Semantic score: {score:.4f}",
            })
        return results

    def _hybrid_search(
        self, query: str, limit: int, semantic_weight: float
    ) -> list[dict[str, Any]]:
        """Combine FTS5 and CLIP scores with weighted RRF fusion.

        Both sources are converted to RRF scores before weighting so that
        FTS5 BM25 magnitudes and CLIP cosine magnitudes (which live in
        completely different numerical ranges) are made commensurable.
        ``semantic_weight`` controls the relative contribution:
        0.0 = text only, 1.0 = semantic only, 0.5 = equal weight.
        """
        pool = limit * _POOL_FACTOR
        text_results     = self._fts_search(query, pool)
        semantic_results = self._semantic_search(query, pool)

        # Convert each result list to RRF scores (already sorted best-first)
        text_rrf: dict[int, float] = {
            r["image_id"]: _rrf_score(rank)
            for rank, r in enumerate(text_results)
        }
        sem_rrf: dict[int, float] = {
            r["image_id"]: _rrf_score(rank)
            for rank, r in enumerate(semantic_results)
        }

        # Weighted combination of RRF scores
        all_ids = set(text_rrf.keys()) | set(sem_rrf.keys())
        combined: list[tuple[int, float]] = []
        for iid in all_ids:
            t = text_rrf.get(iid, 0.0)
            s = sem_rrf.get(iid, 0.0)
            score = (1.0 - semantic_weight) * t + semantic_weight * s
            combined.append((iid, score))

        combined.sort(key=lambda x: -x[1])
        top = combined[:limit]

        # Build path cache
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
                "snippet": f"Combined score: {score:.4f}",
            })
        return results
