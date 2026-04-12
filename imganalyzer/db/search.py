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

Vectorized scoring (performance)
---------------------------------
At 500K images, the naive Python loop over ``get_all_embeddings()`` takes
10–60 seconds per query (3 GB of BLOBs loaded, per-row ``np.dot``, per-row
``np.linalg.norm``).  Since CLIP embeddings are already L2-normalised,
cosine similarity == dot product.  We pre-load all embeddings into a single
``(N, 768)`` float32 numpy matrix and compute all scores in one BLAS call:

    scores = matrix @ query_vec          # (N,) = (N,768) @ (768,)

This is ~100–1000x faster (single-digit milliseconds on any modern CPU).
The matrix is cached per ``SearchEngine`` instance and invalidated via a
simple row-count check — if new embeddings have been added since the last
query, the matrix is rebuilt.
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable
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
_FTS_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
_FTS_QUOTED_PHRASE_RE = re.compile(r'"([^"]+)"|\'([^\']+)\'')
_SEMANTIC_PROFILE_WEIGHTS: dict[str, tuple[float, float]] = {
    "image_dominant": (1.0, 0.10),
    "balanced": (1.0, _DESC_WEIGHT),
    "description_dominant": (0.60, 1.0),
}

# Callback type for search progress: (phase, message, progress_fraction)
ProgressCallback = Callable[[str, str, float], None] | None


def _rrf_score(rank: int, k: int = _RRF_K) -> float:
    """Reciprocal Rank Fusion score for a result at zero-based *rank*."""
    return 1.0 / (rank + 1 + k)  # +1 so rank-0 → 1/(1+k), not 1/k


def _rank_results(
    scored: list[tuple[int, float]]
) -> dict[int, int]:
    """Return {image_id: zero_based_rank} for a list sorted best-first."""
    scored_sorted = sorted(scored, key=lambda x: -x[1])
    return {image_id: rank for rank, (image_id, _) in enumerate(scored_sorted)}


def _build_fts_match_query(query: str) -> str:
    """Convert freeform user text into a safe FTS5 MATCH query."""
    tokens = [token.strip("_") for token in _FTS_TOKEN_RE.findall(query)]
    quoted_tokens = [f'"{token.replace("\"", "\"\"")}"' for token in tokens if token]
    return " AND ".join(quoted_tokens)


def _extract_quoted_phrases(query: str) -> list[str]:
    phrases: list[str] = []
    seen: set[str] = set()
    for match in _FTS_QUOTED_PHRASE_RE.finditer(query):
        phrase = (match.group(1) or match.group(2) or "").strip()
        if not phrase:
            continue
        normalized = " ".join(phrase.split())
        lowered = normalized.casefold()
        if lowered in seen:
            continue
        seen.add(lowered)
        phrases.append(normalized)
    return phrases


def _build_fts_phrase_match_query(query: str) -> str:
    """Build OR query of explicitly quoted phrases from user input."""
    phrases = _extract_quoted_phrases(query)
    if not phrases:
        return ""
    safe = [f'"{phrase.replace("\"", "\"\"")}"' for phrase in phrases]
    return " OR ".join(safe)


def _build_fts_subphrase_match_query(query: str, window: int = 3, max_clauses: int = 8) -> str:
    """Build OR query of sliding token windows to support partial text overlap."""
    tokens = [token.strip("_") for token in _FTS_TOKEN_RE.findall(query) if token.strip("_")]
    if len(tokens) < window:
        return ""
    clauses: list[str] = []
    seen: set[str] = set()
    for idx in range(0, len(tokens) - window + 1):
        phrase = " ".join(tokens[idx: idx + window])
        lowered = phrase.casefold()
        if lowered in seen:
            continue
        seen.add(lowered)
        clauses.append(f'"{phrase.replace("\"", "\"\"")}"')
        if len(clauses) >= max_clauses:
            break
    return " OR ".join(clauses)


def _build_fts_soft_match_query(query: str, max_tokens: int = 16) -> str:
    """Build OR token query for partial/paraphrased lexical recall."""
    raw_tokens = [token.strip("_") for token in _FTS_TOKEN_RE.findall(query)]
    tokens: list[str] = []
    seen: set[str] = set()
    # Prefer informative tokens; fallback to all tokens when none qualify.
    preferred = [token for token in raw_tokens if len(token) >= 4]
    source = preferred if preferred else raw_tokens
    for token in source:
        if not token:
            continue
        lowered = token.casefold()
        if lowered in seen:
            continue
        seen.add(lowered)
        tokens.append(token)
        if len(tokens) >= max_tokens:
            break
    quoted_tokens = [f'"{token.replace("\"", "\"\"")}"' for token in tokens]
    return " OR ".join(quoted_tokens)


def _semantic_profile_weights(profile: str | None) -> tuple[float, float]:
    normalized = (profile or "balanced").strip().lower().replace("-", "_")
    return _SEMANTIC_PROFILE_WEIGHTS.get(normalized, _SEMANTIC_PROFILE_WEIGHTS["balanced"])


def _fuse_semantic_tier1_scores(
    image_ranks: dict[int, int],
    desc_ranks: dict[int, int],
    image_weight: float,
    desc_weight: float,
) -> list[tuple[int, float]]:
    fused: list[tuple[int, float]] = []
    for image_id, rank in image_ranks.items():
        score = image_weight * _rrf_score(rank)
        if image_id in desc_ranks:
            score += desc_weight * _rrf_score(desc_ranks[image_id])
        fused.append((image_id, score))
    fused.sort(key=lambda x: (-x[1], x[0]))
    return fused


class _EmbeddingMatrix:
    """Cached (N, 768) float32 matrix for a single embedding type.

    Loads all vectors from the DB into a contiguous numpy array and an
    aligned ``image_ids`` list.  Staleness is detected via both row count
    and MAX(rowid).  When the DB has only grown (append-only), new rows
    are fetched incrementally instead of rebuilding the whole matrix.
    """

    def __init__(self) -> None:
        self.matrix: np.ndarray | None = None      # (N, dim) float32
        self.image_ids: list[int] = []              # aligned with matrix rows
        self._row_count: int = 0                    # last-known DB row count
        self._max_rowid: int = 0                    # last-known MAX(rowid)

    def get(
        self, conn: sqlite3.Connection, embedding_type: str
    ) -> tuple[np.ndarray, list[int]]:
        """Return ``(matrix, image_ids)`` -- rebuilds or appends if stale."""
        row = conn.execute(
            "SELECT COUNT(*) AS cnt, MAX(rowid) AS max_rid FROM embeddings"
            " WHERE embedding_type = ?",
            [embedding_type],
        ).fetchone()
        current_count = row["cnt"] if row else 0
        current_max = int(row["max_rid"]) if row and row["max_rid"] is not None else 0

        if (
            self.matrix is not None
            and current_count == self._row_count
            and current_max == self._max_rowid
        ):
            return self.matrix, self.image_ids

        # Incremental append: only new rows were added (no deletes/updates)
        if (
            self.matrix is not None
            and self.matrix.size > 0
            and current_count > self._row_count
            and current_count - self._row_count == current_max - self._max_rowid
        ):
            new_rows = conn.execute(
                "SELECT image_id, vector FROM embeddings"
                " WHERE embedding_type = ? AND rowid > ?"
                " ORDER BY image_id",
                [embedding_type, self._max_rowid],
            ).fetchall()
            if new_rows:
                new_ids = [r["image_id"] for r in new_rows]
                new_vecs = [np.frombuffer(r["vector"], dtype=np.float32) for r in new_rows]
                self.matrix = np.vstack([self.matrix, np.vstack(new_vecs)])
                self.image_ids.extend(new_ids)
            self._row_count = current_count
            self._max_rowid = current_max
            return self.matrix, self.image_ids

        # Full rebuild
        rows = conn.execute(
            "SELECT image_id, vector FROM embeddings WHERE embedding_type = ? ORDER BY image_id",
            [embedding_type],
        ).fetchall()

        if not rows:
            empty = np.empty((0, 0), dtype=np.float32)
            self.matrix = empty
            self.image_ids = []
            self._row_count = 0
            self._max_rowid = 0
            return empty, []

        ids: list[int] = []
        vecs: list[np.ndarray] = []
        for r in rows:
            ids.append(r["image_id"])
            vecs.append(np.frombuffer(r["vector"], dtype=np.float32))

        self.matrix = np.vstack(vecs)  # (N, dim)
        self.image_ids = ids
        self._row_count = current_count
        self._max_rowid = current_max
        return self.matrix, self.image_ids


class SearchEngine:
    """Hybrid search: FTS5 for text matching + CLIP embeddings for semantic."""

    _RICH_DESC_TTL = 30.0  # seconds to cache _get_rich_desc_image_ids()

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.repo = Repository(conn)
        # Cached embedding matrices -- rebuilt automatically when new
        # embeddings are inserted (detected via row-count + max-rowid check).
        self._image_clip_cache = _EmbeddingMatrix()
        self._desc_clip_cache = _EmbeddingMatrix()
        self._rich_desc_ids: frozenset[int] | None = None
        self._rich_desc_ts: float = 0.0

    def search(
        self,
        query: str,
        limit: int = 20,
        semantic_weight: float = 0.5,
        mode: str = "hybrid",
        semantic_profile: str | None = None,
        progress_cb: ProgressCallback = None,
        candidate_ids: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Search images by text query.

        *mode* can be:
        - ``text``: FTS5 only (BM25 ranking)
        - ``semantic``: CLIP embedding similarity only
        - ``hybrid``: weighted combination of both

        Returns list of {image_id, file_path, score, match_type, snippet}.
        """
        if mode == "text":
            return self._fts_search(query, limit, candidate_ids=candidate_ids)
        elif mode == "semantic":
            return self._semantic_search(
                query, limit, semantic_profile, progress_cb=progress_cb, candidate_ids=candidate_ids
            )
        else:
            return self._hybrid_search(
                query,
                limit,
                semantic_weight,
                semantic_profile,
                progress_cb=progress_cb,
                candidate_ids=candidate_ids,
            )

    def _resolve_face_rows(self, name: str, limit: int | None = 50) -> list[dict[str, Any]]:
        """Resolve one face/person query into image rows without text fallback."""
        identities = self.repo.find_face_identities_by_alias(name)
        persons = self.repo.find_persons_by_name(name)
        clusters = self.repo.find_clusters_by_label(name)
        if not identities and not persons and not clusters:
            return []

        results: list[dict[str, Any]] = []
        seen: set[int] = set()

        def add_rows(rows: list[dict[str, Any]], snippet: str) -> None:
            for row in rows:
                image_id = int(row["image_id"])
                if image_id in seen:
                    continue
                seen.add(image_id)
                results.append({
                    "image_id": image_id,
                    "file_path": row["file_path"],
                    "score": 1.0,
                    "match_type": "face",
                    "snippet": snippet,
                })

        for person in persons:
            add_rows(
                self.repo.get_images_for_person(int(person["id"]), limit=limit),
                f"Person: {person['name']}",
            )

        for cluster in clusters:
            add_rows(
                self.repo.get_images_for_cluster(int(cluster["cluster_id"]), limit=limit),
                f"Face: {cluster['display_name']}",
            )

        for identity in identities:
            search_names = [identity["canonical_name"]]
            if identity.get("display_name"):
                search_names.append(identity["display_name"])
            aliases = json.loads(identity.get("aliases") or "[]")
            search_names.extend(aliases)
            snippet_name = identity.get("display_name") or identity["canonical_name"]

            for search_name in dict.fromkeys(search_names):
                add_rows(
                    self.repo.get_images_for_face(search_name, limit=limit),
                    f"Face: {snippet_name}",
                )

        if limit is None or limit <= 0:
            return results
        return results[:limit]

    def search_face(self, name: str, limit: int = 50) -> list[dict[str, Any]]:
        """Search for images containing a face identity (by name or alias)."""
        rows = self._resolve_face_rows(name, limit=limit)
        if rows:
            return rows
        # Fall back to FTS text search
        return self._fts_search(name, limit)

    def search_faces(
        self,
        names: list[str],
        limit: int = 50,
        match_mode: str = "all",
    ) -> list[dict[str, Any]]:
        """Search for images containing one or more face/person filters."""
        clean_names: list[str] = []
        seen_names: set[str] = set()
        for name in names:
            clean = str(name).strip()
            lowered = clean.casefold()
            if not clean or lowered in seen_names:
                continue
            seen_names.add(lowered)
            clean_names.append(clean)

        if not clean_names:
            return []
        if len(clean_names) == 1:
            return self.search_face(clean_names[0], limit)

        all_results = [self._resolve_face_rows(name, limit=None) for name in clean_names]
        if match_mode != "any" and any(not rows for rows in all_results):
            return []

        aggregate: dict[int, dict[str, Any]] = {}
        for rows in all_results:
            for row in rows:
                image_id = int(row["image_id"])
                entry = aggregate.setdefault(image_id, {
                    "image_id": image_id,
                    "file_path": row["file_path"],
                    "score": 0.0,
                    "matches": 0,
                })
                entry["score"] = float(entry["score"]) + float(row["score"])
                entry["matches"] = int(entry["matches"]) + 1

        required_matches = len(clean_names) if match_mode == "all" else 1
        snippet = f"People: {', '.join(clean_names)}"
        ranked = sorted(
            (
                {
                    "image_id": int(entry["image_id"]),
                    "file_path": entry["file_path"],
                    "score": float(entry["score"]),
                    "match_type": "face",
                    "snippet": snippet,
                }
                for entry in aggregate.values()
                if int(entry["matches"]) >= required_matches
            ),
            key=lambda item: (-float(item["score"]), str(item["file_path"])),
        )
        return ranked[:limit]

    def resolve_face_queries(self, query: str) -> tuple[list[str], str, str]:
        """Extract all face aliases/names from a freeform query."""
        normalized = " ".join(query.split())
        if not normalized:
            return [], "", "all"

        remainder = normalized
        matches: list[str] = []
        while remainder:
            span = self._extract_face_query_span(remainder)
            if span is None:
                break
            candidate, start, end = span
            if not any(candidate.casefold() == existing.casefold() for existing in matches):
                matches.append(candidate)
            remainder = f"{remainder[:start]} {remainder[end:]}"
            remainder = " ".join(remainder.split())

        cleaned_remainder = self._cleanup_face_query_remainder(remainder) if matches else normalized
        match_mode = "all" if len(matches) > 1 else "any"
        return matches, cleaned_remainder, match_mode

    def resolve_face_query(self, query: str) -> tuple[str | None, str]:
        """Extract the first alias/name span from a freeform query, if any."""
        faces, remaining_query, _ = self.resolve_face_queries(query)
        return (faces[0] if faces else None), remaining_query

    def _extract_face_query_span(self, normalized: str) -> tuple[str, int, int] | None:
        if self._matches_face_query(normalized):
            return normalized, 0, len(normalized)

        tokens = list(re.finditer(r"\S+", normalized))
        for span_len in range(len(tokens), 0, -1):
            for start_idx in range(len(tokens) - span_len + 1):
                start = tokens[start_idx].start()
                end = tokens[start_idx + span_len - 1].end()
                candidate = normalized[start:end].strip(" ,.;:!?()[]{}\"'")
                if not candidate:
                    continue
                if self._matches_face_query(candidate):
                    return candidate, start, end
        return None

    def _cleanup_face_query_remainder(self, remainder: str) -> str:
        cleaned = re.sub(r"[,&]+", " ", remainder)
        cleaned = re.sub(r"\b(with|and|together)\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(
            r"\b(?:is|are)\s+in\s+the\s+(?:picture|photo)\b",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
        return " ".join(cleaned.split())

    def _matches_face_query(self, candidate: str) -> bool:
        return bool(
            self.repo.find_face_identities_by_alias(candidate)
            or self.repo.find_persons_by_name(candidate)
            or self.repo.find_clusters_by_label(candidate)
        )

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

    def search_similar_image(self, image_id: int, limit: int = 50) -> list[dict[str, Any]]:
        """Search for images visually similar to an existing image."""
        seed_row = self.conn.execute(
            """SELECT embedding_type, vector
               FROM embeddings
               WHERE image_id = ?
                 AND embedding_type IN ('image_clip', 'description_clip')
               ORDER BY CASE embedding_type
                   WHEN 'image_clip' THEN 0
                   WHEN 'description_clip' THEN 1
                   ELSE 2
               END
               LIMIT 1""",
            [image_id],
        ).fetchone()
        if seed_row is None:
            return []

        embedding_type = str(seed_row["embedding_type"])
        query_vec = np.frombuffer(seed_row["vector"], dtype=np.float32)
        cache = self._image_clip_cache if embedding_type == "image_clip" else self._desc_clip_cache
        matrix, image_ids = cache.get(self.conn, embedding_type)
        if matrix.size == 0:
            return []

        scores = matrix @ query_vec
        scored = [
            (candidate_image_id, float(scores[idx]))
            for idx, candidate_image_id in enumerate(image_ids)
            if candidate_image_id != image_id
        ]
        scored.sort(key=lambda item: -item[1])

        results: list[dict[str, Any]] = []
        for candidate_image_id, score in scored[:limit]:
            image = self.repo.get_image(candidate_image_id)
            if image is None:
                continue
            results.append(
                {
                    "image_id": candidate_image_id,
                    "file_path": image["file_path"],
                    "score": score,
                    "match_type": "similar",
                    "snippet": "Similar photo",
                }
            )
        return results

    # ── Internal search methods ────────────────────────────────────────────

    def _fts_search(self, query: str, limit: int, candidate_ids: set[int] | None = None) -> list[dict[str, Any]]:
        """Full-text search via blended lexical channels + weighted BM25.

        Channels:
        - strict all-token match (precision)
        - explicit quoted phrase match (boost)
        - sliding subphrase windows (partial overlap recall)
        - soft OR-token match (broad partial recall)
        """
        strict_query = _build_fts_match_query(query)
        phrase_query = _build_fts_phrase_match_query(query)
        subphrase_query = _build_fts_subphrase_match_query(query)
        soft_query = _build_fts_soft_match_query(query)

        channels: list[tuple[str, str, float, int]] = []
        if strict_query:
            channels.append(("text_strict", strict_query, 1.0, 1))
        if phrase_query and phrase_query != strict_query:
            channels.append(("text_phrase", phrase_query, 1.2, 1))
        if subphrase_query and subphrase_query != strict_query and subphrase_query != phrase_query:
            channels.append(("text_subphrase", subphrase_query, 0.8, 2))
        if (
            soft_query
            and soft_query != strict_query
            and soft_query != phrase_query
            and soft_query != subphrase_query
        ):
            channels.append(("text_soft", soft_query, 0.45, 2))

        if not channels:
            return []

        informative_tokens = [
            token.casefold()
            for token in _FTS_TOKEN_RE.findall(query)
            if len(token) >= 4
        ]
        # Require that all informative query tokens exist in the authoritative
        # current analysis text for lexical FTS hits. This filters stale token
        # ghosts and weak OR-only matches (e.g. "woman in tube" matching images
        # that only contain "woman").
        required_tokens: list[str] = []
        if strict_query:
            required_tokens = informative_tokens
        elif phrase_query:
            required_tokens = informative_tokens
        elif subphrase_query:
            subphrase_tokens = [
                token.casefold()
                for token in _FTS_TOKEN_RE.findall(subphrase_query)
                if len(token) >= 4
            ]
            required_tokens = list(dict.fromkeys(subphrase_tokens)) or informative_tokens
        else:
            required_tokens = []
        enforce_consistency_guard = len(required_tokens) >= 2
        min_token_overlap = (
            len(required_tokens)
            if len(required_tokens) <= 2
            else 2 if len(required_tokens) == 3 else 1
        )
        table_exists_cache: dict[str, bool] = {}
        table_columns_cache: dict[str, set[str]] = {}
        current_text_cache: dict[int, str] = {}
        search_row_text_cache: dict[int, str] = {}
        has_analysis_rows_cache: dict[int, bool] = {}
        has_any_analysis_data: bool | None = None

        def _table_exists(name: str) -> bool:
            exists = table_exists_cache.get(name)
            if exists is None:
                exists = self.repo._table_exists(name)
                table_exists_cache[name] = exists
            return exists

        def _table_columns(name: str) -> set[str]:
            cols = table_columns_cache.get(name)
            if cols is None:
                try:
                    rows = self.conn.execute(f"PRAGMA table_info({name})").fetchall()
                except sqlite3.OperationalError:
                    cols = set()
                else:
                    cols = {str(row["name"]) for row in rows}
                table_columns_cache[name] = cols
            return cols

        def _first_available(columns: set[str], *candidates: str) -> str | None:
            for candidate in candidates:
                if candidate in columns:
                    return candidate
            return None

        def _extend_json_text(parts: list[str], raw: Any) -> None:
            if raw is None:
                return
            if isinstance(raw, list):
                parts.extend(str(item) for item in raw if item is not None)
                return
            if not isinstance(raw, str):
                parts.append(str(raw))
                return
            text = raw.strip()
            if not text:
                return
            try:
                decoded = json.loads(text)
            except (TypeError, ValueError):
                parts.append(text)
                return
            if isinstance(decoded, list):
                parts.extend(str(item) for item in decoded if item is not None)
            elif decoded is not None:
                parts.append(str(decoded))

        def _build_authoritative_text_blob(image_id: int) -> str:
            cached = current_text_cache.get(image_id)
            if cached is not None:
                return cached

            parts: list[str] = []

            if _table_exists("analysis_caption"):
                caption_cols = _table_columns("analysis_caption")
                if not caption_cols:
                    caption_cols = {"description", "scene_type", "main_subject", "keywords", "mood", "lighting", "detected_objects", "face_identities", "ocr_text"}
                desc_col = _first_available(caption_cols, "description")
                scene_col = _first_available(caption_cols, "scene_type")
                subject_col = _first_available(caption_cols, "main_subject")
                keywords_col = _first_available(caption_cols, "keywords")
                mood_col = _first_available(caption_cols, "mood")
                lighting_col = _first_available(caption_cols, "lighting")
                objects_col = _first_available(caption_cols, "detected_objects")
                faces_col = _first_available(caption_cols, "face_identities")
                ocr_col = _first_available(caption_cols, "ocr_text")

                select_cols = [col for col in (
                    desc_col,
                    scene_col,
                    subject_col,
                    keywords_col,
                    mood_col,
                    lighting_col,
                    objects_col,
                    faces_col,
                    ocr_col,
                ) if col]
                if not select_cols:
                    select_cols = ["description"]
                row = self.conn.execute(
                    f"SELECT {', '.join(select_cols)} FROM analysis_caption WHERE image_id = ?",
                    [image_id],
                ).fetchone()
                if row:
                    for field in (desc_col, scene_col, subject_col, mood_col, lighting_col, ocr_col):
                        if field and field in row.keys() and row[field]:
                            parts.append(str(row[field]))
                    if keywords_col and keywords_col in row.keys():
                        _extend_json_text(parts, row[keywords_col])
                    if objects_col and objects_col in row.keys():
                        _extend_json_text(parts, row[objects_col])
                    if faces_col and faces_col in row.keys():
                        _extend_json_text(parts, row[faces_col])
            elif _table_exists("search_index"):
                row = self.conn.execute(
                    """
                    SELECT description_text, subjects_text, keywords_text, faces_text, exif_text
                    FROM search_index
                    WHERE rowid = ?
                    """,
                    [image_id],
                ).fetchone()
                if row:
                    for field in ("description_text", "subjects_text", "keywords_text", "faces_text", "exif_text"):
                        if row[field]:
                            parts.append(str(row[field]))

            if _table_exists("analysis_blip2"):
                blip_cols = _table_columns("analysis_blip2")
                desc_col = _first_available(blip_cols, "description")
                scene_col = _first_available(blip_cols, "scene_type")
                subject_col = _first_available(blip_cols, "main_subject")
                keywords_col = _first_available(blip_cols, "keywords")
                mood_col = _first_available(blip_cols, "mood")
                lighting_col = _first_available(blip_cols, "lighting")
                select_cols = [col for col in (desc_col, scene_col, subject_col, keywords_col, mood_col, lighting_col) if col]
                if not select_cols:
                    select_cols = ["description"]
                row = self.conn.execute(
                    f"SELECT {', '.join(select_cols)} FROM analysis_blip2 WHERE image_id = ?",
                    [image_id],
                ).fetchone()
                if row:
                    for field in (desc_col, scene_col, subject_col, mood_col, lighting_col):
                        if field and field in row.keys() and row[field]:
                            parts.append(str(row[field]))
                    if keywords_col and keywords_col in row.keys():
                        _extend_json_text(parts, row[keywords_col])

            if _table_exists("analysis_cloud_ai"):
                cloud_cols = _table_columns("analysis_cloud_ai")
                desc_col = _first_available(cloud_cols, "description")
                scene_col = _first_available(cloud_cols, "scene_type")
                subject_col = _first_available(cloud_cols, "main_subject")
                keywords_col = _first_available(cloud_cols, "keywords")
                analyzed_at_col = _first_available(cloud_cols, "analyzed_at")
                id_col = _first_available(cloud_cols, "id")
                select_cols = [col for col in (desc_col, scene_col, subject_col, keywords_col) if col]
                if not select_cols:
                    select_cols = ["description"]
                order_by_parts: list[str] = []
                if analyzed_at_col:
                    order_by_parts.append(f"{analyzed_at_col} DESC")
                if id_col:
                    order_by_parts.append(f"{id_col} DESC")
                order_by = f" ORDER BY {', '.join(order_by_parts)}" if order_by_parts else ""
                rows = self.conn.execute(
                    f"SELECT {', '.join(select_cols)} FROM analysis_cloud_ai WHERE image_id = ?{order_by}",
                    [image_id],
                ).fetchall()
                for row in rows:
                    for field in (desc_col, scene_col, subject_col):
                        if field and field in row.keys() and row[field]:
                            parts.append(str(row[field]))
                    if keywords_col and keywords_col in row.keys():
                        _extend_json_text(parts, row[keywords_col])

            if _table_exists("analysis_faces"):
                row = self.conn.execute(
                    "SELECT face_identities FROM analysis_faces WHERE image_id = ?",
                    [image_id],
                ).fetchone()
                if row:
                    _extend_json_text(parts, row["face_identities"])

            if _table_exists("analysis_metadata"):
                row = self.conn.execute(
                    """
                    SELECT camera_make, camera_model, lens_model,
                           location_city, location_state, location_country
                    FROM analysis_metadata
                    WHERE image_id = ?
                    """,
                    [image_id],
                ).fetchone()
                if row:
                    for field in (
                        "camera_make",
                        "camera_model",
                        "lens_model",
                        "location_city",
                        "location_state",
                        "location_country",
                    ):
                        if row[field]:
                            parts.append(str(row[field]))

            blob = " ".join(parts).casefold()
            current_text_cache[image_id] = blob
            return blob

        def _build_search_index_row_blob(image_id: int) -> str:
            cached = search_row_text_cache.get(image_id)
            if cached is not None:
                return cached
            row = self.conn.execute(
                """
                SELECT description_text, subjects_text, keywords_text, faces_text, exif_text
                FROM search_index
                WHERE rowid = ?
                """,
                [image_id],
            ).fetchone()
            if row is None:
                blob = ""
            else:
                blob = " ".join(
                    str(row[field])
                    for field in ("description_text", "subjects_text", "keywords_text", "faces_text", "exif_text")
                    if row[field]
                ).casefold()
            search_row_text_cache[image_id] = blob
            return blob

        def _has_analysis_rows(image_id: int) -> bool:
            cached = has_analysis_rows_cache.get(image_id)
            if cached is not None:
                return cached

            checks = (
                "analysis_caption",
                "analysis_blip2",
                "analysis_cloud_ai",
                "analysis_faces",
            )
            found = False
            for table in checks:
                if not _table_exists(table):
                    continue
                row = self.conn.execute(
                    f"SELECT 1 FROM {table} WHERE image_id = ? LIMIT 1",
                    [image_id],
                ).fetchone()
                if row is not None:
                    found = True
                    break

            has_analysis_rows_cache[image_id] = found
            return found

        def _has_any_text_analysis_data() -> bool:
            nonlocal has_any_analysis_data
            if has_any_analysis_data is not None:
                return has_any_analysis_data

            checks = (
                "analysis_caption",
                "analysis_blip2",
                "analysis_cloud_ai",
                "analysis_faces",
            )
            has_any = False
            for table in checks:
                if not _table_exists(table):
                    continue
                row = self.conn.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone()
                if row is not None:
                    has_any = True
                    break
            has_any_analysis_data = has_any
            return has_any

        def _passes_consistency_guard(image_id: int) -> bool:
            if not enforce_consistency_guard:
                return True
            # Test fixtures and minimal DBs may only have search_index rows and no
            # analysis tables populated. In that case, keep legacy behavior.
            if not _has_any_text_analysis_data():
                return True
            if _has_analysis_rows(image_id):
                blob = _build_authoritative_text_blob(image_id)
            else:
                # For rows without current analysis tables, only trust visible
                # FTS row text. Legacy ghost-token rows in contentless FTS often
                # have empty visible columns but still match via stale postings.
                blob = _build_search_index_row_blob(image_id)
                if not blob:
                    return False
            overlap = sum(1 for token in required_tokens if token in blob)
            return overlap >= min_token_overlap

        channel_limit_base = max(limit, 40)
        fused_scores: dict[int, float] = {}
        path_by_id: dict[int, str] = {}
        snippet_by_id: dict[int, str] = {}
        match_type_by_id: dict[int, str] = {}
        channel_priority = {"text_strict": 0, "text_phrase": 1, "text_subphrase": 2, "text_soft": 3}

        for match_type, match_query, weight, multiplier in channels:
            channel_limit = channel_limit_base * multiplier
            rows = self.conn.execute(
                """SELECT si.rowid AS image_id,
                          i.file_path,
                          bm25(search_index, 0.0, 3.8, 3.2, 2.2, 2.6, 0.6) AS bm25_score,
                          snippet(search_index, 1, '<b>', '</b>', '...', 32) AS snippet
                   FROM search_index si
                   JOIN images i ON i.id = si.rowid
                   WHERE search_index MATCH ?
                   ORDER BY bm25_score
                   LIMIT ?""",
                [match_query, channel_limit],
            ).fetchall()
            for rank, row in enumerate(rows):
                image_id = int(row["image_id"])
                if candidate_ids is not None and image_id not in candidate_ids:
                    continue
                if enforce_consistency_guard and not _passes_consistency_guard(image_id):
                    continue
                fused_scores[image_id] = fused_scores.get(image_id, 0.0) + (weight * _rrf_score(rank))
                path_by_id.setdefault(image_id, row["file_path"])
                snippet_by_id.setdefault(image_id, row["snippet"])
                existing = match_type_by_id.get(image_id)
                if existing is None or channel_priority[match_type] < channel_priority[existing]:
                    match_type_by_id[image_id] = match_type

        ranked_ids = sorted(fused_scores.items(), key=lambda item: -item[1])[:limit]
        return [
            {
                "image_id": image_id,
                "file_path": path_by_id.get(image_id, "?"),
                "score": score,
                "match_type": match_type_by_id.get(image_id, "text"),
                "snippet": snippet_by_id.get(image_id, ""),
            }
            for image_id, score in ranked_ids
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

        Results are cached with a short TTL to avoid repeating this expensive
        query on back-to-back searches.
        """
        import time as _time

        now = _time.monotonic()
        if self._rich_desc_ids is not None and (now - self._rich_desc_ts) < self._RICH_DESC_TTL:
            return self._rich_desc_ids

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
                FROM analysis_caption la
                LEFT JOIN analysis_cloud_ai ca ON ca.image_id = la.image_id
                UNION ALL
                SELECT ca.image_id,
                       NULL AS local_desc,
                       ca.description AS cloud_desc
                FROM analysis_cloud_ai ca
                WHERE ca.image_id NOT IN (SELECT image_id FROM analysis_caption)
            )
            GROUP BY image_id
            HAVING total_len >= ?
            """,
            [_DESC_QUALITY_THRESHOLD],
        ).fetchall()
        result = frozenset(r[0] for r in rows)
        self._rich_desc_ids = result
        self._rich_desc_ts = now
        return result

    def _semantic_search(
        self, query: str, limit: int, semantic_profile: str | None = None,
        *, progress_cb: ProgressCallback = None, candidate_ids: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        """CLIP embedding semantic search — two-tier strategy (vectorized).

        **Tier 1 — images with ``image_clip``** (primary visual signal):
            Ranked by ``image_clip`` cosine (query: ``"a photo of {query}"``).
            For images that *also* have a rich ``description_clip``, a small
            weighted boost is added as a tiebreaker:

                fused = 1.0 * rrf(image_rank) + _DESC_WEIGHT * rrf(desc_rank)

        **Tier 2 — images with only ``description_clip``, no ``image_clip``**:
            These have no visual signal.  Included only when their
            ``description_clip`` cosine z-score >= ``_DESC_ONLY_ZSCORE_MIN``.
            Appended *after* all Tier-1 results.

        Performance: all cosine similarities are computed via a single BLAS
        matrix-vector multiply (``matrix @ query_vec``) instead of a Python
        loop.  At 500K images this reduces query time from 10-60s to <10ms.
        """
        from imganalyzer.embeddings.clip_embedder import CLIPEmbedder, vector_from_bytes

        if progress_cb:
            progress_cb("loading_model", "Loading AI model…", 0.15)
        embedder = CLIPEmbedder()
        image_weight, desc_weight = _semantic_profile_weights(semantic_profile)

        # Fetch a larger pool so RRF has more candidates to re-rank.
        pool = max(limit * _POOL_FACTOR, 100)

        # Encode two query variants:
        # - visual query: prefixed for better text->image retrieval
        # - text query: verbatim for text->text description matching
        if progress_cb:
            progress_cb("encoding", "Encoding query…", 0.25)
        visual_query = query if query.lower().startswith("a photo of") else f"a photo of {query}"
        visual_query_vec = vector_from_bytes(embedder.embed_text(visual_query))
        text_query_vec   = vector_from_bytes(embedder.embed_text(query))

        # Load cached embedding matrices (rebuilt automatically if stale)
        if progress_cb:
            progress_cb("loading_embeddings", "Loading search index…", 0.35)
        img_matrix, img_ids = self._image_clip_cache.get(self.conn, "image_clip")
        desc_matrix, desc_ids = self._desc_clip_cache.get(self.conn, "description_clip")

        has_image = img_matrix.size > 0
        has_desc = desc_matrix.size > 0

        if not has_image and not has_desc:
            return []
        if candidate_ids is not None and not candidate_ids:
            return []

        # Determine which images have rich-enough descriptions to trust.
        rich_desc_ids = self._get_rich_desc_image_ids() if has_desc else frozenset()

        # ── Vectorized scoring ────────────────────────────────────────────
        # CLIP embeddings are L2-normalised, so dot product == cosine similarity.
        # One BLAS call computes all N scores: scores = matrix @ query_vec
        if progress_cb:
            progress_cb("scoring", "Computing relevance…", 0.55)

        image_scored: list[tuple[int, float]] = []
        image_id_set: set[int] = set()
        if has_image:
            # (N,768) @ (768,) -> (N,)
            img_scores = img_matrix @ visual_query_vec
            for i, iid in enumerate(img_ids):
                if candidate_ids is not None and iid not in candidate_ids:
                    continue
                image_scored.append((iid, float(img_scores[i])))
                image_id_set.add(iid)

        # Only score rich descriptions
        desc_cosines: dict[int, float] = {}
        if has_desc:
            desc_scores = desc_matrix @ text_query_vec
            for i, iid in enumerate(desc_ids):
                if candidate_ids is not None and iid not in candidate_ids:
                    continue
                if iid in rich_desc_ids:
                    desc_cosines[iid] = float(desc_scores[i])

        # ── Tier 1: images that have image_clip ───────────────────────────

        image_scored.sort(key=lambda x: -x[1])
        image_pool = image_scored[:pool]
        image_ranks = {iid: rank for rank, (iid, _) in enumerate(image_pool)}

        # Build rank map for description_clip (only for Tier-1 boost)
        tier1_desc = [(iid, s) for iid, s in desc_cosines.items() if iid in image_id_set]
        tier1_desc.sort(key=lambda x: -x[1])
        tier1_desc_pool = tier1_desc[:pool]
        desc_ranks_t1 = {iid: rank for rank, (iid, _) in enumerate(tier1_desc_pool)}

        tier1_fused = _fuse_semantic_tier1_scores(
            image_ranks=image_ranks,
            desc_ranks=desc_ranks_t1,
            image_weight=image_weight,
            desc_weight=desc_weight,
        )
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
        self, query: str, limit: int, semantic_weight: float,
        semantic_profile: str | None = None,
        *, progress_cb: ProgressCallback = None, candidate_ids: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Combine FTS5 and CLIP scores with weighted RRF fusion.

        Both sources are converted to RRF scores before weighting so that
        FTS5 BM25 magnitudes and CLIP cosine magnitudes (which live in
        completely different numerical ranges) are made commensurable.
        ``semantic_weight`` controls the relative contribution:
        0.0 = text only, 1.0 = semantic only, 0.5 = equal weight.

        Candidate-set strategy:
        - When ``semantic_weight >= 0.5`` (CLIP is primary), the final
          candidate union is restricted to images that appear in the CLIP
          pool.  FTS can only *boost* a CLIP candidate; it cannot introduce
          a new candidate.  This prevents FTS false-positives (e.g. images
          whose description mentions "bird of paradise" or "bird's eye view")
          from displacing genuinely bird-like images ranked highly by CLIP.
        - When ``semantic_weight < 0.5`` (text is primary), FTS candidates
          are included unrestricted (original union behaviour).
        """
        pool = limit * _POOL_FACTOR
        if progress_cb:
            progress_cb("text_search", "Searching text index…", 0.05)
        text_results     = self._fts_search(query, pool, candidate_ids=candidate_ids)
        try:
            semantic_results = self._semantic_search(
                query,
                pool,
                semantic_profile,
                progress_cb=progress_cb,
                candidate_ids=candidate_ids,
            )
        except RuntimeError as exc:
            if "CUDA" in str(exc):
                import sys
                print(
                    f"[SearchEngine] Semantic search failed due to CUDA error, "
                    f"falling back to text-only: {exc}",
                    file=sys.stderr,
                )
                return text_results[:limit]
            raise

        if progress_cb:
            progress_cb("ranking", "Ranking results…", 0.70)

        # Convert each result list to RRF scores (already sorted best-first)
        text_rrf: dict[int, float] = {
            r["image_id"]: _rrf_score(rank)
            for rank, r in enumerate(text_results)
        }
        sem_rrf: dict[int, float] = {
            r["image_id"]: _rrf_score(rank)
            for rank, r in enumerate(semantic_results)
        }

        # Candidate set policy:
        # - text-primary (semantic_weight < 0.5): full union.
        # - semantic-primary (semantic_weight >= 0.5): keep CLIP pool, but
        #   always include a lexical guarantee bucket so strong description
        #   matches are not dropped.
        if semantic_weight >= 0.5:
            lexical_floor = max(limit, min(pool // 3, 200))
            lexical_guarantee_ids = {
                int(r["image_id"]) for r in text_results[:lexical_floor]
            }
            all_ids = set(sem_rrf.keys()) | lexical_guarantee_ids
        else:
            all_ids = set(text_rrf.keys()) | set(sem_rrf.keys())
        if candidate_ids is not None:
            all_ids &= candidate_ids

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
