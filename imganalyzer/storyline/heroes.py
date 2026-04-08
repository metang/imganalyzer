"""Hero image selection for story moments and chapters.

Scores images by a weighted combination of aesthetic quality, sharpness,
face presence, and visual diversity (CLIP embedding distance) to pick the
best representative image per moment and per chapter.
"""
from __future__ import annotations

import sqlite3
import struct
from dataclasses import dataclass
from typing import Sequence

from imganalyzer.storyline.clustering import Moment, Chapter


@dataclass
class ImageScore:
    image_id: int
    aesthetic: float
    sharpness: float
    has_face: bool
    embedding: list[float] | None


# ── Scoring weights ──────────────────────────────────────────────────────────

WEIGHT_AESTHETIC = 0.35
WEIGHT_SHARPNESS = 0.15
WEIGHT_FACE = 0.25
WEIGHT_DIVERSITY = 0.25


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """1 - cosine_similarity.  Higher = more different."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - dot / (norm_a * norm_b)


def _decode_embedding(blob: bytes | None) -> list[float] | None:
    """Decode a CLIP embedding stored as packed float32."""
    if blob is None:
        return None
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


def _load_image_scores(
    conn: sqlite3.Connection,
    image_ids: list[int],
) -> dict[int, ImageScore]:
    """Bulk-load scoring data for a set of images."""
    if not image_ids:
        return {}

    scores: dict[int, ImageScore] = {}

    # Batch in chunks to avoid SQL variable limits
    chunk_size = 500
    for start in range(0, len(image_ids), chunk_size):
        chunk = image_ids[start : start + chunk_size]
        placeholders = ",".join("?" for _ in chunk)

        rows = conn.execute(
            f"SELECT sf.image_id, "
            f"       COALESCE(sf.perception_iaa, 0.0) AS aesthetic, "
            f"       COALESCE(sf.sharpness_score, 0.0) AS sharpness, "
            f"       COALESCE(sf.face_count, 0) AS face_count, "
            f"       e.vector AS image_clip "
            f"FROM search_features sf "
            f"LEFT JOIN embeddings e ON e.image_id = sf.image_id "
            f"  AND e.embedding_type = 'image_clip' "
            f"WHERE sf.image_id IN ({placeholders})",
            chunk,
        ).fetchall()

        for r in rows:
            scores[r["image_id"]] = ImageScore(
                image_id=r["image_id"],
                aesthetic=r["aesthetic"] or 0.0,
                sharpness=r["sharpness"] or 0.0,
                has_face=(r["face_count"] or 0) > 0,
                embedding=_decode_embedding(r["image_clip"]),
            )

    # Fill in missing entries
    for iid in image_ids:
        if iid not in scores:
            scores[iid] = ImageScore(
                image_id=iid,
                aesthetic=0.0,
                sharpness=0.0,
                has_face=False,
                embedding=None,
            )
    return scores


def _compute_base_score(s: ImageScore) -> float:
    """Weighted base score (without diversity)."""
    return (
        s.aesthetic * WEIGHT_AESTHETIC
        + s.sharpness * WEIGHT_SHARPNESS
        + (1.0 if s.has_face else 0.0) * WEIGHT_FACE
    )


def select_moment_hero(
    conn: sqlite3.Connection,
    moment_image_ids: list[int],
) -> int | None:
    """Select the best hero image for a moment.

    Returns the image_id of the hero, or None if no images.
    """
    if not moment_image_ids:
        return None
    if len(moment_image_ids) == 1:
        return moment_image_ids[0]

    scores = _load_image_scores(conn, moment_image_ids)

    # Simple case: pick by base score (diversity not needed within a moment)
    best_id = max(moment_image_ids, key=lambda iid: _compute_base_score(scores[iid]))
    return best_id


def select_chapter_heroes(
    conn: sqlite3.Connection,
    moments: list[tuple[str, list[int]]],
) -> dict[str, int]:
    """Select hero images for all moments in a chapter.

    Also ensures diversity across the chapter — heroes should not all
    look similar.

    Parameters
    ----------
    moments:
        List of (moment_id, [image_ids]) pairs.

    Returns
    -------
    dict mapping moment_id → hero_image_id
    """
    if not moments:
        return {}

    all_image_ids = []
    for _, img_ids in moments:
        all_image_ids.extend(img_ids)
    scores = _load_image_scores(conn, all_image_ids)

    heroes: dict[str, int] = {}
    selected_embeddings: list[list[float]] = []

    for moment_id, img_ids in moments:
        if not img_ids:
            continue
        if len(img_ids) == 1:
            heroes[moment_id] = img_ids[0]
            emb = scores[img_ids[0]].embedding
            if emb:
                selected_embeddings.append(emb)
            continue

        # Score each candidate with diversity bonus
        best_id = img_ids[0]
        best_score = -1.0

        for iid in img_ids:
            s = scores[iid]
            base = _compute_base_score(s)

            # Diversity bonus: distance from already-selected heroes
            diversity = 0.0
            if s.embedding and selected_embeddings:
                distances = [
                    _cosine_distance(s.embedding, prev)
                    for prev in selected_embeddings
                ]
                diversity = sum(distances) / len(distances)
            elif not selected_embeddings:
                diversity = 0.5  # neutral when no previous heroes

            total = base + diversity * WEIGHT_DIVERSITY
            if total > best_score:
                best_score = total
                best_id = iid

        heroes[moment_id] = best_id
        emb = scores[best_id].embedding
        if emb:
            selected_embeddings.append(emb)

    return heroes


def select_chapter_cover(
    conn: sqlite3.Connection,
    hero_image_ids: list[int],
) -> int | None:
    """Select the chapter cover from moment heroes (highest aesthetic)."""
    if not hero_image_ids:
        return None
    scores = _load_image_scores(conn, hero_image_ids)
    return max(hero_image_ids, key=lambda iid: scores[iid].aesthetic)
