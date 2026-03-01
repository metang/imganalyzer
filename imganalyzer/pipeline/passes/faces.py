"""InsightFace face-analysis pass — runs independently as the ``faces`` module key.

Prerequisites: ``objects`` module must have run first (provides ``has_person``
from ``analysis_objects``).  If the prerequisite row is missing or
``has_person`` is falsy, the pass returns an empty dict without writing.
"""
from __future__ import annotations

import sqlite3
from typing import Any

from imganalyzer.db.repository import Repository


def run_faces(
    image_data: dict[str, Any],
    repo: Repository,
    image_id: int,
    conn: sqlite3.Connection,
    face_match_threshold: float | None = None,
    face_det_score_threshold: float = 0.65,
    face_min_size: int = 40,
) -> dict[str, Any]:
    """Run InsightFace on *image_data* using the person flag from ``analysis_objects``.

    Returns the result dict, or ``{}`` if no person was detected by the objects pass.
    """
    # Read prerequisite output from DB
    objects_row = repo.get_analysis(image_id, "objects")
    if not objects_row:
        return {}

    has_person = bool(objects_row.get("has_person"))
    if not has_person:
        return {}

    from imganalyzer.analysis.ai.faces import FaceAnalyzer
    from imganalyzer.analysis.ai.face_db import FaceDatabase
    face_db = FaceDatabase()
    result = FaceAnalyzer().analyze(
        image_data,
        face_db=face_db if len(face_db) > 0 else None,
        match_threshold=face_match_threshold,
        det_score_threshold=face_det_score_threshold,
        min_face_pixels=face_min_size,
    )

    from imganalyzer.pipeline.modules import _transaction
    with _transaction(conn):
        repo.upsert_faces(image_id, result)
        # Store per-face occurrences (bbox, embedding, age, gender)
        occurrences = result.get("face_occurrences", [])
        if occurrences:
            repo.upsert_face_occurrences(image_id, occurrences)
        repo.update_search_index(image_id)

    return result
