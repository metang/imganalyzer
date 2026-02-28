"""GroundingDINO object-detection pass — runs independently as the ``objects`` module key."""
from __future__ import annotations

import sqlite3
from typing import Any

from imganalyzer.db.repository import Repository


def run_objects(
    image_data: dict[str, Any],
    repo: Repository,
    image_id: int,
    conn: sqlite3.Connection,
    prompt: str | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Run GroundingDINO object detection on *image_data* and write to ``analysis_objects``.

    Returns the result dict (includes ``has_person``, ``has_text``, ``text_boxes``).
    """
    from imganalyzer.analysis.ai.objects import ObjectDetector
    result = ObjectDetector().analyze(image_data, prompt=prompt, threshold=threshold)

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    from imganalyzer.pipeline.modules import _transaction
    with _transaction(conn):
        repo.upsert_objects(image_id, result)

    return result
