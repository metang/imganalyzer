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


def run_objects_batch(
    image_data_list: list[dict[str, Any]],
    repo: Repository,
    image_ids: list[int],
    conn: sqlite3.Connection,
    prompt: str | None = None,
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Run GroundingDINO on a batch of images and write results atomically.

    Uses ``ObjectDetector.analyze_batch()`` for a single batched forward
    pass across all images, then writes each result in a single transaction.
    """
    from imganalyzer.analysis.ai.objects import ObjectDetector
    results = ObjectDetector().analyze_batch(image_data_list, prompt=prompt, threshold=threshold)

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    from imganalyzer.pipeline.modules import _transaction
    with _transaction(conn):
        for image_id, result in zip(image_ids, results):
            repo.upsert_objects(image_id, result)

    return results
