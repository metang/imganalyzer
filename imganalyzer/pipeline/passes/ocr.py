"""TrOCR pass — runs independently as the ``ocr`` module key.

Prerequisites: ``objects`` module must have run first (provides ``has_text``
and ``text_boxes`` from ``analysis_objects``).  If the prerequisite row is
missing, the pass returns an empty dict without writing anything.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any

from imganalyzer.db.repository import Repository


def run_ocr(
    image_data: dict[str, Any],
    repo: Repository,
    image_id: int,
    conn: sqlite3.Connection,
) -> dict[str, Any]:
    """Run TrOCR on *image_data* using text regions from ``analysis_objects``.

    Returns the result dict (``{"ocr_text": "..."}``), or ``{}`` if no text
    was detected by the objects pass.
    """
    # Read prerequisite output from DB
    objects_row = repo.get_analysis(image_id, "objects")
    if not objects_row:
        return {}

    has_text = bool(objects_row.get("has_text"))
    if not has_text:
        return {}

    # Decode text_boxes JSON if stored as a string
    raw_boxes = objects_row.get("text_boxes")
    if isinstance(raw_boxes, str):
        try:
            text_boxes = json.loads(raw_boxes)
        except (json.JSONDecodeError, TypeError):
            text_boxes = []
    else:
        text_boxes = raw_boxes or []

    from imganalyzer.analysis.ai.ocr import OCRAnalyzer
    result = OCRAnalyzer().analyze(image_data, text_boxes=text_boxes)

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    from imganalyzer.pipeline.modules import _transaction
    with _transaction(conn):
        repo.upsert_ocr(image_id, result)
        repo.update_search_index(image_id)

    return result
