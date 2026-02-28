"""BLIP-2 captioning pass — runs independently as the ``blip2`` module key."""
from __future__ import annotations

import sqlite3
from typing import Any

from imganalyzer.db.repository import Repository


def run_blip2(
    image_data: dict[str, Any],
    repo: Repository,
    image_id: int,
    conn: sqlite3.Connection,
) -> dict[str, Any]:
    """Run BLIP-2 captioning + VQA on *image_data* and write to ``analysis_blip2``.

    Returns the result dict.
    """
    from imganalyzer.analysis.ai.local import LocalAI
    result = LocalAI().analyze(image_data)

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    from imganalyzer.pipeline.modules import _transaction
    with _transaction(conn):
        repo.upsert_blip2(image_id, result)
        repo.update_search_index(image_id)

    return result
