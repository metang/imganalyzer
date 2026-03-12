"""Ollama AI pass — runs qwen3.5 via Ollama as the ``blip2`` module key.

Drop-in replacement for the BLIP-2 captioning pass.  Also writes
aesthetic fields to ``analysis_aesthetic`` (same piggy-back pattern
that cloud_ai used).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from imganalyzer.db.repository import Repository


def run_ollama_ai(
    image_data: dict[str, Any],
    repo: Repository,
    image_id: int,
    conn: sqlite3.Connection,
    *,
    path: Path | None = None,
) -> dict[str, Any]:
    """Run qwen3.5 via Ollama on one image and write to ``analysis_blip2``.

    *path* is the original file path (needed for Ollama image encoding).
    Returns the result dict.
    """
    from imganalyzer.analysis.ai.ollama import OllamaAI

    if path is None:
        raise ValueError("run_ollama_ai requires 'path' to encode the image for Ollama")

    result = OllamaAI().analyze(path, image_data)

    # Split aesthetic fields before writing to blip2 table
    aesthetic_score = result.pop("aesthetic_score", None)
    aesthetic_label = result.pop("aesthetic_label", None)
    aesthetic_reason = result.pop("aesthetic_reason", None)

    from imganalyzer.pipeline.modules import _transaction
    with _transaction(conn):
        repo.upsert_blip2(image_id, result)
        if aesthetic_score is not None:
            repo.upsert_aesthetic(image_id, {
                "aesthetic_score": aesthetic_score,
                "aesthetic_label": aesthetic_label or "",
                "aesthetic_reason": aesthetic_reason or "",
                "provider": "ollama-qwen3.5",
            })
        repo.update_search_index(image_id)

    # Re-attach aesthetic for return value
    if aesthetic_score is not None:
        result["aesthetic_score"] = aesthetic_score
        result["aesthetic_label"] = aesthetic_label
        result["aesthetic_reason"] = aesthetic_reason

    return result


def run_ollama_ai_for_cloud(
    image_data: dict[str, Any],
    repo: Repository,
    image_id: int,
    conn: sqlite3.Connection,
    *,
    path: Path | None = None,
) -> dict[str, Any]:
    """Run qwen3.5 via Ollama and write to ``analysis_cloud_ai``.

    Used as drop-in replacement for the cloud_ai module.
    """
    from imganalyzer.analysis.ai.ollama import OllamaAI

    if path is None:
        raise ValueError("run_ollama_ai_for_cloud requires 'path'")

    result = OllamaAI().analyze(path, image_data)

    aesthetic_score = result.pop("aesthetic_score", None)
    aesthetic_label = result.pop("aesthetic_label", None)
    aesthetic_reason = result.pop("aesthetic_reason", None)

    from imganalyzer.pipeline.modules import _transaction
    with _transaction(conn):
        repo.upsert_cloud_ai(image_id, "ollama-qwen3.5", result)
        if aesthetic_score is not None:
            repo.upsert_aesthetic(image_id, {
                "aesthetic_score": aesthetic_score,
                "aesthetic_label": aesthetic_label or "",
                "aesthetic_reason": aesthetic_reason or "",
                "provider": "ollama-qwen3.5",
            })
        repo.update_search_index(image_id)

    if aesthetic_score is not None:
        result["aesthetic_score"] = aesthetic_score
        result["aesthetic_label"] = aesthetic_label
        result["aesthetic_reason"] = aesthetic_reason

    return result
