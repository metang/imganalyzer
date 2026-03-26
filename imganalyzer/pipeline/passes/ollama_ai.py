"""Ollama AI pass — runs qwen3.5 via Ollama as the ``caption`` module key.

Legacy helper module kept for backward compatibility with external scripts.
The main pipeline now calls OllamaAI directly from modules.py._run_caption().
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

    # Keep this payload caption-focused. Aesthetic metrics come from UniPercept.
    result.pop("aesthetic_score", None)
    result.pop("aesthetic_label", None)
    result.pop("aesthetic_reason", None)

    from imganalyzer.pipeline.modules import _transaction
    with _transaction(conn):
        repo.upsert_blip2(image_id, result)
        repo.update_search_artifacts(image_id)

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

    Legacy pass — the cloud_ai module has been replaced by caption.
    Kept for backward compatibility with external scripts.
    """
    from imganalyzer.analysis.ai.ollama import OllamaAI

    if path is None:
        raise ValueError("run_ollama_ai_for_cloud requires 'path'")

    result = OllamaAI().analyze(path, image_data)

    result.pop("aesthetic_score", None)
    result.pop("aesthetic_label", None)
    result.pop("aesthetic_reason", None)

    from imganalyzer.pipeline.modules import _transaction
    with _transaction(conn):
        repo.upsert_cloud_ai(image_id, "ollama-qwen3.5", result)
        repo.update_search_artifacts(image_id)

    return result
