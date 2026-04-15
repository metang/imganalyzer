"""Helpers for loading Hugging Face models without unnecessary network checks."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def _env_truthy(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def has_local_snapshot(model_id: str, cache_dir: str | Path) -> bool:
    """Return True when the model snapshot already exists locally."""
    if _env_truthy("IMGANALYZER_HF_LOCAL_ONLY"):
        return True

    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError:
        return False

    try:
        snapshot_download(
            repo_id=model_id,
            cache_dir=str(cache_dir),
            local_files_only=True,
        )
        return True
    except LocalEntryNotFoundError:
        return False


def load_pretrained(
    loader: Callable[..., T],
    model_id: str,
    *,
    cache_dir: str | Path,
    **kwargs: Any,
) -> T:
    """Prefer the local HF cache to avoid repeated HEAD timeouts on slow networks."""
    load_kwargs = {"cache_dir": str(cache_dir), **kwargs}
    force_local_only = _env_truthy("IMGANALYZER_HF_LOCAL_ONLY")
    prefer_local_only = force_local_only or has_local_snapshot(model_id, cache_dir)

    if prefer_local_only:
        try:
            return loader(model_id, local_files_only=True, **load_kwargs)
        except OSError:
            if force_local_only:
                raise

    return loader(model_id, **load_kwargs)
