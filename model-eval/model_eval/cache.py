"""JSON-based result caching to avoid re-running expensive inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"


def _cache_key(model_name: str, image_path: Path) -> str:
    """Create a deterministic cache key from model name and image file."""
    abs_path = image_path.resolve()
    stat = abs_path.stat()
    content_id = f"{abs_path}:{stat.st_size}:{stat.st_mtime_ns}"
    h = hashlib.sha256(content_id.encode()).hexdigest()[:16]
    return f"{model_name}__{abs_path.stem}__{h}"


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def get_cached(model_name: str, image_path: Path) -> dict[str, Any] | None:
    """Return cached result or None if not cached."""
    key = _cache_key(model_name, image_path)
    path = _cache_path(key)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def store_result(model_name: str, image_path: Path, result: dict[str, Any]) -> None:
    """Store a result in the cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(model_name, image_path)
    path = _cache_path(key)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def clear_cache() -> int:
    """Delete all cached results. Returns number of files removed."""
    if not CACHE_DIR.exists():
        return 0
    files = list(CACHE_DIR.glob("*.json"))
    for f in files:
        f.unlink()
    return len(files)
