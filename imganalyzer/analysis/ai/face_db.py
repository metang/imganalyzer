"""Face embedding database — register, persist and match face identities."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

_DEFAULT_DB_PATH = Path.home() / ".cache" / "imganalyzer" / "faces.json"
_DEFAULT_THRESHOLD = float(os.getenv("IMGANALYZER_FACE_DB_THRESHOLD", "0.40"))


def _db_path() -> Path:
    raw = os.getenv("IMGANALYZER_FACE_DB", "")
    if raw:
        return Path(raw).expanduser()
    return _DEFAULT_DB_PATH


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D float arrays."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class FaceDatabase:
    """Persistent face identity store backed by a JSON file.

    Storage format::

        {
            "Alice": {
                "embeddings": [[...512 floats...], ...],
                "registered_at": "2026-01-01T00:00:00"
            },
            ...
        }

    Embeddings are 512-d float32 vectors produced by InsightFace buffalo_l.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path: Path = path or _db_path()
        self._data: dict[str, Any] = {}
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self.path.exists():
            try:
                with open(self.path, encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    # ── Public API ────────────────────────────────────────────────────────────

    def register(self, name: str, embedding: np.ndarray) -> None:
        """Add a face embedding for *name*.  Multiple embeddings per name are
        supported (improves matching accuracy across lighting/angle variations)."""
        import datetime

        emb_list = embedding.astype(np.float32).flatten().tolist()
        if name not in self._data:
            self._data[name] = {
                "embeddings": [],
                "registered_at": datetime.datetime.now().isoformat(timespec="seconds"),
            }
        self._data[name]["embeddings"].append(emb_list)
        self._save()

    def match(
        self,
        embedding: np.ndarray,
        threshold: float | None = None,
    ) -> tuple[str, float]:
        """Return (name, similarity) for the best matching registered face.

        Returns ("Unknown", 0.0) if the database is empty or no match exceeds
        *threshold*.
        """
        thr = threshold if threshold is not None else _DEFAULT_THRESHOLD
        query = embedding.astype(np.float32).flatten()
        best_name = "Unknown"
        best_sim = 0.0

        for name, entry in self._data.items():
            for emb_list in entry.get("embeddings", []):
                ref = np.array(emb_list, dtype=np.float32)
                sim = _cosine_similarity(query, ref)
                if sim > best_sim:
                    best_sim = sim
                    best_name = name

        if best_sim < thr:
            return "Unknown", best_sim
        return best_name, best_sim

    def list_names(self) -> list[str]:
        """Return all registered identity names."""
        return list(self._data.keys())

    def remove(self, name: str) -> bool:
        """Remove all embeddings for *name*.  Returns True if found."""
        if name in self._data:
            del self._data[name]
            self._save()
            return True
        return False

    def embedding_count(self, name: str) -> int:
        """Number of stored embeddings for *name*."""
        return len(self._data.get(name, {}).get("embeddings", []))

    def __len__(self) -> int:
        return len(self._data)
