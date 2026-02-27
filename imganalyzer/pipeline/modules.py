"""Module runners — per-module analysis executors with override guard and atomic DB writes.

Each module runner:
1. Checks if already analyzed (cache check) — skips unless force=True
2. Runs the full analysis in memory (no partial writes)
3. Applies override mask (removes protected fields)
4. Writes atomically in a single transaction
"""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any

from rich.console import Console

from imganalyzer.db.repository import Repository

console = Console()


def compute_file_hash(path: Path, chunk_size: int = 65536) -> str:
    """SHA-256 hash of the file contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _read_image(path: Path) -> dict[str, Any]:
    """Read an image file and return image_data dict (same as Analyzer does)."""
    from imganalyzer.analyzer import RAW_EXTENSIONS

    suffix = path.suffix.lower()
    is_raw = suffix in RAW_EXTENSIONS

    if is_raw:
        from imganalyzer.readers.raw import RawReader
        reader = RawReader(path)
    else:
        from imganalyzer.readers.standard import StandardReader
        reader = StandardReader(path)

    return reader.read()


class ModuleRunner:
    """Dispatches analysis to the appropriate module and writes results atomically."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        repo: Repository,
        force: bool = False,
        cloud_provider: str = "openai",
        detection_prompt: str | None = None,
        detection_threshold: float | None = None,
        face_match_threshold: float | None = None,
        verbose: bool = False,
    ) -> None:
        self.conn = conn
        self.repo = repo
        self.force = force
        self.cloud_provider = cloud_provider
        self.detection_prompt = detection_prompt
        self.detection_threshold = detection_threshold
        self.face_match_threshold = face_match_threshold
        self.verbose = verbose

    def should_run(self, image_id: int, module: str) -> bool:
        """Return False if the module is already analyzed and force is off."""
        if self.force:
            return True
        return not self.repo.is_analyzed(image_id, module)

    def run(self, image_id: int, module: str) -> dict[str, Any]:
        """Execute the analysis module and write results atomically.

        Returns the result dict, or an empty dict if skipped/cached.
        Raises on analysis failure (caller should handle).
        """
        image = self.repo.get_image(image_id)
        if image is None:
            raise ValueError(f"Image id={image_id} not found in database")

        path = Path(image["file_path"])

        # For the embedding module, skip the file-existence check when the image
        # already has a description in the DB — the text embedding can be computed
        # entirely from stored data without touching the original file.
        if module == "embedding":
            return self._run_embedding(image_id, path)

        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        # Dispatch to the appropriate module
        if module == "metadata":
            return self._run_metadata(image_id, path)
        elif module == "technical":
            return self._run_technical(image_id, path)
        elif module == "local_ai":
            return self._run_local_ai(image_id, path)
        elif module == "cloud_ai":
            return self._run_cloud_ai(image_id, path)
        elif module == "aesthetic":
            return self._run_aesthetic(image_id, path)
        else:
            raise ValueError(f"Unknown module: {module}")

    # ── Individual module runners ──────────────────────────────────────────

    def _run_metadata(self, image_id: int, path: Path) -> dict[str, Any]:
        image_data = _read_image(path)

        # Update image dimensions if not set
        self.repo.update_image(
            image_id,
            width=image_data.get("width"),
            height=image_data.get("height"),
            format=image_data.get("format"),
        )

        from imganalyzer.analysis.metadata import MetadataExtractor
        result = MetadataExtractor(path, image_data).extract()

        # Atomic write
        with _transaction(self.conn):
            self.repo.upsert_metadata(image_id, result)
            self.repo.update_search_index(image_id)

        if self.verbose:
            console.print(f"  [dim]Metadata extracted for image {image_id}[/dim]")
        return result

    def _run_technical(self, image_id: int, path: Path) -> dict[str, Any]:
        image_data = _read_image(path)

        from imganalyzer.analysis.technical import TechnicalAnalyzer
        result = TechnicalAnalyzer(image_data).analyze()

        with _transaction(self.conn):
            self.repo.upsert_technical(image_id, result)

        if self.verbose:
            console.print(f"  [dim]Technical analysis done for image {image_id}[/dim]")
        return result

    def _run_local_ai(self, image_id: int, path: Path) -> dict[str, Any]:
        image_data = _read_image(path)

        from imganalyzer.analysis.ai.local_full import LocalAIFull
        result = LocalAIFull().analyze(
            image_data,
            detection_prompt=self.detection_prompt,
            detection_threshold=self.detection_threshold,
            face_match_threshold=self.face_match_threshold,
        )

        # Track has_people for cloud_ai gating
        has_people = bool(result.get("face_count", 0) > 0)
        result["has_people"] = has_people

        with _transaction(self.conn):
            self.repo.upsert_local_ai(image_id, result)
            self.repo.update_search_index(image_id)

        if self.verbose:
            console.print(f"  [dim]Local AI analysis done for image {image_id}[/dim]")
        return result

    def _run_cloud_ai(self, image_id: int, path: Path) -> dict[str, Any]:
        # People guard: skip cloud AI for images with people
        local_data = self.repo.get_analysis(image_id, "local_ai")
        if local_data and local_data.get("has_people"):
            if self.verbose:
                console.print(
                    f"  [dim]Skipping cloud AI for image {image_id} (has people)[/dim]"
                )
            return {}

        image_data = _read_image(path)

        from imganalyzer.analysis.ai.cloud import CloudAI
        result = CloudAI(backend=self.cloud_provider).analyze(path, image_data)

        with _transaction(self.conn):
            self.repo.upsert_cloud_ai(image_id, self.cloud_provider, result)
            self.repo.update_search_index(image_id)

        if self.verbose:
            console.print(
                f"  [dim]Cloud AI ({self.cloud_provider}) done for image {image_id}[/dim]"
            )
        return result

    def _run_aesthetic(self, image_id: int, path: Path) -> dict[str, Any]:
        # Aesthetic analysis uses a cloud model — skip for images with people
        local_data = self.repo.get_analysis(image_id, "local_ai")
        if local_data and local_data.get("has_people"):
            if self.verbose:
                console.print(
                    f"  [dim]Skipping aesthetic for image {image_id} (has people)[/dim]"
                )
            return {}

        image_data = _read_image(path)

        # Use the cloud AI backend for aesthetic scoring
        # The cloud model is asked specifically for aesthetic analysis
        from imganalyzer.analysis.ai.cloud import CloudAI
        cloud = CloudAI(backend=self.cloud_provider)

        # We call the normal cloud analysis and extract aesthetic fields
        # In future, this could be a dedicated aesthetic prompt
        result = cloud.analyze(path, image_data)

        aesthetic_data = {
            "aesthetic_score": result.get("aesthetic_score"),
            "aesthetic_label": result.get("aesthetic_label"),
            "aesthetic_reason": result.get("aesthetic_reason", ""),
            "provider": self.cloud_provider,
        }

        with _transaction(self.conn):
            self.repo.upsert_aesthetic(image_id, aesthetic_data)

        if self.verbose:
            console.print(
                f"  [dim]Aesthetic analysis done for image {image_id}[/dim]"
            )
        return aesthetic_data

    def _run_embedding(self, image_id: int, path: Path) -> dict[str, Any]:
        from imganalyzer.embeddings.clip_embedder import CLIPEmbedder

        embedder = CLIPEmbedder()

        # Text embedding: combine description, scene, and subject from local AI
        # and all cloud providers so the vector reflects rich semantic content.
        text_parts: list[str] = []
        local_data = self.repo.get_analysis(image_id, "local_ai")
        if local_data:
            for field in ("description", "scene_type", "main_subject"):
                val = local_data.get(field)
                if val:
                    text_parts.append(val)

        cloud_data = self.repo.get_analysis(image_id, "cloud_ai")
        if cloud_data:
            for prov in cloud_data.get("providers", []):
                for field in ("description", "scene_type", "main_subject"):
                    val = prov.get(field)
                    if val and val not in text_parts:
                        text_parts.append(val)

        desc_text = " ".join(text_parts)

        # Always compute image_clip from the visual file when available.
        # image_clip (text→image cosine) is the most reliable signal for
        # semantic search because it directly encodes visual content.
        # description_clip supplements it for images with rich AI descriptions,
        # but should never be the *only* embedding — short local-AI captions
        # produce centroid-clustered embeddings that poison search ranking.
        has_image_clip = False
        if path.exists():
            image_vector = embedder.embed_image(path)
            with _transaction(self.conn):
                self.repo.upsert_embedding(
                    image_id, "image_clip",
                    image_vector, embedder.model_version,
                )
            has_image_clip = True

        has_desc_clip = False
        if desc_text:
            text_vector = embedder.embed_text(desc_text)
            with _transaction(self.conn):
                self.repo.upsert_embedding(
                    image_id, "description_clip",
                    text_vector, embedder.model_version,
                )
            has_desc_clip = True

        if not has_image_clip and not has_desc_clip:
            raise FileNotFoundError(
                f"Image file not found and no description available for image {image_id}: {path}"
            )

        if self.verbose:
            console.print(f"  [dim]CLIP embeddings computed for image {image_id}[/dim]")
        return {"image_clip": has_image_clip, "description_clip": has_desc_clip}


class _transaction:
    """Context manager for SQLite transactions (BEGIN IMMEDIATE ... COMMIT)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def __enter__(self) -> None:
        self.conn.execute("BEGIN IMMEDIATE")

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is None:
            self.conn.commit()
        else:
            self.conn.rollback()


# ── XMP generation from DB data ───────────────────────────────────────────────

def write_xmp_from_db(repo: Repository, image_id: int) -> Path | None:
    """Build an AnalysisResult from DB data and write the XMP sidecar.

    Returns the XMP path written, or None if the image was not found.
    """
    import json as _json
    from imganalyzer.analyzer import AnalysisResult

    image = repo.get_image(image_id)
    if image is None:
        return None

    path = Path(image["file_path"])
    full = repo.get_full_result(image_id)

    # Reconstruct AnalysisResult from DB data
    result = AnalysisResult(
        source_path=path,
        format=image.get("format") or "",
        width=image.get("width") or 0,
        height=image.get("height") or 0,
    )

    # Metadata — strip internal keys
    meta = full.get("metadata", {})
    for key in ("image_id", "analyzed_at"):
        meta.pop(key, None)
    result.metadata = meta

    # Technical — strip internal keys, decode JSON arrays
    tech = full.get("technical", {})
    for key in ("image_id", "analyzed_at"):
        tech.pop(key, None)
    if isinstance(tech.get("dominant_colors"), str):
        try:
            tech["dominant_colors"] = _json.loads(tech["dominant_colors"])
        except (ValueError, TypeError):
            pass
    result.technical = tech

    # AI analysis — merge local_ai and cloud_ai into a single dict
    # (XMPWriter expects result.ai_analysis to be a flat dict)
    ai: dict[str, Any] = {}

    local = full.get("local_ai", {})
    for key in ("image_id", "analyzed_at", "has_people"):
        local.pop(key, None)
    # Decode JSON list fields
    for field in ("keywords", "detected_objects", "face_identities"):
        if isinstance(local.get(field), str):
            try:
                local[field] = _json.loads(local[field])
            except (ValueError, TypeError):
                pass
    if isinstance(local.get("face_details"), str):
        try:
            local["face_details"] = _json.loads(local["face_details"])
        except (ValueError, TypeError):
            pass
    ai.update(local)

    # Cloud AI — layer on top (cloud descriptions override local if present)
    cloud = full.get("cloud_ai", {})
    if cloud and "providers" in cloud:
        for prov_data in cloud["providers"]:
            for key in ("id", "image_id", "analyzed_at", "provider", "raw_response"):
                prov_data.pop(key, None)
            for field in ("keywords", "detected_objects", "dominant_colors_ai"):
                if isinstance(prov_data.get(field), str):
                    try:
                        prov_data[field] = _json.loads(prov_data[field])
                    except (ValueError, TypeError):
                        pass
            # Merge: cloud values override local values for same keys
            for k, v in prov_data.items():
                if v is not None and v != "" and v != []:
                    ai[k] = v

    result.ai_analysis = ai

    xmp_path = path.with_suffix(".xmp")
    result.write_xmp(xmp_path)
    return xmp_path
