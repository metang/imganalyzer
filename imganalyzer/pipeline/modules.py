"""Module runners — per-module analysis executors with override guard and atomic DB writes.

Each module runner:
1. Checks if already analyzed (cache check) — skips unless force=True
2. Runs the full analysis in memory (no partial writes)
3. Applies override mask (removes protected fields)
4. Writes atomically in a single transaction
"""
from __future__ import annotations

import hashlib
import re
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


def compute_file_fingerprint(path: Path) -> str:
    """Fast fingerprint using path, size, and mtime.

    At 500K images, SHA-256 hashing reads every byte of every file (12.5 TB+
    at 25 MB average) taking ~7 hours of pure I/O.  A (path, size, mtime)
    fingerprint is effectively free — a single ``stat()`` call per file.

    The fingerprint is deterministic for a given file state and changes when
    the file is modified.  Collisions are astronomically unlikely for the
    deduplication use case (same path + same size + same mtime = same file).
    """
    st = path.stat()
    # Combine path, size, and mtime into a stable string key.
    # Using the resolved path ensures symlinks are handled correctly.
    return f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}"


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


def _read_image_headers(path: Path) -> dict[str, Any]:
    """Read only image metadata/headers — no full pixel decode.

    Used by the metadata module which only needs EXIF headers and sensor
    info, not the actual pixel data.  Saves the entire demosaic/decode
    cost (~2-5s per RAW file, ~300-600 CPU-hours at 500K images).
    """
    from imganalyzer.analyzer import RAW_EXTENSIONS

    suffix = path.suffix.lower()
    is_raw = suffix in RAW_EXTENSIONS

    if is_raw:
        from imganalyzer.readers.raw import RawReader
        return RawReader(path).read_headers()
    else:
        from imganalyzer.readers.standard import StandardReader
        return StandardReader(path).read_headers()


def _is_windows_style_path(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", value)) or value.startswith("\\\\")


def _rewrite_path_with_mappings(original: str, mappings: list[tuple[str, str]]) -> str:
    """Rewrite *original* using the first matching prefix mapping."""
    for source_prefix, target_prefix in mappings:
        source = source_prefix.rstrip("\\/")
        target = target_prefix.rstrip("\\/")
        if not source or not target:
            continue

        if _is_windows_style_path(source):
            matches = original.casefold().startswith(source.casefold())
        else:
            matches = original.startswith(source)
        if not matches:
            continue

        remainder = original[len(source):]
        if remainder and remainder[0] not in ("\\", "/"):
            continue

        parts = [segment for segment in re.split(r"[\\/]+", remainder.lstrip("\\/")) if segment]
        separator = "\\" if _is_windows_style_path(target) else "/"
        mapped = target
        if parts:
            mapped = f"{mapped}{separator}{separator.join(parts)}"
        return mapped
    return original


def _apply_path_mappings(path: Path, mappings: list[tuple[str, str]]) -> Path:
    """Rewrite *path* using the first matching prefix mapping."""
    return Path(_rewrite_path_with_mappings(str(path), mappings))


# Maximum long-edge pixels for AI modules.  All downstream models
# All AI models (CLIP 224px, GroundingDINO 800px, InsightFace 640px)
# internally resize well below this limit.  1024 px matches the Ollama
# caption resolution and is the best trade-off for detail recognition
# (benchmarked across qwen3.5:9b and gpt-4.1 at 480–1920 px).
# Pre-shrinking once avoids redundant resizes and cuts per-image memory
# from ~600 MB (50 MP × 12 bytes) to ~6 MB.
_AI_MAX_LONG_EDGE = 1024


def _pre_resize(image_data: dict[str, Any]) -> dict[str, Any]:
    """Downsize ``rgb_array`` to at most ``_AI_MAX_LONG_EDGE`` on the long edge.

    Modifies *image_data* in-place (replaces ``rgb_array``, updates
    ``width``/``height``) and returns it.  If the image is already small
    enough, no copy is made.
    """
    rgb = image_data.get("rgb_array")
    if rgb is None:
        return image_data

    import numpy as _np

    h, w = rgb.shape[:2]
    long_edge = max(h, w)
    if long_edge <= _AI_MAX_LONG_EDGE:
        return image_data

    # Compute new size preserving aspect ratio
    scale = _AI_MAX_LONG_EDGE / long_edge
    new_w = int(w * scale)
    new_h = int(h * scale)

    from PIL import Image as _PILImage
    pil_img = _PILImage.fromarray(rgb)
    pil_img = pil_img.resize((new_w, new_h), _PILImage.LANCZOS)
    image_data["rgb_array"] = _np.asarray(pil_img)
    image_data["width"] = new_w
    image_data["height"] = new_h
    return image_data


class ModuleRunner:
    """Dispatches analysis to the appropriate module and writes results atomically."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        repo: Repository,
        force: bool = False,
        detection_prompt: str | None = None,
        detection_threshold: float | None = None,
        face_match_threshold: float | None = None,
        verbose: bool = False,
        profiler: Any = None,
        path_mappings: list[tuple[str, str]] | None = None,
    ) -> None:
        self.conn = conn
        self.repo = repo
        self.force = force
        self.detection_prompt = detection_prompt
        self.detection_threshold = detection_threshold
        self.face_match_threshold = face_match_threshold
        self.verbose = verbose
        self.path_mappings = path_mappings or []
        from imganalyzer.pipeline.profiler import NullProfiler
        self.profiler: Any = profiler or NullProfiler()
        # Per-image decode cache: avoids re-reading the same image from disk
        # when multiple modules run on the same image sequentially.  The cache
        # is keyed by file path and stores the decoded image_data dict.  It
        # holds at most one entry (the "current" image) and is invalidated
        # automatically when a different path is requested.
        self._image_cache_path: Path | None = None
        self._image_cache_data: dict[str, Any] | None = None

    def _cached_read_image(self, path: Path, image_id: int | None = None) -> dict[str, Any]:
        """Return decoded image_data, using a single-entry cache.

        When the worker processes multiple modules for the same image (the
        common case), the image is decoded once and reused.  At 500K images ×
        7+ modules each, this eliminates ~3.5M redundant file reads.

        The cached ``rgb_array`` is pre-resized to at most
        ``_AI_MAX_LONG_EDGE`` pixels on the long edge.  Every AI module
        (CLIP, GroundingDINO, BLIP-2, InsightFace, TrOCR) internally
        downsamples to ≤1920 px anyway, so passing a 50 MP array just
        wastes memory and CPU cycles on a redundant resize per module.
        Doing it once here saves those costs across all 7+ modules.
        """
        if self._image_cache_path == path and self._image_cache_data is not None:
            self.profiler.record_event("cache_hit", 0.0, image_id=image_id)
            return self._image_cache_data
        file_size = path.stat().st_size if path.exists() else None
        fmt = path.suffix.lower()
        with self.profiler.span("io_read", image_id=image_id,
                                image_file_size=file_size, image_format=fmt):
            data = _read_image(path)
        w = data.get("width")
        h = data.get("height")
        with self.profiler.span("io_resize", image_id=image_id,
                                image_width=w, image_height=h):
            data = _pre_resize(data)
        self._image_cache_path = path
        self._image_cache_data = data
        return data

    def invalidate_image_cache(self) -> None:
        """Clear the per-image decode cache (call between different images)."""
        self._image_cache_path = None
        self._image_cache_data = None

    def prime_image_cache(self, path: Path, data: dict[str, Any]) -> None:
        """Populate the single-entry cache with pre-read image data.

        Used by the prefetch pipeline to overlap IO with GPU inference.
        """
        self._image_cache_path = path
        self._image_cache_data = data

    @staticmethod
    def _has_perception_scores(row: dict[str, Any] | None) -> bool:
        if not row:
            return False
        return all(
            row.get(key) is not None
            for key in ("perception_iaa", "perception_iqa", "perception_ista")
        )

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
        with self.profiler.span("module_run", image_id=image_id, module=module):
            return self._run_dispatch(image_id, module)

    def _run_dispatch(self, image_id: int, module: str) -> dict[str, Any]:
        """Internal dispatch — separated so profiler.span wraps the whole call."""
        image = self.repo.get_image(image_id)
        if image is None:
            raise ValueError(f"Image id={image_id} not found in database")

        path = _apply_path_mappings(Path(image["file_path"]), self.path_mappings)

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
        elif module == "caption":
            return self._run_caption(image_id, path)
        elif module == "objects":
            return self._run_objects(image_id, path)
        elif module == "faces":
            return self._run_faces(image_id, path)
        elif module == "perception":
            return self._run_perception(image_id, path)
        else:
            raise ValueError(f"Unknown module: {module}")

    # ── Individual module runners ──────────────────────────────────────────

    def _run_metadata(self, image_id: int, path: Path) -> dict[str, Any]:
        # Header-only read: skip full pixel decode (demosaic) since metadata
        # extraction only needs EXIF headers and RAW sensor info.
        image_data = _read_image_headers(path)

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


        if self.verbose:
            console.print(f"  [dim]Metadata extracted for image {image_id}[/dim]")
        return result

    def _run_technical(self, image_id: int, path: Path) -> dict[str, Any]:
        image_data = self._cached_read_image(path, image_id)

        from imganalyzer.analysis.technical import TechnicalAnalyzer
        result = TechnicalAnalyzer(image_data).analyze()

        with _transaction(self.conn):
            self.repo.upsert_technical(image_id, result)

        if self.verbose:
            console.print(f"  [dim]Technical analysis done for image {image_id}[/dim]")
        return result

    def _run_caption(self, image_id: int, path: Path) -> dict[str, Any]:
        from imganalyzer.analysis.ai.ollama import OllamaAI

        ollama = OllamaAI()
        result = ollama.analyze(path, {})

        with _transaction(self.conn):
            self.repo.upsert_caption(image_id, result)

        if self.verbose:
            console.print(f"  [dim]Caption analysis done for image {image_id}[/dim]")
        return result

    def _run_objects(self, image_id: int, path: Path) -> dict[str, Any]:
        image_data = self._cached_read_image(path, image_id)

        from imganalyzer.pipeline.passes.objects import run_objects
        result = run_objects(
            image_data, self.repo, image_id, self.conn,
            prompt=self.detection_prompt,
            threshold=self.detection_threshold,
        )

        if self.verbose:
            console.print(f"  [dim]Object detection done for image {image_id}[/dim]")
        return result

    def _run_faces(self, image_id: int, path: Path) -> dict[str, Any]:
        image_data = self._cached_read_image(path, image_id)

        from imganalyzer.pipeline.passes.faces import run_faces
        result = run_faces(
            image_data, self.repo, image_id, self.conn,
            face_match_threshold=self.face_match_threshold,
        )

        if self.verbose:
            console.print(f"  [dim]Face analysis done for image {image_id}[/dim]")
        return result

    def _run_perception(self, image_id: int, path: Path) -> dict[str, Any]:
        from imganalyzer.analysis.perception import analyze as perception_analyze

        result = perception_analyze(path)

        with _transaction(self.conn):
            self.repo.upsert_perception(image_id, result)

        if self.verbose:
            iaa = result.get("perception_iaa", "?")
            console.print(
                f"  [dim]Perception analysis done for image {image_id} "
                f"(IAA={iaa})[/dim]"
            )
        return result

    def _run_embedding(self, image_id: int, path: Path) -> dict[str, Any]:
        from imganalyzer.embeddings.clip_embedder import CLIPEmbedder

        embedder = CLIPEmbedder()

        # Text embedding: combine description, scene, and subject from caption
        # so the vector reflects rich semantic content.
        text_parts: list[str] = []
        local_data = self.repo.get_analysis(image_id, "caption")
        if local_data:
            for field in ("description", "scene_type", "main_subject"):
                val = local_data.get(field)
                if val:
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
            # Use cached decoded image if available (avoids redundant decode)
            cached = self._image_cache_data if self._image_cache_path == path else None
            if cached is not None:
                from PIL import Image as _PILImage
                pil_img = _PILImage.fromarray(cached["rgb_array"]).convert("RGB")
                image_vector = embedder.embed_image_pil(pil_img)
            else:
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

    def run_embedding_batch(
        self,
        jobs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Batch CLIP image embedding for multiple images in a single GPU forward pass.

        Text (description_clip) embeddings are computed sequentially — they are
        cheap CPU-side tokeniser + tiny GPU forward passes.  The image_clip
        computation is the expensive part and benefits enormously from batching
        (batch 32 → ~32x fewer CUDA kernel launches).

        Returns a list of result dicts, one per job.
        """
        from imganalyzer.embeddings.clip_embedder import CLIPEmbedder
        from PIL import Image as _PILImage

        embedder = CLIPEmbedder()
        results: list[dict[str, Any]] = []

        # ── Collect PIL images for batch visual encoding ──────────────
        batch_pil_images: list[_PILImage.Image] = []
        batch_indices: list[int] = []  # index into jobs list

        for idx, job in enumerate(jobs):
            image_id = job["image_id"]
            path = Path(self.repo.get_image(image_id)["file_path"])

            if not path.exists():
                continue

            # Use cached decoded image if available
            cached = self._image_cache_data if self._image_cache_path == path else None
            if cached is not None:
                pil_img = _PILImage.fromarray(cached["rgb_array"]).convert("RGB")
            else:
                try:
                    image_data = _read_image(path)
                    image_data = _pre_resize(image_data)
                    pil_img = _PILImage.fromarray(image_data["rgb_array"]).convert("RGB")
                except Exception:
                    continue

            batch_pil_images.append(pil_img)
            batch_indices.append(idx)

        # ── Batched image CLIP forward pass ───────────────────────────
        image_vectors: list[bytes] = []
        if batch_pil_images:
            try:
                image_vectors = embedder.embed_images_batch(batch_pil_images)
            except Exception:
                # Fallback to sequential on OOM or error
                image_vectors = []
                for img in batch_pil_images:
                    try:
                        image_vectors.append(embedder.embed_image_pil(img))
                    except Exception:
                        image_vectors.append(b"")

        # Map batch results back to job indices
        image_vector_map: dict[int, bytes] = {}
        for i, job_idx in enumerate(batch_indices):
            if i < len(image_vectors) and image_vectors[i]:
                image_vector_map[job_idx] = image_vectors[i]

        # ── Per-image: write image_clip + compute/write description_clip ──
        for idx, job in enumerate(jobs):
            image_id = job["image_id"]
            path = Path(self.repo.get_image(image_id)["file_path"])
            result: dict[str, Any] = {"image_clip": False, "description_clip": False}

            # Write image_clip if we got a vector
            if idx in image_vector_map:
                with _transaction(self.conn):
                    self.repo.upsert_embedding(
                        image_id, "image_clip",
                        image_vector_map[idx], embedder.model_version,
                    )
                result["image_clip"] = True

            # Text embedding (cheap, sequential)
            text_parts: list[str] = []
            caption_data = self.repo.get_analysis(image_id, "caption")
            if caption_data:
                for field in ("description", "scene_type", "main_subject"):
                    val = caption_data.get(field)
                    if val:
                        text_parts.append(val)

            cloud_data = self.repo.get_analysis(image_id, "cloud_ai")
            if cloud_data:
                # Legacy cloud_ai data — include for backward compat
                for prov in cloud_data.get("providers", []):
                    for field in ("description", "scene_type", "main_subject"):
                        val = prov.get(field)
                        if val and val not in text_parts:
                            text_parts.append(val)

            desc_text= " ".join(text_parts)
            if desc_text:
                text_vector = embedder.embed_text(desc_text)
                with _transaction(self.conn):
                    self.repo.upsert_embedding(
                        image_id, "description_clip",
                        text_vector, embedder.model_version,
                    )
                result["description_clip"] = True

            if not result["image_clip"] and not result["description_clip"]:
                # Neither embedding could be computed — will be treated as failure
                pass

            results.append(result)

            if self.verbose:
                console.print(f"  [dim]CLIP embeddings computed for image {image_id}[/dim]")

        try:
            from imganalyzer.device import empty_cache
            empty_cache()
        except Exception:
            pass

        return results


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


def write_local_ai_split_tables(
    conn: sqlite3.Connection,
    repo: Repository,
    image_id: int,
    merged: dict[str, Any],
    *,
    wrap_transactions: bool = True,
) -> None:
    """Populate split-pass tables from a merged caption result dict.

    Legacy helper: distributes fields from the monolithic caption (formerly
    local_ai) result into individual module tables (blip2, objects, ocr, faces)
    for backward compatibility with older queries.
    """
    import json as _json

    blip2_data: dict[str, Any] = {}
    for key in ("description", "scene_type", "main_subject", "lighting", "mood", "keywords"):
        if key in merged:
            blip2_data[key] = merged[key]
    if blip2_data:
        try:
            if wrap_transactions:
                with _transaction(conn):
                    repo.upsert_blip2(image_id, blip2_data)
            else:
                repo.upsert_blip2(image_id, blip2_data)
        except Exception:
            pass

    objects_data: dict[str, Any] = {}
    if "detected_objects" in merged:
        objects_data["detected_objects"] = merged["detected_objects"]
    face_count = merged.get("face_count") or 0
    objects_data["has_person"] = 1 if int(face_count) > 0 else 0
    ocr_text = merged.get("ocr_text") or ""
    objects_data["has_text"] = 1 if ocr_text else 0
    objects_data["text_boxes"] = _json.dumps([])
    if objects_data:
        try:
            if wrap_transactions:
                with _transaction(conn):
                    repo.upsert_objects(image_id, objects_data)
            else:
                repo.upsert_objects(image_id, objects_data)
        except Exception:
            pass

    if merged.get("ocr_text"):
        try:
            if wrap_transactions:
                with _transaction(conn):
                    repo.upsert_ocr(image_id, {"ocr_text": merged["ocr_text"]})
            else:
                repo.upsert_ocr(image_id, {"ocr_text": merged["ocr_text"]})
        except Exception:
            pass

    faces_data: dict[str, Any] = {}
    for key in ("face_count", "face_identities", "face_details"):
        if key in merged:
            faces_data[key] = merged[key]
    if faces_data:
        try:
            if wrap_transactions:
                with _transaction(conn):
                    repo.upsert_faces(image_id, faces_data)
            else:
                repo.upsert_faces(image_id, faces_data)
        except Exception:
            pass


def unload_gpu_model(module: str) -> None:
    """Unload the GPU model(s) used by *module* to free VRAM.

    Called by the worker between GPU passes so that at most one large
    model is resident in VRAM at any time.  Each model will be lazily
    reloaded if a later pass needs it again (negligible cost — loading
    happens once per 500K-image run, not once per image).

    With model unloading, peak VRAM drops significantly, leaving
    ample headroom for batch inference within the GPU ceiling.
    """
    if module == "caption":
        from imganalyzer.analysis.ai.ollama import OllamaAI
        OllamaAI.unload_model()
    if module in ("objects",):
        from imganalyzer.analysis.ai.objects import ObjectDetector
        ObjectDetector._unload()
    if module in ("faces",):
        from imganalyzer.analysis.ai.faces import FaceAnalyzer
        FaceAnalyzer._unload()
    if module == "embedding":
        from imganalyzer.embeddings.clip_embedder import CLIPEmbedder
        CLIPEmbedder._unload()
    if module == "perception":
        from imganalyzer.analysis.perception import unload_model
        unload_model()


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

    # AI analysis — merge caption and cloud_ai into a single dict
    # (XMPWriter expects result.ai_analysis to be a flat dict)
    ai: dict[str, Any] = {}

    local = full.get("caption", {})
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

    # Cloud AI — legacy data, layer on top for backward compat with existing analyses
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

    # Perception — merge into ai_analysis so XMP writer picks it up
    perception = full.get("perception", {})
    for key in ("image_id", "analyzed_at"):
        perception.pop(key, None)
    if perception:
        ai.update(perception)

    xmp_path = path.with_suffix(".xmp")
    result.write_xmp(xmp_path)
    return xmp_path
