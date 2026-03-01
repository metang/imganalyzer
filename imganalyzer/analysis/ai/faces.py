"""Face detection, recognition and attribute estimation using InsightFace buffalo_l."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

CACHE_DIR = os.getenv("IMGANALYZER_MODEL_CACHE", str(Path.home() / ".cache" / "imganalyzer"))

# InsightFace downloads models to INSIGHTFACE_HOME; redirect to our cache dir.
# Must be set before insightface is imported.
_INSIGHTFACE_HOME = str(Path(CACHE_DIR) / "insightface")


def _crop_thumbnail(
    rgb: np.ndarray, bbox: np.ndarray, max_dim: int = 200, quality: int = 85
) -> bytes:
    """Crop a face from *rgb* using *bbox* [x1,y1,x2,y2] and return JPEG bytes."""
    import io
    from PIL import Image

    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

    # Add 20% padding
    fw, fh = x2 - x1, y2 - y1
    pad_x, pad_y = int(fw * 0.2), int(fh * 0.2)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    crop_arr = rgb[y1:y2, x1:x2]
    pil_crop = Image.fromarray(crop_arr)

    cw, ch = pil_crop.size
    if cw > max_dim or ch > max_dim:
        scale = max_dim / max(cw, ch)
        pil_crop = pil_crop.resize(
            (int(cw * scale), int(ch * scale)), Image.LANCZOS
        )

    buf = io.BytesIO()
    pil_crop.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class FaceAnalyzer:
    """Face detection + recognition using InsightFace buffalo_l.

    - Detects all faces in the image.
    - Extracts 512-d embeddings per face.
    - Matches against registered identities in FaceDatabase.
    - Returns age estimate and gender per face.

    Requires: insightface, onnxruntime-gpu  (pip install 'imganalyzer[local-ai]')
    """

    _app = None

    @classmethod
    def _unload(cls) -> None:
        """Unload InsightFace models from GPU to free VRAM.

        InsightFace uses ONNX Runtime which manages GPU memory through
        its own allocator.  Deleting the FaceAnalysis app releases the
        underlying ONNX InferenceSession objects and their GPU buffers.
        """
        if cls._app is not None:
            del cls._app
            cls._app = None
        # ONNX RT does not use the PyTorch CUDA allocator, so
        # empty_cache() won't help here — but gc.collect() ensures
        # the ONNX sessions are actually freed.
        import gc
        gc.collect()

    def analyze(
        self,
        image_data: dict[str, Any],
        face_db: "Any | None" = None,
        match_threshold: float | None = None,
        det_score_threshold: float = 0.65,
        min_face_pixels: int = 40,
    ) -> dict[str, Any]:
        try:
            import insightface  # noqa: F401
        except ImportError:
            raise ImportError(
                "insightface is required for face analysis:\n"
                "  pip install 'imganalyzer[local-ai]'"
            )

        rgb: np.ndarray = image_data["rgb_array"]
        # InsightFace expects BGR
        bgr = rgb[:, :, ::-1].copy()

        self._load_model()
        app = FaceAnalyzer._app

        faces = app.get(bgr)

        # Filter out low-confidence and tiny detections
        if faces:
            filtered = []
            for face in faces:
                score = getattr(face, "det_score", 1.0)
                if score < det_score_threshold:
                    continue
                bbox = face.bbox
                fw = bbox[2] - bbox[0]
                fh = bbox[3] - bbox[1]
                if min(fw, fh) < min_face_pixels:
                    continue
                filtered.append(face)
            faces = filtered

        if not faces:
            return {
                "face_count": 0,
                "face_identities": [],
                "face_details": [],
                "face_occurrences": [],
            }

        identities: list[str] = []
        details: list[str] = []
        occurrences: list[dict[str, Any]] = []

        for idx, face in enumerate(faces):
            embedding: np.ndarray = face.embedding  # 512-d float32
            bbox = face.bbox  # [x1, y1, x2, y2] float array
            age: int = int(getattr(face, "age", -1))
            # InsightFace gender: 0=Female, 1=Male (attribute name varies by version)
            gender_raw = getattr(face, "gender", None)
            if gender_raw is None:
                gender_raw = getattr(face, "sex", None)
            gender = _parse_gender(gender_raw)

            if face_db is not None and embedding is not None:
                name, _sim = face_db.match(embedding, threshold=match_threshold)
            else:
                name = "Unknown"

            identities.append(name)
            age_str = str(age) if age >= 0 else "?"
            details.append(f"{name}:{age_str}:{gender}")

            # Per-face occurrence data for storage
            occ: dict[str, Any] = {
                "face_idx": idx,
                "bbox_x1": float(bbox[0]),
                "bbox_y1": float(bbox[1]),
                "bbox_x2": float(bbox[2]),
                "bbox_y2": float(bbox[3]),
                "det_score": float(getattr(face, "det_score", 0.0)),
                "age": age if age >= 0 else None,
                "gender": gender,
                "identity_name": name,
            }
            if embedding is not None:
                occ["embedding"] = embedding.astype(np.float32).tobytes()

            # Pre-generate thumbnail from the in-memory rgb_array
            occ["thumbnail"] = _crop_thumbnail(rgb, bbox)
            occurrences.append(occ)

        return {
            "face_count": len(faces),
            "face_identities": identities,
            "face_details": details,
            "face_occurrences": occurrences,
        }

    @classmethod
    def _load_model(cls) -> None:
        if cls._app is not None:
            return

        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError(
                "insightface is required for face analysis:\n"
                "  pip install 'imganalyzer[local-ai]'"
            )

        from rich.console import Console
        Console().print("[dim]Loading InsightFace buffalo_l model (first run downloads ~300MB)...[/dim]")

        # Redirect InsightFace model download to our cache directory
        os.environ.setdefault("INSIGHTFACE_HOME", _INSIGHTFACE_HOME)
        Path(_INSIGHTFACE_HOME).mkdir(parents=True, exist_ok=True)

        # Determine execution provider
        providers = _get_providers()

        app = FaceAnalysis(
            name="buffalo_l",
            root=_INSIGHTFACE_HOME,
            providers=providers,
        )
        # ctx_id=0 → GPU 0; ctx_id=-1 → CPU
        # providers may be strings or (name, opts) tuples
        def _has_gpu(prov_list: list) -> bool:
            for p in prov_list:
                name = p[0] if isinstance(p, tuple) else p
                if "CUDA" in name or "GPU" in name:
                    return True
            return False
        ctx_id = 0 if _has_gpu(providers) else -1
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        cls._app = app


def _get_providers() -> list:
    """Return ONNX Runtime execution providers, preferring CUDA.

    When CUDA is available, applies a GPU memory limit (1 GB) and
    conservative arena strategy.  InsightFace buffalo_l weights total
    ~400 MB; the extra headroom allows ONNX RT to keep intermediate
    tensors in GPU memory across inference calls, avoiding repeated
    allocation overhead.  The VRAM budget system (vram_budget.py)
    manages co-residency at a higher level.
    """
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            cuda_opts = {
                "gpu_mem_limit": str(1024 * 1024 * 1024),  # 1 GB
                "arena_extend_strategy": "kSameAsRequested",
            }
            return [
                ("CUDAExecutionProvider", cuda_opts),
                "CPUExecutionProvider",
            ]
    except Exception:
        pass
    return ["CPUExecutionProvider"]


def _parse_gender(raw: Any) -> str:
    if raw is None:
        return "Unknown"
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in ("m", "male", "1"):
            return "Male"
        if s in ("f", "female", "0"):
            return "Female"
        return raw.capitalize()
    try:
        v = int(raw)
        return "Male" if v == 1 else "Female"
    except Exception:
        return str(raw)


def extract_embedding_from_image(image_path: "Path | str") -> np.ndarray | None:
    """Extract the dominant face embedding from a single image file.

    Used by the ``register-face`` CLI command.  Returns None if no face is
    detected.
    """
    from pathlib import Path as _Path
    from PIL import Image

    path = _Path(image_path)
    img = Image.open(path).convert("RGB")
    rgb = np.array(img, dtype=np.uint8)

    analyzer = FaceAnalyzer()
    analyzer._load_model()
    app = FaceAnalyzer._app

    bgr = rgb[:, :, ::-1].copy()
    faces = app.get(bgr)

    if not faces:
        return None

    # Pick the largest face by bounding-box area
    def _area(f: Any) -> float:
        bbox = f.bbox  # [x1, y1, x2, y2]
        return float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

    best = max(faces, key=_area)
    return best.embedding
