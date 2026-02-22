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


class FaceAnalyzer:
    """Face detection + recognition using InsightFace buffalo_l.

    - Detects all faces in the image.
    - Extracts 512-d embeddings per face.
    - Matches against registered identities in FaceDatabase.
    - Returns age estimate and gender per face.

    Requires: insightface, onnxruntime-gpu  (pip install 'imganalyzer[local-ai]')
    """

    _app = None

    def analyze(
        self,
        image_data: dict[str, Any],
        face_db: "Any | None" = None,
        match_threshold: float | None = None,
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

        if not faces:
            return {
                "face_count": 0,
                "face_identities": [],
                "face_details": [],
            }

        identities: list[str] = []
        details: list[str] = []

        for face in faces:
            embedding: np.ndarray = face.embedding  # 512-d float32
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

        return {
            "face_count": len(faces),
            "face_identities": identities,
            "face_details": details,
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
        ctx_id = 0 if any("CUDA" in p or "GPU" in p for p in providers) else -1
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        cls._app = app


def _get_providers() -> list[str]:
    """Return ONNX Runtime execution providers, preferring CUDA."""
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
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
