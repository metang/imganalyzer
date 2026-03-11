"""pyiqa-based aesthetic scoring adapters (TOPIQ, CLIPIQA+, MANIQA, NIMA, MUSIQ)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from model_eval.models.base import ModelAdapter


class _PyIQABaseAdapter(ModelAdapter):
    """Base adapter for pyiqa metrics."""

    category = "aesthetic"
    _metric_name: str = ""

    def __init__(self) -> None:
        self._model: Any = None
        self._device: str = "cuda"

    def load(self, device: str = "cuda") -> None:
        import pyiqa

        self._device = device
        self._model = pyiqa.create_metric(self._metric_name, device=device)

    def run(self, image_path: Path) -> dict[str, Any]:
        import tempfile

        from PIL import Image

        try:
            from pillow_heif import register_heif_opener
            register_heif_opener()
        except ImportError:
            pass

        # Always convert through PIL: handles HEIC/RAW and caps resolution
        # to avoid OOM on multi-megapixel images
        MAX_DIM = 1024
        img = Image.open(image_path).convert("RGB")
        if max(img.size) > MAX_DIM:
            img.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img.save(f.name, "JPEG", quality=95)
            tmp = Path(f.name)

        try:
            score = self._model(str(tmp)).cpu().item()
        finally:
            tmp.unlink(missing_ok=True)

        return {
            "score": round(score, 4),
            "scale": "metric-specific",
            "metric": self._metric_name,
            "details": {},
        }

    def unload(self) -> None:
        del self._model
        self._model = None
        super().unload()


class PyIQAAdapter(_PyIQABaseAdapter):
    name = "pyiqa-topiq"
    model_id = "pyiqa:topiq_nr"
    _metric_name = "topiq_nr"


class PyIQACLIPIQAAdapter(_PyIQABaseAdapter):
    name = "pyiqa-clipiqa"
    model_id = "pyiqa:clipiqa+"
    _metric_name = "clipiqa+"


class PyIQAMANIQAAdapter(_PyIQABaseAdapter):
    name = "pyiqa-maniqa"
    model_id = "pyiqa:maniqa"
    _metric_name = "maniqa"


class PyIQANIMAAdapter(_PyIQABaseAdapter):
    name = "pyiqa-nima"
    model_id = "pyiqa:nima"
    _metric_name = "nima"


class PyIQAMUSIQAdapter(_PyIQABaseAdapter):
    name = "pyiqa-musiq"
    model_id = "pyiqa:musiq"
    _metric_name = "musiq"
