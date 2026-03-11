"""Abstract base class for model adapters."""

from __future__ import annotations

import gc
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from PIL import Image

# Cap input image resolution to avoid OOM with large RAW/HEIC photos
MAX_IMAGE_DIM = 1280


class ModelAdapter(ABC):
    """Base class that all model adapters must implement."""

    name: str = "unnamed"
    category: str = "unknown"  # "aesthetic" or "caption"
    model_id: str = ""  # HuggingFace model ID or package name

    @abstractmethod
    def load(self, device: str = "cuda") -> None:
        """Load model weights onto the specified device."""

    @abstractmethod
    def run(self, image_path: Path) -> dict[str, Any]:
        """Run inference on a single image. Returns result dict."""

    def unload(self) -> None:
        """Unload model and free GPU memory.

        Subclasses should override to delete their specific model attributes,
        then call super().unload() to trigger GPU cleanup.
        """
        self._release_gpu()

    @staticmethod
    def _release_gpu() -> None:
        """Aggressively release GPU memory — call after deleting model refs."""
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    _RAW_EXTENSIONS = {".arw", ".cr2", ".cr3", ".nef", ".dng", ".raf", ".rw2", ".orf"}

    @staticmethod
    def load_image(image_path: Path, max_dim: int = MAX_IMAGE_DIM) -> Image.Image:
        """Load and resize image, handling HEIC via pillow-heif and RAW via rawpy."""
        try:
            from pillow_heif import register_heif_opener
            register_heif_opener()
        except ImportError:
            pass

        suffix = image_path.suffix.lower()
        if suffix in ModelAdapter._RAW_EXTENSIONS:
            import rawpy
            with rawpy.imread(str(image_path)) as raw:
                rgb = raw.postprocess(use_camera_wb=True, half_size=True)
            img = Image.fromarray(rgb)
        else:
            img = Image.open(image_path)

        img = img.convert("RGB")
        if max_dim and max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.LANCZOS)
        return img

    def run_timed(self, image_path: Path) -> dict[str, Any]:
        """Run inference and include timing information."""
        start = time.perf_counter()
        result = self.run(image_path)
        elapsed = time.perf_counter() - start
        result["inference_time_s"] = round(elapsed, 3)
        result["model"] = self.name
        return result

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
