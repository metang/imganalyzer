"""Standard image reader using Pillow."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def read(path: Path) -> dict[str, Any]:
    """Read a standard image file and return image data dict."""
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is required: pip install Pillow")

    img = Image.open(path)  # Path objects work on both Windows and macOS
    fmt = img.format or path.suffix.upper().lstrip(".")
    mode = img.mode

    # Convert to RGB for consistent processing
    if mode not in ("RGB", "L"):
        img_rgb = img.convert("RGB")
    else:
        img_rgb = img if mode == "RGB" else img.convert("RGB")

    w, h = img_rgb.size
    rgb_array = np.array(img_rgb, dtype=np.uint8)

    # Preserve original for EXIF
    exif_bytes: bytes | None = None
    try:
        exif_bytes = img.info.get("exif")
    except Exception:
        pass

    return {
        "format": fmt,
        "width": w,
        "height": h,
        "rgb_array": rgb_array,
        "is_raw": False,
        "pil_image": img,
        "original_mode": mode,
        "exif_bytes": exif_bytes,
        "icc_profile": img.info.get("icc_profile"),
        "dpi": img.info.get("dpi"),
    }


class StandardReader:
    def __init__(self, path: Path) -> None:
        self.path = path

    def read(self) -> dict[str, Any]:
        return read(self.path)
