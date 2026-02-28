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

    # Register HEIC/HEIF support if available
    if path.suffix.lower() in (".heic", ".heif"):
        try:
            from pillow_heif import register_heif_opener
            register_heif_opener()
        except ImportError:
            raise ImportError(
                "pillow-heif is required for HEIC/HEIF files: pip install pillow-heif"
            )

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

    def read_headers(self) -> dict[str, Any]:
        return read_headers(self.path)


def read_headers(path: Path) -> dict[str, Any]:
    """Read only metadata from an image file — no full pixel decode.

    Opens the file lazily (Pillow defers decoding until pixel access),
    extracts format, dimensions, EXIF, and DPI info, then returns without
    converting to a numpy array.  This avoids the expensive RGB decode
    for the metadata extractor which only needs EXIF headers.
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is required: pip install Pillow")

    # Register HEIC/HEIF support if available
    if path.suffix.lower() in (".heic", ".heif"):
        try:
            from pillow_heif import register_heif_opener
            register_heif_opener()
        except ImportError:
            raise ImportError(
                "pillow-heif is required for HEIC/HEIF files: pip install pillow-heif"
            )

    img = Image.open(path)
    fmt = img.format or path.suffix.upper().lstrip(".")
    w, h = img.size

    exif_bytes: bytes | None = None
    try:
        exif_bytes = img.info.get("exif")
    except Exception:
        pass

    return {
        "format": fmt,
        "width": w,
        "height": h,
        "rgb_array": None,
        "is_raw": False,
        "pil_image": img,
        "original_mode": img.mode,
        "exif_bytes": exif_bytes,
        "icc_profile": img.info.get("icc_profile"),
        "dpi": img.info.get("dpi"),
    }
