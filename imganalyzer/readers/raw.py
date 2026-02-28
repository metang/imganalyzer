"""RAW image reader using rawpy (LibRaw bindings)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def read(path: Path, *, half_size: bool = True) -> dict[str, Any]:
    """Read a RAW file and return image data dict.

    Args:
        half_size: When True (default), demosaic at half resolution — 4x faster
            decode and 4x less memory.  All AI models resize to ≤1920px anyway,
            so half of a typical 20–50 MP sensor (≥2500px per side) is more than
            sufficient.  Pass False only when full native resolution is required.
    """
    try:
        import rawpy
    except ImportError:
        raise ImportError("rawpy is required for RAW files: pip install rawpy")

    try:
        raw_ctx = rawpy.imread(str(path))
    except Exception as exc:
        raise ValueError(
            f"LibRaw cannot decode {path.name}: {exc}"
        ) from exc

    with raw_ctx as raw:
        # Get raw dimensions first to check size
        raw_h, raw_w = raw.raw_image.shape[:2]
        try:
            rgb = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=False,
                output_bps=8,
                half_size=half_size,
            )
        except Exception as exc:
            raise ValueError(
                f"LibRaw postprocess failed for {path.name}: {exc}"
            ) from exc

        # Also expose the raw Bayer data for technical analysis
        try:
            raw_image = None  # skip full Bayer copy to save memory
            raw_colors = raw.color_desc.decode()
            raw_pattern = raw.raw_pattern.tolist()
            black_level = raw.black_level_per_channel
            white_level = raw.white_level
            camera_wb = raw.camera_whitebalance
            daylight_wb = raw.daylight_whitebalance
        except Exception:
            raw_image = None
            raw_colors = ""
            raw_pattern = None
            black_level = None
            white_level = None
            camera_wb = None
            daylight_wb = None

        h, w = rgb.shape[:2]

        return {
            "format": path.suffix.upper().lstrip("."),
            "width": w,
            "height": h,
            "rgb_array": rgb,          # numpy uint8 H×W×3
            "is_raw": True,
            "raw_image": raw_image,    # Bayer data (optional)
            "raw_colors": raw_colors,
            "raw_pattern": raw_pattern,
            "black_level": black_level,
            "white_level": white_level,
            "camera_wb": camera_wb,
            "daylight_wb": daylight_wb,
            "color_desc": raw_colors,
        }


class RawReader:
    def __init__(self, path: Path) -> None:
        self.path = path

    def read(self, *, half_size: bool = True) -> dict[str, Any]:
        return read(self.path, half_size=half_size)

    def read_headers(self) -> dict[str, Any]:
        return read_headers(self.path)


def read_headers(path: Path) -> dict[str, Any]:
    """Read only metadata from a RAW file — no pixel decode (demosaic).

    Returns a minimal image_data dict with format, dimensions, and RAW
    metadata (white balance, black/white levels) but no ``rgb_array``.
    This is sufficient for the metadata extractor which only reads EXIF
    headers and RAW sensor info.
    """
    try:
        import rawpy
    except ImportError:
        raise ImportError("rawpy is required for RAW files: pip install rawpy")

    try:
        raw_ctx = rawpy.imread(str(path))
    except Exception as exc:
        raise ValueError(
            f"LibRaw cannot decode {path.name}: {exc}"
        ) from exc

    with raw_ctx as raw:
        raw_h, raw_w = raw.raw_image.shape[:2]
        try:
            raw_colors = raw.color_desc.decode()
            camera_wb = raw.camera_whitebalance
            daylight_wb = raw.daylight_whitebalance
            white_level = raw.white_level
            black_level = raw.black_level_per_channel
        except Exception:
            raw_colors = ""
            camera_wb = None
            daylight_wb = None
            white_level = None
            black_level = None

        return {
            "format": path.suffix.upper().lstrip("."),
            "width": raw_w,
            "height": raw_h,
            "is_raw": True,
            "rgb_array": None,
            "camera_wb": camera_wb,
            "daylight_wb": daylight_wb,
            "white_level": white_level,
            "black_level": black_level,
            "color_desc": raw_colors,
        }
