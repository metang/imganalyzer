"""RAW image reader using rawpy (LibRaw bindings)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def read(path: Path) -> dict[str, Any]:
    """Read a RAW file and return image data dict."""
    try:
        import rawpy
    except ImportError:
        raise ImportError("rawpy is required for RAW files: pip install rawpy")

    with rawpy.imread(str(path)) as raw:
        # Develop the RAW into an 8-bit RGB array
        rgb = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=False,
            output_bps=8,
        )

        # Also expose the raw Bayer data for technical analysis
        try:
            raw_image = raw.raw_image.copy()
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

    def read(self) -> dict[str, Any]:
        return read(self.path)
