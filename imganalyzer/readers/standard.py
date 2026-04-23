"""Standard image reader using Pillow."""
from __future__ import annotations

import contextlib
import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np


_PIL_TIFF_LOGGER = logging.getLogger("PIL.TiffImagePlugin")
_SUPPRESSED_PILLOW_DECODE_PATTERNS = (
    re.compile(r"More samples per pixel than can be decoded:\s*\d+", re.IGNORECASE),
)


class _PillowDecodeLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(pattern.search(message) for pattern in _SUPPRESSED_PILLOW_DECODE_PATTERNS)


@contextlib.contextmanager
def pillow_decode_guard(path: Path) -> Iterator[None]:
    """Suppress known Pillow TIFF noise and normalize decode failures."""
    log_filter = _PillowDecodeLogFilter()
    _PIL_TIFF_LOGGER.addFilter(log_filter)
    try:
        yield
    except Exception as exc:
        raise ValueError(f"Pillow cannot decode {path.name}: {exc}") from exc
    finally:
        _PIL_TIFF_LOGGER.removeFilter(log_filter)


_HEIF_REGISTERED = False


def register_optional_pillow_opener(path: Path) -> None:
    """Register optional Pillow plugins needed for certain formats.

    ``register_heif_opener`` from ``pillow_heif`` is idempotent but costs
    ~10 ms per call, which adds up during large ingests.  We guard the
    call with a module-level flag so the registration runs at most once
    per process.
    """
    global _HEIF_REGISTERED
    if _HEIF_REGISTERED:
        return
    if path.suffix.lower() in (".heic", ".heif", ".avif"):
        try:
            from pillow_heif import register_heif_opener
            register_heif_opener()
            _HEIF_REGISTERED = True
        except ImportError:
            raise ImportError(
                "pillow-heif is required for HEIC/HEIF/AVIF files: pip install pillow-heif"
            )


def read(path: Path) -> dict[str, Any]:
    """Read a standard image file and return image data dict."""
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is required: pip install Pillow")

    register_optional_pillow_opener(path)

    with pillow_decode_guard(path):
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

    register_optional_pillow_opener(path)

    with pillow_decode_guard(path):
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
