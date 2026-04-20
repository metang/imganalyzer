"""Image readers — Pillow (standard) and rawpy (camera RAW).

The convenience function :func:`open_as_pil` is the preferred way to obtain a
``PIL.Image`` from an arbitrary image path.  It transparently handles camera RAW
formats (MRW, CR2, NEF, ARW …) via *rawpy* and standard formats via Pillow.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image


_DECODE_ERROR_MARKERS = (
    "libraw cannot decode",
    "libraw postprocess failed",
    "librawdataerror",
    "data error or unsupported file format",
    "unsupported file format",
    "pillow cannot decode",
    "cannot identify image file",
)


def is_decode_error(exc: BaseException) -> bool:
    """Return True when *exc* represents a corrupt/unsupported image decode error."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _DECODE_ERROR_MARKERS)


def open_as_pil(path: Path, *, mode: str = "RGB") -> Image.Image:
    """Open any supported image file and return a PIL Image.

    For camera RAW formats (listed in ``analyzer.RAW_EXTENSIONS``) this uses
    the shared RAW reader stack, including embedded-preview / Pillow fallbacks
    when LibRaw cannot fully demosaic the file. For everything else it delegates
    to Pillow. HEIC/HEIF openers are registered automatically when needed.

    Args:
        path: Image file path.
        mode: Target colour mode (default ``"RGB"``).
    """
    from imganalyzer.analyzer import RAW_EXTENSIONS

    suffix = path.suffix.lower()
    if suffix in RAW_EXTENSIONS:
        from imganalyzer.readers.raw import read as read_raw

        image_data = read_raw(path, half_size=True)
        rgb = image_data.get("rgb_array")
        if rgb is None:
            raise ValueError(f"LibRaw postprocess failed for {path.name}: decode produced no pixels")
        img = Image.fromarray(rgb)
    else:
        from imganalyzer.readers.standard import (
            pillow_decode_guard,
            register_optional_pillow_opener,
        )

        register_optional_pillow_opener(path)
        with pillow_decode_guard(path):
            img = Image.open(path)

    if img.mode != mode:
        img = img.convert(mode)
    return img
