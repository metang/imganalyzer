"""Image readers — Pillow (standard) and rawpy (camera RAW).

The convenience function :func:`open_as_pil` is the preferred way to obtain a
``PIL.Image`` from an arbitrary image path.  It transparently handles camera RAW
formats (MRW, CR2, NEF, ARW …) via *rawpy* and standard formats via Pillow.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image


def open_as_pil(path: Path, *, mode: str = "RGB") -> Image.Image:
    """Open any supported image file and return a PIL Image.

    For camera RAW formats (listed in ``analyzer.RAW_EXTENSIONS``) this uses
    *rawpy* to demosaic; for everything else it delegates to Pillow.  HEIC/HEIF
    openers are registered automatically when needed.

    Args:
        path: Image file path.
        mode: Target colour mode (default ``"RGB"``).
    """
    from imganalyzer.analyzer import RAW_EXTENSIONS

    suffix = path.suffix.lower()
    if suffix in RAW_EXTENSIONS:
        from imganalyzer.readers.raw import _suppress_c_stderr

        import rawpy

        try:
            with _suppress_c_stderr():
                raw_ctx = rawpy.imread(str(path))
        except Exception as exc:
            raise ValueError(f"LibRaw cannot decode {path.name}: {exc}") from exc

        with raw_ctx as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=False,
                output_bps=8,
                half_size=True,
            )
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
