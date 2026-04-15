"""RAW image reader using rawpy (LibRaw bindings)."""
from __future__ import annotations

import contextlib
import io
import logging
import os
import threading
from pathlib import Path
from typing import Any, Generator

import numpy as np

log = logging.getLogger(__name__)

_STDERR_FD_LOCK = threading.Lock()


@contextlib.contextmanager
def _suppress_c_stderr() -> Generator[None, None, None]:
    """Temporarily redirect C-level stderr to devnull.

    LibRaw writes "unknown file: data corrupted at ..." directly to the C
    file descriptor, bypassing Python's sys.stderr.  This silences those
    messages so they don't flood the Electron DevTools console.
    """
    with _STDERR_FD_LOCK:
        devnull_fd = None
        old_stderr_fd = None
        try:
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            old_stderr_fd = os.dup(2)
            os.dup2(devnull_fd, 2)
            yield
        except OSError:
            # If fd manipulation fails (e.g. on some platforms), just proceed
            yield
        finally:
            if old_stderr_fd is not None:
                try:
                    os.dup2(old_stderr_fd, 2)
                    os.close(old_stderr_fd)
                except OSError:
                    pass
            if devnull_fd is not None:
                try:
                    os.close(devnull_fd)
                except OSError:
                    pass


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
        with _suppress_c_stderr():
            raw_ctx = rawpy.imread(str(path))
    except Exception as exc:
        raise ValueError(
            f"LibRaw cannot decode {path.name}: {exc}"
        ) from exc

    with raw_ctx as raw:
        rgb = None
        postprocess_err = None
        try:
            rgb = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=False,
                output_bps=8,
                half_size=half_size,
            )
        except Exception as exc:
            postprocess_err = exc

        if rgb is None:
            # Fallback 1: extract embedded thumbnail (most DNG/RAW files have one)
            rgb = _try_extract_thumb(raw, path)

        if rgb is None:
            # Fallback 2: try Pillow (works for DNG since it's TIFF-based)
            rgb = _try_pillow_fallback(path)

        if rgb is None:
            raise ValueError(
                f"LibRaw postprocess failed for {path.name}: {postprocess_err}"
            ) from postprocess_err

        if postprocess_err is not None:
            log.info(
                "Decoded %s via fallback (postprocess failed: %s)",
                path.name, postprocess_err,
            )

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


def _try_extract_thumb(raw: Any, path: Path) -> np.ndarray | None:
    """Try to extract the embedded thumbnail/preview from a RAW file.

    Most DNG and RAW files embed a JPEG preview that can be used when full
    demosaic fails.  Returns an RGB uint8 numpy array or None.
    """
    try:
        thumb = raw.extract_thumb()
    except Exception:
        return None

    try:
        import rawpy

        if thumb.format == rawpy.ThumbFormat.JPEG:
            from PIL import Image

            img = Image.open(io.BytesIO(thumb.data))
            img = img.convert("RGB")
            return np.array(img, dtype=np.uint8)
        elif thumb.format == rawpy.ThumbFormat.BITMAP:
            arr = np.asarray(thumb.data, dtype=np.uint8)
            if arr.ndim == 3 and arr.shape[2] == 3:
                return arr
    except Exception:
        pass
    return None


def _try_pillow_fallback(path: Path) -> np.ndarray | None:
    """Try opening the file with Pillow as a last resort.

    Works for DNG files (TIFF-based) and some other formats Pillow supports.
    """
    try:
        from PIL import Image

        img = Image.open(path)
        img = img.convert("RGB")
        return np.array(img, dtype=np.uint8)
    except Exception:
        return None


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
        with _suppress_c_stderr():
            raw_ctx = rawpy.imread(str(path))
    except Exception as exc:
        raise ValueError(
            f"LibRaw cannot decode {path.name}: {exc}"
        ) from exc

    with raw_ctx as raw:
        # Use raw.sizes (header-only) instead of raw.raw_image (triggers unpack
        # which fails for some DNG variants with LibRawDataError).
        raw_h = raw.sizes.raw_height
        raw_w = raw.sizes.raw_width
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
