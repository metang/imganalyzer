"""Centralized GPU device selection with CUDA, MPS, and CPU support."""

from __future__ import annotations

import os

# Some models (e.g. GroundingDINO) use ops not yet implemented on MPS.
# Enable automatic CPU fallback so they run instead of crashing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch


def get_device() -> str:
    """Return the best available device: ``"cuda"``, ``"mps"``, or ``"cpu"``."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_device_type() -> str:
    """Return the device *type* string for :func:`torch.autocast`.

    Same as :func:`get_device` but guaranteed to be a valid autocast backend.
    """
    return get_device()


def supports_fp16() -> bool:
    """Return ``True`` if the active device can run fp16 efficiently."""
    dev = get_device()
    return dev in ("cuda", "mps")


def empty_cache() -> None:
    """Release cached GPU memory on whichever backend is active."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
