"""CLIP embedder — image and text embedding for semantic search.

Uses CLIP ViT-L/14 from OpenAI via the ``open_clip`` library.
Embeddings are 768-d float32 vectors stored as raw bytes in the DB.

Pre-resize policy
-----------------
CLIP ViT-L/14 internally rescales every image to 224×224 px, so feeding it a
50 MP RAW file wastes decode time and GPU memory without improving the embedding.
``embed_image`` therefore downsizes the image to at most ``EMBED_MAX_LONG_EDGE``
pixels on the long edge *before* handing it to the CLIP pre-processor.

``thumbnail()`` is used (not ``resize()``) so the image is only ever shrunk,
never upscaled, and the aspect ratio is always preserved.
"""
from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Any

import numpy as np

CACHE_DIR = os.getenv("IMGANALYZER_MODEL_CACHE", str(Path.home() / ".cache" / "imganalyzer"))

# Maximum size (pixels) on the long edge before feeding to CLIP.
# Anything larger is downsampled first to reduce decode/GPU memory overhead.
# CLIP itself rescales to 224 px, so values above ~1280 give diminishing returns.
EMBED_MAX_LONG_EDGE = 1280


class CLIPEmbedder:
    """Lazy-loading CLIP encoder for both images and text queries."""

    _model = None
    _preprocess = None
    _tokenizer = None
    _device = None
    model_version: str = "ViT-L-14/openai"

    def embed_image(self, path: Path | str) -> bytes:
        """Encode an image file → float32 bytes (768-d).

        Supports all formats handled by Pillow *and* RAW camera files
        (any extension in ``analyzer.RAW_EXTENSIONS``) by routing them
        through rawpy before handing the decoded RGB array to CLIP.
        """
        self._load_model()
        import torch
        from PIL import Image

        path = Path(path)
        suffix = path.suffix.lower()

        # Check if this is a RAW camera file that Pillow cannot open directly.
        try:
            from imganalyzer.analyzer import RAW_EXTENSIONS
            is_raw = suffix in RAW_EXTENSIONS
        except ImportError:
            is_raw = False

        if is_raw:
            import rawpy
            with rawpy.imread(str(path)) as raw:
                rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, output_bps=8)
            img = Image.fromarray(rgb).convert("RGB")
        else:
            # Register HEIC/HEIF support before opening — Pillow cannot decode
            # these formats without the pillow-heif plugin.
            if suffix in (".heic", ".heif"):
                try:
                    from pillow_heif import register_heif_opener
                    register_heif_opener()
                except ImportError:
                    raise ImportError(
                        "pillow-heif is required for HEIC/HEIF files: pip install pillow-heif"
                    )
            img = Image.open(str(path)).convert("RGB")

        # Downsize to EMBED_MAX_LONG_EDGE on the long edge before CLIP pre-processing.
        # CLIP rescales to 224 px internally, so there is no quality loss.
        # thumbnail() only ever shrinks — it never upscales a small image.
        img.thumbnail((EMBED_MAX_LONG_EDGE, EMBED_MAX_LONG_EDGE), Image.LANCZOS)

        image_input = CLIPEmbedder._preprocess(img).unsqueeze(0).to(CLIPEmbedder._device)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(CLIPEmbedder._device != "cpu")):
            features = CLIPEmbedder._model.encode_image(image_input)
            features = features / features.norm(dim=-1, keepdim=True)

        vec = features.cpu().numpy().flatten().astype(np.float32)
        return vec.tobytes()

    def embed_image_pil(self, img: "Image.Image") -> bytes:
        """Encode a pre-loaded PIL Image → float32 bytes (768-d).

        Same pipeline as :meth:`embed_image` but skips the file read and
        RAW/HEIC decode — used by the per-image decode cache in
        :class:`ModuleRunner` to avoid redundant disk I/O when the image
        has already been decoded for a prior module.
        """
        self._load_model()
        import torch
        from PIL import Image

        img = img.convert("RGB")
        img.thumbnail((EMBED_MAX_LONG_EDGE, EMBED_MAX_LONG_EDGE), Image.LANCZOS)

        image_input = CLIPEmbedder._preprocess(img).unsqueeze(0).to(CLIPEmbedder._device)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(CLIPEmbedder._device != "cpu")):
            features = CLIPEmbedder._model.encode_image(image_input)
            features = features / features.norm(dim=-1, keepdim=True)

        vec = features.cpu().numpy().flatten().astype(np.float32)
        return vec.tobytes()

    def embed_text(self, text: str) -> bytes:
        """Encode a text query → float32 bytes (768-d)."""
        self._load_model()
        import torch

        tokens = CLIPEmbedder._tokenizer([text]).to(CLIPEmbedder._device)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(CLIPEmbedder._device != "cpu")):
            features = CLIPEmbedder._model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)

        vec = features.cpu().numpy().flatten().astype(np.float32)
        return vec.tobytes()

    @classmethod
    def _load_model(cls) -> None:
        if cls._model is not None:
            return

        try:
            import open_clip
        except ImportError:
            raise ImportError(
                "open_clip is required for CLIP embeddings:\n"
                "  pip install open-clip-torch"
            )
        import torch

        from rich.console import Console
        Console().print("[dim]Loading CLIP ViT-L/14 model...[/dim]")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load weights directly in fp16 on CUDA — saves ~0.75 GB vs fp32 default.
        # Inference already runs under autocast so this is fully consistent.
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14",
            pretrained="openai",
            precision="fp16" if device == "cuda" else "fp32",
            cache_dir=str(Path(CACHE_DIR) / "clip"),
        )
        tokenizer = open_clip.get_tokenizer("ViT-L-14")

        model = model.to(device).eval()
        cls._model = model
        cls._preprocess = preprocess
        cls._tokenizer = tokenizer
        cls._device = device


def vector_from_bytes(data: bytes) -> np.ndarray:
    """Convert raw bytes back to float32 numpy array."""
    return np.frombuffer(data, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D float arrays."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
