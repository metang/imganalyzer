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
from PIL import Image


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

    @classmethod
    def _unload(cls) -> None:
        """Unload CLIP model from GPU to free VRAM.

        Called by the worker between GPU passes so that only the model
        needed for the current pass is resident.  The model will be
        lazily reloaded on the next ``embed_*()`` call if needed.
        """
        if cls._model is not None:
            del cls._model
            cls._model = None
        cls._preprocess = None
        cls._tokenizer = None
        cls._device = None
        try:
            from imganalyzer.device import empty_cache
            empty_cache()
        except Exception:
            pass

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

        from imganalyzer.readers import open_as_pil
        img = open_as_pil(path)

        # Downsize to EMBED_MAX_LONG_EDGE on the long edge before CLIP pre-processing.
        # CLIP rescales to 224 px internally, so there is no quality loss.
        # thumbnail() only ever shrinks — it never upscales a small image.
        img.thumbnail((EMBED_MAX_LONG_EDGE, EMBED_MAX_LONG_EDGE), Image.LANCZOS)

        image_input = CLIPEmbedder._preprocess(img).unsqueeze(0).to(CLIPEmbedder._device)

        with torch.no_grad(), torch.autocast(CLIPEmbedder._device, enabled=(CLIPEmbedder._device != "cpu")):
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

        with torch.no_grad(), torch.autocast(CLIPEmbedder._device, enabled=(CLIPEmbedder._device != "cpu")):
            features = CLIPEmbedder._model.encode_image(image_input)
            features = features / features.norm(dim=-1, keepdim=True)

        vec = features.cpu().numpy().flatten().astype(np.float32)
        return vec.tobytes()

    def embed_images_batch(self, images: "list[Image.Image]") -> list[bytes]:
        """Encode a batch of pre-loaded PIL Images → list of float32 bytes (768-d each).

        Processes up to ``batch_size`` images in a single GPU forward pass,
        dramatically improving throughput by amortising CUDA launch overhead
        and saturating the GPU's tensor cores.

        At batch_size=32 with ViT-L/14, activation memory is ~450 MB —
        safe within the 14 GB VRAM ceiling even with model weights (~0.45 GB).
        """
        if not images:
            return []

        self._load_model()
        import torch
        from PIL import Image

        # Pre-process all images: resize + CLIP transforms → stacked tensor
        tensors = []
        for img in images:
            img = img.convert("RGB")
            img.thumbnail((EMBED_MAX_LONG_EDGE, EMBED_MAX_LONG_EDGE), Image.LANCZOS)
            tensors.append(CLIPEmbedder._preprocess(img))

        # Stack into (N, C, H, W) batch tensor
        batch_input = torch.stack(tensors).to(CLIPEmbedder._device)

        with torch.no_grad(), torch.autocast(CLIPEmbedder._device, enabled=(CLIPEmbedder._device != "cpu")):
            features = CLIPEmbedder._model.encode_image(batch_input)
            features = features / features.norm(dim=-1, keepdim=True)

        # Split back into per-image embedding bytes
        vecs = features.cpu().numpy().astype(np.float32)
        return [vecs[i].tobytes() for i in range(vecs.shape[0])]

    def embed_text(self, text: str) -> bytes:
        """Encode a text query → float32 bytes (768-d)."""
        self._load_model()
        import torch

        tokens = CLIPEmbedder._tokenizer([text]).to(CLIPEmbedder._device)

        with torch.no_grad(), torch.autocast(CLIPEmbedder._device, enabled=(CLIPEmbedder._device != "cpu")):
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

        from imganalyzer.device import get_device, supports_fp16
        device = get_device()
        # Load weights directly in fp16 on GPU — saves ~0.75 GB vs fp32 default.
        # Inference already runs under autocast so this is fully consistent.
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14",
            pretrained="openai",
            precision="fp16" if supports_fp16() else "fp32",
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
