"""CLIP embedder — image and text embedding for semantic search.

Uses CLIP ViT-L/14 from OpenAI via the ``open_clip`` library.
Embeddings are 768-d float32 vectors stored as raw bytes in the DB.
"""
from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Any

import numpy as np

CACHE_DIR = os.getenv("IMGANALYZER_MODEL_CACHE", str(Path.home() / ".cache" / "imganalyzer"))


class CLIPEmbedder:
    """Lazy-loading CLIP encoder for both images and text queries."""

    _model = None
    _preprocess = None
    _tokenizer = None
    _device = None
    model_version: str = "ViT-L-14/openai"

    def embed_image(self, path: Path | str) -> bytes:
        """Encode an image file → float32 bytes (768-d)."""
        self._load_model()
        import torch
        from PIL import Image

        img = Image.open(str(path)).convert("RGB")
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
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14",
            pretrained="openai",
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
