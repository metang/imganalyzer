"""Aesthetic scoring using LAION Aesthetic Predictor V2 (CLIP-based)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

CACHE_DIR = os.getenv("IMGANALYZER_MODEL_CACHE", str(Path.home() / ".cache" / "imganalyzer"))

# Aesthetic score label breakpoints (0–10 scale)
_AESTHETIC_THRESHOLDS = [
    (3.0, "Very Low"),
    (5.0, "Low"),
    (6.5, "Medium"),
    (8.0, "High"),
    (10.0, "Exceptional"),
]


def _aesthetic_label(score: float) -> str:
    for threshold, label in _AESTHETIC_THRESHOLDS:
        if score <= threshold:
            return label
    return "Exceptional"


class AestheticScorer:
    """LAION Aesthetic Predictor V2 — outputs a 0–10 aesthetic quality score.

    Model: shunk031/aesthetic-predictor-v2-sac-logos-ava1-l14-linearMSE
    Requires: open_clip_torch, torch  (pip install 'imganalyzer[local-ai]')
    """

    _model = None
    _preprocess = None
    _clip_model = None

    def analyze(self, image_data: dict[str, Any]) -> dict[str, Any]:
        try:
            import torch
            from PIL import Image
        except ImportError:
            raise ImportError(
                "Local AI requires torch and open_clip_torch:\n"
                "  pip install 'imganalyzer[local-ai]'"
            )

        rgb: np.ndarray = image_data["rgb_array"]
        pil_img = Image.fromarray(rgb)

        self._load_models()

        clip_model = AestheticScorer._clip_model
        preprocess = AestheticScorer._preprocess
        model = AestheticScorer._model
        device = next(model.parameters()).device

        with torch.inference_mode():
            image_tensor = preprocess(pil_img).unsqueeze(0).to(device)
            # Extract CLIP image features
            image_features = clip_model.encode_image(image_tensor)
            # Normalise
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.to(torch.float32)
            # Predict aesthetic score
            score_tensor = model(image_features)
            score = float(score_tensor.squeeze())

        # Clamp to [0, 10]
        score = max(0.0, min(10.0, score))

        return {
            "aesthetic_score": round(score, 2),
            "aesthetic_label": _aesthetic_label(score),
        }

    @classmethod
    def _load_models(cls) -> None:
        if cls._model is not None:
            return

        try:
            import torch
            import open_clip
        except ImportError:
            raise ImportError(
                "open_clip_torch is required for aesthetic scoring:\n"
                "  pip install 'imganalyzer[local-ai]'"
            )

        from rich.console import Console
        Console().print("[dim]Loading Aesthetic Predictor V2 model...[/dim]")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load CLIP ViT-L/14 backbone (same as used by the aesthetic predictor)
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14",
            pretrained="openai",
            cache_dir=CACHE_DIR,
        )
        clip_model = clip_model.to(device)
        clip_model.eval()

        # Load the aesthetic linear head from HuggingFace
        aesthetic_model = _AestheticLinearModel(768)
        _load_aesthetic_weights(aesthetic_model, device)
        aesthetic_model = aesthetic_model.to(device)
        aesthetic_model.eval()

        cls._clip_model = clip_model
        cls._preprocess = preprocess
        cls._model = aesthetic_model


def _load_aesthetic_weights(model: "Any", device: str) -> None:
    """Download and load the aesthetic predictor weights from HuggingFace."""
    import torch
    from pathlib import Path as _Path

    cache_path = _Path(CACHE_DIR) / "aesthetic_predictor_v2.pth"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not cache_path.exists():
        try:
            from huggingface_hub import hf_hub_download
            weights_path = hf_hub_download(
                repo_id="shunk031/aesthetic-predictor-v2-sac-logos-ava1-l14-linearMSE",
                filename="pytorch_model.bin",
                cache_dir=CACHE_DIR,
            )
        except Exception:
            # Fallback: direct URL download
            import urllib.request
            url = (
                "https://huggingface.co/shunk031/aesthetic-predictor-v2-sac-logos-ava1-l14-linearMSE"
                "/resolve/main/pytorch_model.bin"
            )
            urllib.request.urlretrieve(url, str(cache_path))
            weights_path = str(cache_path)

        # Copy to our cache path for future use
        import shutil
        if str(weights_path) != str(cache_path):
            shutil.copy2(weights_path, cache_path)
    else:
        weights_path = str(cache_path)

    state = torch.load(weights_path, map_location=device, weights_only=True)
    # The HF model saves as a plain OrderedDict of linear weights
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)


class _AestheticLinearModel(object if True else None):
    """Simple linear head on top of CLIP features (768 → 1)."""

    def __new__(cls, input_size: int) -> Any:
        import torch
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self, input_size: int) -> None:
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_size, 1024),
                    nn.Dropout(0.2),
                    nn.Linear(1024, 128),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.Dropout(0.1),
                    nn.Linear(64, 16),
                    nn.Linear(16, 1),
                )

            def forward(self, x: Any) -> Any:
                return self.layers(x)

        return _Model(input_size)
