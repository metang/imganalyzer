"""CLIP ViT-L/14 + MLP aesthetic predictor (Improved Aesthetic Predictor)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from model_eval.models.base import ModelAdapter


class CLIPMLPAestheticAdapter(ModelAdapter):
    name = "clip-mlp"
    category = "aesthetic"
    model_id = "christoph-schuhmann/improved-aesthetic-predictor"

    def __init__(self) -> None:
        self._model: Any = None
        self._clip_model: Any = None
        self._preprocess: Any = None
        self._device: str = "cuda"

    def load(self, device: str = "cuda") -> None:
        import open_clip
        import torch
        import torch.nn as nn
        from huggingface_hub import hf_hub_download

        self._device = device

        # Load CLIP
        self._clip_model, _, self._preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        self._clip_model.eval()
        if device == "cuda" and torch.cuda.is_available():
            self._clip_model = self._clip_model.cuda()

        # Load MLP head
        class MLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(768, 1024),
                    nn.Dropout(0.2),
                    nn.Linear(1024, 128),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.Dropout(0.1),
                    nn.Linear(64, 16),
                    nn.Linear(16, 1),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.layers(x)

        self._model = MLP()
        weights_path = hf_hub_download(
            repo_id="camenduru/improved-aesthetic-predictor",
            filename="sac+logos+ava1-l14-linearMSE.pth",
        )
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        self._model.load_state_dict(state_dict)
        self._model.eval()
        if device == "cuda" and torch.cuda.is_available():
            self._model = self._model.cuda()

    def run(self, image_path: Path) -> dict[str, Any]:
        import torch

        image = self.load_image(image_path)
        image_tensor = self._preprocess(image).unsqueeze(0)
        if self._device == "cuda" and torch.cuda.is_available():
            image_tensor = image_tensor.cuda()

        with torch.inference_mode():
            embedding = self._clip_model.encode_image(image_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            prediction = self._model(embedding).squeeze().cpu().item()

        return {
            "score": round(prediction, 4),
            "scale": "1-10",
            "details": {},
        }

    def unload(self) -> None:
        del self._model
        del self._clip_model
        del self._preprocess
        self._model = None
        self._clip_model = None
        self._preprocess = None
        super().unload()
