"""SigLIP-based Aesthetic Predictor V2.5 adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from model_eval.models.base import ModelAdapter


class SigLIPAestheticAdapter(ModelAdapter):
    name = "siglip-v2.5"
    category = "aesthetic"
    model_id = "discus0434/aesthetic-predictor-v2-5"

    def __init__(self) -> None:
        self._model: Any = None
        self._preprocessor: Any = None
        self._device: str = "cuda"

    def load(self, device: str = "cuda") -> None:
        import torch
        from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip

        self._device = device
        self._model, self._preprocessor = convert_v2_5_from_siglip(
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        if device == "cuda" and torch.cuda.is_available():
            self._model = self._model.to(torch.bfloat16).cuda()

    def run(self, image_path: Path) -> dict[str, Any]:
        import torch

        image = self.load_image(image_path)
        inputs = self._preprocessor(images=image, return_tensors="pt")
        if self._device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.to(torch.bfloat16).cuda() for k, v in inputs.items()}

        with torch.inference_mode():
            prediction = self._model(**inputs).logits.squeeze().float().cpu().item()

        return {
            "score": round(prediction, 4),
            "scale": "1-10",
            "details": {},
        }

    def unload(self) -> None:
        del self._model
        del self._preprocessor
        self._model = None
        self._preprocessor = None
        super().unload()
