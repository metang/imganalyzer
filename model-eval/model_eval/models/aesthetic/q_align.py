"""Q-Align / OneAlign aesthetic scoring adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from model_eval.models.base import ModelAdapter


class QAlignAdapter(ModelAdapter):
    name = "q-align"
    category = "aesthetic"
    model_id = "q-future/one-align"

    def __init__(self) -> None:
        self._model: Any = None
        self._device: str = "cuda"

    def load(self, device: str = "cuda") -> None:
        from transformers import AutoModelForCausalLM

        self._device = device
        self._model = AutoModelForCausalLM.from_pretrained(
            "q-future/one-align",
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        )

    def run(self, image_path: Path) -> dict[str, Any]:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")

        # Q-Align supports task_="aesthetic" for aesthetic scoring
        aesthetic_score = self._model.score(
            [image],
            task_="aesthetic",
            input_="image",
        )

        # Also get quality score for comparison
        quality_score = self._model.score(
            [image],
            task_="quality",
            input_="image",
        )

        return {
            "score": round(float(aesthetic_score[0]), 4),
            "scale": "1-5",
            "details": {
                "aesthetic_score": round(float(aesthetic_score[0]), 4),
                "quality_score": round(float(quality_score[0]), 4),
            },
        }

    def unload(self) -> None:
        del self._model
        self._model = None
        super().unload()
