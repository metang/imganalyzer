"""BLIP2 captioning adapter (baseline)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from model_eval.models.base import ModelAdapter


class BLIP2Adapter(ModelAdapter):
    name = "blip2"
    category = "caption"
    model_id = "Salesforce/blip2-flan-t5-xl"

    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None
        self._device: str = "cuda"

    def load(self, device: str = "cuda") -> None:
        import torch
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        self._device = device
        self._processor = Blip2Processor.from_pretrained(self.model_id)
        self._model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            max_memory={0: "14GiB", "cpu": "24GiB"} if device == "cuda" else None,
        )

    def run(self, image_path: Path) -> dict[str, Any]:
        import torch

        image = self.load_image(image_path)
        prompt = "Describe this image in detail."
        inputs = self._processor(image, text=prompt, return_tensors="pt").to(
            self._model.device, dtype=torch.float16 if self._device == "cuda" else torch.float32
        )

        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, max_new_tokens=256)

        caption = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return {"caption": caption}

    def unload(self) -> None:
        del self._model
        del self._processor
        self._model = None
        self._processor = None
        super().unload()
