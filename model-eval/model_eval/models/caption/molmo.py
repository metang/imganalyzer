"""Molmo-7B-D captioning adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from model_eval.models.base import ModelAdapter


class MolmoAdapter(ModelAdapter):
    name = "molmo"
    category = "caption"
    model_id = "allenai/molmo-7b-d-0924"

    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None

    def load(self, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory={0: "14GiB", "cpu": "24GiB"},
        )

    def run(self, image_path: Path) -> dict[str, Any]:
        import torch

        image = self.load_image(image_path)

        inputs = self._processor.process(
            images=[image],
            text="Describe this image in detail.",
        )
        # Move to device
        inputs = {
            k: v.to(self._model.device).unsqueeze(0) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            output = self._model.generate_from_batch(
                inputs,
                max_new_tokens=512,
                tokenizer=self._processor.tokenizer,
            )
            caption = self._processor.tokenizer.decode(
                output[0], skip_special_tokens=True
            ).strip()

        return {"caption": caption}

    def unload(self) -> None:
        del self._model
        del self._processor
        self._model = None
        self._processor = None
        super().unload()
