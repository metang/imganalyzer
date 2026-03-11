"""Gemma3-12B captioning adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from model_eval.models.base import ModelAdapter


class Gemma3Adapter(ModelAdapter):
    name = "gemma3"
    category = "caption"
    model_id = "google/gemma-3-12b-it"

    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None

    def load(self, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory={0: "14GiB", "cpu": "24GiB"},
        )

    def run(self, image_path: Path) -> dict[str, Any]:
        import torch

        image = self.load_image(image_path)
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image in detail."},
            ]},
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, max_new_tokens=512)
            generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            caption = self._processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

        return {"caption": caption}

    def unload(self) -> None:
        del self._model
        del self._processor
        self._model = None
        self._processor = None
        super().unload()
