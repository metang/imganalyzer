"""Qwen3-VL-8B captioning adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from model_eval.models.base import ModelAdapter


class Qwen3VLAdapter(ModelAdapter):
    name = "qwen3-vl"
    category = "caption"
    model_id = "Qwen/Qwen3-VL-8B-Instruct"

    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None

    def load(self, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
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

        text = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self._processor(
            text=[text], images=[image], return_tensors="pt", padding=True
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
