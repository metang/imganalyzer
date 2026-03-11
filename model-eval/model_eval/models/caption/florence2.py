"""Florence-2-large-ft captioning adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from model_eval.models.base import ModelAdapter


class Florence2Adapter(ModelAdapter):
    name = "florence2"
    category = "caption"
    model_id = "microsoft/Florence-2-large-ft"

    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None

    def load(self, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True, use_fast=False
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            max_memory={0: "14GiB", "cpu": "24GiB"} if device == "cuda" else None,
        )

    def run(self, image_path: Path) -> dict[str, Any]:
        import torch

        image = self.load_image(image_path)

        # Florence-2 uses task prompts — <MORE_DETAILED_CAPTION> for richest output
        prompt = "<MORE_DETAILED_CAPTION>"
        inputs = self._processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=1024,
                num_beams=3,
            )

        caption = self._processor.batch_decode(
            output_ids, skip_special_tokens=False
        )[0]

        # Florence-2 wraps output in task tokens; extract the text
        caption = self._processor.post_process_generation(
            caption, task=prompt, image_size=image.size
        )
        if isinstance(caption, dict):
            caption = caption.get(prompt, str(caption))

        return {"caption": str(caption).strip()}

    def unload(self) -> None:
        del self._model
        del self._processor
        self._model = None
        self._processor = None
        super().unload()
