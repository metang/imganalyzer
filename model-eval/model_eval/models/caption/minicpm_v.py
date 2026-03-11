"""MiniCPM-V-2.6 captioning adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from model_eval.models.base import ModelAdapter


class MiniCPMVAdapter(ModelAdapter):
    name = "minicpm-v"
    category = "caption"
    model_id = "openbmb/MiniCPM-V-2_6"

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None

    def load(self, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            extra = {"quantization_config": quantization_config}
        except (ImportError, Exception):
            extra = {}

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self._model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory={0: "14GiB", "cpu": "24GiB"},
            **extra,
        )
        self._model.eval()

    def run(self, image_path: Path) -> dict[str, Any]:
        import torch

        image = self.load_image(image_path)

        question = "Describe this image in detail."
        msgs = [{"role": "user", "content": [image, question]}]

        with torch.inference_mode():
            answer = self._model.chat(
                image=None,
                msgs=msgs,
                tokenizer=self._tokenizer,
                max_new_tokens=512,
            )

        return {"caption": answer.strip()}

    def unload(self) -> None:
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        super().unload()
