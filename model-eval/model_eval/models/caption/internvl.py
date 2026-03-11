"""InternVL2.5-8B captioning adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from model_eval.models.base import ModelAdapter


class InternVLAdapter(ModelAdapter):
    name = "internvl"
    category = "caption"
    model_id = "OpenGVLab/InternVL2_5-8B"

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None

    def load(self, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self._model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory={0: "14GiB", "cpu": "24GiB"},
        )
        self._model.eval()

    def run(self, image_path: Path) -> dict[str, Any]:
        import torch

        image = self.load_image(image_path)

        # InternVL2.5 uses a chat interface
        question = "<image>\nDescribe this image in detail."
        pixel_values = self._load_image(image)

        generation_config = {
            "max_new_tokens": 512,
            "do_sample": False,
        }

        with torch.inference_mode():
            response = self._model.chat(
                self._tokenizer,
                pixel_values,
                question,
                generation_config,
            )

        return {"caption": response.strip()}

    def _load_image(self, image: Any) -> Any:
        """Process image for InternVL format."""
        import torch
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        transform = T.Compose([
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        pixel_values = transform(image).unsqueeze(0).to(
            self._model.device, dtype=torch.bfloat16
        )
        return pixel_values

    def unload(self) -> None:
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        super().unload()
