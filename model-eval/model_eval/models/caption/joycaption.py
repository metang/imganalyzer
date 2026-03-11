"""JoyCaption Alpha Two captioning adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from model_eval.models.base import ModelAdapter


class JoyCaptionAdapter(ModelAdapter):
    name = "joycaption"
    category = "caption"
    model_id = "fancyfeast/llama-joycaption-alpha-two-hf-llava"

    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None

    def load(self, device: str = "cuda") -> None:
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            extra = {"quantization_config": quantization_config}
        except (ImportError, Exception):
            extra = {}

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory={0: "14GiB", "cpu": "24GiB"},
            **extra,
        )

    def run(self, image_path: Path) -> dict[str, Any]:
        import torch

        image = self.load_image(image_path)

        prompt = (
            "Write a descriptive caption for this image in a formal tone. "
            "Describe the image in detail."
        )

        # Use raw LLaVA prompt format (chat template is broken with transformers 5.0)
        text = f"USER: <image>\n{prompt}\nASSISTANT:"
        inputs = self._processor(
            text=text, images=image, return_tensors="pt"
        ).to(self._model.device)

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
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
