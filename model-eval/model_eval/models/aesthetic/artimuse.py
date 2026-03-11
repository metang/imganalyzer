"""ArtiMuse fine-grained image aesthetics assessment adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from model_eval.models.base import ModelAdapter


class ArtiMuseAdapter(ModelAdapter):
    name = "artimuse"
    category = "aesthetic"
    model_id = "Thunderbolt215215/ArtiMuse"

    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None
        self._device: str = "cuda"

    def load(self, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._device = device
        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self._model.eval()

    def run(self, image_path: Path) -> dict[str, Any]:
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")

        # ArtiMuse supports multi-attribute aesthetic assessment
        prompt = (
            "Evaluate this image's aesthetic quality. Score each attribute from 1-10 "
            "and provide an overall score:\n"
            "- Composition\n- Color harmony\n- Lighting\n- Subject\n"
            "- Mood/Atmosphere\n- Technical quality\n- Creativity\n- Overall"
        )
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]},
        ]

        text = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(
            text=[text], images=[image], return_tensors="pt", padding=True
        ).to(self._model.device)

        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, max_new_tokens=512)
            generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            response = self._processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

        # Extract overall score
        score = self._extract_score(response)
        attributes = self._extract_attributes(response)

        return {
            "score": score,
            "scale": "1-10",
            "response": response,
            "details": {"attributes": attributes},
        }

    @staticmethod
    def _extract_score(text: str) -> float:
        """Extract overall score from response."""
        import re

        patterns = [
            r"overall\s*(?::|score|rating)?\s*(?::|is|=)?\s*(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*/\s*10",
        ]
        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                val = float(match.group(1))
                if 0 <= val <= 10:
                    return round(val, 2)
        return -1.0

    @staticmethod
    def _extract_attributes(text: str) -> dict[str, float]:
        """Extract per-attribute scores from response."""
        import re

        attrs: dict[str, float] = {}
        for attr in [
            "composition", "color", "lighting", "subject",
            "mood", "atmosphere", "technical", "creativity",
        ]:
            match = re.search(
                rf"{attr}\w*\s*(?::|=)?\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE
            )
            if match:
                attrs[attr] = float(match.group(1))
        return attrs

    def unload(self) -> None:
        del self._model
        del self._processor
        self._model = None
        self._processor = None
        super().unload()
