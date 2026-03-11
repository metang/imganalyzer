"""Generic Ollama vision-model captioning adapter.

Works with any Ollama model that supports the /api/chat vision endpoint.
Subclass and set `name`, `_model_name`, and optionally `prompt` to create
a new model adapter.
"""

from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path
from typing import Any
from urllib import request

from model_eval.models.base import ModelAdapter

_OLLAMA_URL = "http://localhost:11434"

_NATURAL_PROMPT = (
    "You are a photo librarian writing a brief catalog entry. "
    "Write a short, natural paragraph describing what this image shows — "
    "the main subject, setting, and notable details. "
    "Keep it under 80 words. Plain text only."
)


class OllamaVisionAdapter(ModelAdapter):
    """Generic adapter for any Ollama vision model."""

    name = "ollama-generic"
    category = "caption"
    model_id = ""
    prompt = _NATURAL_PROMPT
    _model_name = ""
    _max_empty_retries = 3
    _min_caption_chars = 24

    def load(self, device: str = "cuda") -> None:
        try:
            resp = request.urlopen(f"{_OLLAMA_URL}/api/tags", timeout=5)
            data = json.loads(resp.read())
            names = [m["name"] for m in data.get("models", [])]
            if not any(self._model_name in n for n in names):
                raise RuntimeError(
                    f"Model {self._model_name} not found in Ollama. "
                    f"Run: ollama pull {self._model_name}"
                )
        except Exception as e:
            if "connection" in str(e).lower() or "urlopen" in str(e).lower():
                raise RuntimeError(
                    "Cannot connect to Ollama at localhost:11434. "
                    "Run: ollama serve"
                ) from e
            raise

    def run(self, image_path: Path) -> dict[str, Any]:
        image_b64 = self._encode_image(image_path)
        for attempt in range(1, self._max_empty_retries + 1):
            payload = json.dumps({
                "model": self._model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": self.prompt,
                        "images": [image_b64],
                    }
                ],
                "stream": False,
                "options": {
                    "num_predict": 1024,
                },
            }).encode()

            req = request.Request(
                f"{_OLLAMA_URL}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read())

            caption = result.get("message", {}).get("content", "").strip()
            # Strip <think> tags from reasoning models
            if "<think>" in caption:
                caption = re.sub(
                    r"<think>.*?</think>\s*", "", caption, flags=re.DOTALL
                ).strip()
            if caption and len(caption) >= self._min_caption_chars:
                return {"caption": caption}

        return {
            "caption": "",
            "error": (
                f"Ollama returned empty/too-short caption for {self._model_name} "
                f"after {self._max_empty_retries} attempts"
            ),
        }

    def _encode_image(self, image_path: Path) -> str:
        """Return base64-encoded JPEG suitable for Ollama (resized to 1280px max)."""
        img = self.load_image(image_path)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()

    def unload(self) -> None:
        pass  # Ollama manages its own GPU memory


# ── Qwen 3.5 VL models ──────────────────────────────────────────────────────


class Qwen35VL2BAdapter(OllamaVisionAdapter):
    name = "qwen3.5-vl-2b"
    model_id = "qwen3.5:2b"
    _model_name = "qwen3.5:2b"


class Qwen35VL4BAdapter(OllamaVisionAdapter):
    name = "qwen3.5-vl-4b"
    model_id = "qwen3.5:4b"
    _model_name = "qwen3.5:4b"


class Qwen35VL9BAdapter(OllamaVisionAdapter):
    name = "qwen3.5-vl-9b"
    model_id = "qwen3.5:9b"
    _model_name = "qwen3.5:9b"


# ── Qwen 2.5 VL models ──────────────────────────────────────────────────────


class Qwen25VL3BAdapter(OllamaVisionAdapter):
    name = "qwen2.5-vl-3b"
    model_id = "qwen2.5vl:3b"
    _model_name = "qwen2.5vl:3b"


class Qwen25VL7BAdapter(OllamaVisionAdapter):
    name = "qwen2.5-vl-7b"
    model_id = "qwen2.5vl:7b"
    _model_name = "qwen2.5vl:7b"
