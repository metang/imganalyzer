"""Qwen3-VL-8B quantized captioning adapter (via Ollama).

Includes the default adapter plus prompt variants for style tuning.
Register each variant separately in config.py.
"""

from __future__ import annotations

import base64
import io
import json
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib import request

from model_eval.models.base import ModelAdapter

_OLLAMA_URL = "http://localhost:11434"


class Qwen3VLQuantAdapter(ModelAdapter):
    """Base Ollama Qwen3-VL adapter — default verbose prompt."""

    name = "qwen3-vl-int4"
    category = "caption"
    model_id = "qwen3-vl:8b (Ollama Q4)"
    prompt = "Describe this image in detail."

    def __init__(self) -> None:
        self._model_name = "qwen3-vl:8b"

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
        if "<think>" in caption:
            caption = re.sub(r"<think>.*?</think>\s*", "", caption, flags=re.DOTALL).strip()

        return {"caption": caption}

    def _encode_image(self, image_path: Path) -> str:
        """Return base64-encoded JPEG suitable for Ollama (resized to 1280px max)."""
        img = self.load_image(image_path)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()

    def unload(self) -> None:
        pass  # Ollama manages its own GPU memory


# ── Smaller Ollama Qwen3-VL variants ─────────────────────────────────────────


class Qwen3VL2BAdapter(Qwen3VLQuantAdapter):
    """Ollama Qwen3-VL 2B adapter."""

    name = "qwen3-vl-2b"
    model_id = "qwen3-vl:2b (Ollama)"

    def __init__(self) -> None:
        self._model_name = "qwen3-vl:2b"


class Qwen3VL4BAdapter(Qwen3VLQuantAdapter):
    """Ollama Qwen3-VL 4B adapter."""

    name = "qwen3-vl-4b"
    model_id = "qwen3-vl:4b (Ollama)"

    def __init__(self) -> None:
        self._model_name = "qwen3-vl:4b"


# ── Prompt variants ──────────────────────────────────────────────────────────


class Qwen3VLConciseAdapter(Qwen3VLQuantAdapter):
    """Prompt A: direct length constraint."""

    name = "qwen3-vl-concise"
    prompt = (
        "Describe this image in 2-4 sentences. "
        "Be concise and factual. No bullet points, no markdown, no headers."
    )


class Qwen3VLNaturalAdapter(Qwen3VLQuantAdapter):
    """Prompt B: natural prose with role framing."""

    name = "qwen3-vl-natural"
    prompt = (
        "You are a photo librarian writing a brief catalog entry. "
        "Write a short, natural paragraph describing what this image shows — "
        "the main subject, setting, and notable details. "
        "Keep it under 80 words. Plain text only."
    )


class Qwen3VLObserverAdapter(Qwen3VLQuantAdapter):
    """Prompt C: observational style matching cloud AI output."""

    name = "qwen3-vl-observer"
    prompt = (
        "Look at this image and write a concise description of what you see. "
        "Mention the main subject, the setting or background, lighting, "
        "and any notable details. Write 2-4 plain sentences, no formatting."
    )
