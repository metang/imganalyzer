"""Ollama-based AI analysis using Qwen 3.5 VL (local, no API key).

Replaces both BLIP-2 (local captioning) and CloudAI (API-based analysis)
with a single local Ollama call to Qwen 3.5 vision-language model.

Model default: qwen3.5:9b  (configurable via OLLAMA_MODEL env var)
Ollama URL:    http://localhost:11434  (configurable via OLLAMA_URL env var)
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_DEFAULT_MODEL = "qwen3.5:9b"
_DEFAULT_URL = "http://localhost:11434"
_MAX_DIM = 1024  # Optimal resolution for qwen3.5 (validated via resolution eval)
_NUM_PREDICT = 2800  # Must be >=2800 to avoid thinking-token exhaustion
_RETRIES = 3
_TIMEOUT = 300  # seconds per Ollama request


def _encode_image(path: Path, max_dim: int = _MAX_DIM) -> str:
    """Return base64-encoded JPEG string, resized to *max_dim* px."""
    from PIL import Image
    from imganalyzer.readers import open_as_pil

    img = open_as_pil(path)

    w, h = img.size
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode()


def _build_prompt() -> str:
    """Build the full Ollama prompt with /no_think prefix."""
    from imganalyzer.analysis.ai.cloud import SYSTEM_PROMPT_WITH_AESTHETIC

    return f"/no_think\nAnalyze this image.\n\n{SYSTEM_PROMPT_WITH_AESTHETIC}"


def _parse_json_response(text: str) -> dict[str, Any]:
    """Extract JSON from Ollama response, handling markdown fences and thinking tags."""
    text = text.strip()
    # Strip <think>...</think> blocks if present
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip markdown code fences
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"description": text, "keywords": []}


class OllamaAI:
    """Ollama-based vision analysis using qwen3.5 model."""

    def __init__(
        self,
        model: str | None = None,
        url: str | None = None,
    ) -> None:
        self.model = model or os.environ.get("OLLAMA_MODEL", _DEFAULT_MODEL)
        self.url = (url or os.environ.get("OLLAMA_URL", _DEFAULT_URL)).rstrip("/")
        self._prompt = _build_prompt()

    @classmethod
    def unload_model(cls) -> None:
        """Tell Ollama to unload the model from GPU, freeing VRAM."""
        from urllib import request as urllib_request
        from urllib.error import URLError

        model = os.environ.get("OLLAMA_MODEL", _DEFAULT_MODEL)
        url = os.environ.get("OLLAMA_URL", _DEFAULT_URL).rstrip("/")
        payload = json.dumps({
            "model": model,
            "keep_alive": 0,
        }).encode("utf-8")
        try:
            req = urllib_request.Request(
                f"{url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib_request.urlopen(req, timeout=30) as resp:
                resp.read()
            log.info("Ollama model %s unloaded from GPU", model)
        except (URLError, OSError) as exc:
            log.warning("Failed to unload Ollama model: %s", exc)

    def analyze(self, path: Path, image_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze a single image via Ollama.

        Args:
            path: Path to the image file.
            image_data: Pipeline image data dict (rgb_array, etc.). Currently
                        unused — we re-encode from *path* for Ollama at 1024px.

        Returns:
            Dict with description, scene_type, main_subject, lighting, mood,
            keywords, technical_notes, aesthetic_score, aesthetic_label,
            aesthetic_reason.
        """
        b64 = _encode_image(path)
        return self._call_ollama(b64)

    def analyze_batch(
        self,
        paths: list[Path],
        image_data_list: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Analyze a batch of images sequentially via Ollama.

        Ollama doesn't support true batching — each image is a separate call.
        """
        results: list[dict[str, Any]] = []
        for p in paths:
            results.append(self.analyze(p, {}))
        return results

    def _call_ollama(self, b64_image: str) -> dict[str, Any]:
        """Send image to Ollama and parse JSON response with retries."""
        from urllib import request as urllib_request
        from urllib.error import URLError

        payload = json.dumps({
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": self._prompt,
                    "images": [b64_image],
                },
            ],
            "stream": False,
            "options": {
                "num_predict": _NUM_PREDICT,
                "temperature": 0,
            },
        }).encode("utf-8")

        last_text = ""
        for attempt in range(1, _RETRIES + 1):
            start = time.perf_counter()
            try:
                req = urllib_request.Request(
                    f"{self.url}/api/chat",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib_request.urlopen(req, timeout=_TIMEOUT) as resp:
                    body = json.loads(resp.read())
            except (URLError, OSError, json.JSONDecodeError) as exc:
                log.warning(
                    "Ollama request attempt %d/%d failed: %s",
                    attempt, _RETRIES, exc,
                )
                if attempt == _RETRIES:
                    return {"description": "", "keywords": [], "error": str(exc)}
                time.sleep(1)
                continue

            elapsed = time.perf_counter() - start
            text = (body.get("message", {}).get("content") or "").strip()
            last_text = text

            parsed = _parse_json_response(text)
            keywords = parsed.get("keywords", [])
            if isinstance(keywords, list) and len(keywords) > 0:
                log.debug(
                    "Ollama %s: %d keywords, %.1fs (attempt %d)",
                    self.model, len(keywords), elapsed, attempt,
                )
                return self._normalize(parsed)

            log.warning(
                "Ollama %s attempt %d/%d: empty keywords (text=%d chars)",
                self.model, attempt, _RETRIES, len(text),
            )

        log.error(
            "Ollama %s: failed after %d attempts, last_text=%s",
            self.model, _RETRIES, last_text[:200],
        )
        return self._normalize(_parse_json_response(last_text))

    @staticmethod
    def _normalize(parsed: dict[str, Any]) -> dict[str, Any]:
        """Ensure all expected fields exist with proper types."""
        result: dict[str, Any] = {}

        result["description"] = str(parsed.get("description", "")).strip()
        result["scene_type"] = str(parsed.get("scene_type", "")).strip()
        result["main_subject"] = str(parsed.get("main_subject", "")).strip()
        result["lighting"] = str(parsed.get("lighting", "")).strip()
        result["mood"] = str(parsed.get("mood", "")).strip()
        result["technical_notes"] = str(parsed.get("technical_notes", "")).strip()

        # Keywords
        kw = parsed.get("keywords", [])
        if isinstance(kw, list):
            result["keywords"] = [str(k).strip() for k in kw if str(k).strip()]
        else:
            result["keywords"] = []

        # Aesthetic fields
        try:
            result["aesthetic_score"] = float(parsed.get("aesthetic_score", 0))
        except (ValueError, TypeError):
            result["aesthetic_score"] = 0.0
        result["aesthetic_label"] = str(parsed.get("aesthetic_label", "")).strip()
        result["aesthetic_reason"] = str(parsed.get("aesthetic_reason", "")).strip()

        return result
