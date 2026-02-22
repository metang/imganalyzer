"""Local AI analysis using HuggingFace BLIP-2 (offline, no API key)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


CACHE_DIR = os.getenv("IMGANALYZER_MODEL_CACHE", str(Path.home() / ".cache" / "imganalyzer"))


class LocalAI:
    _processor = None
    _model = None

    def analyze(self, image_data: dict[str, Any]) -> dict[str, Any]:
        try:
            from PIL import Image
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            import torch
        except ImportError:
            raise ImportError(
                "Local AI requires transformers and torch:\n"
                "  pip install 'imganalyzer[local-ai]'"
            )

        rgb: np.ndarray = image_data["rgb_array"]
        pil_img = Image.fromarray(rgb)

        # Load model lazily (cached after first load)
        if LocalAI._processor is None:
            from rich.console import Console
            Console().print("[dim]Loading BLIP-2 model (first run downloads ~3GB)...[/dim]")
            LocalAI._processor = Blip2Processor.from_pretrained(
                "Salesforce/blip2-opt-2.7b", cache_dir=CACHE_DIR
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            LocalAI._model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                cache_dir=CACHE_DIR,
            ).to(device)

        processor = LocalAI._processor
        model = LocalAI._model
        device = next(model.parameters()).device

        results: dict[str, Any] = {}

        # 1. Image captioning
        inputs = processor(pil_img, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=100)
        caption = processor.decode(out[0], skip_special_tokens=True).strip()
        results["description"] = caption

        # 2. VQA: scene type
        for question, key in [
            ("What type of scene is this? (landscape, portrait, street, architecture, macro, etc.)", "scene_type"),
            ("What is the main subject of this image?", "main_subject"),
            ("What time of day or lighting condition is present?", "lighting"),
            ("What is the mood or aesthetic of this image?", "mood"),
        ]:
            try:
                inputs = processor(pil_img, question, return_tensors="pt").to(device)
                out = model.generate(**inputs, max_new_tokens=50)
                answer = processor.decode(out[0], skip_special_tokens=True).strip()
                results[key] = answer
            except Exception:
                pass

        # 3. Keywords from caption
        results["keywords"] = _extract_keywords(caption)

        return results


def _extract_keywords(text: str) -> list[str]:
    """Extract simple keyword tags from text."""
    stopwords = {"a", "an", "the", "is", "are", "was", "were", "with", "of", "in", "on",
                 "at", "to", "and", "or", "this", "that", "it", "its"}
    words = text.lower().replace(",", " ").replace(".", " ").split()
    return list(dict.fromkeys(w for w in words if w not in stopwords and len(w) > 2))[:20]
