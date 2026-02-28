"""Local AI analysis using HuggingFace BLIP-2 (offline, no API key).

Model: Salesforce/blip2-flan-t5-xl (~8 GB)
- FlanT5 encoder-decoder backbone handles VQA reliably
- Captioning: image-only input → encoder+decoder generates description
- VQA: image+question input → decoder answers from scratch (no prompt-slicing needed)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


CACHE_DIR = os.getenv("IMGANALYZER_MODEL_CACHE", str(Path.home() / ".cache" / "imganalyzer"))

# Model ID — FlanT5-XL gives reliable VQA answers; opt-2.7b only works for captioning
_MODEL_ID = "Salesforce/blip2-flan-t5-xl"


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
            Console().print(f"[dim]Loading BLIP-2 model {_MODEL_ID} (first run downloads ~8 GB)...[/dim]")
            LocalAI._processor = Blip2Processor.from_pretrained(
                _MODEL_ID, cache_dir=CACHE_DIR
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            LocalAI._model = Blip2ForConditionalGeneration.from_pretrained(
                _MODEL_ID,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                cache_dir=CACHE_DIR,
            ).to(device)

        processor = LocalAI._processor
        model = LocalAI._model
        device = next(model.parameters()).device

        results: dict[str, Any] = {}

        # 1. Image captioning (no text prompt → pure captioning mode)
        # inference_mode disables autograd entirely — no activation graph retained,
        # saving 2–3 GB vs bare generate() calls.
        with torch.inference_mode():
            inputs = processor(pil_img, return_tensors="pt").to(device)
            out = model.generate(**inputs, max_new_tokens=100)
            caption = processor.decode(out[0], skip_special_tokens=True).strip()
        results["description"] = caption

        # 2. VQA — batch all 4 questions into a single forward pass.
        # The ViT-L image encoder output is reused across all questions
        # (the processor duplicates the pixel values internally), and
        # a single model.generate() call amortises the CUDA launch overhead.
        # Removes the per-question empty_cache() calls that caused unnecessary
        # CUDA synchronisation stalls (~4x fewer forward passes).
        vqa_questions = [
            ("What type of scene is this? Answer in 1-3 words.", "scene_type"),
            ("What is the main subject of this image? Answer in 1-5 words.", "main_subject"),
            ("What is the lighting condition or time of day? Answer in 1-3 words.", "lighting"),
            ("What is the mood or aesthetic of this image? Answer in 1-3 words.", "mood"),
        ]

        try:
            with torch.inference_mode():
                # Batch: replicate the image for each question and process together.
                questions = [q for q, _ in vqa_questions]
                images = [pil_img] * len(questions)
                inputs = processor(images, questions, return_tensors="pt", padding=True).to(device)
                outputs = model.generate(**inputs, max_new_tokens=30)

                for idx, (_, key) in enumerate(vqa_questions):
                    answer = processor.decode(outputs[idx], skip_special_tokens=True).strip()
                    if answer:
                        results[key] = answer
        except Exception:
            # Fallback: sequential VQA if batched inference fails (e.g. OOM)
            for question, key in vqa_questions:
                try:
                    with torch.inference_mode():
                        inputs = processor(pil_img, question, return_tensors="pt").to(device)
                        out = model.generate(**inputs, max_new_tokens=30)
                        answer = processor.decode(out[0], skip_special_tokens=True).strip()
                    if answer:
                        results[key] = answer
                except Exception:
                    pass

        # Free activation tensors once after all VQA is done (not between questions).
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 3. Keywords from caption
        results["keywords"] = _extract_keywords(caption)

        return results


def _extract_keywords(text: str) -> list[str]:
    """Extract simple keyword tags from text."""
    stopwords = {"a", "an", "the", "is", "are", "was", "were", "with", "of", "in", "on",
                 "at", "to", "and", "or", "this", "that", "it", "its"}
    words = text.lower().replace(",", " ").replace(".", " ").split()
    return list(dict.fromkeys(w for w in words if w not in stopwords and len(w) > 2))[:20]
