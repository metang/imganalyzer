"""SigLIP-v2.5 aesthetic scoring for the imganalyzer pipeline.

Uses ``aesthetic-predictor-v2-5`` (SigLIP backbone) to produce a 1-10
aesthetic quality score.  Replaces cloud-derived aesthetic scoring.

Install: ``pip install aesthetic-predictor-v2-5``
VRAM:    ~1.5 GB in bfloat16
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_SCORE_LABELS: list[tuple[float, str]] = [
    (3.0, "Poor"),
    (5.0, "Average"),
    (7.0, "Good"),
    (9.0, "Excellent"),
    (10.0, "Masterpiece"),
]


def _score_to_label(score: float) -> str:
    """Map a 0–10 score to a human-readable label."""
    for threshold, label in _SCORE_LABELS:
        if score <= threshold:
            return label
    return "Masterpiece"


class SigLIPAesthetic:
    """SigLIP-v2.5 aesthetic predictor for the pipeline."""

    _model: Any = None
    _preprocessor: Any = None
    _device: str = "cpu"

    @classmethod
    def load(cls, device: str = "cuda") -> None:
        """Load the SigLIP aesthetic model onto *device*."""
        if cls._model is not None:
            return  # already loaded

        import torch
        try:
            from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
        except ImportError as exc:
            raise ImportError(
                "SigLIP aesthetic dependency missing. Install with "
                "`pip install aesthetic-predictor-v2-5` "
                "or `pip install -e \".[local-ai]\"`."
            ) from exc

        cls._device = device
        cls._model, cls._preprocessor = convert_v2_5_from_siglip(
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        if device == "cuda" and torch.cuda.is_available():
            cls._model = cls._model.to(torch.bfloat16).cuda()
        log.info("SigLIP aesthetic model loaded on %s", device)

    @classmethod
    def analyze(cls, path: Path) -> dict[str, Any]:
        """Score a single image.

        Returns dict with aesthetic_score (0-10), aesthetic_label,
        aesthetic_reason (empty — SigLIP doesn't explain), and provider.
        """
        import torch
        from imganalyzer.readers import open_as_pil

        if cls._model is None:
            cls.load()

        img = open_as_pil(path)

        inputs = cls._preprocessor(images=img, return_tensors="pt")
        if cls._device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.to(torch.bfloat16).cuda() for k, v in inputs.items()}

        with torch.inference_mode():
            score = cls._model(**inputs).logits.squeeze().float().cpu().item()

        score = round(max(0.0, min(10.0, score)), 2)
        label = _score_to_label(score)

        return {
            "aesthetic_score": score,
            "aesthetic_label": label,
            "aesthetic_reason": "",
            "provider": "siglip-v2.5",
        }

    @classmethod
    def unload(cls) -> None:
        """Free GPU memory."""
        import torch
        import gc

        del cls._model
        del cls._preprocessor
        cls._model = None
        cls._preprocessor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("SigLIP aesthetic model unloaded")
