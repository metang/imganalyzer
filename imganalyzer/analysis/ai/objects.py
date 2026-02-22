"""Object detection using GroundingDINO (zero-shot, open-vocabulary)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

CACHE_DIR = os.getenv("IMGANALYZER_MODEL_CACHE", str(Path.home() / ".cache" / "imganalyzer"))

# Default detection prompt — common photography subjects.
# Each category separated by " . " as required by GroundingDINO.
DEFAULT_DETECTION_PROMPT = (
    "person . animal . dog . cat . bird . horse . vehicle . car . bicycle . "
    "motorcycle . bus . truck . building . house . tree . plant . flower . "
    "food . furniture . chair . table . sky . water . mountain . road . text ."
)

DEFAULT_DETECTION_THRESHOLD = float(
    os.getenv("IMGANALYZER_DETECTION_THRESHOLD", "0.30")
)

# Max image dimension fed to GroundingDINO (speed optimisation)
_MAX_DIM = 800


class ObjectDetector:
    """Zero-shot object detector using GroundingDINO (transformers AutoModel).

    Model: IDEA-Research/grounding-dino-base
    Requires: transformers>=4.38, torch  (pip install 'imganalyzer[local-ai]')
    """

    _processor = None
    _model = None

    def analyze(
        self,
        image_data: dict[str, Any],
        prompt: str | None = None,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        try:
            import torch
            from PIL import Image
        except ImportError:
            raise ImportError(
                "Local AI requires transformers and torch:\n"
                "  pip install 'imganalyzer[local-ai]'"
            )

        detection_prompt = (
            prompt
            or os.getenv("IMGANALYZER_DETECTION_PROMPT")
            or DEFAULT_DETECTION_PROMPT
        )
        conf_threshold = threshold if threshold is not None else DEFAULT_DETECTION_THRESHOLD

        rgb: np.ndarray = image_data["rgb_array"]
        pil_img = Image.fromarray(rgb)

        # Resize for speed — GroundingDINO accuracy is good at 800px
        w, h = pil_img.size
        if max(w, h) > _MAX_DIM:
            scale = _MAX_DIM / max(w, h)
            pil_img = pil_img.resize(
                (int(w * scale), int(h * scale)), Image.LANCZOS
            )

        self._load_models()

        processor = ObjectDetector._processor
        model = ObjectDetector._model
        device = next(model.parameters()).device

        with torch.inference_mode():
            inputs = processor(
                images=pil_img,
                text=detection_prompt,
                return_tensors="pt",
            ).to(device)
            outputs = model(**inputs)
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=conf_threshold,
                text_threshold=conf_threshold,
                target_sizes=[pil_img.size[::-1]],  # (height, width)
            )[0]

        labels: list[str] = []
        has_person = False

        scores = results["scores"].cpu().tolist()
        text_labels = results["labels"]

        for label, score in zip(text_labels, scores):
            label_clean = str(label).strip().lower()
            pct = int(round(score * 100))
            labels.append(f"{label_clean}:{pct}%")
            if label_clean in ("person", "people", "man", "woman", "child", "boy", "girl", "human"):
                has_person = True

        return {
            "detected_objects": labels,
            "has_person": has_person,
        }

    @classmethod
    def _load_models(cls) -> None:
        if cls._model is not None:
            return

        try:
            import torch
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        except ImportError:
            raise ImportError(
                "transformers>=4.38 is required for object detection:\n"
                "  pip install 'imganalyzer[local-ai]'"
            )

        from rich.console import Console
        Console().print("[dim]Loading GroundingDINO model (first run downloads ~700MB)...[/dim]")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "IDEA-Research/grounding-dino-base"

        cls._processor = AutoProcessor.from_pretrained(
            model_id, cache_dir=CACHE_DIR
        )
        cls._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            cache_dir=CACHE_DIR,
        ).to(device)
        cls._model.eval()
