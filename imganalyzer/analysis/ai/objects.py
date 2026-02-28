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

    @classmethod
    def _unload(cls) -> None:
        """Unload GroundingDINO model from GPU to free VRAM.

        Called by the worker between GPU passes so that only the model
        needed for the current pass is resident.  The model will be
        lazily reloaded on the next ``analyze()`` call if needed.
        """
        if cls._model is not None:
            del cls._model
            cls._model = None
        if cls._processor is not None:
            del cls._processor
            cls._processor = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

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

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            inputs = processor(
                images=pil_img,
                text=detection_prompt,
                return_tensors="pt",
            ).to(device)
            outputs = model(**inputs)
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=conf_threshold,
                text_threshold=conf_threshold,
                target_sizes=[pil_img.size[::-1]],  # (height, width)
            )[0]

        labels: list[str] = []
        has_person = False
        has_text = False
        text_boxes: list[list[float]] = []

        scores = results["scores"].cpu().tolist()
        boxes = results["boxes"].cpu().tolist()
        # Use "text_labels" (string names) — "labels" returns int IDs in transformers>=4.51
        text_labels = results.get("text_labels") or results.get("labels") or []

        for label, score, box in zip(text_labels, scores, boxes):
            label_clean = str(label).strip().lower()
            pct = int(round(score * 100))
            labels.append(f"{label_clean}:{pct}%")
            if label_clean in ("person", "people", "man", "woman", "child", "boy", "girl", "human"):
                has_person = True
            if label_clean == "text":
                has_text = True
                text_boxes.append(box)

        return {
            "detected_objects": labels,
            "has_person": has_person,
            "has_text": has_text,
            "text_boxes": text_boxes,
        }

    def analyze_batch(
        self,
        image_data_list: list[dict[str, Any]],
        prompt: str | None = None,
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Run GroundingDINO on a batch of images in a single forward pass.

        At batch_size=8 with 800px images, activation memory is ~2.4 GB —
        safe within the 14 GB VRAM ceiling with model unloading active.

        Falls back to per-image ``analyze()`` on OOM or other errors.
        """
        if not image_data_list:
            return []
        if len(image_data_list) == 1:
            return [self.analyze(image_data_list[0], prompt=prompt, threshold=threshold)]

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

        # Prepare all PIL images (resize to 800px for speed)
        pil_images: list[Image.Image] = []
        for image_data in image_data_list:
            rgb: np.ndarray = image_data["rgb_array"]
            pil_img = Image.fromarray(rgb)
            w, h = pil_img.size
            if max(w, h) > _MAX_DIM:
                scale = _MAX_DIM / max(w, h)
                pil_img = pil_img.resize(
                    (int(w * scale), int(h * scale)), Image.LANCZOS
                )
            pil_images.append(pil_img)

        self._load_models()
        processor = ObjectDetector._processor
        model = ObjectDetector._model
        device = next(model.parameters()).device

        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                inputs = processor(
                    images=pil_images,
                    text=[detection_prompt] * len(pil_images),
                    return_tensors="pt",
                    padding=True,
                ).to(device)
                outputs = model(**inputs)
                target_sizes = [img.size[::-1] for img in pil_images]  # (height, width) each
                batch_results = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs["input_ids"],
                    threshold=conf_threshold,
                    text_threshold=conf_threshold,
                    target_sizes=target_sizes,
                )
        except Exception:
            # OOM or batching failure — fall back to sequential processing
            return [self.analyze(d, prompt=prompt, threshold=threshold) for d in image_data_list]

        # Parse per-image results
        all_results: list[dict[str, Any]] = []
        for results in batch_results:
            labels: list[str] = []
            has_person = False
            has_text = False
            text_boxes: list[list[float]] = []

            scores = results["scores"].cpu().tolist()
            boxes = results["boxes"].cpu().tolist()
            text_labels = results.get("text_labels") or results.get("labels") or []

            for label, score, box in zip(text_labels, scores, boxes):
                label_clean = str(label).strip().lower()
                pct = int(round(score * 100))
                labels.append(f"{label_clean}:{pct}%")
                if label_clean in ("person", "people", "man", "woman", "child", "boy", "girl", "human"):
                    has_person = True
                if label_clean == "text":
                    has_text = True
                    text_boxes.append(box)

            all_results.append({
                "detected_objects": labels,
                "has_person": has_person,
                "has_text": has_text,
                "text_boxes": text_boxes,
            })

        return all_results

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
        # Load in float32 first — GroundingDINO's text enhancer layers
        # have internal operations that fail with fp16 weights.
        cls._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            cache_dir=CACHE_DIR,
        ).to(device)
        cls._model.eval()

        # Selectively convert the visual backbone (Swin Transformer) to
        # fp16 on CUDA.  The text enhancer layers in the encoder/decoder
        # must stay fp32 (they mix dtype-sensitive ops that crash with
        # fp16 weights), but the Swin backbone is a standard vision
        # transformer that works fine in fp16.  Saves ~0.35 GB VRAM.
        # Inference already runs under autocast(fp16) so this is
        # consistent; we are just reducing static weight memory.
        if device == "cuda":
            try:
                cls._model.model.backbone.half()
            except Exception:
                pass  # Fallback: keep full fp32 — no harm done
