"""OCR using Microsoft TrOCR (transformer-based, offline).

Model: microsoft/trocr-large-printed
- VisionEncoderDecoderModel (ViT encoder + RoBERTa decoder)
- Optimised for printed text; good on signs, labels, overlays, captions
- ~1.3 GB download, runs on CPU or CUDA

Pipeline:
  1. Receive optional bounding boxes from GroundingDINO (text regions).
  2. Crop each region, pad to square, run TrOCR.
  3. If no boxes are supplied, run one pass on the full image resized to 384px.
  4. Deduplicate lines, join into a single string, return as ``ocr_text``.
  5. If no text is read, return an empty dict (caller skips writing to XMP).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

CACHE_DIR = os.getenv("IMGANALYZER_MODEL_CACHE", str(Path.home() / ".cache" / "imganalyzer"))

_MODEL_ID = "microsoft/trocr-large-printed"

# Minimum crop dimension — ignore tiny boxes (likely false detections)
_MIN_CROP_PX = 32

# TrOCR canonical input size
_INPUT_SIZE = 384


class OCRAnalyzer:
    """Extract text from images using Microsoft TrOCR.

    Model is loaded once per process (class-level singleton) and shared across
    all images in a batch.

    ``analyze()`` accepts an optional *text_boxes* list of ``[x0, y0, x1, y1]``
    bounding boxes (pixel coordinates in the original image) from GroundingDINO.
    When provided, each box is cropped and fed to TrOCR individually for
    higher accuracy than a full-image pass.
    """

    _processor = None
    _model = None

    def analyze(
        self,
        image_data: dict[str, Any],
        text_boxes: list[list[float]] | None = None,
    ) -> dict[str, Any]:
        try:
            import torch
            from PIL import Image
        except ImportError:
            raise ImportError(
                "Local AI requires transformers and torch:\n"
                "  pip install 'imganalyzer[local-ai]'"
            )

        rgb: np.ndarray = image_data["rgb_array"]
        pil_full = Image.fromarray(rgb)

        self._load_models()
        processor = OCRAnalyzer._processor
        model = OCRAnalyzer._model
        device = next(model.parameters()).device  # type: ignore[union-attr]

        regions: list[Image.Image] = []

        if text_boxes:
            w_full, h_full = pil_full.size
            for box in text_boxes:
                x0, y0, x1, y1 = (
                    max(0, int(box[0])),
                    max(0, int(box[1])),
                    min(w_full, int(box[2])),
                    min(h_full, int(box[3])),
                )
                if (x1 - x0) < _MIN_CROP_PX or (y1 - y0) < _MIN_CROP_PX:
                    continue
                # Add a 10 % padding around the crop for context
                pad_x = int((x1 - x0) * 0.10)
                pad_y = int((y1 - y0) * 0.10)
                x0 = max(0, x0 - pad_x)
                y0 = max(0, y0 - pad_y)
                x1 = min(w_full, x1 + pad_x)
                y1 = min(h_full, y1 + pad_y)
                regions.append(pil_full.crop((x0, y0, x1, y1)))

        if not regions:
            # Fallback: full image resized — fits within TrOCR's sweet spot
            w, h = pil_full.size
            if max(w, h) > _INPUT_SIZE:
                scale = _INPUT_SIZE / max(w, h)
                pil_full = pil_full.resize(
                    (int(w * scale), int(h * scale)), Image.LANCZOS
                )
            regions = [pil_full]

        lines: list[str] = []
        with torch.inference_mode():
            for region in regions:
                # Convert to RGB (handles RGBA / palette images)
                region_rgb = region.convert("RGB")
                pixel_values = processor(
                    images=region_rgb, return_tensors="pt"
                ).pixel_values.to(device)
                generated_ids = model.generate(  # type: ignore[union-attr]
                    pixel_values,
                    max_new_tokens=128,
                )
                text = processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0].strip()
                if text:
                    lines.append(text)

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for line in lines:
            if line not in seen:
                seen.add(line)
                unique.append(line)

        ocr_text = "\n".join(unique)
        if not ocr_text:
            return {}

        return {"ocr_text": ocr_text}

    @classmethod
    def _load_models(cls) -> None:
        if cls._model is not None:
            return

        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError:
            raise ImportError(
                "transformers>=4.40 and torch are required for OCR:\n"
                "  pip install 'imganalyzer[local-ai]'"
            )

        from rich.console import Console
        Console().print(
            "[dim]Loading TrOCR model (first run downloads ~1.3 GB)...[/dim]"
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        cls._processor = TrOCRProcessor.from_pretrained(
            _MODEL_ID, cache_dir=CACHE_DIR
        )
        cls._model = VisionEncoderDecoderModel.from_pretrained(
            _MODEL_ID,
            cache_dir=CACHE_DIR,
        ).to(device)  # type: ignore[union-attr]

        # Workaround: when accelerate is installed and another model (e.g.
        # GroundingDINO) was loaded first, TrOCRSinusoidalPositionalEmbedding
        # keeps its `weights` tensor on the meta device (non-persistent buffer,
        # so .to() doesn't move it).  Re-compute the sinusoidal table on the
        # correct device using the module's own get_embedding() method.
        for mod in cls._model.modules():  # type: ignore[union-attr]
            if type(mod).__name__ == "TrOCRSinusoidalPositionalEmbedding":
                w = vars(mod).get("weights")
                if isinstance(w, torch.Tensor) and w.device.type == "meta":
                    num_embeddings = w.shape[0]
                    # get_embedding creates on CPU; move result to target device
                    mod.weights = mod.get_embedding(
                        num_embeddings, mod.embedding_dim, mod.padding_idx
                    ).to(device)

        cls._model.eval()  # type: ignore[union-attr]
