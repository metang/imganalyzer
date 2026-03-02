"""OCR using Microsoft TrOCR (transformer-based, offline).

Model: microsoft/trocr-large-printed
- VisionEncoderDecoderModel (ViT encoder + RoBERTa decoder)
- Optimised for printed text; good on signs, labels, overlays, captions
- ~1.3 GB download, runs on CPU or CUDA
- Uses beam search (num_beams=5) for better quality vs greedy decoding

Pipeline:
  1. Receive optional bounding boxes from GroundingDINO (text regions).
  2. Evaluate coverage: if boxes cover >= _COVERAGE_THRESHOLD of image area,
     crop each box (10% padding) and run TrOCR per region.
  3. Otherwise (document/receipt — whole image is text):
     a. Resize to 1920px on long edge (8x faster, minimal quality loss).
     b. Tile the resized region into horizontal strips scaled to _INPUT_SIZE
        wide with _STRIP_H_PX native-pixel height each, run TrOCR per strip.
     c. Use adaptive per-strip ink detection to crop each strip to text bounds.
     d. Pad strips to square before resizing to prevent aspect ratio distortion.
  4. Fallback when no boxes given: same resize + tiling strategy.
  5. Deduplicate lines, join with ``\\n``, return as ``ocr_text``.
  6. If no text is read, return {} (caller skips writing to XMP).

Environment variables:
  - IMGANALYZER_OCR_NUM_BEAMS: Beam search width (default: 5). Higher is slower
    but better quality. Set to 1 for greedy decoding (fast but may hallucinate).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

CACHE_DIR = os.getenv("IMGANALYZER_MODEL_CACHE", str(Path.home() / ".cache" / "imganalyzer"))

_MODEL_ID = "microsoft/trocr-large-printed"

# Beam search width for generation. Higher values improve quality but are slower.
# num_beams=1 (greedy) is fast but produces hallucinations on receipts/documents.
# num_beams=5 is 3-4x slower but gives much better results (actual text vs garbage).
# Set IMGANALYZER_OCR_NUM_BEAMS environment variable to override.
_NUM_BEAMS = int(os.getenv("IMGANALYZER_OCR_NUM_BEAMS", "5"))

# Minimum crop dimension — ignore tiny boxes (likely false detections)
_MIN_CROP_PX = 32

# TrOCR canonical input size
_INPUT_SIZE = 384

# If GroundingDINO boxes cover less than this fraction of image area, the image
# is likely a document/receipt (whole image = text) → use tiling instead.
_COVERAGE_THRESHOLD = 0.50

# Tiling parameters for document/receipt mode.
# Strips are cut at native resolution then scaled to _INPUT_SIZE wide.
# _STRIP_H_PX is the strip height *in the content-cropped image* (native px).
# Pick it so a receipt line (~40–80 px tall at iPhone resolution) fills a
# reasonable fraction of the TrOCR input height after scaling.
# For a ~2000 px wide crop the scale factor is 384/2000 ≈ 0.19, so a 400 px
# strip becomes ~76 px tall — a few text lines, good for TrOCR.
_STRIP_H_PX = 400    # strip height in cropped-image native pixels
_STRIP_OVERLAP = 40  # overlap in native pixels to avoid cutting mid-line

# Maximum number of strips to process (safety cap for very tall images)
_MAX_STRIPS = 40

# Minimum fraction of pixels in a column/row that must be "light" (>200
# brightness) for that column/row to be considered part of the paper content.
# Used by _crop_to_content() to discard surrounding desk / shadow.
_CONTENT_THRESH = 0.08


class OCRAnalyzer:
    """Extract text from images using Microsoft TrOCR.

    Model is loaded once per process (class-level singleton) and shared across
    all images in a batch.

    ``analyze()`` accepts an optional *text_boxes* list of ``[x0, y0, x1, y1]``
    bounding boxes (pixel coordinates in the original image) from GroundingDINO.

    Strategy selection:
    - If boxes cover >= 50% of image area → crop each box individually.
    - Otherwise (document/receipt) → tile the full image into horizontal strips.
    """

    _processor = None
    _model = None

    @classmethod
    def _unload(cls) -> None:
        """Unload TrOCR model from GPU to free VRAM.

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
        w_full, h_full = pil_full.size

        self._load_models()
        processor = OCRAnalyzer._processor
        model = OCRAnalyzer._model
        device = next(model.parameters()).device  # type: ignore[union-attr]

        regions: list[Image.Image] = []

        # ── Decide strategy ────────────────────────────────────────────────
        use_box_crops = False
        if text_boxes:
            image_area = w_full * h_full
            box_area = sum(
                (max(0, min(w_full, int(b[2])) - max(0, int(b[0]))) *
                 max(0, min(h_full, int(b[3])) - max(0, int(b[1]))))
                for b in text_boxes
            )
            coverage = box_area / image_area if image_area > 0 else 0
            use_box_crops = coverage >= _COVERAGE_THRESHOLD

        if use_box_crops:
            # Individual box crops — good for incidental text in scene images
            for box in (text_boxes or []):
                x0, y0, x1, y1 = (
                    max(0, int(box[0])),
                    max(0, int(box[1])),
                    min(w_full, int(box[2])),
                    min(h_full, int(box[3])),
                )
                if (x1 - x0) < _MIN_CROP_PX or (y1 - y0) < _MIN_CROP_PX:
                    continue
                pad_x = int((x1 - x0) * 0.10)
                pad_y = int((y1 - y0) * 0.10)
                x0 = max(0, x0 - pad_x)
                y0 = max(0, y0 - pad_y)
                x1 = min(w_full, x1 + pad_x)
                y1 = min(h_full, y1 + pad_y)
                regions.append(pil_full.crop((x0, y0, x1, y1)))
        else:
            # Document/receipt tiling: dynamically resize based on font size
            # to balance OCR quality and performance, then _tile_image detects
            # ink bounds per strip for curved/partial documents.
            target_size = _compute_target_size(pil_full, min_font_px=24)
            if target_size:
                pil_resized = pil_full.resize(target_size, Image.LANCZOS)
            else:
                pil_resized = pil_full
            regions = _tile_image(pil_resized)

        lines: list[str] = []
        with torch.inference_mode():
            # Process regions in batches for speed
            batch_size = 4
            for i in range(0, len(regions), batch_size):
                batch = regions[i:i+batch_size]
                batch_rgb = [r.convert("RGB") for r in batch]
                pixel_values = processor(
                    images=batch_rgb, return_tensors="pt"
                ).pixel_values.to(device)
                generated_ids = model.generate(  # type: ignore[union-attr]
                    pixel_values,
                    max_new_tokens=256,
                    num_beams=_NUM_BEAMS,
                )
                texts = processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                for text in texts:
                    text = text.strip()
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
        # Load in fp16 on CUDA — halves static weight memory (~1.3 GB → ~0.65 GB)
        # and also halves activation/beam-search tensors during inference.
        load_dtype = torch.float16 if device == "cuda" else torch.float32

        cls._processor = TrOCRProcessor.from_pretrained(
            _MODEL_ID, cache_dir=CACHE_DIR
        )
        cls._model = VisionEncoderDecoderModel.from_pretrained(
            _MODEL_ID,
            torch_dtype=load_dtype,
            cache_dir=CACHE_DIR,
        ).to(device)  # type: ignore[union-attr]

        # Workaround: when accelerate is installed and another model (e.g.
        # GroundingDINO) was loaded first, TrOCRSinusoidalPositionalEmbedding
        # keeps its `weights` tensor on the meta device (non-persistent buffer,
        # so .to() doesn't move it).  Re-compute the sinusoidal table on the
        # correct device using the module's own get_embedding() method, then
        # cast to the model dtype so it matches fp16 weights.
        for mod in cls._model.modules():  # type: ignore[union-attr]
            if type(mod).__name__ == "TrOCRSinusoidalPositionalEmbedding":
                w = vars(mod).get("weights")
                if isinstance(w, torch.Tensor) and w.device.type == "meta":
                    num_embeddings = w.shape[0]
                    # get_embedding creates on CPU float32; cast to model dtype/device
                    mod.weights = mod.get_embedding(
                        num_embeddings, mod.embedding_dim, mod.padding_idx
                    ).to(device=device, dtype=load_dtype)

        cls._model.eval()  # type: ignore[union-attr]


def _estimate_font_size(img: "Image.Image") -> int:
    """Estimate typical font size (line height) in pixels by analyzing text strokes.
    
    Uses edge detection and horizontal projection to find typical text line spacing.
    Returns estimated font height in pixels, clamped to 8-100px range.
    """
    from PIL import Image
    import cv2
    
    # Work with a downsampled version for speed (max 800px on long edge)
    w, h = img.size
    max_dim = max(w, h)
    if max_dim > 800:
        scale = 800 / max_dim
        sample = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        scale_factor = max_dim / 800
    else:
        sample = img
        scale_factor = 1.0
    
    # Convert to grayscale numpy array
    gray = np.array(sample.convert("L"), dtype=np.uint8)
    
    # Edge detection to find text strokes
    edges = cv2.Canny(gray, 50, 150)
    
    # Horizontal projection: sum edge pixels in each row
    h_projection = edges.sum(axis=1)
    
    # Find peaks in projection (text lines) using simple threshold
    threshold = np.percentile(h_projection, 75)  # Top 25% rows
    in_text = h_projection > threshold
    
    # Find runs of consecutive text rows (text line heights)
    line_heights = []
    run_length = 0
    for val in in_text:
        if val:
            run_length += 1
        elif run_length > 0:
            line_heights.append(run_length)
            run_length = 0
    if run_length > 0:
        line_heights.append(run_length)
    
    if not line_heights:
        # Fallback: assume medium font ~30px at original resolution
        return int(30 * scale_factor)
    
    # Use median line height as font size estimate
    median_height = int(np.median(line_heights))
    # Scale back to original resolution
    font_size = int(median_height * scale_factor)
    
    # Clamp to reasonable range
    return max(8, min(100, font_size))


def _compute_target_size(img: "Image.Image", min_font_px: int = 24) -> "tuple[int, int] | None":
    """Compute optimal resize dimensions to balance OCR quality and performance.
    
    Simple heuristic: downscale to 1920px if larger for speed.
    
    Returns:
        (width, height) tuple if resize needed, None if current size is good
    """
    w, h = img.size
    max_dim = max(w, h)
    
    # Downscale large images to 1920px for speed/memory
    if max_dim > 1920:
        scale = 1920 / max_dim
        return (int(w * scale), int(h * scale))
    
    return None


def _ink_column_bounds(gray: "np.ndarray", y0: int, y1: int) -> "tuple[int, int] | None":
    """Return the (left, right) column indices of ink pixels in a row slice.

    "Ink" = dark pixels (< 80) that sit on a light background (column mean
    > 170).  Returns None if no ink is found in the slice.
    
    If the detected bounds are very wide (>1500px), refines them by finding
    the densest 800-1200px text window within those bounds.
    """
    band = gray[y0:y1, :]
    col_mean = band.mean(axis=0)
    col_dark = (band < 80).mean(axis=0)
    # A column carries ink if it is mostly light (paper) and has some dark dots
    ink_mask = (col_mean > 170) & (col_dark > 0.006)
    cols = np.where(ink_mask)[0]
    if cols.size < 5:
        return None
    
    x0_init, x1_init = int(cols[0]), int(cols[-1])
    width_init = x1_init - x0_init
    
    # If bounds are very wide, refine by finding the densest text region
    if width_init > 1500:
        # Try windows of 800-1200px and pick the one with most dark pixels
        best_score = 0
        best_x0, best_x1 = x0_init, x1_init
        
        for win_w in [800, 1000, 1200]:
            if win_w > width_init:
                continue
            for x in range(x0_init, x1_init - win_w + 1, 50):
                window_dark = col_dark[x : x + win_w].sum()
                if window_dark > best_score:
                    best_score = window_dark
                    best_x0, best_x1 = x, x + win_w
        
        if best_score > 0:
            return best_x0, best_x1
    
    return x0_init, x1_init


def _crop_to_content(img: "Image.Image") -> "Image.Image":
    """Crop *img* to the global bounding box of all ink on paper.

    Uses the _ink_column_bounds helper to find where actual printed text
    lives, then returns a crop with a small margin.  Falls back to the
    full image if no ink region is detected.
    """
    gray = np.array(img.convert("L"), dtype=np.float32)
    h, w = gray.shape

    bounds = _ink_column_bounds(gray, 0, h)
    if bounds is None:
        return img

    margin = 30
    x0 = max(0, bounds[0] - margin)
    x1 = min(w, bounds[1] + margin)

    # Row extent: rows where any ink-column pixel is dark
    ink_cols = slice(bounds[0], bounds[1] + 1)
    row_dark = (gray[:, ink_cols] < 80).mean(axis=1)
    row_ok = np.where(row_dark > 0.002)[0]
    if row_ok.size == 0:
        return img
    y0_c = max(0, int(row_ok[0]) - margin)
    y1_c = min(h, int(row_ok[-1]) + margin)

    if (x1 - x0) < _MIN_CROP_PX or (y1_c - y0_c) < _MIN_CROP_PX:
        return img

    return img.crop((x0, y0_c, x1, y1_c))


def _tile_image(img: "Image.Image") -> "list[Image.Image]":
    """Slice *img* into adaptive horizontal strips for TrOCR.

    For each strip the ink column bounds are re-detected so that curved or
    partially-visible documents (e.g. a receipt that curls away from the
    camera) are cropped tightly per-strip rather than using a single global
    column range that would include lots of blank margin.

    Each strip is _STRIP_H_PX native pixels tall (with _STRIP_OVERLAP
    overlap), then cropped to ink bounds, padded to square (to avoid
    distortion), and resized to _INPUT_SIZE × _INPUT_SIZE for TrOCR.

    Returns at most _MAX_STRIPS strips.
    """
    from PIL import Image

    gray = np.array(img.convert("L"), dtype=np.float32)
    full_w, full_h = img.size
    step = _STRIP_H_PX - _STRIP_OVERLAP
    strips: list[Image.Image] = []

    y = 0
    while y < full_h and len(strips) < _MAX_STRIPS:
        y1 = min(y + _STRIP_H_PX, full_h)
        strip_h = y1 - y
        if strip_h >= _MIN_CROP_PX:
            # Find ink bounds for this specific slice
            bounds = _ink_column_bounds(gray, y, y1)
            if bounds is not None:
                margin = 20
                sx0 = max(0, bounds[0] - margin)
                sx1 = min(full_w, bounds[1] + margin)
            else:
                sx0, sx1 = 0, full_w  # fallback: full width

            if (sx1 - sx0) >= _MIN_CROP_PX:
                strip = img.crop((sx0, y, sx1, y1))
                sw, sh = strip.size
                
                # Pad to square to prevent distortion when TrOCR processor
                # resizes. Add white padding on the shorter dimension.
                if sw > sh:
                    # Wide strip: add top/bottom padding
                    pad_total = sw - sh
                    pad_top = pad_total // 2
                    pad_bottom = pad_total - pad_top
                    square = Image.new("RGB", (sw, sw), (255, 255, 255))
                    square.paste(strip, (0, pad_top))
                elif sh > sw:
                    # Tall strip: add left/right padding
                    pad_total = sh - sw
                    pad_left = pad_total // 2
                    pad_right = pad_total - pad_left
                    square = Image.new("RGB", (sh, sh), (255, 255, 255))
                    square.paste(strip, (pad_left, 0))
                else:
                    square = strip
                
                # Resize to TrOCR input size
                strip_resized = square.resize((_INPUT_SIZE, _INPUT_SIZE), Image.LANCZOS)
                strips.append(strip_resized)
        y += step

    return strips if strips else [img.resize((_INPUT_SIZE, _INPUT_SIZE), Image.LANCZOS)]
