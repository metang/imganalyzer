"""Generate a self-contained HTML comparison report."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader
from PIL import Image

from model_eval.config import AESTHETIC_MODELS, CAPTION_MODELS

TEMPLATE_DIR = Path(__file__).parent / "templates"
THUMBNAIL_SIZE = (400, 400)


def _image_to_base64_thumbnail(image_path: Path) -> str:
    """Convert an image to a base64-encoded JPEG thumbnail."""
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass
    try:
        img = Image.open(image_path).convert("RGB")
        img.thumbnail(THUMBNAIL_SIZE, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return ""


def generate_report(
    results: dict[str, dict[str, Any]],
    image_dir: Path,
    output_path: Path,
) -> None:
    """Generate an HTML report from evaluation results.

    Args:
        results: Nested dict results[model_name][image_stem] = result_dict
        image_dir: Directory containing the original test images
        output_path: Where to write the HTML file
    """
    # Collect all image stems that have at least one result
    image_stems: set[str] = set()
    for model_results in results.values():
        image_stems.update(model_results.keys())
    image_stems_sorted = sorted(image_stems)

    # Build image data (thumbnails)
    image_data: dict[str, dict[str, Any]] = {}
    for stem in image_stems_sorted:
        # Find the actual file (could be .jpg, .heic, etc.)
        matching = [
            p for p in image_dir.iterdir()
            if p.stem == stem and p.is_file()
        ]
        if matching:
            path = matching[0]
            image_data[stem] = {
                "name": path.name,
                "thumbnail_b64": _image_to_base64_thumbnail(path),
            }
        else:
            image_data[stem] = {"name": stem, "thumbnail_b64": ""}

    # Separate aesthetic vs caption models
    aesthetic_models = [m for m in results if m in AESTHETIC_MODELS]
    caption_models = [m for m in results if m in CAPTION_MODELS]

    # Compute max score per aesthetic model for bar normalization
    score_maxes: dict[str, float] = {}
    for model in aesthetic_models:
        scores = [
            r["score"] for r in results[model].values()
            if isinstance(r, dict) and "score" in r
        ]
        if scores:
            score_maxes[model] = max(scores) * 1.1  # 10% headroom
        else:
            score_maxes[model] = 10.0

    # Render
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=True,
    )
    template = env.get_template("report.html.j2")
    html = template.render(
        image_stems=image_stems_sorted,
        image_data=image_data,
        aesthetic_models=aesthetic_models,
        caption_models=caption_models,
        results=results,
        score_maxes=score_maxes,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
