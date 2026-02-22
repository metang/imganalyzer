"""Cloud AI analysis: OpenAI GPT-4o Vision, Anthropic Claude, Google Vision."""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are an expert photography analyst. Analyze the provided image and return a JSON object with the following fields:
- description: (string) A detailed 2-3 sentence description of the image content
- scene_type: (string) e.g. "landscape", "portrait", "street", "architecture", "macro", "wildlife", "abstract", "product", etc.
- main_subject: (string) Primary subject(s) in the image
- lighting: (string) Lighting conditions and quality e.g. "golden hour", "overcast", "harsh midday", "studio softbox"
- mood: (string) Emotional tone/aesthetic e.g. "serene", "dramatic", "intimate", "moody"
- keywords: (array of strings) 10-15 descriptive keywords suitable as Lightroom tags
- technical_notes: (string) Any notable photographic/technical observations

Return ONLY valid JSON with no extra text."""


def _encode_image(path: Path, max_size_kb: int = 1024) -> tuple[str, str]:
    """Return base64-encoded image and MIME type, resized if needed."""
    from PIL import Image
    import io

    img = Image.open(str(path))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    # Resize large images for API efficiency
    w, h = img.size
    max_dim = 1568  # Anthropic recommended max
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode(), "image/jpeg"


def _parse_json_response(text: str) -> dict[str, Any]:
    """Extract JSON from model response."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Best-effort: return description as-is
        return {"description": text, "keywords": []}


class CloudAI:
    def __init__(self, backend: str) -> None:
        self.backend = backend

    def analyze(self, path: Path, image_data: dict[str, Any]) -> dict[str, Any]:
        if self.backend == "openai":
            return self._openai(path, image_data)
        elif self.backend == "anthropic":
            return self._anthropic(path, image_data)
        elif self.backend == "google":
            return self._google(path, image_data)
        raise ValueError(f"Unknown cloud AI backend: {self.backend}")

    def _openai(self, path: Path, image_data: dict[str, Any]) -> dict[str, Any]:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package required: pip install 'imganalyzer[openai]'")

        b64, mime = _encode_image(path)
        client = OpenAI()  # reads OPENAI_API_KEY from env

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"},
                        },
                        {"type": "text", "text": "Analyze this image."},
                    ],
                },
            ],
            max_tokens=800,
        )
        return _parse_json_response(response.choices[0].message.content or "")

    def _anthropic(self, path: Path, image_data: dict[str, Any]) -> dict[str, Any]:
        try:
            import anthropic
        except ImportError:
            raise ImportError("Anthropic package required: pip install 'imganalyzer[anthropic]'")

        b64, mime = _encode_image(path)
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=800,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": mime, "data": b64},
                        },
                        {"type": "text", "text": "Analyze this image."},
                    ],
                }
            ],
        )
        return _parse_json_response(message.content[0].text)

    def _google(self, path: Path, image_data: dict[str, Any]) -> dict[str, Any]:
        """Google Vision API — uses feature detection rather than generative."""
        try:
            from google.cloud import vision
        except ImportError:
            raise ImportError("Google Vision required: pip install 'imganalyzer[google]'")

        client = vision.ImageAnnotatorClient()

        with open(str(path), "rb") as f:
            content = f.read()

        image = vision.Image(content=content)

        # Run multiple detection features in one request
        response = client.annotate_image({
            "image": image,
            "features": [
                {"type_": vision.Feature.Type.LABEL_DETECTION, "max_results": 15},
                {"type_": vision.Feature.Type.OBJECT_LOCALIZATION, "max_results": 10},
                {"type_": vision.Feature.Type.IMAGE_PROPERTIES},
                {"type_": vision.Feature.Type.SAFE_SEARCH_DETECTION},
                {"type_": vision.Feature.Type.LANDMARK_DETECTION, "max_results": 5},
            ],
        })

        result: dict[str, Any] = {}

        # Labels → description + keywords
        labels = [l.description for l in response.label_annotations]
        result["keywords"] = labels
        result["description"] = "Scene detected: " + ", ".join(labels[:5]) + "."

        # Objects
        objects = [o.name for o in response.localized_object_annotations]
        if objects:
            result["detected_objects"] = objects
            result["main_subject"] = objects[0]

        # Dominant colors
        colors = response.image_properties_annotation.dominant_colors.colors
        hex_colors = []
        for c in sorted(colors, key=lambda x: -x.pixel_fraction)[:5]:
            rgb = c.color
            hex_colors.append(f"#{int(rgb.red):02x}{int(rgb.green):02x}{int(rgb.blue):02x}")
        if hex_colors:
            result["dominant_colors_ai"] = hex_colors

        # Landmark
        landmarks = [lm.description for lm in response.landmark_annotations]
        if landmarks:
            result["landmark"] = landmarks[0]

        return result
