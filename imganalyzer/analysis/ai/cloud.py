"""Cloud AI analysis: OpenAI GPT-4o Vision, Anthropic Claude, Google Vision, GitHub Copilot."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import tempfile
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert photography analyst. Analyze the provided image and return a JSON object with the following fields:
- description: (string) A detailed 2-3 sentence description of the image content
- scene_type: (string) e.g. "landscape", "portrait", "street", "architecture", "macro", "wildlife", "abstract", "product", etc.
- main_subject: (string) Primary subject(s) in the image
- lighting: (string) Lighting conditions and quality e.g. "golden hour", "overcast", "harsh midday", "studio softbox"
- mood: (string) Emotional tone/aesthetic e.g. "serene", "dramatic", "intimate", "moody"
- keywords: (array of strings) 10-15 descriptive keywords suitable as Lightroom tags. If the image contains animals, birds, insects, or plants, include specific species or common names as keywords (e.g. "Red-tailed Hawk", "Monarch Butterfly", "Japanese Maple", "Golden Retriever"). Prefer the most specific identification possible over generic terms like "bird" or "flower".
- technical_notes: (string) Any notable photographic/technical observations

Return ONLY valid JSON with no extra text."""

# Extended prompt for backends that also produce aesthetic scoring.
SYSTEM_PROMPT_WITH_AESTHETIC = """You are an expert photography analyst. Analyze the provided image and return a JSON object with exactly these fields:
- description: (string) A detailed 2-3 sentence description of the image content
- scene_type: (string) e.g. "landscape", "portrait", "street", "architecture", "macro", "wildlife", "abstract", "product"
- main_subject: (string) Primary subject(s) in the image
- lighting: (string) Lighting conditions e.g. "golden hour", "overcast", "harsh midday", "studio softbox"
- mood: (string) Emotional tone e.g. "serene", "dramatic", "intimate", "moody"
- keywords: (array of strings) 10-15 descriptive keywords suitable as photo tags. If the image contains animals, birds, insects, or plants, include specific species or common names as keywords (e.g. "Red-tailed Hawk", "Monarch Butterfly", "Japanese Maple", "Golden Retriever"). Prefer the most specific identification possible over generic terms like "bird" or "flower".
- technical_notes: (string) Notable photographic or technical observations
- aesthetic_score: (number) Overall aesthetic quality score from 0.0 to 10.0. Consider composition, lighting, subject interest, technical quality, and emotional impact. Be critical and realistic: 0-3 = poor, 4-5 = average, 6-7 = good, 8-9 = excellent, 10 = exceptional/masterpiece
- aesthetic_label: (string) One-word label matching the score: "Poor" (0-3), "Average" (4-5), "Good" (6-7), "Excellent" (8-9), "Masterpiece" (10)
- aesthetic_reason: (string) 1-2 sentence explanation of the score, referencing specific strengths and weaknesses in composition, lighting, subject, or technical quality

Return ONLY valid JSON with no extra text, no markdown fences."""

# RAW file extensions that need conversion to JPEG before cloud API submission.
_RAW_EXTENSIONS = frozenset({
    ".arw", ".cr2", ".cr3", ".crw", ".dng", ".nef", ".nrw", ".orf", ".pef",
    ".raf", ".raw", ".rw2", ".rwl", ".sr2", ".srf", ".srw", ".x3f", ".iiq",
    ".3fr", ".fff", ".mef", ".mos", ".mrw", ".rwz", ".erf",
})
# Non-RAW formats that also need JPEG conversion for the Copilot SDK attachment.
_NEEDS_JPEG_CONVERSION = _RAW_EXTENSIONS | frozenset({".heic", ".heif", ".avif"})

# Copilot SDK session lifecycle is not safe to run concurrently across
# multiple worker threads (can raise "cannot schedule new futures after shutdown").
_COPILOT_LOCK = threading.Lock()


async def _stop_copilot_client(client: Any) -> None:
    """Shut down a Copilot client and its spawned CLI process."""
    stop = getattr(client, "stop", None)
    if not callable(stop):
        return

    try:
        errors = await asyncio.wait_for(stop(), timeout=5.0)
        if errors:
            log.warning("Copilot client stop reported cleanup errors: %s", errors)
    except asyncio.TimeoutError:
        force_stop = getattr(client, "force_stop", None)
        if callable(force_stop):
            log.warning("Copilot client stop timed out; forcing shutdown")
            await force_stop()
        else:
            log.warning("Copilot client stop timed out and force_stop is unavailable")
    except Exception:
        log.exception("Failed to stop Copilot client cleanly")


def _encode_image(path: Path, max_size_kb: int = 1024) -> tuple[str, str]:
    """Return base64-encoded image and MIME type, resized if needed."""
    from PIL import Image
    import io
    from imganalyzer.readers.standard import pillow_decode_guard, register_optional_pillow_opener

    register_optional_pillow_opener(path)
    with pillow_decode_guard(path):
        img = Image.open(path)  # Path objects work on both Windows and macOS
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
        elif self.backend == "copilot":
            return self._copilot(path, image_data)
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
                {"role": "system", "content": SYSTEM_PROMPT_WITH_AESTHETIC},
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
            max_tokens=1000,
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
            max_tokens=1000,
            system=SYSTEM_PROMPT_WITH_AESTHETIC,
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

        with open(path, "rb") as f:  # Path objects work on both Windows and macOS
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
        labels = [label.description for label in response.label_annotations]
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

    def _copilot(self, path: Path, image_data: dict[str, Any]) -> dict[str, Any]:
        """GitHub Copilot SDK backend — uses gpt-4.1 vision model.

        The SDK is async; we run it synchronously via asyncio.run().
        Each analysis gets a fresh session which is deleted after use so
        sessions don't accumulate in the user's Copilot history.

        RAW files are converted to a temporary JPEG first (same approach
        as the Electron copilot-analyzer.ts reference implementation).
        Returns all standard cloud_ai fields PLUS aesthetic_score /
        aesthetic_label so both modules are populated in one API call.
        """
        try:
            from copilot import CopilotClient, PermissionHandler
        except ImportError:
            raise ImportError(
                "GitHub Copilot SDK required: pip install 'imganalyzer[copilot]'"
            )

        async def _run(analysis_path: str) -> dict[str, Any]:
            # Always create a fresh client — asyncio.run() closes its event
            # loop on return, so any cached client becomes invalid.
            client = CopilotClient()
            session = None
            try:
                session = await client.create_session({
                    "model": "gpt-4.1",
                    "on_permission_request": PermissionHandler.approve_all,
                })
                event = await session.send_and_wait(
                    {
                        "prompt": "Analyze this image.\n\n" + SYSTEM_PROMPT_WITH_AESTHETIC,
                        "attachments": [{"type": "file", "path": analysis_path}],
                    },
                    timeout=120.0,
                )
                if event is None:
                    raise RuntimeError("Copilot returned no response")
                content: str = getattr(event.data, "content", "") or ""
                if not content:
                    raise RuntimeError("Copilot returned an empty response")
                return _parse_json_response(content)
            finally:
                # Delete the session to prevent "Analyze this image" clutter.
                if session is not None:
                    try:
                        await asyncio.wait_for(
                            client.delete_session(session.session_id), timeout=10.0
                        )
                    except Exception:
                        log.debug(
                            "Failed to delete Copilot session %s",
                            session.session_id,
                            exc_info=True,
                        )
                await _stop_copilot_client(client)

        # RAW and HEIC/HEIF/AVIF files must be converted to JPEG before submission.
        temp_jpeg: Path | None = None
        analysis_path = path
        if path.suffix.lower() in _NEEDS_JPEG_CONVERSION:
            temp_jpeg = _convert_raw_to_jpeg(path)
            analysis_path = temp_jpeg

        try:
            with _COPILOT_LOCK:
                return asyncio.run(_run(str(analysis_path)))
        finally:
            if temp_jpeg is not None:
                try:
                    temp_jpeg.unlink()
                except OSError:
                    pass


def _convert_raw_to_jpeg(raw_path: Path) -> Path:
    """Decode a RAW or HEIC/HEIF/AVIF file, resize to ≤1568 px, return a temp JPEG path.

    - True RAW files (in _RAW_EXTENSIONS) are decoded via rawpy.
    - HEIC/HEIF/AVIF and other non-RAW formats are opened directly with Pillow
      (requires pillow-heif plugin for HEIC/HEIF).
    Caller is responsible for deleting the returned file.
    """
    from PIL import Image

    if raw_path.suffix.lower() in _RAW_EXTENSIONS:
        try:
            import rawpy
        except ImportError:
            raise ImportError("rawpy required for RAW conversion: pip install rawpy")
        from imganalyzer.readers.raw import _suppress_c_stderr
        with _suppress_c_stderr():
            raw_ctx = rawpy.imread(str(raw_path))
        with raw_ctx as raw:
            rgb = raw.postprocess(use_camera_wb=True, output_bps=8, half_size=True)
        img = Image.fromarray(rgb)
    else:
        # HEIC/HEIF/AVIF — Pillow opens these directly (with pillow-heif registered).
        from imganalyzer.readers.standard import pillow_decode_guard, register_optional_pillow_opener

        register_optional_pillow_opener(raw_path)
        with pillow_decode_guard(raw_path):
            img = Image.open(raw_path)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            w, h = img.size
            max_dim = 1568
            if max(w, h) > max_dim:
                ratio = max_dim / max(w, h)
                img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    if raw_path.suffix.lower() in _RAW_EXTENSIONS:
        w, h = img.size
        max_dim = 1568
        if max(w, h) > max_dim:
            ratio = max_dim / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    img.save(tmp.name, format="JPEG", quality=85)
    log.debug("Converted %s to temp JPEG: %s", raw_path.suffix, tmp.name)
    return Path(tmp.name)
