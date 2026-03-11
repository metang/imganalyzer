"""GitHub Copilot (GPT-4.1 vision) captioning adapter — cloud baseline.

Uses the Copilot SDK (``github-copilot-sdk``) which authenticates via the
user's GitHub Copilot subscription.  No separate API key is needed.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

from model_eval.models.base import ModelAdapter

# Extensions that must be converted to JPEG before submission.
_NEEDS_JPEG = frozenset({
    ".heic", ".heif", ".avif",
    ".arw", ".cr2", ".cr3", ".dng", ".nef", ".orf", ".raf", ".rw2",
})


class CopilotAdapter(ModelAdapter):
    name = "copilot"
    category = "caption"
    model_id = "gpt-4.1 (via Copilot SDK)"

    def __init__(self) -> None:
        self._loaded = False

    def load(self, device: str = "cuda") -> None:
        # Validate the SDK is importable at load time.
        from copilot import CopilotClient  # noqa: F401
        self._loaded = True

    def run(self, image_path: Path) -> dict[str, Any]:
        from copilot import CopilotClient

        # Convert non-JPEG-compatible formats to a temp JPEG.
        temp_jpeg: Path | None = None
        analysis_path = image_path
        if image_path.suffix.lower() in _NEEDS_JPEG:
            temp_jpeg = self._convert_to_jpeg(image_path)
            analysis_path = temp_jpeg

        try:
            result = asyncio.run(self._query(CopilotClient, str(analysis_path)))
        finally:
            if temp_jpeg is not None:
                try:
                    temp_jpeg.unlink()
                except OSError:
                    pass

        return result

    async def _query(self, client_cls: type, image_path: str) -> dict[str, Any]:
        client = client_cls()
        session = None
        try:
            session = await client.create_session({
                "model": "gpt-4.1",
                "on_permission_request": lambda _req, _ctx: {
                    "kind": "approved",
                    "rules": [],
                },
            })
            event = await session.send_and_wait(
                {
                    "prompt": "Describe this image in detail.",
                    "attachments": [{"type": "file", "path": image_path}],
                },
                timeout=120.0,
            )
            if event is None:
                raise RuntimeError("Copilot returned no response")
            content: str = getattr(event.data, "content", "") or ""
            if not content:
                raise RuntimeError("Copilot returned an empty response")
            return {"caption": content.strip()}
        finally:
            if session is not None:
                try:
                    await asyncio.wait_for(
                        client.delete_session(session.session_id), timeout=10.0
                    )
                except Exception:
                    pass
            stop = getattr(client, "stop", None)
            if callable(stop):
                try:
                    await asyncio.wait_for(stop(), timeout=5.0)
                except Exception:
                    force = getattr(client, "force_stop", None)
                    if callable(force):
                        await force()

    @staticmethod
    def _convert_to_jpeg(path: Path) -> Path:
        """Convert HEIC/RAW to a temp JPEG for the Copilot attachment."""
        from PIL import Image
        from model_eval.models.base import ModelAdapter
        try:
            from pillow_heif import register_heif_opener
            register_heif_opener()
        except ImportError:
            pass

        # Reuse shared loader so RAW formats (ARW/CR2/NEF...) are decoded
        # consistently and resized to our evaluation cap.
        try:
            img = ModelAdapter.load_image(path, max_dim=1280)
        except Exception:
            img = Image.open(path).convert("RGB")
            w, h = img.size
            max_dim = 1280
            if max(w, h) > max_dim:
                ratio = max_dim / max(w, h)
                img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.close()
        img.save(tmp.name, format="JPEG", quality=85)
        return Path(tmp.name)

    def unload(self) -> None:
        self._loaded = False
