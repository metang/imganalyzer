"""GPT-4o captioning adapter (cloud baseline)."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from model_eval.models.base import ModelAdapter


class GPT4oAdapter(ModelAdapter):
    name = "gpt-4o"
    category = "caption"
    model_id = "gpt-4o"

    def __init__(self) -> None:
        self._client: Any = None

    def load(self, device: str = "cuda") -> None:
        import openai

        self._client = openai.OpenAI()  # Uses OPENAI_API_KEY env var

    def run(self, image_path: Path) -> dict[str, Any]:
        # Read and encode image as base64
        image_data = base64.b64encode(image_path.read_bytes()).decode("utf-8")

        # Determine media type
        suffix = image_path.suffix.lower()
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }
        media_type = media_types.get(suffix, "image/jpeg")

        response = self._client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_data}",
                                "detail": "high",
                            },
                        },
                        {
                            "type": "text",
                            "text": "Describe this image in detail.",
                        },
                    ],
                }
            ],
            max_tokens=1024,
        )

        caption = response.choices[0].message.content or ""
        return {
            "caption": caption.strip(),
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            },
        }

    def unload(self) -> None:
        self._client = None
