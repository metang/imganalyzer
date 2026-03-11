"""Compare keyword extraction across Copilot and qwen3.5 variants.

Uses the exact keyword prompt from `imganalyzer.analysis.ai.cloud` and writes
an HTML table report with columns:
  - image thumbnail
  - Copilot keywords
  - qwen3.5:2b keywords
  - qwen3.5:4b keywords
  - qwen3.5:9b keywords
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib import request

# Ensure model-eval package is importable from repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "model-eval"))

from imganalyzer.analysis.ai.cloud import SYSTEM_PROMPT_WITH_AESTHETIC, _parse_json_response
from model_eval.cache import get_cached, store_result
from model_eval.models.base import ModelAdapter
from model_eval.runner import collect_images

PROMPT = "Analyze this image.\n\n" + SYSTEM_PROMPT_WITH_AESTHETIC
OLLAMA_PROMPT = "/no_think\n" + PROMPT
OLLAMA_URL = "http://localhost:11434"
IMAGE_DIR = REPO_ROOT / "test_images"
REPORT_PATH = REPO_ROOT / "model-eval" / "report_keywords_copilot_qwen35.html"

MODEL_SPECS = [
    ("kw-copilot", "copilot", "Copilot (gpt-4.1)"),
    ("kw-qwen35-9b", "qwen3.5:9b", "qwen3.5:9b"),
    ("kw-qwen35-4b", "qwen3.5:4b", "qwen3.5:4b"),
    ("kw-qwen35-2b", "qwen3.5:2b", "qwen3.5:2b"),
]

# Caption model cache names (from caption eval) used as descriptions in the report.
CAPTION_MODELS = [
    ("copilot", "kw-copilot"),
    ("qwen3.5-vl-9b", "kw-qwen35-9b"),
    ("qwen3.5-vl-4b", "kw-qwen35-4b"),
    ("qwen3.5-vl-2b", "kw-qwen35-2b"),
]

_RAW_EXTENSIONS = {
    ".arw", ".cr2", ".cr3", ".crw", ".dng", ".nef", ".nrw", ".orf", ".pef",
    ".raf", ".raw", ".rw2", ".rwl", ".sr2", ".srf", ".srw", ".x3f", ".iiq",
    ".3fr", ".fff", ".mef", ".mos", ".mrw", ".rwz", ".erf",
}
_NEEDS_JPEG = _RAW_EXTENSIONS | {".heic", ".heif", ".avif"}


def _encode_for_vision(path: Path, max_dim: int = 1568) -> str:
    img = ModelAdapter.load_image(path, max_dim=max_dim)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _to_jpeg_temp(path: Path, max_dim: int = 1568) -> Path:
    img = ModelAdapter.load_image(path, max_dim=max_dim)
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    img.save(tmp.name, format="JPEG", quality=85)
    return Path(tmp.name)


def _normalize_keywords(data: dict[str, Any]) -> list[str]:
    kw = data.get("keywords", [])
    items: list[str]
    if isinstance(kw, list):
        items = [str(x).strip() for x in kw]
    elif isinstance(kw, str):
        items = [x.strip() for x in kw.split(",")]
    else:
        items = []
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _extract_with_ollama(model_tag: str, image_path: Path, retries: int = 3) -> dict[str, Any]:
    b64 = _encode_for_vision(image_path, max_dim=1024)
    last_text = ""
    start = time.perf_counter()
    for attempt in range(1, retries + 1):
        payload = json.dumps(
            {
                "model": model_tag,
                "messages": [{"role": "user", "content": OLLAMA_PROMPT, "images": [b64]}],
                "stream": False,
                "options": {"num_predict": 2800},
            }
        ).encode("utf-8")
        req = request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read())
        text = (result.get("message", {}).get("content") or "").strip()
        last_text = text
        parsed = _parse_json_response(text)
        keywords = _normalize_keywords(parsed)
        if keywords:
            result: dict[str, Any] = {
                "keywords": keywords,
                "attempts": attempt,
                "inference_time_s": round(time.perf_counter() - start, 3),
            }
            desc = parsed.get("description", "")
            if desc:
                result["description"] = str(desc).strip()
            return result

    return {
        "keywords": [],
        "attempts": retries,
        "inference_time_s": round(time.perf_counter() - start, 3),
        "error": f"Empty keywords after {retries} attempts",
        "raw_text": last_text[:1000],
    }


async def _extract_with_copilot_async(analysis_path: str, retries: int = 3) -> dict[str, Any]:
    from copilot import CopilotClient

    start = time.perf_counter()
    last_content = ""
    client = CopilotClient()
    session = None
    try:
        session = await client.create_session(
            {
                "model": "gpt-4.1",
                "on_permission_request": lambda _req, _ctx: {"kind": "approved", "rules": []},
            }
        )
        for attempt in range(1, retries + 1):
            try:
                event = await session.send_and_wait(
                    {"prompt": PROMPT, "attachments": [{"type": "file", "path": analysis_path}]},
                    timeout=180.0,
                )
            except (asyncio.TimeoutError, TimeoutError) as exc:
                last_content = f"TimeoutError: {exc}"
                continue
            content = getattr(event.data, "content", "") if event is not None else ""
            content = (content or "").strip()
            last_content = content
            parsed = _parse_json_response(content)
            keywords = _normalize_keywords(parsed)
            if keywords:
                result: dict[str, Any] = {
                    "keywords": keywords,
                    "attempts": attempt,
                    "inference_time_s": round(time.perf_counter() - start, 3),
                }
                desc = parsed.get("description", "")
                if desc:
                    result["description"] = str(desc).strip()
                return result
        return {
            "keywords": [],
            "attempts": retries,
            "inference_time_s": round(time.perf_counter() - start, 3),
            "error": f"Empty keywords after {retries} attempts",
            "raw_text": last_content[:1000],
        }
    finally:
        if session is not None:
            try:
                await client.delete_session(session.session_id)
            except Exception:
                pass
        stop = getattr(client, "stop", None)
        if callable(stop):
            try:
                await stop()
            except Exception:
                force_stop = getattr(client, "force_stop", None)
                if callable(force_stop):
                    await force_stop()


def _extract_with_copilot(path: Path) -> dict[str, Any]:
    tmp: Path | None = None
    analysis_path = path
    if path.suffix.lower() in _NEEDS_JPEG:
        tmp = _to_jpeg_temp(path)
        analysis_path = tmp
    try:
        return asyncio.run(_extract_with_copilot_async(str(analysis_path)))
    finally:
        if tmp is not None:
            try:
                tmp.unlink()
            except OSError:
                pass


def _thumb_b64(path: Path) -> str:
    img = ModelAdapter.load_image(path, max_dim=300)
    img.thumbnail((220, 220))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def main() -> None:
    images = [p.resolve() for p in collect_images(IMAGE_DIR)]
    print(f"Images: {len(images)}")

    # Evaluate one model at a time to avoid cross-model churn.
    table_by_image: dict[Path, dict[str, Any]] = {
        img: {"image_name": img.name} for img in images
    }

    for cache_name, model_tag, label in MODEL_SPECS:
        print(f"\n=== {label} ===")
        for img in images:
            cached = get_cached(cache_name, img)
            cached_kw = _normalize_keywords(cached or {})
            needs_run = cached is None or not cached_kw
            if needs_run:
                if model_tag == "copilot":
                    result = _extract_with_copilot(img)
                else:
                    result = _extract_with_ollama(model_tag, img)
                store_result(cache_name, img, result)
                cached = result
                status = "computed"
            else:
                status = "cache"

            kw = _normalize_keywords(cached or {})
            err = (cached or {}).get("error")
            print(
                f"  {img.name:30} {status:8} keywords={len(kw):2d}"
                + (f" err={err}" if err else "")
            )
            table_by_image[img][cache_name] = cached or {}

    table: list[dict[str, Any]] = [table_by_image[img] for img in images]

    # Load captions from the caption eval cache to use as descriptions.
    for img in images:
        for cap_model, kw_key in CAPTION_MODELS:
            cached_cap = get_cached(cap_model, img)
            table_by_image[img][f"desc-{kw_key}"] = (cached_cap or {}).get("caption", "")

    # Generate thumbnails (deferred to avoid blocking cache reads).
    print("\nGenerating thumbnails...")
    for i, img in enumerate(images):
        table_by_image[img]["thumb"] = _thumb_b64(img)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(images)} thumbnails")
    print(f"  {len(images)}/{len(images)} thumbnails done")

    headers = [label for _, _, label in MODEL_SPECS]
    html_parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>Keyword &amp; Description Comparison</title>",
        "<style>",
        "body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:#0b1020;color:#e5e7eb;margin:0}",
        ".wrap{max-width:1800px;margin:0 auto;padding:24px}",
        "table{width:100%;border-collapse:collapse;table-layout:fixed;background:#111827;border:1px solid #374151}",
        "th,td{border:1px solid #374151;vertical-align:top;padding:10px}",
        "th{position:sticky;top:0;background:#1f2937;text-align:left}",
        "td:nth-child(1),th:nth-child(1){width:240px}",
        ".thumb{max-width:220px;max-height:220px;display:block;margin-bottom:8px;border-radius:6px}",
        ".name{font-size:12px;color:#cbd5e1;word-break:break-word}",
        ".kw{white-space:pre-wrap;word-break:break-word;line-height:1.5}",
        ".desc{white-space:pre-wrap;word-break:break-word;line-height:1.5;color:#93c5fd;font-style:italic;margin-bottom:8px;border-bottom:1px solid #374151;padding-bottom:8px}",
        ".meta{font-size:11px;color:#9ca3af;margin-top:8px}",
        ".err{color:#fca5a5}",
        "</style></head><body><div class='wrap'>",
        f"<h1>Keyword &amp; Description Comparison</h1><div class='meta'>{len(images)} images · prompt from cloud.py</div>",
        "<table><thead><tr>",
        "<th>Image thumbnail</th>",
    ]
    for h in headers:
        html_parts.append(f"<th>{h}</th>")
    html_parts.append("</tr></thead><tbody>")

    import html as html_escape

    for row in table:
        html_parts.append("<tr>")
        html_parts.append(
            "<td>"
            f"<img class='thumb' src='data:image/jpeg;base64,{row['thumb']}' alt='{html_escape.escape(row['image_name'])}'>"
            f"<div class='name'>{html_escape.escape(row['image_name'])}</div>"
            "</td>"
        )
        for cache_name, _model_tag, _label in MODEL_SPECS:
            data = row.get(cache_name, {})
            keywords = data.get("keywords", [])
            text = ", ".join(keywords) if keywords else "[no keywords]"
            error = data.get("error")
            meta = f"keywords={len(keywords)} · attempts={data.get('attempts', '?')} · {data.get('inference_time_s', '?')}s"

            # Description from the caption eval cache
            description = row.get(f"desc-{cache_name}", "")
            desc_html = f"<div class='desc'>{html_escape.escape(description)}</div>" if description else ""

            cell = f"{desc_html}<div class='kw'>{html_escape.escape(text)}</div><div class='meta'>{html_escape.escape(meta)}</div>"
            if error:
                cell += f"<div class='meta err'>{html_escape.escape(str(error))}</div>"
            html_parts.append(f"<td>{cell}</td>")
        html_parts.append("</tr>")

    html_parts.append("</tbody></table></div></body></html>")
    REPORT_PATH.write_text("".join(html_parts), encoding="utf-8")
    print(f"\nReport written: {REPORT_PATH}")


if __name__ == "__main__":
    main()

