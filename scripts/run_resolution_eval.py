"""Compare keyword/description quality across resolutions for Copilot and qwen3.5:9b.

Runs 10 random images at resolutions 480–1920 to find the best trade-off between
image quality and model reliability/output quality.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib import request

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "model-eval"))

from imganalyzer.analysis.ai.cloud import SYSTEM_PROMPT_WITH_AESTHETIC, _parse_json_response
from model_eval.models.base import ModelAdapter
from model_eval.runner import collect_images

PROMPT = "Analyze this image.\n\n" + SYSTEM_PROMPT_WITH_AESTHETIC
OLLAMA_PROMPT = "/no_think\n" + PROMPT
OLLAMA_URL = "http://localhost:11434"
IMAGE_DIR = REPO_ROOT / "test_images"
REPORT_PATH = REPO_ROOT / "model-eval" / "report_resolution_comparison.html"
CACHE_PATH = REPO_ROOT / "model-eval" / ".cache" / "_resolution_eval_cache.json"

RESOLUTIONS = [480, 640, 768, 1024, 1280, 1568, 1920]
NUM_IMAGES = 10
SEED = 42

_RAW_EXTENSIONS = {
    ".arw", ".cr2", ".cr3", ".crw", ".dng", ".nef", ".nrw", ".orf", ".pef",
    ".raf", ".raw", ".rw2", ".rwl", ".sr2", ".srf", ".srw", ".x3f", ".iiq",
    ".3fr", ".fff", ".mef", ".mos", ".mrw", ".rwz", ".erf",
}
_NEEDS_JPEG = _RAW_EXTENSIONS | {".heic", ".heif", ".avif"}


def _encode_for_vision(path: Path, max_dim: int) -> str:
    img = ModelAdapter.load_image(path, max_dim=max_dim)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _image_dims(path: Path, max_dim: int) -> tuple[int, int, int]:
    """Return (width, height, jpeg_bytes) for the image at the given max_dim."""
    img = ModelAdapter.load_image(path, max_dim=max_dim)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return img.size[0], img.size[1], len(buf.getvalue())


def _normalize_keywords(data: dict[str, Any]) -> list[str]:
    kw = data.get("keywords", [])
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


def _call_ollama(b64: str) -> dict[str, Any]:
    """Single Ollama call, returns full result dict."""
    payload = json.dumps({
        "model": "qwen3.5:9b",
        "messages": [{"role": "user", "content": OLLAMA_PROMPT, "images": [b64]}],
        "stream": False,
        "options": {"num_predict": 4000},
    }).encode("utf-8")
    req = request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def _extract_ollama(image_path: Path, max_dim: int) -> dict[str, Any]:
    b64 = _encode_for_vision(image_path, max_dim)
    start = time.perf_counter()
    for attempt in range(1, 4):
        result = _call_ollama(b64)
        text = (result.get("message", {}).get("content") or "").strip()
        parsed = _parse_json_response(text)
        keywords = _normalize_keywords(parsed)
        elapsed = round(time.perf_counter() - start, 3)
        if keywords:
            return {
                "keywords": keywords,
                "description": str(parsed.get("description", "")).strip(),
                "attempts": attempt,
                "inference_time_s": elapsed,
                "done_reason": result.get("done_reason", ""),
                "eval_count": result.get("eval_count", 0),
                "prompt_eval_count": result.get("prompt_eval_count", 0),
            }
    return {
        "keywords": [],
        "description": "",
        "attempts": 3,
        "inference_time_s": round(time.perf_counter() - start, 3),
        "error": "Empty keywords after 3 attempts",
        "raw_text": text[:500],
        "done_reason": result.get("done_reason", ""),
        "eval_count": result.get("eval_count", 0),
        "prompt_eval_count": result.get("prompt_eval_count", 0),
    }


async def _extract_copilot_async(
    analysis_path: str, max_dim: int,
) -> dict[str, Any]:
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
        for attempt in range(1, 4):
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
                return {
                    "keywords": keywords,
                    "description": str(parsed.get("description", "")).strip(),
                    "attempts": attempt,
                    "inference_time_s": round(time.perf_counter() - start, 3),
                }
        return {
            "keywords": [],
            "description": "",
            "attempts": 3,
            "inference_time_s": round(time.perf_counter() - start, 3),
            "error": "Empty keywords after 3 attempts",
            "raw_text": last_content[:500],
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


def _extract_copilot(image_path: Path, max_dim: int) -> dict[str, Any]:
    tmp: Path | None = None
    # Copilot SDK takes a file path — resize to a temp JPEG.
    img = ModelAdapter.load_image(image_path, max_dim=max_dim)
    tmp = Path(tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name)
    img.save(str(tmp), format="JPEG", quality=85)
    try:
        return asyncio.run(_extract_copilot_async(str(tmp), max_dim))
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def _load_cache() -> dict[str, Any]:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict[str, Any]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def _cache_key(model: str, image_name: str, res: int) -> str:
    return f"{model}__{image_name}__{res}"


def main() -> None:
    all_images = sorted(collect_images(IMAGE_DIR), key=lambda p: p.name)
    # Filter to JPEG only for consistency (avoid RAW/HEIC conversion variability).
    jpeg_images = [p for p in all_images if p.suffix.lower() in (".jpg", ".jpeg")]
    rng = random.Random(SEED)
    images = sorted(rng.sample(jpeg_images, min(NUM_IMAGES, len(jpeg_images))), key=lambda p: p.name)
    print(f"Selected {len(images)} images for resolution comparison:", flush=True)
    for img in images:
        print(f"  {img.name}", flush=True)

    cache = _load_cache()
    results: dict[str, dict[int, dict[str, Any]]] = {}  # image_name -> res -> result

    # --- qwen3.5:9b ---
    print(f"\n{'='*60}", flush=True)
    print("qwen3.5:9b", flush=True)
    print(f"{'='*60}", flush=True)
    for img in images:
        results.setdefault(img.name, {})
        for res in RESOLUTIONS:
            key = _cache_key("qwen35-9b", img.name, res)
            if key in cache:
                r = cache[key]
                print(f"  {img.name:40} {res:5d}px  cache  kw={len(r.get('keywords', []))}", flush=True)
            else:
                print(f"  {img.name:40} {res:5d}px  running...", end="", flush=True)
                r = _extract_ollama(img, res)
                cache[key] = r
                _save_cache(cache)
                kw_count = len(r.get("keywords", []))
                err = r.get("error", "")
                print(
                    f"\r  {img.name:40} {res:5d}px  done   kw={kw_count}"
                    + (f"  ERR={err}" if err else ""),
                    flush=True,
                )
            results[img.name][res] = {"qwen35-9b": cache[key]}

    # --- Copilot ---
    print(f"\n{'='*60}", flush=True)
    print("Copilot (gpt-4.1)", flush=True)
    print(f"{'='*60}", flush=True)
    for img in images:
        for res in RESOLUTIONS:
            key = _cache_key("copilot", img.name, res)
            if key in cache:
                r = cache[key]
                print(f"  {img.name:40} {res:5d}px  cache  kw={len(r.get('keywords', []))}", flush=True)
            else:
                print(f"  {img.name:40} {res:5d}px  running...", end="", flush=True)
                r = _extract_copilot(img, res)
                cache[key] = r
                _save_cache(cache)
                kw_count = len(r.get("keywords", []))
                err = r.get("error", "")
                print(
                    f"\r  {img.name:40} {res:5d}px  done   kw={kw_count}"
                    + (f"  ERR={err}" if err else ""),
                    flush=True,
                )
            results[img.name].setdefault(res, {})
            results[img.name][res]["copilot"] = cache[key]

    # --- Generate HTML report ---
    _generate_report(images, results)


def _generate_report(
    images: list[Path],
    results: dict[str, dict[int, dict[str, Any]]],
) -> None:
    import html as html_escape

    print("\nGenerating HTML report...", flush=True)

    html_parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>Resolution Impact Comparison</title>",
        "<style>",
        "body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:#0b1020;color:#e5e7eb;margin:0}",
        ".wrap{max-width:100%;margin:0 auto;padding:24px}",
        "h1{margin-bottom:4px}",
        ".subtitle{font-size:13px;color:#9ca3af;margin-bottom:24px}",
        "table{width:100%;border-collapse:collapse;background:#111827;border:1px solid #374151;margin-bottom:40px}",
        "th,td{border:1px solid #374151;vertical-align:top;padding:8px;font-size:13px}",
        "th{position:sticky;top:0;background:#1f2937;text-align:left}",
        ".img-header{display:flex;align-items:center;gap:12px;margin-bottom:16px}",
        ".thumb{max-width:180px;max-height:180px;border-radius:6px}",
        ".img-name{font-size:14px;font-weight:600}",
        ".desc{color:#93c5fd;font-style:italic;line-height:1.4;margin-bottom:6px}",
        ".kw{line-height:1.5;word-break:break-word}",
        ".meta{font-size:11px;color:#9ca3af;margin-top:4px}",
        ".err{color:#fca5a5}",
        ".res-label{font-weight:700;font-size:14px;white-space:nowrap}",
        ".model-header{background:#1e3a5f;color:#93c5fd;font-weight:700;font-size:14px;text-align:center}",
        ".summary-table{margin-top:32px}",
        ".summary-table td,.summary-table th{text-align:center;padding:6px 10px}",
        ".good{color:#86efac}",
        ".ok{color:#fde68a}",
        ".bad{color:#fca5a5}",
        "</style></head><body><div class='wrap'>",
        "<h1>Resolution Impact on Keyword &amp; Description Quality</h1>",
        f"<div class='subtitle'>{len(images)} images &middot; resolutions: {', '.join(str(r) for r in RESOLUTIONS)}px &middot; models: Copilot (gpt-4.1), qwen3.5:9b</div>",
    ]

    # Per-image tables
    for img in images:
        # Thumbnail
        thumb_img = ModelAdapter.load_image(img, max_dim=300)
        thumb_img.thumbnail((180, 180))
        buf = io.BytesIO()
        thumb_img.save(buf, format="JPEG", quality=82)
        thumb_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        html_parts.append(
            f"<div class='img-header'>"
            f"<img class='thumb' src='data:image/jpeg;base64,{thumb_b64}' alt='{html_escape.escape(img.name)}'>"
            f"<div class='img-name'>{html_escape.escape(img.name)}</div>"
            f"</div>"
        )
        html_parts.append("<table><thead><tr>")
        html_parts.append("<th>Resolution</th>")
        html_parts.append("<th class='model-header'>Copilot (gpt-4.1)</th>")
        html_parts.append("<th class='model-header'>qwen3.5:9b</th>")
        html_parts.append("</tr></thead><tbody>")

        for res in RESOLUTIONS:
            data = results.get(img.name, {}).get(res, {})
            html_parts.append("<tr>")

            # Resolution info
            w, h, jpeg_sz = _image_dims(img, res)
            html_parts.append(
                f"<td class='res-label'>{res}px<br>"
                f"<span class='meta'>{w}×{h} · {jpeg_sz // 1024}KB</span></td>"
            )

            for model in ["copilot", "qwen35-9b"]:
                d = data.get(model, {})
                desc = d.get("description", "")
                keywords = d.get("keywords", [])
                error = d.get("error", "")
                attempts = d.get("attempts", "?")
                t = d.get("inference_time_s", "?")
                prompt_tokens = d.get("prompt_eval_count", "")
                eval_tokens = d.get("eval_count", "")

                desc_html = f"<div class='desc'>{html_escape.escape(desc)}</div>" if desc else ""
                kw_html = html_escape.escape(", ".join(keywords)) if keywords else "<span class='err'>[no keywords]</span>"
                meta_parts = [f"kw={len(keywords)}", f"attempts={attempts}", f"{t}s"]
                if prompt_tokens:
                    meta_parts.append(f"prompt_tok={prompt_tokens}")
                if eval_tokens:
                    meta_parts.append(f"eval_tok={eval_tokens}")
                meta = " · ".join(meta_parts)
                err_html = f"<div class='meta err'>{html_escape.escape(error)}</div>" if error else ""

                html_parts.append(
                    f"<td>{desc_html}<div class='kw'>{kw_html}</div>"
                    f"<div class='meta'>{html_escape.escape(meta)}</div>{err_html}</td>"
                )
            html_parts.append("</tr>")
        html_parts.append("</tbody></table>")

    # --- Summary statistics table ---
    html_parts.append("<h2>Summary Statistics</h2>")
    html_parts.append("<table class='summary-table'><thead><tr>")
    html_parts.append("<th>Resolution</th>")
    for model_label in ["Copilot", "qwen3.5:9b"]:
        html_parts.append(f"<th>{model_label}<br>Avg Keywords</th>")
        html_parts.append(f"<th>{model_label}<br>Avg Time (s)</th>")
        html_parts.append(f"<th>{model_label}<br>Failures</th>")
    html_parts.append("</tr></thead><tbody>")

    for res in RESOLUTIONS:
        html_parts.append("<tr>")
        html_parts.append(f"<td class='res-label'>{res}px</td>")
        for model in ["copilot", "qwen35-9b"]:
            kw_counts = []
            times = []
            failures = 0
            for img in images:
                d = results.get(img.name, {}).get(res, {}).get(model, {})
                kw = d.get("keywords", [])
                kw_counts.append(len(kw))
                if d.get("inference_time_s"):
                    times.append(d["inference_time_s"])
                if d.get("error") or not kw:
                    failures += 1
            avg_kw = sum(kw_counts) / len(kw_counts) if kw_counts else 0
            avg_time = sum(times) / len(times) if times else 0

            kw_class = "good" if avg_kw >= 13 else ("ok" if avg_kw >= 10 else "bad")
            fail_class = "good" if failures == 0 else ("ok" if failures <= 1 else "bad")

            html_parts.append(f"<td class='{kw_class}'>{avg_kw:.1f}</td>")
            html_parts.append(f"<td>{avg_time:.1f}</td>")
            html_parts.append(f"<td class='{fail_class}'>{failures}/{len(images)}</td>")
        html_parts.append("</tr>")
    html_parts.append("</tbody></table>")

    html_parts.append("</div></body></html>")
    REPORT_PATH.write_text("".join(html_parts), encoding="utf-8")
    print(f"Report written: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
