"""Run Copilot vs qwen3.5:9b comparison on all test images.

Produces an HTML table with 3 columns:
1) image thumbnail
2) Copilot result (description, scene_type, main_subject, keywords)
3) qwen3.5:9b result (description, scene_type, main_subject, keywords)
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "model-eval"))

from model_eval.cache import get_cached, store_result
from model_eval.models.base import ModelAdapter
from model_eval.runner import collect_images

OLLAMA_URL = "http://localhost:11434"
IMAGE_DIR = REPO_ROOT / "test_images"
REPORT_PATH = REPO_ROOT / "model-eval" / "report_compare_copilot_qwen35_9b.html"

# Cache keys (not model names) for this specific comparison run.
COPILOT_KEY = "cmp-copilot-v3"
QWEN_KEY = "cmp-qwen35-9b-v3"

PROMPT = """Analyze this image and output only JSON matching this structure:
{
  "description": "",
  "scene_type": "",
  "main_subject": "",
  "lighting": "",
  "mood": "",
  "keywords": ["k1", "k2"],
  "technical_notes": ""
}

For description, use this style:
"You are a photo librarian writing a brief catalog entry. Write a short, natural paragraph describing what this image shows — the main subject, setting, and notable details. Keep it under 80 words. Plain text only."

Requirements:
- Fill every field.
- keywords must be a non-empty array with 10-15 concise tags.
- No markdown, no extra text, JSON only."""

_RAW_EXTENSIONS = {
    ".arw", ".cr2", ".cr3", ".crw", ".dng", ".nef", ".nrw", ".orf", ".pef",
    ".raf", ".raw", ".rw2", ".rwl", ".sr2", ".srf", ".srw", ".x3f", ".iiq",
    ".3fr", ".fff", ".mef", ".mos", ".mrw", ".rwz", ".erf",
}
_NEEDS_JPEG = _RAW_EXTENSIONS | {".heic", ".heif", ".avif"}


def _parse_json_response(text: str) -> dict[str, Any]:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1]
        t = t.rsplit("```", 1)[0]
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {"description": t, "keywords": []}


def _normalize_keywords(value: Any) -> list[str]:
    if isinstance(value, str):
        items = [x.strip() for x in value.split(",")]
    elif isinstance(value, list):
        items = [str(x).strip() for x in value]
    else:
        items = []

    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if not item:
            continue
        k = item.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(item)
    return out


def _normalize_result(parsed: dict[str, Any]) -> dict[str, Any]:
    out = {
        "description": str(parsed.get("description") or "").strip(),
        "scene_type": str(parsed.get("scene_type") or "").strip(),
        "main_subject": str(parsed.get("main_subject") or "").strip(),
        "lighting": str(parsed.get("lighting") or "").strip(),
        "mood": str(parsed.get("mood") or "").strip(),
        "technical_notes": str(parsed.get("technical_notes") or "").strip(),
        "keywords": _normalize_keywords(parsed.get("keywords")),
    }
    return out


def _is_valid_result(result: dict[str, Any]) -> bool:
    if not result.get("description"):
        return False
    if not result.get("scene_type"):
        return False
    if not result.get("main_subject"):
        return False
    kws = result.get("keywords") or []
    if not isinstance(kws, list) or len(kws) == 0:
        return False
    return True


def _encode_for_vision(path: Path, max_dim: int = 1568) -> str:
    img = ModelAdapter.load_image(path, max_dim=max_dim)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _thumb_b64(path: Path) -> str:
    img = ModelAdapter.load_image(path, max_dim=320)
    img.thumbnail((220, 220))
    b = io.BytesIO()
    img.save(b, format="JPEG", quality=82)
    return base64.b64encode(b.getvalue()).decode("utf-8")


def _temp_jpeg(path: Path, max_dim: int = 1568) -> Path:
    img = ModelAdapter.load_image(path, max_dim=max_dim)
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    img.save(tmp.name, format="JPEG", quality=85)
    return Path(tmp.name)


def _run_qwen(image_path: Path, retries: int = 6) -> dict[str, Any]:
    b64 = _encode_for_vision(image_path)
    start = time.perf_counter()
    last_text = ""

    for attempt in range(1, retries + 1):
        payload: dict[str, Any] = {
            "model": "qwen3.5:9b",
            "messages": [{"role": "user", "content": PROMPT, "images": [b64]}],
            "stream": False,
            "format": "json",
            "options": {"num_predict": 1000, "temperature": 0},
            "keep_alive": "0s",
        }
        req = request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read())

        text = (result.get("message", {}).get("content") or "").strip()
        last_text = text
        parsed = _parse_json_response(text)
        normalized = _normalize_result(parsed)
        if _is_valid_result(normalized):
            normalized["attempts"] = attempt
            normalized["inference_time_s"] = round(time.perf_counter() - start, 3)
            return normalized

    return {
        "description": "",
        "scene_type": "",
        "main_subject": "",
        "lighting": "",
        "mood": "",
        "technical_notes": "",
        "keywords": [],
        "attempts": retries,
        "inference_time_s": round(time.perf_counter() - start, 3),
        "error": f"Invalid/empty structured response after {retries} attempts",
        "raw_text": last_text[:1500],
    }


async def _run_copilot_async(analysis_path: str, retries: int = 3) -> dict[str, Any]:
    from copilot import CopilotClient

    start = time.perf_counter()
    last_text = ""
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
            event = await session.send_and_wait(
                {"prompt": PROMPT, "attachments": [{"type": "file", "path": analysis_path}]},
                timeout=180.0,
            )
            text = (getattr(event.data, "content", "") or "").strip() if event is not None else ""
            last_text = text
            parsed = _parse_json_response(text)
            normalized = _normalize_result(parsed)
            if _is_valid_result(normalized):
                normalized["attempts"] = attempt
                normalized["inference_time_s"] = round(time.perf_counter() - start, 3)
                return normalized

        return {
            "description": "",
            "scene_type": "",
            "main_subject": "",
            "lighting": "",
            "mood": "",
            "technical_notes": "",
            "keywords": [],
            "attempts": retries,
            "inference_time_s": round(time.perf_counter() - start, 3),
            "error": f"Invalid/empty structured response after {retries} attempts",
            "raw_text": last_text[:1500],
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


def _run_copilot(image_path: Path) -> dict[str, Any]:
    tmp: Path | None = None
    path = image_path
    if image_path.suffix.lower() in _NEEDS_JPEG:
        tmp = _temp_jpeg(image_path)
        path = tmp
    try:
        return asyncio.run(_run_copilot_async(str(path)))
    finally:
        if tmp is not None:
            try:
                tmp.unlink()
            except OSError:
                pass


def _result_html(model_label: str, result: dict[str, Any]) -> str:
    import html

    desc = html.escape(result.get("description") or "[missing]")
    scene = html.escape(result.get("scene_type") or "[missing]")
    subject = html.escape(result.get("main_subject") or "[missing]")
    kws = result.get("keywords") or []
    kw_html = "<ul class='kw'>" + "".join(
        f"<li>{html.escape(str(k))}</li>" for k in kws
    ) + "</ul>" if kws else "<div class='err'>[no keywords]</div>"

    meta = f"attempts={result.get('attempts', '?')} · {result.get('inference_time_s', '?')}s"
    error = result.get("error")
    err_html = f"<div class='meta err'>{html.escape(str(error))}</div>" if error else ""
    return (
        f"<div class='block'>"
        f"<div class='meta'><strong>{html.escape(model_label)}</strong></div>"
        f"<div><strong>Description:</strong> {desc}</div>"
        f"<div><strong>Scene type:</strong> {scene}</div>"
        f"<div><strong>Main subject:</strong> {subject}</div>"
        f"<div><strong>Keywords:</strong>{kw_html}</div>"
        f"<div class='meta'>{html.escape(meta)}</div>"
        f"{err_html}"
        f"</div>"
    )


def build_report(rows: list[dict[str, Any]]) -> None:
    import html

    parts: list[str] = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'>")
    parts.append("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    parts.append("<title>Copilot vs qwen3.5:9b comparison</title>")
    parts.append(
        "<style>"
        "body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:#0b1020;color:#e5e7eb;margin:0}"
        ".wrap{max-width:1900px;margin:0 auto;padding:24px}"
        "table{width:100%;border-collapse:collapse;table-layout:fixed;background:#111827;border:1px solid #374151}"
        "th,td{border:1px solid #374151;vertical-align:top;padding:10px}"
        "th{position:sticky;top:0;background:#1f2937;text-align:left}"
        "td:nth-child(1),th:nth-child(1){width:240px}"
        "td:nth-child(2),th:nth-child(2){width:780px}"
        "td:nth-child(3),th:nth-child(3){width:780px}"
        ".thumb{max-width:220px;max-height:220px;display:block;margin-bottom:8px;border-radius:6px}"
        ".name{font-size:12px;color:#cbd5e1;word-break:break-word}"
        ".kw{margin:4px 0 0 0;padding-left:18px;line-height:1.45}"
        ".kw li{margin:0 0 2px 0}"
        ".meta{font-size:11px;color:#9ca3af;margin-top:6px}"
        ".err{color:#fca5a5}"
        ".block > div{margin-bottom:6px}"
        "</style>"
    )
    parts.append("</head><body><div class='wrap'>")
    parts.append(
        f"<h1>Copilot vs qwen3.5:9b (description/scene/main subject/keywords)</h1>"
        f"<div class='meta'>{len(rows)} images · prompt without aesthetic fields</div>"
    )
    parts.append("<table><thead><tr>")
    parts.append("<th>Image thumbnail</th>")
    parts.append("<th>Copilot result</th>")
    parts.append("<th>qwen3.5:9b result</th>")
    parts.append("</tr></thead><tbody>")

    for row in rows:
        parts.append("<tr>")
        parts.append(
            f"<td><img class='thumb' src='data:image/jpeg;base64,{row['thumb']}' alt='{html.escape(row['image_name'])}'>"
            f"<div class='name'>{html.escape(row['image_name'])}</div></td>"
        )
        parts.append(f"<td>{_result_html('Copilot', row['copilot'])}</td>")
        parts.append(f"<td>{_result_html('qwen3.5:9b', row['qwen'])}</td>")
        parts.append("</tr>")

    parts.append("</tbody></table></div></body></html>")
    REPORT_PATH.write_text("".join(parts), encoding="utf-8")


def main() -> None:
    images = [p.resolve() for p in collect_images(IMAGE_DIR)]
    print(f"Images found: {len(images)}")

    # Run model comparisons one model at a time.
    rows: list[dict[str, Any]] = []
    for img in images:
        rows.append(
            {
                "path": img,
                "image_name": img.name,
                "thumb": _thumb_b64(img),
                "copilot": {},
                "qwen": {},
            }
        )

    print("\n=== Running Copilot ===")
    for i, row in enumerate(rows, 1):
        img = row["path"]
        cached = get_cached(COPILOT_KEY, img)
        if cached is None or not _is_valid_result(cached):
            result = _run_copilot(img)
            store_result(COPILOT_KEY, img, result)
            status = "computed"
            row["copilot"] = result
        else:
            status = "cache"
            row["copilot"] = cached
        print(f"  [{i}/{len(rows)}] {img.name} -> {status} kw={len(row['copilot'].get('keywords', []))}")

    print("\n=== Running qwen3.5:9b ===")
    for i, row in enumerate(rows, 1):
        img = row["path"]
        cached = get_cached(QWEN_KEY, img)
        if cached is None or not _is_valid_result(cached):
            result = _run_qwen(img)
            store_result(QWEN_KEY, img, result)
            status = "computed"
            row["qwen"] = result
        else:
            status = "cache"
            row["qwen"] = cached
        print(f"  [{i}/{len(rows)}] {img.name} -> {status} kw={len(row['qwen'].get('keywords', []))}")

    # Drop internal path before report.
    for row in rows:
        row.pop("path", None)

    build_report(rows)
    print(f"\nReport written: {REPORT_PATH}")


if __name__ == "__main__":
    main()

