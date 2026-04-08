"""Export a story album as a standalone HTML page.

Generates a self-contained HTML file with embedded CSS and base64-encoded
hero thumbnails.  The exported page uses a responsive timeline layout
that works offline.
"""
from __future__ import annotations

import base64
import sqlite3
from io import BytesIO
from pathlib import Path
from typing import Any

from imganalyzer.readers import open_as_pil
from imganalyzer.storyline.generator import (
    get_chapter_moments,
    get_story_chapters,
)


def export_story_html(
    conn: sqlite3.Connection,
    album_id: str,
    output_path: str | Path,
    *,
    include_thumbnails: bool = True,
    max_heroes_per_chapter: int = 6,
) -> Path:
    """Export an album's story as a standalone HTML file.

    Parameters
    ----------
    conn:
        Database connection.
    album_id:
        Smart album ID.
    output_path:
        Where to write the HTML file.
    include_thumbnails:
        If True, embed hero thumbnails as base64 data URLs.
    max_heroes_per_chapter:
        Max number of hero images to show per chapter.

    Returns
    -------
    Path
        The output file path.
    """
    output_path = Path(output_path)

    album = conn.execute(
        "SELECT * FROM smart_albums WHERE id = ?", [album_id]
    ).fetchone()
    if album is None:
        raise ValueError(f"Album {album_id} not found")

    chapters = get_story_chapters(conn, album_id)

    # Build chapter data with moments
    chapter_data: list[dict[str, Any]] = []
    for ch in chapters:
        moments = get_chapter_moments(conn, ch["id"])
        hero_ids = [
            m["hero_image_id"]
            for m in moments[:max_heroes_per_chapter]
            if m.get("hero_image_id")
        ]

        # Load hero thumbnails from files
        thumbs: list[str] = []
        if include_thumbnails and hero_ids:
            for hid in hero_ids:
                row = conn.execute(
                    "SELECT file_path FROM images WHERE id = ?", [hid]
                ).fetchone()
                if row and row["file_path"]:
                    thumb_data = _load_thumbnail_base64(row["file_path"])
                    if thumb_data:
                        thumbs.append(thumb_data)

        chapter_data.append({
            "title": ch.get("title") or "Untitled",
            "summary": ch.get("summary") or "",
            "location": ch.get("location") or "",
            "start_date": ch.get("start_date") or "",
            "end_date": ch.get("end_date") or "",
            "image_count": ch.get("image_count", 0),
            "moment_count": ch.get("moment_count", 0),
            "thumbnails": thumbs,
        })

    html = _render_html(
        album_name=album["name"],
        album_description=album["description"] or "",
        total_images=album["item_count"],
        chapters=chapter_data,
    )

    output_path.write_text(html, encoding="utf-8")
    return output_path


def _load_thumbnail_base64(file_path: str) -> str | None:
    """Load an image file and return a resized JPEG thumbnail data URL."""
    try:
        p = Path(file_path)
        if not p.exists():
            return None
        img = open_as_pil(p)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.thumbnail((400, 300))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except (OSError, ValueError, TypeError):
        return None


def _render_html(
    album_name: str,
    album_description: str,
    total_images: int,
    chapters: list[dict[str, Any]],
) -> str:
    """Render the HTML page."""
    chapter_html_parts: list[str] = []
    for idx, ch in enumerate(chapters):
        thumbs_html = ""
        if ch["thumbnails"]:
            thumb_imgs = "".join(
                f'<img src="{t}" alt="" class="hero-thumb" />'
                for t in ch["thumbnails"]
            )
            thumbs_html = f'<div class="hero-grid">{thumb_imgs}</div>'

        date_range = ""
        if ch["start_date"]:
            date_range = ch["start_date"][:10]
            if ch["end_date"] and ch["end_date"][:10] != ch["start_date"][:10]:
                date_range += f' – {ch["end_date"][:10]}'

        chapter_html_parts.append(f"""
    <div class="chapter">
      <div class="chapter-marker">{idx + 1}</div>
      <div class="chapter-content">
        <h2>{_escape(ch["title"])}</h2>
        <div class="meta">
          {f'<span class="location">{_escape(ch["location"])}</span>' if ch["location"] else ''}
          {f'<span class="dates">{date_range}</span>' if date_range else ''}
          <span class="counts">{ch["image_count"]} photos · {ch["moment_count"]} moments</span>
        </div>
        {f'<p class="summary">{_escape(ch["summary"])}</p>' if ch["summary"] else ''}
        {thumbs_html}
      </div>
    </div>""")

    chapters_html = "\n".join(chapter_html_parts)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{_escape(album_name)}</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           background: #1a1a1a; color: #e0e0e0; line-height: 1.6; }}
    .header {{ text-align: center; padding: 3rem 1rem 2rem; border-bottom: 1px solid #333; }}
    .header h1 {{ font-size: 2rem; color: #fff; margin-bottom: 0.5rem; }}
    .header .desc {{ color: #888; font-size: 0.9rem; }}
    .header .stats {{ color: #666; font-size: 0.8rem; margin-top: 0.5rem; }}
    .timeline {{ max-width: 900px; margin: 2rem auto; padding: 0 1rem; }}
    .chapter {{ display: flex; gap: 1.5rem; margin-bottom: 2rem;
               padding: 1.5rem; background: #222; border-radius: 12px; }}
    .chapter-marker {{ width: 40px; height: 40px; border-radius: 50%; background: #2563eb;
                       color: #fff; display: flex; align-items: center; justify-content: center;
                       font-weight: 600; font-size: 0.9rem; flex-shrink: 0; }}
    .chapter-content {{ flex: 1; min-width: 0; }}
    .chapter-content h2 {{ font-size: 1.1rem; color: #fff; margin-bottom: 0.4rem; }}
    .meta {{ display: flex; gap: 1rem; flex-wrap: wrap; font-size: 0.8rem; color: #888; margin-bottom: 0.5rem; }}
    .location {{ color: #60a5fa; }}
    .summary {{ font-size: 0.9rem; color: #aaa; margin: 0.5rem 0; }}
    .hero-grid {{ display: flex; gap: 6px; flex-wrap: wrap; margin-top: 0.75rem; }}
    .hero-thumb {{ width: 120px; height: 90px; object-fit: cover; border-radius: 6px; }}
    @media (max-width: 600px) {{
      .chapter {{ flex-direction: column; gap: 0.75rem; }}
      .hero-thumb {{ width: 100px; height: 75px; }}
    }}
  </style>
</head>
<body>
  <div class="header">
    <h1>{_escape(album_name)}</h1>
    {f'<p class="desc">{_escape(album_description)}</p>' if album_description else ''}
    <p class="stats">{total_images} photos · {len(chapters)} chapters</p>
  </div>
  <div class="timeline">
{chapters_html}
  </div>
</body>
</html>"""


def _escape(text: str) -> str:
    """HTML-escape a string."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )
