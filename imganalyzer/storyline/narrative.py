"""AI narrative generation for story chapters.

Uses Qwen 3.5 VL via Ollama (same infrastructure as the caption module)
to generate chapter summaries from image descriptions, keywords, and
metadata.  Falls back to heuristic templates when Ollama is unavailable.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any


def generate_chapter_narrative(
    conn: sqlite3.Connection,
    chapter_id: str,
    *,
    use_ai: bool = True,
    max_descriptions: int = 20,
) -> str:
    """Generate an AI narrative summary for a chapter.

    Collects descriptions and keywords from the chapter's images,
    then asks Ollama to summarize them into a 1-2 sentence narrative.
    Falls back to a heuristic template if Ollama is not available.
    """
    chapter = conn.execute(
        "SELECT * FROM story_chapters WHERE id = ?", [chapter_id]
    ).fetchone()
    if chapter is None:
        return ""

    # Collect image descriptions from this chapter's moments (single query)
    descriptions: list[str] = []
    keywords_set: set[str] = set()
    locations: set[str] = set()

    rows = conn.execute(
        "SELECT ac.description, ac.keywords, am.location_city "
        "FROM story_moments sm "
        "JOIN moment_images mi ON mi.moment_id = sm.id "
        "JOIN analysis_caption ac ON ac.image_id = mi.image_id "
        "LEFT JOIN analysis_metadata am ON am.image_id = mi.image_id "
        "WHERE sm.chapter_id = ? "
        "ORDER BY sm.sort_order, mi.sort_order "
        "LIMIT ?",
        [chapter_id, max_descriptions],
    ).fetchall()

    for r in rows:
        if r["description"]:
            descriptions.append(r["description"])
        if r["keywords"]:
            try:
                kws = json.loads(r["keywords"]) if isinstance(r["keywords"], str) else r["keywords"]
                if isinstance(kws, list):
                    keywords_set.update(str(k) for k in kws[:10])
            except (json.JSONDecodeError, TypeError):
                pass
        if r["location_city"]:
            locations.add(r["location_city"])

    if not descriptions:
        return _heuristic_summary(chapter, keywords_set, locations)

    if not use_ai:
        return _heuristic_summary(chapter, keywords_set, locations)

    # Try AI generation via Ollama
    try:
        return _generate_with_ollama(
            descriptions[:max_descriptions],
            keywords_set,
            locations,
            chapter,
        )
    except (RuntimeError, OSError, ValueError, KeyError):
        return _heuristic_summary(chapter, keywords_set, locations)


def _generate_with_ollama(
    descriptions: list[str],
    keywords: set[str],
    locations: set[str],
    chapter: sqlite3.Row,
) -> str:
    """Use Ollama to generate a narrative summary."""
    from imganalyzer.analysis.ai.ollama import OllamaAI

    desc_text = "\n".join(f"- {d}" for d in descriptions[:15])
    kw_text = ", ".join(sorted(keywords)[:20])
    loc_text = ", ".join(sorted(locations)) if locations else "unknown location"

    prompt = (
        f"You are writing a photo album caption. Based on these photo descriptions "
        f"from a chapter titled \"{chapter['title'] or 'Untitled'}\", write a "
        f"1-2 sentence narrative summary that captures the mood and story.\n\n"
        f"Location: {loc_text}\n"
        f"Keywords: {kw_text}\n"
        f"Photo descriptions:\n{desc_text}\n\n"
        f"Write ONLY the 1-2 sentence summary, nothing else."
    )

    ai = OllamaAI()
    # Use text-only mode (no image needed)
    result = ai.generate_text(prompt)
    return result.strip() if result else _heuristic_summary(chapter, keywords, locations)


def _heuristic_summary(
    chapter: sqlite3.Row,
    keywords: set[str],
    locations: set[str],
) -> str:
    """Generate a template-based summary when AI is unavailable."""
    parts: list[str] = []

    count = chapter["image_count"] or 0
    title = chapter["title"] or ""

    if locations:
        parts.append(f"Captured in {', '.join(sorted(locations))}")
    elif title:
        parts.append(title)

    if count > 0:
        parts.append(f"with {count} photos")

    top_keywords = sorted(keywords)[:5]
    if top_keywords:
        parts.append(f"featuring {', '.join(top_keywords)}")

    return " ".join(parts) + "." if parts else ""


def generate_all_chapter_narratives(
    conn: sqlite3.Connection,
    album_id: str,
    *,
    use_ai: bool = True,
) -> int:
    """Generate narratives for all chapters in an album.

    Returns the number of chapters updated.
    """
    chapters = conn.execute(
        "SELECT id FROM story_chapters WHERE album_id = ? ORDER BY sort_order",
        [album_id],
    ).fetchall()

    updated = 0
    for (chapter_id,) in chapters:
        summary = generate_chapter_narrative(conn, chapter_id, use_ai=use_ai)
        if summary:
            conn.execute(
                "UPDATE story_chapters SET summary = ? WHERE id = ?",
                [summary, chapter_id],
            )
            updated += 1

    if updated:
        conn.commit()
    return updated
