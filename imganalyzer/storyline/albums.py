"""Smart-album CRUD operations with materialized membership."""
from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from imganalyzer.storyline.rules import compile_rules, evaluate_rules


@dataclass
class SmartAlbum:
    """In-memory representation of a smart album."""

    id: str
    name: str
    description: str | None
    cover_image_id: int | None
    rules: dict[str, Any]
    story_enabled: bool
    sort_order: str
    item_count: int
    chapter_count: int
    created_at: str
    updated_at: str


def create_album(
    conn: sqlite3.Connection,
    name: str,
    rules: dict[str, Any],
    *,
    description: str | None = None,
    story_enabled: bool = True,
    sort_order: str = "chronological",
) -> SmartAlbum:
    """Create a smart album and materialize its membership."""
    album_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rules_json = json.dumps(rules)

    conn.execute(
        "INSERT INTO smart_albums "
        "(id, name, description, rules, story_enabled, sort_order, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [album_id, name, description, rules_json, int(story_enabled), sort_order, now, now],
    )

    # Materialize membership
    image_ids = evaluate_rules(conn, rules)
    if image_ids:
        conn.executemany(
            "INSERT OR IGNORE INTO album_items (album_id, image_id) VALUES (?, ?)",
            [(album_id, iid) for iid in image_ids],
        )
    conn.execute(
        "UPDATE smart_albums SET item_count = ? WHERE id = ?",
        [len(image_ids), album_id],
    )
    conn.commit()

    # Select cover image (first image by date, or highest quality)
    cover_row = conn.execute(
        "SELECT ai.image_id FROM album_items ai "
        "LEFT JOIN search_features sf ON sf.image_id = ai.image_id "
        "WHERE ai.album_id = ? "
        "ORDER BY sf.perception_iaa DESC NULLS LAST LIMIT 1",
        [album_id],
    ).fetchone()
    if cover_row:
        conn.execute(
            "UPDATE smart_albums SET cover_image_id = ? WHERE id = ?",
            [cover_row[0], album_id],
        )
        conn.commit()

    return get_album(conn, album_id)  # type: ignore[return-value]


def get_album(conn: sqlite3.Connection, album_id: str) -> SmartAlbum | None:
    """Fetch a single smart album by ID."""
    row = conn.execute(
        "SELECT * FROM smart_albums WHERE id = ?", [album_id]
    ).fetchone()
    if row is None:
        return None
    return _row_to_album(row)


def list_albums(conn: sqlite3.Connection) -> list[SmartAlbum]:
    """List all smart albums, ordered by creation date."""
    rows = conn.execute(
        "SELECT * FROM smart_albums ORDER BY created_at DESC"
    ).fetchall()
    return [_row_to_album(r) for r in rows]


def update_album(
    conn: sqlite3.Connection,
    album_id: str,
    *,
    name: str | None = None,
    description: str | None = ...,  # type: ignore[assignment]
    rules: dict[str, Any] | None = None,
    story_enabled: bool | None = None,
    sort_order: str | None = None,
) -> SmartAlbum | None:
    """Update an album.  If rules change, re-materialize membership."""
    album = get_album(conn, album_id)
    if album is None:
        return None

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sets: list[str] = ["updated_at = ?"]
    params: list[Any] = [now]

    if name is not None:
        sets.append("name = ?")
        params.append(name)
    if description is not ...:
        sets.append("description = ?")
        params.append(description)
    if story_enabled is not None:
        sets.append("story_enabled = ?")
        params.append(int(story_enabled))
    if sort_order is not None:
        sets.append("sort_order = ?")
        params.append(sort_order)

    rules_changed = False
    if rules is not None:
        sets.append("rules = ?")
        params.append(json.dumps(rules))
        rules_changed = True

    params.append(album_id)
    conn.execute(
        f"UPDATE smart_albums SET {', '.join(sets)} WHERE id = ?",
        params,
    )

    if rules_changed:
        refresh_membership(conn, album_id, rules)

    conn.commit()
    return get_album(conn, album_id)


def delete_album(conn: sqlite3.Connection, album_id: str) -> bool:
    """Delete a smart album and all its story data (CASCADE)."""
    cur = conn.execute("DELETE FROM smart_albums WHERE id = ?", [album_id])
    conn.commit()
    return cur.rowcount > 0


def refresh_membership(
    conn: sqlite3.Connection,
    album_id: str,
    rules: dict[str, Any] | None = None,
) -> int:
    """Re-evaluate rules and refresh album_items.  Returns new item count."""
    if rules is None:
        row = conn.execute(
            "SELECT rules FROM smart_albums WHERE id = ?", [album_id]
        ).fetchone()
        if row is None:
            return 0
        rules = json.loads(row[0])

    image_ids = evaluate_rules(conn, rules)

    conn.execute("DELETE FROM album_items WHERE album_id = ?", [album_id])
    if image_ids:
        conn.executemany(
            "INSERT INTO album_items (album_id, image_id) VALUES (?, ?)",
            [(album_id, iid) for iid in image_ids],
        )
    conn.execute(
        "UPDATE smart_albums SET item_count = ?, updated_at = datetime('now') WHERE id = ?",
        [len(image_ids), album_id],
    )
    conn.commit()
    return len(image_ids)


def get_album_image_ids(
    conn: sqlite3.Connection,
    album_id: str,
) -> list[int]:
    """Return all image IDs in an album (materialized)."""
    rows = conn.execute(
        "SELECT image_id FROM album_items WHERE album_id = ?", [album_id]
    ).fetchall()
    return [r[0] for r in rows]


def check_image_against_rules(
    conn: sqlite3.Connection,
    image_id: int,
    rules: dict[str, Any],
) -> bool:
    """Check whether a single image matches an album's rules.

    Used for incremental refresh when a new image is analyzed.
    """
    sql, params = compile_rules(rules)
    # Add image_id filter to the query
    sql += " AND i.id = ?" if " WHERE " in sql else " WHERE i.id = ?"
    params.append(image_id)
    row = conn.execute(sql, params).fetchone()
    return row is not None


def _row_to_album(row: sqlite3.Row) -> SmartAlbum:
    return SmartAlbum(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        cover_image_id=row["cover_image_id"],
        rules=json.loads(row["rules"]),
        story_enabled=bool(row["story_enabled"]),
        sort_order=row["sort_order"],
        item_count=row["item_count"],
        chapter_count=row["chapter_count"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
