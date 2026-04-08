"""Incremental album refresh — check new images against existing album rules.

Called after image analysis completes to add matching images to smart albums
without re-running the full rule query.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from typing import Any


def check_and_add_image(
    conn: sqlite3.Connection,
    image_id: int,
) -> list[str]:
    """Check if a newly analyzed image matches any smart album rules.

    Returns list of album IDs the image was added to.
    """
    albums = conn.execute(
        "SELECT id, rules FROM smart_albums"
    ).fetchall()

    if not albums:
        return []

    added_to: list[str] = []
    for album in albums:
        album_id = album["id"]
        rules = json.loads(album["rules"])

        # Skip if image already in album
        existing = conn.execute(
            "SELECT 1 FROM album_items WHERE album_id = ? AND image_id = ?",
            [album_id, image_id],
        ).fetchone()
        if existing:
            continue

        # Check if image matches rules
        from imganalyzer.storyline.albums import check_image_against_rules
        if check_image_against_rules(conn, image_id, rules):
            conn.execute(
                "INSERT OR IGNORE INTO album_items (album_id, image_id) VALUES (?, ?)",
                [album_id, image_id],
            )
            conn.execute(
                "UPDATE smart_albums SET item_count = item_count + 1, "
                "updated_at = datetime('now') WHERE id = ?",
                [album_id],
            )
            added_to.append(album_id)

    if added_to:
        conn.commit()

    return added_to


def refresh_all_albums(conn: sqlite3.Connection) -> dict[str, int]:
    """Refresh membership of all smart albums.  Returns {album_id: new_count}."""
    from imganalyzer.storyline.albums import refresh_membership

    albums = conn.execute("SELECT id FROM smart_albums").fetchall()
    results: dict[str, int] = {}
    for album in albums:
        count = refresh_membership(conn, album["id"])
        results[album["id"]] = count
    return results
