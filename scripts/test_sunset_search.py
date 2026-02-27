"""Run semantic search for 'sunset' directly against the DB and print top results with descriptions."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from imganalyzer.db.connection import get_db
from imganalyzer.db.search import SearchEngine

QUERY = "sunset"
LIMIT = 20

conn = get_db()
engine = SearchEngine(conn)

print(f"Searching for: '{QUERY}' (mode=semantic, limit={LIMIT})\n")
results = engine.search(QUERY, limit=LIMIT, mode="semantic")

for i, r in enumerate(results):
    image_id = r["image_id"]
    score = r["score"]

    # Fetch description from DB
    row = conn.execute(
        """
        SELECT
            la.description AS local_desc,
            la.scene_type,
            la.main_subject,
            ca.description AS cloud_desc
        FROM images i
        LEFT JOIN analysis_local_ai la ON la.image_id = i.id
        LEFT JOIN analysis_cloud_ai ca ON ca.image_id = i.id
        WHERE i.id = ?
        LIMIT 1
        """,
        [image_id],
    ).fetchone()

    local_desc = row["local_desc"] if row else None
    cloud_desc = row["cloud_desc"] if row else None
    scene = row["scene_type"] if row else None
    subject = row["main_subject"] if row else None

    best_desc = cloud_desc or local_desc or "(no description)"
    print(f"Rank {i+1:2d} | id={image_id:5d} | score={score:.5f}")
    print(f"         scene={scene!r}  subject={subject!r}")
    print(f"         desc={best_desc[:200]!r}")
    print()
