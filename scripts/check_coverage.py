"""Quick coverage check: how many images now have image_clip embeddings."""
import sys
sys.path.insert(0, ".")
from imganalyzer.db.connection import get_db

conn = get_db()
rows = conn.execute(
    "SELECT embedding_type, COUNT(*) as cnt FROM embeddings GROUP BY embedding_type"
).fetchall()
for r in rows:
    print(f"  {r[0]}: {r[1]}")
total = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
print(f"total images: {total}")
