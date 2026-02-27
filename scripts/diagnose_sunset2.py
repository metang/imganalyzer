"""Deep diagnostic: understand why irrelevant images rank above actual sunsets."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from imganalyzer.db.connection import get_db
from imganalyzer.db.search import SearchEngine, _DESC_QUALITY_THRESHOLD
from imganalyzer.embeddings.clip_embedder import CLIPEmbedder, vector_from_bytes, cosine_similarity

conn = get_db()
conn.row_factory = __import__("sqlite3").Row
embedder = CLIPEmbedder()

QUERY = "sunset"
visual_q = f"a photo of {QUERY}"
visual_vec = vector_from_bytes(embedder.embed_text(visual_q))
text_vec   = vector_from_bytes(embedder.embed_text(QUERY))

# ── Section 1: Raw image_clip cosine scores (top 20) ──────────────────────────
print("=" * 70)
print("SECTION 1: Top 20 by image_clip cosine (visual signal only)")
print("=" * 70)
img_rows = conn.execute(
    "SELECT image_id, vector FROM embeddings WHERE embedding_type='image_clip'"
).fetchall()
img_sims = [(r["image_id"], cosine_similarity(visual_vec, vector_from_bytes(r["vector"])))
            for r in img_rows]
img_sims.sort(key=lambda x: -x[1])

for rank, (iid, sim) in enumerate(img_sims[:20]):
    row = conn.execute(
        """SELECT la.description, la.scene_type, ca.description AS cloud_desc
           FROM images i
           LEFT JOIN analysis_local_ai la ON la.image_id=i.id
           LEFT JOIN analysis_cloud_ai ca ON ca.image_id=i.id
           WHERE i.id=? LIMIT 1""", [iid]).fetchone()
    d = (row["cloud_desc"] or row["description"] or "") if row else ""
    print(f"  #{rank+1:2d} id={iid:5d} cosine={sim:.4f}  {d[:100]!r}")

# ── Section 2: Rich-desc gate — how many pass? ────────────────────────────────
print()
print("=" * 70)
print("SECTION 2: description_clip — rich vs gated-out")
print("=" * 70)
engine = SearchEngine(conn)
rich_ids = engine._get_rich_desc_image_ids()
desc_rows = conn.execute(
    "SELECT image_id, vector FROM embeddings WHERE embedding_type='description_clip'"
).fetchall()
rich_desc = [(r["image_id"], cosine_similarity(text_vec, vector_from_bytes(r["vector"])))
             for r in desc_rows if r["image_id"] in rich_ids]
gated_out = len(desc_rows) - len(rich_desc)
rich_desc.sort(key=lambda x: -x[1])
print(f"  Total description_clip embeddings: {len(desc_rows)}")
print(f"  Rich (pass gate, len>={_DESC_QUALITY_THRESHOLD}): {len(rich_desc)}")
print(f"  Gated out (short local-AI desc): {gated_out}")
print()
print("  Top 20 description_clip (rich only):")
for rank, (iid, sim) in enumerate(rich_desc[:20]):
    row = conn.execute(
        """SELECT ca.description AS cloud_desc
           FROM analysis_cloud_ai ca WHERE ca.image_id=? LIMIT 1""", [iid]).fetchone()
    d = (row["cloud_desc"] or "") if row else ""
    print(f"  #{rank+1:2d} id={iid:5d} cosine={sim:.4f}  {d[:100]!r}")

# ── Section 3: RRF fusion trace for top-20 final results ──────────────────────
print()
print("=" * 70)
print("SECTION 3: RRF fusion — why do irrelevant images rank high?")
print("=" * 70)

img_ranks  = {iid: rank for rank, (iid, _) in enumerate(img_sims)}
desc_ranks = {iid: rank for rank, (iid, _) in enumerate(rich_desc)}

def rrf(rank, k=60):
    return 1.0 / (rank + 1 + k)

all_ids = set(img_ranks) | set(desc_ranks)
fused = []
for iid in all_ids:
    s = 0.0
    if iid in img_ranks:  s += rrf(img_ranks[iid])
    if iid in desc_ranks: s += rrf(desc_ranks[iid])
    fused.append((iid, s))
fused.sort(key=lambda x: -x[1])

print(f"  {'Rank':>4}  {'ID':>5}  {'RRF':>7}  {'img_rank':>8}  {'img_cos':>7}  {'desc_rank':>9}  {'desc_cos':>8}  description")
img_cos_map  = dict(img_sims)
desc_cos_map = dict(rich_desc)
for rank, (iid, score) in enumerate(fused[:20]):
    ir   = img_ranks.get(iid, -1)
    ic   = img_cos_map.get(iid, 0)
    dr   = desc_ranks.get(iid, -1)
    dc   = desc_cos_map.get(iid, 0)
    row = conn.execute(
        """SELECT la.description, ca.description AS cloud_desc
           FROM images i
           LEFT JOIN analysis_local_ai la ON la.image_id=i.id
           LEFT JOIN analysis_cloud_ai ca ON ca.image_id=i.id
           WHERE i.id=? LIMIT 1""", [iid]).fetchone()
    d = (row["cloud_desc"] or row["description"] or "") if row else ""
    ir_str  = str(ir+1) if ir >= 0 else "N/A"
    dr_str  = str(dr+1) if dr >= 0 else "N/A"
    print(f"  #{rank+1:2d}  id={iid:5d}  rrf={score:.5f}  "
          f"img={ir_str:>5}({ic:.3f})  desc={dr_str:>5}({dc:.3f})  {d[:80]!r}")

# ── Section 4: True sunset images — where do they rank? ───────────────────────
print()
print("=" * 70)
print("SECTION 4: Known sunset images — their ranks in each list")
print("=" * 70)
sunset_keywords = ["sunset", "sunrise", "dusk", "golden hour", "sun low", "sun on the horizon"]
sunset_rows = conn.execute("""
    SELECT i.id, la.description, la.scene_type, ca.description AS cloud_desc
    FROM images i
    LEFT JOIN analysis_local_ai la ON la.image_id=i.id
    LEFT JOIN analysis_cloud_ai ca ON ca.image_id=i.id
    WHERE lower(la.description) LIKE '%sunset%'
       OR lower(la.description) LIKE '%sunrise%'
       OR lower(la.description) LIKE '%dusk%'
       OR lower(la.scene_type)  LIKE '%sunset%'
       OR lower(ca.description) LIKE '%sunset%'
       OR lower(ca.description) LIKE '%sunrise%'
       OR lower(ca.description) LIKE '%dusk%'
       OR lower(ca.description) LIKE '%golden hour%'
    LIMIT 30
""").fetchall()

fused_rank_map = {iid: rank for rank, (iid, _) in enumerate(fused)}
print(f"  Found {len(sunset_rows)} images with sunset-related descriptions")
for r in sunset_rows:
    iid  = r["id"]
    ir   = img_ranks.get(iid, -1)
    ic   = img_cos_map.get(iid, 0.0)
    dr   = desc_ranks.get(iid, -1)
    dc   = desc_cos_map.get(iid, 0.0)
    fr   = fused_rank_map.get(iid, -1)
    d    = r["cloud_desc"] or r["description"] or ""
    print(f"  id={iid:5d}  final_rank={fr+1 if fr>=0 else 'N/A':>5}  "
          f"img_rank={ir+1 if ir>=0 else 'N/A':>5}({ic:.3f})  "
          f"desc_rank={dr+1 if dr>=0 else 'N/A':>6}({dc:.3f})  "
          f"{d[:80]!r}")

# ── Section 5: Coverage — how many images have image_clip? ────────────────────
print()
print("=" * 70)
print("SECTION 5: Embedding coverage")
print("=" * 70)
total = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
has_img = conn.execute("SELECT COUNT(DISTINCT image_id) FROM embeddings WHERE embedding_type='image_clip'").fetchone()[0]
has_desc = conn.execute("SELECT COUNT(DISTINCT image_id) FROM embeddings WHERE embedding_type='description_clip'").fetchone()[0]
has_both = conn.execute("""
    SELECT COUNT(*) FROM (
      SELECT image_id FROM embeddings WHERE embedding_type='image_clip'
      INTERSECT
      SELECT image_id FROM embeddings WHERE embedding_type='description_clip'
    )""").fetchone()[0]
print(f"  Total images:           {total}")
print(f"  Has image_clip:         {has_img}  ({100*has_img/total:.1f}%)")
print(f"  Has description_clip:   {has_desc}  ({100*has_desc/total:.1f}%)")
print(f"  Has both:               {has_both}  ({100*has_both/total:.1f}%)")
print(f"  Has neither (invisible):{total - has_img - (has_desc - has_both)}")
print(f"  Rich desc (pass gate):  {len(rich_ids)}")
