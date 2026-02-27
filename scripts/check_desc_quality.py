"""Check description length distribution to calibrate quality threshold."""
import sqlite3
import os

db = os.path.expanduser("~/.cache/imganalyzer/imganalyzer.db")
conn = sqlite3.connect(db)
conn.row_factory = sqlite3.Row

# Distribution of local AI description lengths
print("=== Local AI description length distribution ===")
rows = conn.execute("""
SELECT length(description) as len, COUNT(*) as cnt
FROM analysis_local_ai
WHERE description IS NOT NULL
GROUP BY len / 20
ORDER BY len / 20
""").fetchall()
for r in rows:
    print(f"  len ~{r['len']:4d}  count={r['cnt']}")

print()

# Sample short vs long
print("=== Short descriptions (len < 50) ===")
rows = conn.execute("""
SELECT la.image_id, length(la.description) as len, la.description
FROM analysis_local_ai la
WHERE la.description IS NOT NULL AND length(la.description) < 50
ORDER BY RANDOM() LIMIT 10
""").fetchall()
for r in rows:
    print(f"  id={r['image_id']}  len={r['len']}  {repr(r['description'])}")

print()
print("=== Long descriptions (len > 100) ===")
rows = conn.execute("""
SELECT la.image_id, length(la.description) as len, la.description
FROM analysis_local_ai la
WHERE la.description IS NOT NULL AND length(la.description) > 100
ORDER BY RANDOM() LIMIT 5
""").fetchall()
for r in rows:
    print(f"  id={r['image_id']}  len={r['len']}  {repr(r['description'][:120])}")

print()
print("=== Cloud AI description presence ===")
row = conn.execute("""
SELECT COUNT(*) as total,
  SUM(CASE WHEN ca.description IS NOT NULL AND length(ca.description) > 50 THEN 1 ELSE 0 END) as cloud_rich
FROM images i
LEFT JOIN analysis_cloud_ai ca ON ca.image_id = i.id
""").fetchone()
if row:
    print(f"  total={row['total']}  cloud_rich={row['cloud_rich']}")
