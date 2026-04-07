"""Diagnose faces/embedding requeue loop."""
import sqlite3, pathlib
db = pathlib.Path.home() / ".cache" / "imganalyzer" / "imganalyzer.db"
c = sqlite3.connect(str(db), check_same_thread=False)
c.row_factory = sqlite3.Row
c.execute("PRAGMA journal_mode=WAL")
c.execute("PRAGMA busy_timeout=5000")

# Sample pending faces jobs — check their images' objects status
print("=== SAMPLE PENDING FACES JOBS ===")
for r in c.execute("""
    SELECT jq.id, jq.image_id, jq.module, jq.attempts, jq.queued_at, jq.started_at
    FROM job_queue jq
    WHERE jq.module='faces' AND jq.status='pending' AND jq.attempts <= jq.max_attempts
    LIMIT 10
"""):
    # Check if objects is analyzed for this image
    obj_job = c.execute(
        "SELECT status, attempts FROM job_queue WHERE image_id=? AND module='objects'",
        [r['image_id']]
    ).fetchone()
    obj_status = f"{obj_job['status']}(att={obj_job['attempts']})" if obj_job else "NO JOB"

    # Check analysis_objects table
    obj_analyzed = c.execute(
        "SELECT 1 FROM analysis_objects WHERE image_id=?", [r['image_id']]
    ).fetchone()
    has_objects_data = "YES" if obj_analyzed else "NO"

    print(f"  img={r['image_id']} faces_att={r['attempts']} queued={r['queued_at']}"
          f" objects_job={obj_status} objects_data={has_objects_data}")

# Same for embedding
print("\n=== SAMPLE PENDING EMBEDDING JOBS ===")
for r in c.execute("""
    SELECT jq.id, jq.image_id, jq.module, jq.attempts, jq.queued_at
    FROM job_queue jq
    WHERE jq.module='embedding' AND jq.status='pending' AND jq.attempts <= jq.max_attempts
    LIMIT 10
"""):
    # Check all module statuses for this image
    print(f"  img={r['image_id']} emb_att={r['attempts']} queued={r['queued_at']}")
    for m in c.execute(
        "SELECT module, status, attempts FROM job_queue WHERE image_id=? ORDER BY module",
        [r['image_id']]
    ):
        print(f"    {m['module']:12s}: {m['status']} att={m['attempts']}")

# Check: how many faces-pending images have objects NOT analyzed?
print("\n=== FACES PENDING: OBJECTS STATUS ===")
row = c.execute("""
    SELECT COUNT(DISTINCT fq.image_id) as cnt
    FROM job_queue fq
    WHERE fq.module='faces' AND fq.status='pending'
      AND fq.attempts <= fq.max_attempts
      AND NOT EXISTS (
          SELECT 1 FROM analysis_objects ao WHERE ao.image_id = fq.image_id
      )
""").fetchone()
print(f"  Faces pending WITHOUT objects data: {row['cnt']}")

row = c.execute("""
    SELECT COUNT(DISTINCT fq.image_id) as cnt
    FROM job_queue fq
    WHERE fq.module='faces' AND fq.status='pending'
      AND fq.attempts <= fq.max_attempts
      AND EXISTS (
          SELECT 1 FROM analysis_objects ao WHERE ao.image_id = fq.image_id
      )
""").fetchone()
print(f"  Faces pending WITH objects data: {row['cnt']}")
