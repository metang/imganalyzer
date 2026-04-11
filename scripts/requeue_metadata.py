"""Re-queue metadata extraction for images with incomplete EXIF data.

Bug: _parse_piexif() only extracted camera_make/model, blocking the richer
exifread fallback.  This script finds all affected images and resets their
metadata jobs to 'pending' so they get re-processed with the fixed code.
"""
import sqlite3
import os
import sys

db_path = os.path.expanduser("~/.cache/imganalyzer/imganalyzer.db")
if not os.path.exists(db_path):
    print("Database not found:", db_path)
    sys.exit(1)

db = sqlite3.connect(db_path)
db.execute("PRAGMA journal_mode=WAL")
db.execute("PRAGMA busy_timeout=5000")

# Incomplete metadata: piexif-only extraction (has camera but no date/lens/exposure)
INCOMPLETE = """
    SELECT am.image_id FROM analysis_metadata am
    WHERE am.camera_make IS NOT NULL
      AND am.date_time_original IS NULL
      AND am.lens_model IS NULL
      AND am.f_number IS NULL
      AND am.iso IS NULL
"""

count = db.execute(f"SELECT count(*) FROM ({INCOMPLETE})").fetchone()[0]
print(f"Found {count} images with incomplete metadata (piexif-only)")

if count == 0:
    print("Nothing to do.")
    db.close()
    sys.exit(0)

# Show current job statuses
print("\nCurrent metadata job statuses for affected images:")
for row in db.execute(f"""
    SELECT jq.status, count(*) c FROM job_queue jq
    WHERE jq.module = 'metadata'
      AND jq.image_id IN ({INCOMPLETE})
    GROUP BY jq.status
""").fetchall():
    print(f"  {row[0]}: {row[1]}")

# Clear analyzed_at so the worker doesn't skip via is_analyzed() check
cleared = db.execute(f"""
    UPDATE analysis_metadata
    SET analyzed_at = NULL
    WHERE image_id IN ({INCOMPLETE})
""").rowcount
db.commit()
print(f"\nCleared analyzed_at for {cleared} images")

# Reset done/failed/skipped jobs to pending
updated = db.execute(f"""
    UPDATE job_queue
    SET status = 'pending', attempts = 0,
        started_at = NULL, completed_at = NULL, error_message = NULL,
        skip_reason = NULL
    WHERE module = 'metadata'
      AND status IN ('done', 'failed', 'skipped')
      AND image_id IN ({INCOMPLETE})
""").rowcount
db.commit()
print(f"Reset {updated} metadata jobs to 'pending'")

# Insert jobs for images that have no metadata job row at all
inserted = db.execute(f"""
    INSERT OR IGNORE INTO job_queue (image_id, module, status, attempts)
    SELECT image_id, 'metadata', 'pending', 0
    FROM ({INCOMPLETE}) sub
    WHERE sub.image_id NOT IN (
        SELECT image_id FROM job_queue WHERE module = 'metadata'
    )
""").rowcount
if inserted:
    db.commit()
    print(f"Inserted {inserted} new metadata jobs")

final = db.execute(
    "SELECT count(*) FROM job_queue WHERE module = 'metadata' AND status = 'pending'"
).fetchone()[0]
print(f"\nTotal pending metadata jobs now: {final}")
db.close()
