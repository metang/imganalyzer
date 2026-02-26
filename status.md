# Project Status

## Latest Changes (committed `c7d0895`)

### Skip inaccessible images in `analyze` command
- **File:** `cli.py`
- Added `skipped = 0` counter and explicit `img_path.exists()` check before analysis loop body.
- Files that don't exist on disk are logged (`Skip: <name> — file not found or inaccessible`) and counted as skipped (not errors).
- XMP-exists skip also now increments `skipped`.
- Final summary line updated: shows `N/M file(s) processed, K skipped` when skipped count > 0.

## Previous Changes (committed `70cc815`)

### Aesthetic reason persisted end-to-end
- `SYSTEM_PROMPT_WITH_AESTHETIC` updated to request `aesthetic_reason` field from GPT-4.1.
- `_persist_result_to_db` now pops `aesthetic_reason` from `cloud_data` and passes it to `upsert_aesthetic`.
- All 17 DB rows have `aesthetic_score`, `aesthetic_label`, and `aesthetic_reason` populated.

---

# Phase 8 — Integration & Testing Status

## Completed Steps

### 8.1 Fix sqlite3 autocommit/transaction issue
- **File:** `db/connection.py`
- **Change:** Set `isolation_level=None` on `sqlite3.connect()` to disable Python's implicit transaction management. This ensures our explicit `BEGIN IMMEDIATE ... COMMIT` pattern in `_transaction` (modules.py) works without "cannot start a transaction within a transaction" errors.
- **Verified:** `repository.py` and `queue.py` methods that call `conn.commit()` outside `_transaction` blocks are harmless in autocommit mode. Methods called inside `_transaction` blocks (e.g. `upsert_metadata`, `upsert_technical`) do NOT call `conn.commit()` — the `_transaction.__exit__` handles commit/rollback.

### 8.2 Update old face CLI commands to use DB repository
- **File:** `cli.py` (lines 147–282 rewritten)
- **Change:** Replaced `register-face`, `list-faces`, `remove-face` commands that used the legacy JSON-based `FaceDatabase` (`analysis/ai/face_db.py`) with new versions using `get_db()` + `Repository`.
- `register-face`: Now accepts `--display` option for display name. Uses `repo.register_face_identity()` + `repo.add_face_embedding()`. Stores embedding as raw `float32` bytes in SQLite BLOB.
- `list-faces`: Uses `repo.list_face_identities()`. Now shows display name, aliases, and embedding count columns.
- `remove-face`: Uses `repo.remove_face_identity()`. CASCADE delete also removes associated embeddings.
- **Note:** The old `FaceDatabase` class in `face_db.py` is still used by `faces.py` at runtime for face matching during local AI analysis. That matching logic still reads from the JSON file. Full migration of the runtime matching to use DB embeddings would require changes to `faces.py` — deferred to a future step.

### 8.4 Integrate Analyzer class — single-file `analyze` also writes to DB
- **File:** `cli.py`
- **Change:** Added `_persist_result_to_db()` helper that stores `AnalysisResult` in the DB after `analyzer.analyze()` returns. Registers the image, upserts metadata/technical/AI data (routing to `upsert_local_ai` or `upsert_cloud_ai` based on backend), and updates the search index. All upserts wrapped in a single `BEGIN IMMEDIATE ... COMMIT` transaction for atomicity. Best-effort: if DB write fails, the XMP was already written so the user still gets output.

### 8.3 Wire XMP output into batch pipeline
- **Files:** `pipeline/modules.py`, `pipeline/worker.py`, `cli.py`
- **Change:** Added `write_xmp_from_db(repo, image_id)` function to `modules.py` that reconstructs `AnalysisResult` from DB data and writes XMP sidecar. Updated `Worker` class with `write_xmp` flag, `_xmp_candidates` tracking set, and `_write_pending_xmps()` method. Added `--no-xmp` CLI option to `run` command and wired `write_xmp=not no_xmp` to Worker constructor.
- **Design:** XMP is written after ALL modules for an image complete (not after each module), since XMP sidecars need data from multiple modules (metadata + technical + AI).

### 8.5 Update `__init__.py` exports
- **Status:** Skipped (low priority). CLI is the primary interface; programmatic users can import directly from submodules.

### 8.6 End-to-end testing
- **File:** `tests/test_imganalyzer.py` (18 new test cases added)
- **Test classes:** `TestDatabaseLayer` (8 tests), `TestJobQueue` (4 tests), `TestSearchIndex` (2 tests), `TestFaceIdentityDB` (3 tests), `TestPersistResultToDB` (1 test)
- **Coverage:** Image registration/idempotency, metadata/technical/local-AI/cloud-AI upserts, override protection (mask + get_full_result), queue enqueue/claim/dedup/batch/stats, FTS5 search index population, face identity CRUD with cascade delete, end-to-end DB persistence from AnalysisResult.
- **Result:** All 18 new tests pass. Full suite: 50 pass, 9 fail (all pre-existing failures due to missing `aesthetic.py` module and incomplete ObjectDetector mocks).

---

## Prior Phases (1–7) — All Complete

| Phase | Deliverables | Status |
|-------|-------------|--------|
| 1 — DB foundation | `db/schema.py`, `db/connection.py`, `db/repository.py` | Done |
| 2 — Queue engine | `db/queue.py` | Done |
| 3 — Module runners | `pipeline/modules.py` (override guard, people-guard, atomic writes) | Done |
| 4 — Face identity DB | `face_identities` + `face_embeddings` tables, CRUD in repository | Done |
| 5 — CLI commands | `ingest`, `run`, `status`, `rebuild`, `override`, `alias-face`, `rename-face`, `merge-face`, `search` | Done |
| 6 — CLIP embeddings | `embeddings/clip_embedder.py` | Done |
| 7 — Search | `db/search.py` (FTS5 + CLIP hybrid) | Done |
