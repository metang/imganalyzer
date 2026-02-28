# imganalyzer — Performance Analysis & Improvement Opportunities

**Model:** claude-sonnet-4.6  
**Date:** 2026-02-27  
**Target workload:** Batch ingestion and analysis of 500 K+ images

---

## Executive Summary

At 500 K images the system faces four compounding bottlenecks that will collectively
make a full analysis run take weeks rather than days:

1. **Each image is decoded from disk 5–8 times** — once per analysis module, with no
   shared in-memory buffer between them.
2. **The SQLite search index is rebuilt per-image per-module** — meaning 500 K images
   × 6 modules = 3 M expensive FTS5 DELETE+INSERT sequences.
3. **Reverse geocoding issues one blocking HTTP request per geotagged image** — with
   no cache, no batching, and subject to Nominatim rate-limiting.
4. **CLIP semantic search loads all embedding BLOBs into Python RAM** — at 500 K
   images this is ~2 GB of raw bytes plus the NumPy conversion, making every search
   query an O(N) memory spike.

The recommendations below are ranked by estimated throughput impact.  Items marked
**[CPU]**, **[GPU]**, **[MEM]**, or **[I/O]** indicate which resource is primarily
affected.

---

## Table of Contents

1. [Critical: Redundant Image Decoding](#1-critical-redundant-image-decoding)
2. [Critical: FTS5 Search Index Rebuild Storm](#2-critical-fts5-search-index-rebuild-storm)
3. [Critical: O(N) CLIP Embedding Search](#3-critical-on-clip-embedding-search)
4. [High: Reverse Geocoding — Blocking HTTP With No Cache](#4-high-reverse-geocoding--blocking-http-with-no-cache)
5. [High: SHA-256 Hash on Every File at Ingest](#5-high-sha-256-hash-on-every-file-at-ingest)
6. [High: Per-Job SQLite Commits (5 M Commits at 500 K Scale)](#6-high-per-job-sqlite-commits)
7. [High: Ingest Loop Issues N×M Individual DB Queries](#7-high-ingest-loop-issues-nm-individual-db-queries)
8. [Medium: RAW Full Decode for Every Module](#8-medium-raw-full-decode-for-every-module)
9. [Medium: GPU Modules Fully Serialized — No Micro-Batching](#9-medium-gpu-modules-fully-serialized--no-micro-batching)
10. [Medium: Technical Analysis — Double Resize + Expensive Per-Pixel Ops](#10-medium-technical-analysis--double-resize--expensive-per-pixel-ops)
11. [Medium: `find_face_by_alias` Full Table Scan in Hot Path](#11-medium-find_face_by_alias-full-table-scan-in-hot-path)
12. [Medium: `CLIPEmbedder` Re-reads File Already Decoded by ModuleRunner](#12-medium-clipembedder-re-reads-file-already-decoded-by-modulerunner)
13. [Low: `iter_image_ids()` Loads All IDs Into Memory](#13-low-iter_image_ids-loads-all-ids-into-memory)
14. [Low: Missing DB Indexes on High-Cardinality Filter Columns](#14-low-missing-db-indexes-on-high-cardinality-filter-columns)
15. [Low: Cloud AI — `asyncio.run()` Creates New Event Loop Per Call](#15-low-cloud-ai--asynciorun-creates-new-event-loop-per-call)
16. [Architecture: Multi-Worker GPU Pipeline](#16-architecture-multi-worker-gpu-pipeline)

---

## 1. Critical: Redundant Image Decoding

**Impact:** CPU, I/O, Memory  
**Files:** `imganalyzer/pipeline/modules.py:35–49`, `imganalyzer/embeddings/clip_embedder.py:42–93`

### Problem

`_read_image(path)` is called at the top of **every** `_run_*` method in
`ModuleRunner`:

```python
# modules.py — called independently in each of these:
def _run_metadata(...)  → _read_image(path)   # decode #1
def _run_technical(...) → _read_image(path)   # decode #2
def _run_blip2(...)     → _read_image(path)   # decode #3
def _run_objects(...)   → _read_image(path)   # decode #4
def _run_ocr(...)       → _read_image(path)   # decode #5
def _run_faces(...)     → _read_image(path)   # decode #6
def _run_cloud_ai(...)  → _read_image(path)   # decode #7
```

Additionally, `CLIPEmbedder.embed_image()` (called from `_run_embedding`) independently
re-opens and decodes the original file via rawpy or Pillow without receiving the already-
decoded `rgb_array` from `_run_embedding`.

For a 50 MP RAW file (Sony A7R IV = 61 MP, ~25 MB on disk, ~175 MB when decoded to
uint8 RGB), a single image going through all 7 modules consumes:

| Metric | Per-image | At 500 K images |
|---|---|---|
| File reads | 7 | 3.5 M |
| rawpy postprocess calls | 7 | 3.5 M |
| Peak RAM per image | ~175 MB × 2 (double-buffered) | — |
| Decode CPU time (est.) | ~14 s | ~81 days |

### Root Cause

`ModuleRunner` was designed so each module is independently re-runnable (e.g., in
`rebuild_module`). This is correct for resilience, but when an image is being processed
for the first time and multiple modules run sequentially, the decode is needlessly
repeated.

### Fix

Introduce a **per-job image cache** inside `_process_job`. The cache holds the decoded
`image_data` dict for the current image and is cleared when the image changes:

```python
# pipeline/worker.py — _process_job

# Thread-local dict: {image_id: image_data}
# Evict when image_id changes (each thread processes one image at a time).

def _process_job(self, job):
    image_id = job["image_id"]
    _, repo, queue, runner = self._get_thread_db()

    # --- NEW: populate per-thread image cache before calling runner.run()
    local = self._local
    if getattr(local, "cached_image_id", None) != image_id:
        path = Path(repo.get_image(image_id)["file_path"])
        if path.exists() and job["module"] != "embedding":
            local.cached_image_data = _read_image(path)
        else:
            local.cached_image_data = None
        local.cached_image_id = image_id

    runner.run(image_id, job["module"],
               image_data=local.cached_image_data)   # pass cached data
    ...
```

Each `_run_*` method then accepts an optional `image_data` parameter and falls back to
calling `_read_image` only when the caller did not provide it (preserving backward
compatibility with `rebuild_module` which runs single modules in isolation).

For CLIP specifically, `_run_embedding` should pass the already-decoded PIL image to
`CLIPEmbedder.embed_image_pil(pil_img)` rather than letting the embedder re-open the
file.

**Expected gain:** 6× reduction in file reads and decode CPU time for first-pass
processing. For 500 K RAW images this is the single largest throughput improvement.

---

## 2. Critical: FTS5 Search Index Rebuild Storm

**Impact:** CPU, I/O (SQLite write amplification)  
**Files:** `imganalyzer/db/repository.py:550–676`, `imganalyzer/pipeline/modules.py:143,179,322`

### Problem

`repo.update_search_index(image_id)` is called after every module write that touches
searchable text:

```python
# modules.py
_run_metadata  → upsert_metadata   → update_search_index(image_id)  # line 143
_run_local_ai  → upsert_local_ai   → update_search_index(image_id)  # line 179
_run_cloud_ai  → upsert_cloud_ai   → update_search_index(image_id)  # line 322
```

Each `update_search_index` call:
1. Issues **6+ `SELECT *` queries** (local_ai, blip2, faces, cloud_ai rows, metadata, FTS5 exists check)
2. Calls `find_face_by_alias()` for every detected face name (itself O(N) — see §11)
3. Issues a `DELETE FROM search_index WHERE image_id = ?`
4. Issues an `INSERT INTO search_index ...`

At 500 K images × 3 modules × (6 SELECTs + 1 DELETE + 1 INSERT) = **12 M SQL
statements** just for FTS maintenance.

FTS5 insertions are expensive: each insert triggers tokenisation, index update, and
(periodically) segment merges. At 1.5 M inserts the FTS5 segment merge overhead
dominates wall-clock time.

### Fix A — Defer FTS rebuild to post-processing

Mark images as "FTS-dirty" in a lightweight flag column rather than rebuilding
immediately. Run a single bulk FTS rebuild at the end of a worker session:

```python
# After worker.run() returns (worker.py:_run_loop end):
repo.rebuild_search_index_bulk()

# repository.py:
def rebuild_search_index_bulk(self):
    """Single-pass FTS rebuild for all dirty images."""
    dirty_ids = self.conn.execute(
        "SELECT id FROM images WHERE fts_dirty = 1"
    ).fetchall()
    self.conn.execute("DELETE FROM search_index")
    for img_id in dirty_ids:
        self._build_fts_row(img_id)   # existing logic, no commit per row
    self.conn.execute("UPDATE images SET fts_dirty = 0")
    self.conn.commit()
```

This reduces 1.5 M FTS writes to **one batched FTS populate** at session end.

### Fix B — Batch FTS using `INSERT OR REPLACE` with a WAL checkpoint hint

If Fix A is too invasive, a lighter improvement is to coalesce FTS updates per image
per module-batch rather than per-module. Collect dirty image IDs in a set and flush at
`mark_done` time:

```python
# queue.py — mark_done: add image_id to a flush set
# After each batch_size group: flush_fts(dirty_ids) in one transaction
```

### Fix C — Write FTS entry once, update incrementally

Instead of DELETE+INSERT, use FTS5 `content=` or `contentless=` virtual tables to avoid
storing duplicated text, and update only the changed columns. This is a schema change
but reduces write amplification to a single column update.

---

## 3. Critical: O(N) CLIP Embedding Search

**Impact:** Memory, CPU  
**Files:** `imganalyzer/db/search.py:350–365`, `imganalyzer/db/repository.py:540–546`

### Problem

`_semantic_search` fetches **all** embeddings from the database before scoring:

```python
# search.py:350
image_embeddings = self.repo.get_all_embeddings("image_clip")
desc_embeddings  = self.repo.get_all_embeddings("description_clip")
```

`get_all_embeddings` (repository.py:540) does:
```python
rows = self.conn.execute(
    "SELECT image_id, vector FROM embeddings WHERE embedding_type = ?", ...
).fetchall()
return [(r["image_id"], r["vector"]) for r in rows]
```

At 500 K images × 2 embedding types × 768 dims × 4 bytes:

| | Value |
|---|---|
| Raw BLOB data | ~3.1 GB |
| Python list overhead | ~200 MB |
| NumPy conversion (per search) | +3.1 GB peak |
| Total RAM per query | **~6 GB** |
| Query latency (RAM-only cosine) | ~8–25 s on CPU |

Every search query causes a full 6 GB allocation-and-free cycle.

### Fix A — FAISS ANN Index (primary recommendation)

Replace the brute-force cosine scan with a FAISS approximate nearest-neighbor index,
stored as a file alongside the DB:

```python
# embeddings/faiss_index.py (new file)
import faiss
import numpy as np

class FAISSIndex:
    def __init__(self, index_path: Path):
        self.path = index_path
        self._index = None   # faiss.IndexFlatIP (inner product = cosine on unit vecs)
        self._ids = None     # np.ndarray of image_ids parallel to index rows

    def build(self, embeddings: list[tuple[int, bytes]]):
        vectors = np.stack([np.frombuffer(v, np.float32) for _, v in embeddings])
        ids = np.array([i for i, _ in embeddings], dtype=np.int64)
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)   # exact cosine (unit vecs → inner product)
        index.add(vectors)
        faiss.write_index(index, str(self.path))
        np.save(str(self.path) + ".ids", ids)

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        if self._index is None:
            self._index = faiss.read_index(str(self.path))
            self._ids = np.load(str(self.path) + ".ids.npy")
        D, I = self._index.search(query_vec.reshape(1, -1), k)
        return [(int(self._ids[i]), float(D[0][j])) for j, i in enumerate(I[0]) if i >= 0]
```

Build the index once after `embedding` jobs complete; invalidate/rebuild only when new
embeddings are added (tracked by a `embeddings_dirty` flag).

**Expected gain:**
- Memory per query: ~6 GB → **~50 MB** (index stays resident between queries)
- Query latency: 8–25 s → **<100 ms** at 500 K images (FAISS brute inner product is
  BLAS-optimised; IVF index brings it to <10 ms)

### Fix B — SQLite `vec0` extension (lighter-weight alternative)

If adding FAISS as a dependency is undesirable, the `sqlite-vec` extension provides
vector search directly inside SQLite with a compatible API. Query becomes:

```sql
SELECT image_id, distance
FROM vec_embeddings
WHERE embedding_type = 'image_clip'
  AND embedding MATCH ?   -- KNN via sqlite-vec
ORDER BY distance
LIMIT 100
```

---

## 4. High: Reverse Geocoding — Blocking HTTP With No Cache

**Impact:** I/O (network), CPU (thread stall), Throughput  
**File:** `imganalyzer/analysis/metadata.py:41–60`

### Problem

```python
# metadata.py:41–60
def _reverse_geocode(lat, lon):
    resp = httpx.get(
        "https://nominatim.openstreetmap.org/reverse",
        params={"lat": lat, "lon": lon, "format": "json"},
        timeout=5.0,
    )
```

Every geotagged image triggers a synchronous HTTP call to Nominatim. At 500 K images
(assume 80% have GPS = 400 K geocode calls):

- Nominatim enforces **1 req/sec** per user agent for bulk use. At 1 req/sec it takes
  **~111 hours** just for geocoding.
- The 5-second timeout means up to 5 s of stall per image in the `metadata` thread.
- No two images shot at the same location re-use the result.

### Fix A — GPS coordinate cache (in-process LRU)

Deduplicate by rounding to 4 decimal places (~11 m resolution), which collapses a burst
shoot at one location to a single HTTP call:

```python
import functools

@functools.lru_cache(maxsize=8192)
def _reverse_geocode_cached(lat_r: float, lon_r: float) -> tuple:
    result = _reverse_geocode(lat_r, lon_r)
    return tuple(sorted(result.items()))   # hashable

def reverse_geocode(lat: float, lon: float) -> dict:
    # Round to 4 dp (~11 m)
    return dict(_reverse_geocode_cached(round(lat, 4), round(lon, 4)))
```

### Fix B — Persist cache in DB (across sessions)

Add a `geocode_cache` table keyed on `(lat_4dp, lon_4dp)` with the location fields.
On startup load the cache into a dict; write back new entries in a deferred batch.
At 500 K images from a typical photo library, the cache will converge after a few
hundred unique locations.

### Fix C — Asynchronous batch geocoder

Run reverse geocoding asynchronously and in parallel (Nominatim allows 1 req/s; use
`asyncio` with a rate-limiter semaphore rather than blocking threads). Since `metadata`
already runs in the local I/O thread pool, use `asyncio.run` in the thread to batch
pending coordinates.

---

## 5. High: SHA-256 Hash on Every File at Ingest

**Impact:** I/O, CPU  
**File:** `imganalyzer/pipeline/batch.py:87`, `imganalyzer/pipeline/modules.py:23–32`

### Problem

```python
# batch.py:87
file_hash = compute_file_hash(path) if compute_hash else None
```

`compute_file_hash` reads the entire file in 64 KB chunks to compute a SHA-256 digest.
For 500 K images averaging 25 MB (RAW) each:

| Metric | Value |
|---|---|
| Total data read | ~12.5 TB |
| At 500 MB/s sequential I/O | ~7 hours |
| Adds ~0.05 s per RAW file | 500 K × 0.05 s = 6.9 hours |

The hash is stored in `images.file_hash` and used for deduplication, but
`get_image_by_path` (called first at line 83) already does a path-based dedup check.
SHA-256 is only needed for content-based dedup when the same image appears at different
paths.

### Fix — Lazy/deferred hashing + size-based pre-filter

```python
# batch.py — ingest()
# 1. Always skip hash if path already registered:
existing = self.repo.get_image_by_path(file_path_str)
if existing:
    image_id = existing["id"]
else:
    # 2. Use (size, mtime) as fast pre-filter; only hash on collision:
    file_size = path.stat().st_size
    mtime = int(path.stat().st_mtime)
    candidate = self.repo.get_image_by_size_mtime(file_size, mtime)
    if candidate:
        # Same size + mtime → almost certainly same file; skip SHA-256
        file_hash = None
    else:
        # New file: compute hash only for content-dedup (optional flag)
        file_hash = compute_file_hash(path) if compute_hash else None
```

With `--no-hash` (already supported via `compute_hash=False`), ingest throughput on
first run is limited only by `path.stat()` calls rather than full file reads.

---

## 6. High: Per-Job SQLite Commits

**Impact:** I/O (fsync amplification)  
**Files:** `imganalyzer/db/queue.py:133–156`, `imganalyzer/db/repository.py:84,103`

### Problem

Every job completion calls `mark_done`, `mark_failed`, or `mark_skipped`, each ending
with `self.conn.commit()`:

```python
# queue.py:132–156
def mark_done(self, job_id):
    self.conn.execute("UPDATE job_queue SET status='done' ...")
    self.conn.commit()   # ← fsync (or at least WAL write) per job

def mark_failed(...):
    ...
    self.conn.commit()

def mark_skipped(...):
    ...
    self.conn.commit()
```

Additionally, `register_image` and `update_image` each commit individually.

At 500 K images × 10 modules = **5 M individual commits**. Even in WAL mode each commit
involves a write to the WAL file and a potential WAL checkpoint. On spinning disk this
is catastrophic; on SSD it still adds 100–500 µs per commit = **8–70 minutes** of pure
commit overhead.

### Fix — Commit batching

Accumulate job status changes in a pending list and flush every N jobs or every T
seconds:

```python
# queue.py — new method
def mark_done_batch(self, job_ids: list[int]) -> None:
    placeholders = ",".join("?" * len(job_ids))
    self.conn.execute(
        f"UPDATE job_queue SET status='done', completed_at=? WHERE id IN ({placeholders})",
        [_now()] + job_ids,
    )
    self.conn.commit()   # ONE commit for N jobs
```

In `worker.py` collect `done_job_ids` after each GPU batch (e.g., every 50 jobs) and
flush. This is safe in WAL mode because a crash during a batch just resets the jobs to
their prior state (they will be re-claimed on next run via `recover_stale`).

---

## 7. High: Ingest Loop Issues N×M Individual DB Queries

**Impact:** CPU, I/O  
**File:** `imganalyzer/pipeline/batch.py:79–112`

### Problem

For each file the ingest loop issues:
1. `get_image_by_path(file_path_str)` — SELECT
2. For each of ~10 modules: `is_analyzed(image_id, module)` — SELECT per module
3. `enqueue(image_id, module, ...)` — SELECT + INSERT/UPDATE per module

At 500 K files × (1 + 10 + 10) = **10.5 M queries** for ingest alone. Since `metadata`
and `technical` run in a thread pool, ingest is often blocking the main thread while
each thread is blocked on DB.

### Fix A — Batch path lookup

Load all known paths in a single query at the start of ingest:

```python
# batch.py — ingest() start
known = {
    row["file_path"]: row["id"]
    for row in self.conn.execute("SELECT id, file_path FROM images").fetchall()
}
```

For 500 K already-registered images this replaces 500 K SELECTs with one query.

### Fix B — Bulk `is_analyzed` check

Replace per-module per-image `is_analyzed` with a single query that returns the full
analysis status for all images and modules at once:

```python
analyzed = set(
    (row["image_id"], row["module"])
    for row in self.conn.execute(
        """SELECT image_id, module FROM (
            SELECT image_id, 'metadata'  AS module FROM analysis_metadata  WHERE analyzed_at IS NOT NULL
            UNION ALL
            SELECT image_id, 'technical' AS module FROM analysis_technical WHERE analyzed_at IS NOT NULL
            -- ... all modules
        )"""
    ).fetchall()
)
# Then: skip = (image_id, module) in analyzed
```

### Fix C — Batch INSERT for new images

Register all new images in a single `executemany`:

```python
self.conn.executemany(
    "INSERT INTO images (file_path, file_hash, file_size) VALUES (?, ?, ?)",
    [(str(p), hash, size) for p, hash, size in new_files],
)
self.conn.commit()
```

Combined, Fixes A+B+C reduce ingest DB overhead from 10.5 M queries to **~5 queries**
(bulk SELECT + bulk INSERT + bulk analysis status + bulk enqueue).

---

## 8. Medium: RAW Full Decode for Every Module

**Impact:** CPU, Memory  
**File:** `imganalyzer/readers/raw.py:27–37`

### Problem

`rawpy.postprocess(half_size=False)` always produces the full-resolution demosaiced
RGB array. For a 61 MP Sony A7R IV:

- Full decode: ~175 MB uint8 RGB, ~1.2 s CPU
- `TechnicalAnalyzer` immediately downsamples to 3000 px (line 24, technical.py)
- CLIP embedder immediately downsamples to 1280 px (line 84, clip_embedder.py)
- `MetadataExtractor` does not use `rgb_array` at all

### Fix A — Half-size decode for most modules

`rawpy` supports `half_size=True` which produces a 2× downsampled image (15 MP for a
61 MP RAW) at roughly 1/4 the decode time and memory. This is sufficient for
`technical`, `objects`, `ocr`, `faces`, and `embedding` (all of which internally
downsample further anyway):

```python
# readers/raw.py — add resolution hint
def read(path, half_size=False):
    rgb = raw.postprocess(
        use_camera_wb=True,
        no_auto_bright=False,
        output_bps=8,
        half_size=half_size,     # ← new parameter
    )
```

Expose a `resolution` parameter in `_read_image(path, resolution="full"|"half"|"thumb")`
and have each module request only what it needs. `cloud_ai` already uses `half_size=True`
for its JPEG encoding path.

### Fix B — Embedded JPEG thumbnail for thumbnail-only use cases

RAW files contain an embedded JPEG preview (LibRaw exposes it via
`raw.extract_thumb()`). For modules that only need a preview-quality image (cloud AI,
quick EXIF-only pipelines), extracting the embedded JPEG (~4–8 MP) is ~50× faster than
full postprocessing.

---

## 9. Medium: GPU Modules Fully Serialized — No Micro-Batching

**Impact:** GPU utilisation  
**File:** `imganalyzer/pipeline/worker.py:57–58,325–347`

### Problem

```python
# worker.py:57–58
GPU_MODULES = {"local_ai", "embedding", "blip2", "objects", "ocr", "faces"}
```

All GPU modules run on the main thread one image at a time. GPU utilisation for a
single image through BLIP-2 or GroundingDINO is typically 30–60%: the model runs in
bursts between Python overhead, data transfer, tokenization, and DB writes.

The current architecture processes image N, writes to DB, then fetches image N+1 — the
GPU sits idle during file I/O and DB writes.

### Fix — Prefetch pipeline (double-buffering)

Use a background thread to decode and pre-process the next image while the GPU
processes the current one:

```python
# Conceptual double-buffer in _drain_gpu_module:
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=1) as prefetch_pool:
    # Prefetch image N+1 while GPU processes image N
    next_job = queue.claim(batch_size=1, module=module_name)
    prefetch_future = prefetch_pool.submit(_read_image, next_job[0]["path"])

    for job in current_jobs:
        image_data = prefetch_future.result()  # already decoded
        next_job = queue.claim(batch_size=1, module=module_name)
        if next_job:
            prefetch_future = prefetch_pool.submit(_read_image, next_job[0]["path"])
        gpu_process(image_data)
```

### Fix — Micro-batching for CLIP (most impactful)

`CLIPEmbedder` processes one image at a time. CLIP ViT-L/14 throughput scales nearly
linearly with batch size up to ~32 on a 24 GB GPU. Batching 8–32 images per forward
pass increases GPU utilisation from ~35% to ~90%:

```python
# embeddings/clip_embedder.py — new method
def embed_image_batch(self, paths: list[Path], batch_size=16) -> list[bytes]:
    tensors = []
    for path in paths:
        img = _load_and_resize(path)
        tensors.append(CLIPEmbedder._preprocess(img))
    batch = torch.stack(tensors).to(CLIPEmbedder._device)
    with torch.no_grad(), torch.cuda.amp.autocast(...):
        features = CLIPEmbedder._model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)
    return [row.cpu().numpy().astype(np.float32).tobytes() for row in features]
```

In `worker.py` collect `batch_size` embedding jobs before calling the batch embedder
instead of processing one at a time.

---

## 10. Medium: Technical Analysis — Double Resize + Expensive Per-Pixel Ops

**Impact:** CPU  
**File:** `imganalyzer/analysis/technical.py`

### Problem

`TechnicalAnalyzer.analyze()` does **two** downsamples of the full RGB array in
sequence:

```python
# technical.py:17–25 — first downsample to 3000 px
if max(h, w) > 3000:
    pil_img = PIL.Image.fromarray(rgb).resize(...)
    rgb = np.array(pil_img)

# technical.py:63–70 — second downsample to 1200 px for sharpness
if max(h, w) > 1200:
    pil_s = PIL.Image.fromarray(rgb).resize(...)
    rgb_s = np.array(pil_s)
```

The `_spectral_saliency` function (line 206–227) does:
- `rgb2lab` on the full 1200 px image (float32, 3-channel)
- Three `gaussian_filter` passes (one per LAB channel)
- Squared-difference saliency map

`_patch_sharpness` (line 230–289) iterates over all `(H/64) × (W/64)` patches in a
Python `for` loop, computing `np.var(laplace(patch))` per patch. For a 1200×800 image
this is (18 × 12) = 216 iterations, each calling `scipy.ndimage.laplace`.

### Fix A — Single downsample to 1200 px upfront

Since sharpness uses the 1200 px version and the other metrics are not resolution-
sensitive, downsample once to 1200 px at the start:

```python
# technical.py:analyze()
MAX_DIM = 1200   # single target; not 3000 then 1200
```

This halves the resize work and avoids creating a 3000 px intermediate array.

### Fix B — Vectorise `_patch_sharpness` with strided array views

Replace the Python `for` loop with a NumPy strided approach:

```python
def _patch_sharpness_fast(gray, patch_size=64):
    from scipy.ndimage import laplace
    # Compute Laplacian once across the full image
    lap = laplace(gray)
    # Reshape into patches using stride tricks
    h, w = gray.shape
    rows, cols = h // patch_size, w // patch_size
    patched = lap[:rows*patch_size, :cols*patch_size].reshape(
        rows, patch_size, cols, patch_size
    )
    # Variance per patch = mean(x²) - mean(x)² — no Python loop
    patch_var = patched.var(axis=(1, 3))
    # ... apply center/saliency weights in vectorised form
```

This reduces 216 `scipy.ndimage.laplace` calls to **one** and replaces the Python loop
with NumPy ops.

### Fix C — Skip `estimate_sigma` on downsampled images

`skimage.restoration.estimate_sigma` is O(H×W) and internally runs a wavelet
decomposition. On the already-downsampled 1200 px image this is fast, but for large
inputs it can take 2–5 s. The existing fallback (`std of Laplacian residuals`) is
adequate and 10× faster; consider making it the default for `technical` analysis.

---

## 11. Medium: `find_face_by_alias` Full Table Scan in Hot Path

**Impact:** CPU, I/O  
**File:** `imganalyzer/db/repository.py:491–506`

### Problem

```python
# repository.py:491–506
def find_face_by_alias(self, name: str) -> dict | None:
    row = self.conn.execute(
        "SELECT * FROM face_identities WHERE canonical_name=? OR display_name=?",
        [name, name],
    ).fetchone()
    if row:
        return dict(row)
    # Search in aliases JSON — full table scan in Python
    rows = self.conn.execute("SELECT * FROM face_identities").fetchall()
    for r in rows:
        aliases = json.loads(r["aliases"] or "[]")
        if name in aliases:
            return dict(r)
    return None
```

This is called inside `update_search_index` (repository.py:648–654) for every detected
face name — which is itself called after every module write. At 500 K images with 2
faces each = **1 M calls**, each potentially scanning the entire `face_identities` table
in Python.

### Fix — Store aliases as indexed rows in a separate table

Replace the JSON `aliases` column with a `face_aliases` table:

```sql
CREATE TABLE face_aliases (
    identity_id INTEGER REFERENCES face_identities(id),
    alias       TEXT NOT NULL,
    PRIMARY KEY (identity_id, alias)
);
CREATE INDEX idx_face_aliases_alias ON face_aliases(alias);
```

`find_face_by_alias` becomes a single indexed lookup:

```sql
SELECT fi.* FROM face_identities fi
JOIN face_aliases fa ON fa.identity_id = fi.id
WHERE fi.canonical_name = ?
   OR fi.display_name = ?
   OR fa.alias = ?
LIMIT 1
```

This changes O(N) Python JSON scan to O(log N) B-tree lookup.

---

## 12. Medium: `CLIPEmbedder` Re-reads File Already Decoded by ModuleRunner

**Impact:** I/O, CPU  
**Files:** `imganalyzer/pipeline/modules.py:367–425`, `imganalyzer/embeddings/clip_embedder.py:42–93`

### Problem

`_run_embedding` (modules.py:367) calls `embedder.embed_image(path)` which reopens and
decodes the image file from scratch via `rawpy` or `Pillow` — even though the image
may already be in the per-thread cache (if Fix #1 is applied) or the same process has
just decoded it for another module.

This is a **second independent rawpy decode** specifically for CLIP on top of the
main-pipeline decode.

### Fix — Add `embed_image_pil` overload to CLIPEmbedder

```python
# clip_embedder.py
def embed_image_pil(self, img: "PIL.Image.Image") -> bytes:
    """Encode a pre-loaded PIL image → float32 bytes (768-d).

    Use this overload when the image is already decoded to avoid a
    redundant file read.  The caller is responsible for passing an RGB image.
    """
    self._load_model()
    import torch
    img.thumbnail((EMBED_MAX_LONG_EDGE, EMBED_MAX_LONG_EDGE), Image.LANCZOS)
    image_input = CLIPEmbedder._preprocess(img).unsqueeze(0).to(CLIPEmbedder._device)
    with torch.no_grad(), torch.cuda.amp.autocast(...):
        features = CLIPEmbedder._model.encode_image(image_input)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().flatten().astype(np.float32).tobytes()
```

`_run_embedding` passes the cached PIL image (from Fix #1) rather than the file path.

---

## 13. Low: `iter_image_ids()` Loads All IDs Into Memory

**Impact:** Memory  
**File:** `imganalyzer/db/repository.py:109–112`

### Problem

```python
# repository.py:109
def iter_image_ids(self) -> list[int]:
    rows = self.conn.execute("SELECT id FROM images ORDER BY id").fetchall()
    return [r["id"] for r in rows]
```

`rebuild_module` (batch.py:171) calls this and holds the full 500 K list in RAM. At
500 K images this is ~4 MB of Python integers — small, but `executemany` patterns that
follow iterate this list in Python.

### Fix — Yield in chunks (cursor-based iteration)

```python
def iter_image_ids(self, chunk_size=1000):
    """Yield image IDs in chunks to avoid large list allocation."""
    cursor = self.conn.execute("SELECT id FROM images ORDER BY id")
    while True:
        rows = cursor.fetchmany(chunk_size)
        if not rows:
            break
        for r in rows:
            yield r["id"]
```

Callers that need a list can still wrap in `list(repo.iter_image_ids())`, but callers
like `rebuild_module` can iterate lazily.

---

## 14. Low: Missing DB Indexes on High-Cardinality Filter Columns

**Impact:** CPU, I/O (query planning)  
**Files:** `imganalyzer/db/schema.py`, `imganalyzer/db/search.py:170–243`

### Problem

`search_exif` (search.py:170) builds queries like:

```sql
WHERE m.date_time_original >= ? AND m.date_time_original <= ?
AND   m.iso >= ? AND m.iso <= ?
AND   m.camera_make LIKE ? AND m.camera_model LIKE ?
```

The `analysis_metadata` table has no indexes on `date_time_original`, `iso`,
`camera_make`, or `camera_model`. At 500 K rows, each query does a full table scan
with no LIKE optimisation possible (LIKE with a leading `%` cannot use a B-tree index).

### Fix A — Add indexes for equality/range filters

```sql
-- schema.py migration
CREATE INDEX idx_metadata_date   ON analysis_metadata(date_time_original);
CREATE INDEX idx_metadata_iso    ON analysis_metadata(iso);
CREATE INDEX idx_metadata_camera ON analysis_metadata(camera_make, camera_model);
CREATE INDEX idx_metadata_geo    ON analysis_metadata(gps_latitude, gps_longitude);
```

### Fix B — Replace `LIKE '%text%'` with FTS5

Camera make/model searches should go through the existing FTS5 `search_index` (which
already indexes `exif_text` containing `camera_make`, `camera_model`, `lens_model`) rather
than LIKE scans on raw text columns. This turns O(N) scans into O(log N) FTS lookups.

---

## 15. Low: Cloud AI — `asyncio.run()` Creates New Event Loop Per Call

**Impact:** CPU overhead  
**File:** `imganalyzer/analysis/ai/cloud.py` (Copilot backend)

### Problem

The Copilot backend (and any async cloud client) uses `asyncio.run()` inside a
synchronous method. `asyncio.run()` creates and tears down a new event loop on every
call. At 500 K images × 1 cloud call this is 500 K event loop creates/destroys.

### Fix — Reuse event loop per thread

Use a thread-local event loop:

```python
import asyncio, threading
_loop_local = threading.local()

def _get_loop():
    if not hasattr(_loop_local, "loop") or _loop_local.loop.is_closed():
        _loop_local.loop = asyncio.new_event_loop()
    return _loop_local.loop

# In cloud.py:
result = _get_loop().run_until_complete(_async_analyze(path, image_data))
```

---

## 16. Architecture: Multi-Worker GPU Pipeline

**Impact:** GPU utilisation, Throughput  
**File:** `imganalyzer/pipeline/worker.py`

### Current limitation

The current design serialises all GPU modules on the main thread to avoid VRAM
contention between models (BLIP-2 ~8 GB, GroundingDINO ~0.7 GB, TrOCR ~1.3 GB, CLIP
~1.6 GB). On a 24 GB GPU all four models fit simultaneously (~11.6 GB total fp16), but
they still run serially because worker.py is single-threaded for GPU work.

### Opportunity

On a multi-GPU system (or any system where VRAM permits), separate GPU model servers
can run in parallel:

- **Process A** (GPU 0): BLIP-2 + GroundingDINO (objects gate)
- **Process B** (GPU 0 or 1): CLIP embedder (batched)
- **Process C** (CPU or GPU): InsightFace via ONNX Runtime

Use a Unix domain socket or ZeroMQ to accept image data and return results. The worker
becomes a client that submits to these model servers. This decouples model loading from
the queue-processing loop and allows true GPU parallelism.

A lighter version achieves similar results within a single process by running CLIP
concurrently with BLIP-2/GroundingDINO on the same GPU: since CLIP runs on the
embedding pass (after all other GPU passes are done), it could overlap with cloud I/O
that is waiting on the network.

---

## Prioritised Implementation Roadmap

| Priority | Issue | Est. Speedup at 500 K | Effort |
|---|---|---|---|
| P0 | #1 Redundant image decoding | 4–6× throughput | Medium |
| P0 | #2 FTS rebuild storm | 3–5× write I/O | Low |
| P0 | #3 O(N) CLIP search → FAISS | 100× query latency | Medium |
| P1 | #4 Geocode cache | Unblocks rate limiter | Low |
| P1 | #5 Lazy SHA-256 | 2× ingest speed | Low |
| P1 | #6 Commit batching | 2–3× write throughput | Low |
| P1 | #7 Bulk ingest queries | 10× ingest DB perf | Medium |
| P2 | #8 RAW half-size decode | 30–50% CPU reduction | Low |
| P2 | #9 GPU micro-batching (CLIP) | 2–3× GPU utilisation | Medium |
| P2 | #10 Vectorise patch sharpness | 5× technical analysis | Low |
| P2 | #11 Face alias indexed table | Eliminates O(N) scan | Medium |
| P3 | #12 CLIPEmbedder PIL overload | Eliminates 2nd decode | Low |
| P3 | #13 Chunked iter_image_ids | Memory safety | Low |
| P3 | #14 DB indexes | 10–100× EXIF query | Low |
| P3 | #15 Event loop reuse | Marginal | Low |
| P4 | #16 Multi-GPU architecture | 2–4× GPU throughput | High |

---

## Quantitative Impact Estimate (500 K RAW Images, Single GPU)

**Baseline (current):** Estimated 30–60 days for full pipeline  
*(dominated by 7× redundant decode + FTS storm + geocode blocking)*

**After P0 fixes (#1, #2, #3):** Estimated 5–10 days  
*(single decode, deferred FTS, fast CLIP search)*

**After P0+P1 fixes (#1–#7):** Estimated 3–5 days  
*(adds: geocode cache, fast ingest, batched commits)*

**After all fixes + GPU batching:** Estimated 1–2 days  
*(CLIP batching doubles GPU utilisation; parallel model serving possible)*

*These estimates assume: 4-core CPU, 1 GPU (16 GB VRAM), NVMe SSD, ~25 MB avg RAW,
all modules enabled. Adjust linearly for faster hardware.*
