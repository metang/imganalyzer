# Performance Analysis & Improvement Opportunities

**Codebase:** imganalyzer (Python CLI + Electron GUI)
**Model:** claude-opus-4.6
**Date:** 2026-02-27
**Target Scale:** 500,000+ images

---

## Executive Summary

The imganalyzer system has significant performance bottlenecks at the 500K+ image scale across CPU, GPU, memory, I/O, and database layers. The most impactful issues fall into five categories:

1. **Redundant image I/O** — each image is decoded from disk up to 10 times (once per analysis module)
2. **Database anti-patterns** — millions of individual SQL statements with per-row commits during ingest
3. **No GPU batching** — AI models process one image per forward pass despite supporting batch inference
4. **Brute-force vector search** — every search query loads ~3GB of embeddings and performs O(n) Python-loop cosine similarity
5. **Electron subprocess overhead** — thumbnail generation, search, and status polling each spawn a full `conda run → Python` process

The top 20 improvements below are estimated to deliver a combined **5-15x throughput increase** for batch processing and reduce interactive latency from minutes to seconds.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│ Electron GUI (React 18 + electron-vite)                  │
│   ├── IPC handlers (images.ts, batch.ts, search.ts)      │
│   └── Communicates via: conda run → Python subprocess    │
├──────────────────────────────────────────────────────────┤
│ Python CLI (Typer)                                       │
│   ├── Readers: rawpy (LibRaw), Pillow, pillow-heif       │
│   ├── Analysis: EXIF, technical quality, AI models       │
│   ├── AI Models (GPU): BLIP-2, GroundingDINO, TrOCR,    │
│   │                     InsightFace, CLIP ViT-L/14       │
│   ├── Cloud AI: OpenAI, Anthropic, Google Vision, Copilot│
│   ├── Pipeline: batch scanner, job queue, worker pool    │
│   ├── Database: SQLite WAL + FTS5 + embeddings           │
│   └── Output: XMP sidecar files                          │
└──────────────────────────────────────────────────────────┘
```

**Key constraint:** GPU modules (BLIP-2, GroundingDINO, TrOCR, InsightFace, CLIP) run serially on the main thread to avoid VRAM contention. Cloud API calls run in a thread pool (4 workers). Local I/O modules run in a separate thread pool (1 worker).

---

## Top 20 Performance Improvement Opportunities

### Tier 1: Critical (10x+ impact at 500K scale)

---

#### 1. Eliminate Redundant Image Decoding Across Modules

| Attribute | Detail |
|-----------|--------|
| **Files** | `pipeline/modules.py:35-49`, `pipeline/worker.py:386-401` |
| **Resource** | CPU, Memory, Disk I/O |
| **Current behavior** | Each module (`metadata`, `technical`, `blip2`, `objects`, `ocr`, `faces`, `cloud_ai`, `aesthetic`, `embedding`) calls `_read_image(path)` independently. Each call opens the file, decodes the full image (demosaic for RAW, JPEG decode for standard), and allocates a fresh numpy array. For a 50MP image: ~144 MB per decode, ~2-5 seconds for RAW demosaic. |
| **Impact at 500K** | ~3.5 million redundant file reads. For RAW files: ~1.75M unnecessary demosaic operations at 2-5 seconds each. |
| **Proposed fix** | Introduce a per-image `ImageCache` that lazily loads image data on first access and holds it across all module runs for the same image. The worker already processes modules for the same image sequentially (GPU modules are serial), so a simple `dict` keyed by `image_id` with a single decoded `rgb_array` would suffice. Clear the cache after all modules for an image complete. |
| **Expected gain** | ~7x reduction in image I/O for each image. RAW processing time per image drops from ~20-35s (7 decodes) to ~3-5s (1 decode). |
| **Complexity** | Medium — requires threading the image data through `ModuleRunner.run()` instead of re-reading per module. The CLIP embedder (`clip_embedder.py:63-67`) also needs to accept a pre-decoded image instead of a file path. |

---

#### 2. Batch Database Operations During Ingest

| Attribute | Detail |
|-----------|--------|
| **Files** | `pipeline/batch.py:79-112`, `db/queue.py:25-88`, `db/repository.py:64-85` |
| **Resource** | Disk I/O (SQLite fsync) |
| **Current behavior** | Ingest performs individual SQL operations per image per module: `get_image_by_path` (SELECT), `register_image` (INSERT + COMMIT), `is_analyzed` (SELECT per module), `enqueue` (SELECT + INSERT + COMMIT per module). `enqueue_batch()` is a "fake batch" that loops over `enqueue()`. For 500K images × 10 modules: ~10.5M individual SQL statements with ~5M+ individual COMMITs. |
| **Impact at 500K** | Each SQLite COMMIT in WAL mode requires an fsync (~50-100μs). 5M commits × 75μs = ~375 seconds of pure commit overhead. Total ingest time is measured in hours. |
| **Proposed fix** | (a) Wrap the entire ingest loop in a single transaction (`BEGIN ... COMMIT` every 1000 images). (b) Replace `enqueue_batch` with a true bulk `INSERT ... SELECT ... WHERE NOT EXISTS` statement. (c) Replace `register_image` with `INSERT OR IGNORE` + batch commit. (d) Batch `is_analyzed` checks into a single `SELECT image_id, module FROM ... WHERE image_id IN (...)`. |
| **Expected gain** | Ingest time for 500K images drops from hours to **minutes** (5000x fewer commits, 100x fewer round-trips). |
| **Complexity** | Low-Medium — the SQL refactoring is straightforward; main risk is ensuring the progress reporting still works (the `[PROGRESS]` output on line 96 emits per-image). |

---

#### 3. Vectorized Semantic Search (Replace Brute-Force Loop)

| Attribute | Detail |
|-----------|--------|
| **Files** | `db/search.py:350-374`, `db/repository.py:540-546`, `embeddings/clip_embedder.py:149-153` |
| **Resource** | CPU, Memory |
| **Current behavior** | Every search query: (1) loads ALL embeddings from SQLite into Python lists (`get_all_embeddings` — `fetchall()` materializes all rows), (2) iterates in a Python `for` loop doing `np.frombuffer` + `cosine_similarity` (two `np.linalg.norm` calls + `np.dot`) per embedding. CLIP embeddings are already L2-normalized, making the norms redundant. |
| **Impact at 500K** | Each search loads ~3 GB into Python memory (500K × 768-d float32 × 2 embedding types). The Python loop performs 500K iterations with per-element numpy calls. Estimated latency: 10-60 seconds per search query. |
| **Proposed fix** | (a) Pre-load all embeddings into a single `(N, 768)` numpy matrix at startup or on first search, cached in memory. (b) Replace the Python loop with a single matrix multiply: `scores = matrix @ query_vec` (one BLAS call). Since embeddings are L2-normalized, dot product = cosine similarity. (c) Consider `faiss.IndexFlatIP` for even faster similarity with GPU acceleration. (d) Long-term: use `sqlite-vss` or `faiss` for approximate nearest neighbor search (HNSW or IVF). |
| **Expected gain** | Search latency drops from 10-60s to **<100ms** (vectorized numpy) or **<10ms** (FAISS). Memory stays at ~1.5 GB but is loaded once, not per query. |
| **Complexity** | Low — the matrix vectorization is a ~20-line change. FAISS integration is medium complexity. |

---

#### 4. Batch GPU Inference (Multi-Image Forward Passes)

| Attribute | Detail |
|-----------|--------|
| **Files** | `pipeline/worker.py:280-346`, `analysis/ai/local.py:65-94`, `analysis/ai/objects.py:77-90`, `embeddings/clip_embedder.py:81-90` |
| **Resource** | GPU |
| **Current behavior** | The worker claims `batch_size=1` for GPU modules (worker.py line 281). Each GPU model processes exactly one image per forward pass. BLIP-2 runs 5 sequential `model.generate()` calls per image (1 caption + 4 VQA questions). GroundingDINO, CLIP, and InsightFace each process one image per call. |
| **Impact at 500K** | GPU utilization is severely underutilized. Modern GPUs can process 8-32 images per batch for CLIP/GroundingDINO with near-linear throughput scaling. BLIP-2 VQA questions could be batched (4 questions → 1 batched generate call). |
| **Proposed fix** | (a) For BLIP-2: batch the 4 VQA questions into a single `model.generate()` call with batched inputs (processor supports batch). Re-use the vision encoder output across questions. (b) For CLIP embedder: add `embed_images_batch(paths, batch_size=16)` that stacks tensors and runs a single forward pass. (c) For GroundingDINO: batch 4-8 images per forward pass (processor supports batch). (d) Modify worker to claim `batch_size=8` for GPU modules and pass image batches. |
| **Expected gain** | 3-8x GPU throughput improvement for CLIP and GroundingDINO. 2-4x for BLIP-2 (vision encoder reuse + batched VQA). |
| **Complexity** | Medium-High — requires modifying the `ModuleRunner` to accept image batches, changing the pass wrappers, and handling partial batch failures. |

---

#### 5. Fix Electron `listImages()` — Synchronous Stats at 500K Scale

| Attribute | Detail |
|-----------|--------|
| **Files** | `imganalyzer-app/src/main/images.ts:35-64`, `imganalyzer-app/src/main/index.ts:73-75` |
| **Resource** | CPU (main process event loop) |
| **Current behavior** | `listImages()` performs two synchronous `statSync()` calls per file (one for the image, one for the XMP sidecar) in a tight loop. At 500K images: 1,000,000 synchronous filesystem calls, blocking the entire Electron main process. The result is a full 500K-element array sorted with `localeCompare` (10-50x slower than basic comparison) and serialized over IPC. |
| **Impact at 500K** | Main process freezes for **minutes**. UI is completely unresponsive. IPC payload is ~100-200 MB. |
| **Proposed fix** | (a) Replace `statSync` with `fs.promises.stat` in batches (e.g., `Promise.all` chunks of 100). (b) Implement pagination: return page-sized chunks (e.g., 200 images) instead of the full array. (c) Move listing to a Worker Thread to avoid blocking main process. (d) Replace `localeCompare` sort with numeric/ASCII comparison or move sorting to the renderer. (e) Consider reading the image list from the SQLite database (already contains all registered images with metadata) instead of scanning the filesystem. |
| **Expected gain** | Gallery load time drops from minutes to **<1 second** (paginated from DB) or **5-10 seconds** (async fs scan). |
| **Complexity** | Medium — pagination requires renderer-side virtual scrolling changes. |

---

### Tier 2: High Impact (3-10x improvement in specific paths)

---

#### 6. Eliminate 52x Directory Traversal During Ingest

| Attribute | Detail |
|-----------|--------|
| **Files** | `pipeline/batch.py:51-70` |
| **Resource** | Disk I/O, CPU |
| **Current behavior** | For each of 26 image extensions, two `rglob` calls are made (lowercase + uppercase) = 52 recursive directory traversals per folder. |
| **Impact at 500K** | On network drives or HDDs, each `rglob` traversal can take seconds. 52 traversals of a directory with 500K files takes minutes. |
| **Proposed fix** | Single `rglob("*")` with a suffix filter: `[p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS_SET]`. Or use `os.scandir()` for even better performance. |
| **Expected gain** | ~50x faster directory scanning. |
| **Complexity** | Very Low — 5-line change. |

---

#### 7. Pre-Resize Images Before AI Model Input

| Attribute | Detail |
|-----------|--------|
| **Files** | `analysis/ai/local.py:38-39`, `analysis/ai/objects.py:60-69`, `analysis/ai/faces.py:44-46`, `embeddings/clip_embedder.py:81-86` |
| **Resource** | CPU, Memory |
| **Current behavior** | Full-resolution images (50MP = 8000×6000 = 144 MB) are passed to AI models that internally resize to much smaller dimensions: BLIP-2 → 224×224, GroundingDINO → 800px, CLIP → 224×224, InsightFace → 640×640. Each model independently converts the full-res `rgb_array` to PIL and resizes. |
| **Impact at 500K** | Millions of unnecessary full-resolution PIL allocations and LANCZOS resizes. ~144 MB wasted per module per image. |
| **Proposed fix** | As part of the image cache (Improvement #1), store pre-computed resized versions: `thumb_224` (for BLIP-2, CLIP), `thumb_800` (for GroundingDINO), `thumb_640` (for InsightFace). Generate these once from the original `rgb_array` during cache population. |
| **Expected gain** | ~4x memory reduction per module invocation. Eliminates redundant LANCZOS resizes. |
| **Complexity** | Low — add resize variants to the image cache dict. |

---

#### 8. RAW Half-Size Demosaic for Analysis

| Attribute | Detail |
|-----------|--------|
| **Files** | `readers/raw.py:28-33` |
| **Resource** | CPU, Memory |
| **Current behavior** | RAW files always demosaic at full resolution (`half_size=False`). A 50MP sensor produces 8000×6000 RGB = 144 MB in ~2-5 seconds. No downstream module uses the full resolution — all models resize to ≤1920px. |
| **Impact at 500K** | Full-res demosaic adds ~2-5 seconds CPU time and 144 MB memory per RAW image. For a library with 300K RAW files: ~150-375 hours of unnecessary CPU work. |
| **Proposed fix** | Default to `half_size=True` for analysis pipeline (produces 4000×3000 = 36 MB in ~0.5-1s). Add a parameter to request full-size only when needed (e.g., thumbnail generation at specific sizes). |
| **Expected gain** | 3-5x faster RAW decode, 4x less memory per RAW image. |
| **Complexity** | Very Low — single parameter change, plus a flag for callers that need full resolution. |

---

#### 9. Optimize `update_search_index` — Reduce 5-7 SELECTs to 1

| Attribute | Detail |
|-----------|--------|
| **Files** | `db/repository.py:550-676` |
| **Resource** | Disk I/O (SQLite) |
| **Current behavior** | `update_search_index()` performs 5-7 separate SELECT queries per image to gather data from `analysis_local_ai`, `analysis_blip2`, `analysis_faces`, `analysis_cloud_ai`, `analysis_metadata`. It also calls `find_face_by_alias()` per detected face, which does a full table scan of `face_identities` with JSON parsing. Called after every metadata, cloud_ai, and local_ai module completion. |
| **Impact at 500K** | At minimum 1.5M calls × 5-7 SELECTs = ~10M+ queries. Each `find_face_by_alias` adds a full-table scan. |
| **Proposed fix** | (a) Replace 5-7 SELECTs with a single JOIN query across all analysis tables. (b) Replace `find_face_by_alias` full-table-scan with a `face_alias_index` table (canonical_name, alias_name) for O(1) lookups. (c) Batch `update_search_index` — accumulate image IDs and rebuild the index in bulk every N images. (d) Only call `update_search_index` once after all modules for an image complete, not after each module. |
| **Expected gain** | ~5-10x fewer database queries for search index maintenance. |
| **Complexity** | Medium — JOIN query refactoring is straightforward; batching requires coordination with the worker loop. |

---

#### 10. Replace Per-Job Queue Commits with Batched Transactions

| Attribute | Detail |
|-----------|--------|
| **Files** | `db/queue.py:25-72, 132-156` |
| **Resource** | Disk I/O (SQLite fsync) |
| **Current behavior** | `enqueue()` does SELECT + INSERT/UPDATE + COMMIT per job. `mark_done()`, `mark_failed()`, `mark_skipped()` each COMMIT after a single UPDATE. At 500K × 10 modules: 5M enqueue commits + 5M completion commits = 10M total COMMITs. |
| **Impact at 500K** | ~10M fsyncs × 75μs = ~750 seconds (~12.5 minutes) of pure commit overhead. |
| **Proposed fix** | (a) Batch `enqueue`: wrap in `BEGIN IMMEDIATE ... COMMIT` per 1000 jobs. (b) Batch completion marks: accumulate done/failed job IDs and commit every 100 or every 5 seconds. (c) Add a composite index on `(status, priority DESC, queued_at ASC)` for efficient claim queries. |
| **Expected gain** | ~1000x fewer commits during ingest, ~100x fewer during processing. |
| **Complexity** | Low-Medium — batching requires careful handling of the worker's job lifecycle. |

---

#### 11. BLIP-2 VQA Question Batching

| Attribute | Detail |
|-----------|--------|
| **Files** | `analysis/ai/local.py:65-94` |
| **Resource** | GPU |
| **Current behavior** | 4 VQA questions are processed sequentially, each with its own `processor()` call (re-encodes the image) and `model.generate()` call. The image is preprocessed 5 times total (1 caption + 4 VQA). `torch.cuda.empty_cache()` is called after each question (5 times total), causing CUDA synchronization stalls. |
| **Impact at 500K** | 5 GPU forward passes per image instead of 2 (1 for caption, 1 for batched VQA). 500K × 3 extra forward passes = ~1.5M unnecessary GPU inferences. |
| **Proposed fix** | (a) Batch all 4 VQA questions into a single `processor(images=[pil_img]*4, text=[q1,q2,q3,q4])` + single `model.generate()`. (b) Remove per-question `empty_cache()` calls — one at the end suffices. (c) Explore caching the vision encoder output across caption + VQA (requires minor model API changes). |
| **Expected gain** | 2-3x faster per-image BLIP-2 inference. Eliminates 4 redundant CUDA sync points per image. |
| **Complexity** | Low — Blip2Processor natively supports batched inputs. |

---

#### 12. Unbounded Thumbnail Cache in Electron

| Attribute | Detail |
|-----------|--------|
| **Files** | `imganalyzer-app/src/main/images.ts:67-68` |
| **Resource** | Memory (Electron main process) |
| **Current behavior** | `thumbCache` is a `Map<string, string>` with no eviction policy. Each thumbnail is a base64 data URL (~20-60 KB). There is no `thumbCache.clear()` or size limit anywhere in the codebase. |
| **Impact at 500K** | If the user scrolls through the entire gallery: 500K × ~40 KB = ~20 GB of memory consumed. Electron process will OOM. |
| **Proposed fix** | (a) Implement LRU cache with a configurable max size (e.g., 5000 entries ≈ 200 MB). (b) Write thumbnails to disk cache (`~/.cache/imganalyzer/thumbnails/`) and serve from there. (c) Use a persistent thumbnail database (SQLite BLOB or filesystem) shared between sessions. |
| **Expected gain** | Memory usage capped at ~200 MB instead of growing unbounded. Disk-cached thumbnails persist across sessions. |
| **Complexity** | Low-Medium — LRU cache is ~30 lines. Disk cache requires a hash-based directory structure. |

---

### Tier 3: Medium Impact (2-3x improvement in specific paths)

---

#### 13. Persistent Python Process for Electron Communication

| Attribute | Detail |
|-----------|--------|
| **Files** | `imganalyzer-app/src/main/batch.ts:133-171,233,274`, `imganalyzer-app/src/main/search.ts:134-191`, `imganalyzer-app/src/main/images.ts:159-167`, `imganalyzer-app/src/main/analyzer.ts:46-49` |
| **Resource** | CPU, Memory, Latency |
| **Current behavior** | Every operation (thumbnail, search, status poll, single-image analysis) spawns a new `conda run → Python` subprocess. `conda run` has 1-3 seconds of environment resolution overhead before Python starts. Status polling spawns a subprocess every 1 second. Search spawns a subprocess per query. Thumbnails spawn a subprocess per image (4 concurrent max). |
| **Impact at 500K** | (a) Status polling: 1 subprocess/second for the entire batch processing duration (~hours). (b) Search: 1-3 second minimum latency per query. (c) Thumbnails: 4 concurrent subprocesses with 1-3s conda overhead = ~1-2 thumbnails/second. |
| **Proposed fix** | (a) Launch a persistent Python daemon (e.g., FastAPI/Flask or JSON-RPC over stdio) at Electron startup. (b) All IPC goes through this long-lived process via HTTP or stdio pipes. (c) Eliminates conda resolution overhead on every call. (d) Enables connection pooling, model preloading, and in-memory caching. |
| **Expected gain** | Eliminates 1-3s conda overhead per operation. Search latency drops to <100ms. Thumbnail throughput increases to 20-50/second. |
| **Complexity** | High — significant architectural change. Requires a Python server, protocol design, and replacing all `spawn('conda', ...)` calls. |

---

#### 14. Add Queue Indices for Efficient Claim Queries

| Attribute | Detail |
|-----------|--------|
| **Files** | `db/queue.py:92-128`, `db/schema.py` |
| **Resource** | Disk I/O (SQLite) |
| **Current behavior** | `claim()` runs `SELECT ... WHERE status='pending' AND module=? ORDER BY priority DESC, queued_at ASC LIMIT ?`. Without a composite index, SQLite does a full table scan of the `job_queue` table (up to 5M rows) and sorts in-memory. |
| **Impact at 500K** | The claim query runs after every GPU job completion (millions of times). Each scan touches millions of rows. |
| **Proposed fix** | Add index: `CREATE INDEX idx_queue_claim ON job_queue(status, module, priority DESC, queued_at ASC)`. This turns the claim query into an index-only scan with O(1) seek. |
| **Expected gain** | Claim query drops from O(n) table scan to O(log n) index seek. |
| **Complexity** | Very Low — single `CREATE INDEX` statement in schema migration. |

---

#### 15. Reduce Technical Analysis Array Copies

| Attribute | Detail |
|-----------|--------|
| **Files** | `analysis/technical.py:16-289` |
| **Resource** | CPU, Memory |
| **Current behavior** | The `analyze()` method creates at least 9 intermediate copies of the image data: 2 PIL↔numpy round-trips for 3000px resize, 2 more for 1200px sharpness resize, 2 grayscale conversions (float32), LAB conversion for saliency, noise estimation, and K-means sampling. Each copy of a 3000×2000 image is ~18 MB (uint8) or ~72 MB (float32). |
| **Impact at 500K** | Peak memory during technical analysis: ~300-400 MB per image. With multiple threads, this compounds. |
| **Proposed fix** | (a) Resize once to 1200px (sufficient for all sub-analyses). (b) Compute grayscale once and reuse for sharpness + noise. (c) Use `np.float32` throughout (avoid float64 intermediates in grayscale conversion). (d) Vectorize patch-based sharpness: apply Laplacian to the full image, then compute variance per patch using `reshape` + `var(axis=(-2,-1))`. |
| **Expected gain** | ~50% memory reduction. ~2x CPU speedup from eliminating redundant resizes and vectorizing patch analysis. |
| **Complexity** | Medium — requires careful refactoring of the analysis pipeline to ensure numerical equivalence. |

---

#### 16. Override Check Optimization (Negative Cache)

| Attribute | Detail |
|-----------|--------|
| **Files** | `db/repository.py:359-366` |
| **Resource** | Disk I/O (SQLite) |
| **Current behavior** | `_apply_override_mask()` queries the `overrides` table on every `upsert_*` call. With 9 modules per image: 4.5M override checks at 500K images. In practice, overrides are extremely rare (manually set by users), so ~99.99% of queries return empty results. |
| **Proposed fix** | (a) Pre-fetch all overridden image IDs into a `Set` at worker startup: `SELECT DISTINCT image_id FROM overrides`. (b) Only query the `overrides` table if `image_id in override_set`. (c) Invalidate the cache when the user sets a new override (rare operation). |
| **Expected gain** | Eliminates ~4.5M unnecessary SQLite queries. |
| **Complexity** | Very Low — add a cached set check before the query. |

---

#### 17. Deduplicate File Hashing — Use Filesystem Metadata

| Attribute | Detail |
|-----------|--------|
| **Files** | `pipeline/batch.py:87-88`, `pipeline/modules.py:23-32` |
| **Resource** | Disk I/O |
| **Current behavior** | `compute_file_hash()` reads the entire file with SHA-256 (64KB chunks). For 500K images at 25MB average: ~12.5 TB of I/O just for hashing. |
| **Proposed fix** | (a) Use a composite key of `(file_path, file_size, mtime)` for change detection instead of SHA-256. This is what most tools (rsync, Make) use. (b) Only compute SHA-256 on demand (e.g., when the user requests deduplication). (c) If hashing is required, use `xxhash` (xxh128) which is 10-30x faster than SHA-256 on modern CPUs with 64-bit operations. |
| **Expected gain** | Eliminates 12.5 TB of unnecessary I/O during ingest, or reduces hash time by 10-30x with xxhash. |
| **Complexity** | Low — change the hash function or replace with mtime check. Schema migration needed for the `file_hash` column semantics. |

---

#### 18. IO Job Polling Reduction in Worker

| Attribute | Detail |
|-----------|--------|
| **Files** | `pipeline/worker.py:296-297, 350-352, 230-246` |
| **Resource** | Disk I/O (SQLite) |
| **Current behavior** | After every single GPU job completion, `_submit_io_jobs()` polls the queue for all IO module types (4 separate `claim()` queries for `LOCAL_IO_MODULES` + `CLOUD_MODULES`). At 500K × 5 GPU modules = 2.5M GPU jobs, this generates ~10M queue polling queries. |
| **Proposed fix** | (a) Poll IO jobs every N GPU jobs (e.g., every 10-50) instead of after every single one. (b) Use a trigger-based approach: only poll for a module's IO jobs when a prerequisite module completes for that image. (c) Maintain an in-memory pending count and only poll when the thread pool has free capacity. |
| **Expected gain** | 10-50x fewer polling queries. Reduced SQLite contention between worker threads. |
| **Complexity** | Low — add a counter that gates the polling frequency. |

---

#### 19. Electron Batch Thumbnail IPC

| Attribute | Detail |
|-----------|--------|
| **Files** | `imganalyzer-app/src/main/index.ts:78-80`, `imganalyzer-app/src/main/images.ts:67-167` |
| **Resource** | IPC overhead, subprocess spawning |
| **Current behavior** | Each thumbnail request is a separate IPC round-trip (`fs:getThumbnail`). Each thumbnail generation spawns a separate Python subprocess via `conda run`. With `MAX_CONCURRENT=4`, throughput is ~1-2 thumbnails/second. |
| **Proposed fix** | (a) Add a batch IPC handler: `fs:getThumbnails(paths[])` → `{path: dataUrl}[]`. (b) Generate thumbnails in Python using a single long-lived process (see #13) that accepts batch requests. (c) Use Node.js `sharp` for JPEG/PNG/WebP thumbnails (no Python needed for standard formats), falling back to Python only for RAW/HEIC. (d) Write thumbnails to disk cache with content-addressed filenames. |
| **Expected gain** | 10-50x thumbnail throughput. Eliminates subprocess-per-thumbnail overhead. |
| **Complexity** | Medium — requires `sharp` integration or persistent Python server. |

---

#### 20. Batch Status Polling via File or Shared Memory

| Attribute | Detail |
|-----------|--------|
| **Files** | `imganalyzer-app/src/main/batch.ts:231-269, 274` |
| **Resource** | CPU, Memory (subprocess overhead) |
| **Current behavior** | Status polling spawns `conda run ... status --json` every 1 second. Each spawn has 1-3s conda overhead, meaning polls can overlap (no guard against concurrent polls). The status command queries the `job_queue` table for counts. |
| **Proposed fix** | (a) Write status to a JSON file (e.g., `~/.cache/imganalyzer/status.json`) from the worker process. Electron reads this file directly — no subprocess needed. (b) Use a SQLite WAL read (Electron reads the DB directly via `better-sqlite3`). (c) If using a persistent Python process (#13), expose a `/status` endpoint. (d) Add a guard to prevent overlapping polls (`if (pollInFlight) return`). |
| **Expected gain** | Eliminates 1 subprocess/second during batch processing. Reduces poll latency from 1-3s to <10ms. |
| **Complexity** | Low (file-based) to Medium (direct SQLite read from Electron). |

---

## Impact Summary Matrix

| # | Improvement | CPU | GPU | Memory | Disk I/O | Latency | Complexity | Priority |
|---|------------|-----|-----|--------|----------|---------|------------|----------|
| 1 | Image cache across modules | ★★★ | — | ★★★ | ★★★★★ | ★★★ | Medium | P0 |
| 2 | Batch DB ingest operations | — | — | — | ★★★★★ | ★★★★★ | Low-Med | P0 |
| 3 | Vectorized semantic search | ★★★★★ | — | ★★★ | ★★★ | ★★★★★ | Low | P0 |
| 4 | Batch GPU inference | — | ★★★★★ | ★★ | — | ★★★★ | Med-High | P0 |
| 5 | Fix `listImages()` sync stats | ★★★★★ | — | ★★★ | ★★★★ | ★★★★★ | Medium | P0 |
| 6 | Single directory traversal | ★★ | — | ★ | ★★★★ | ★★★★ | Very Low | P1 |
| 7 | Pre-resize for AI models | ★★★ | — | ★★★★ | — | ★★ | Low | P1 |
| 8 | RAW half-size demosaic | ★★★★ | — | ★★★★ | ★★ | ★★★ | Very Low | P1 |
| 9 | Optimize search index update | — | — | — | ★★★★ | ★★★ | Medium | P1 |
| 10 | Batch queue commits | — | — | — | ★★★★ | ★★★ | Low-Med | P1 |
| 11 | BLIP-2 VQA batching | — | ★★★★ | ★★ | — | ★★★ | Low | P1 |
| 12 | Bounded thumbnail cache | — | — | ★★★★★ | — | — | Low | P1 |
| 13 | Persistent Python process | ★★★ | — | ★★★ | — | ★★★★★ | High | P2 |
| 14 | Queue claim index | — | — | — | ★★★ | ★★★ | Very Low | P2 |
| 15 | Technical analysis copies | ★★★ | — | ★★★ | — | ★★ | Medium | P2 |
| 16 | Override negative cache | — | — | — | ★★★ | ★★ | Very Low | P2 |
| 17 | File hash optimization | — | — | — | ★★★★★ | ★★★ | Low | P2 |
| 18 | IO job polling reduction | — | — | — | ★★★ | ★★ | Low | P2 |
| 19 | Batch thumbnail IPC | ★★ | — | ★★ | — | ★★★★ | Medium | P2 |
| 20 | Status polling optimization | ★★ | — | ★★ | — | ★★★ | Low | P2 |

**Legend:** ★ = minor impact, ★★★★★ = major impact

---

## Recommended Implementation Order

### Phase 1 — Quick Wins (1-2 days, 3-5x overall improvement)
1. **#6** Single directory traversal (5-line change)
2. **#8** RAW half-size demosaic (1-line change)
3. **#14** Add queue claim index (1 SQL statement)
4. **#16** Override negative cache (10-line change)
5. **#17** File hash with xxhash or mtime (20-line change)

### Phase 2 — Core Pipeline Optimization (3-5 days, 5-10x improvement)
6. **#2** Batch DB ingest operations
7. **#10** Batch queue commits
8. **#1** Image cache across modules
9. **#7** Pre-resize for AI models
10. **#11** BLIP-2 VQA batching

### Phase 3 — Search & UI (2-3 days)
11. **#3** Vectorized semantic search
12. **#5** Fix `listImages()` synchronous stats
13. **#12** Bounded thumbnail cache with LRU
14. **#9** Optimize search index update

### Phase 4 — GPU & Architecture (5-7 days)
15. **#4** Batch GPU inference
16. **#13** Persistent Python process
17. **#19** Batch thumbnail IPC
18. **#20** Status polling optimization
19. **#18** IO job polling reduction
20. **#15** Technical analysis array copies

---

## Appendix: Resource Consumption Profile at 500K Images

### Memory Profile (Peak, Single Image Processing)

| Component | Current | After Optimization |
|-----------|---------|-------------------|
| RAW demosaic output | 144 MB | 36 MB (half-size) |
| Image copies (7 modules) | ~1 GB | ~144 MB (1 decode) |
| BLIP-2 model (fp16) | 4 GB VRAM | 4 GB VRAM (no change) |
| GroundingDINO model | 700 MB VRAM | 700 MB VRAM (no change) |
| TrOCR model (fp16) | 650 MB VRAM | 650 MB VRAM (no change) |
| InsightFace model | 300 MB VRAM | 300 MB VRAM (no change) |
| CLIP ViT-L/14 (fp16) | 856 MB VRAM | 856 MB VRAM (no change) |
| Technical analysis arrays | ~400 MB | ~200 MB |
| Electron thumbnail cache | Unbounded (→20 GB) | 200 MB (LRU cap) |
| Semantic search embeddings | 3 GB per query | 1.5 GB cached once |

### Throughput Estimates (per hour, single GPU)

| Stage | Current | After Phase 2 | After Phase 4 |
|-------|---------|---------------|---------------|
| Ingest (register + queue) | ~10K imgs/hr | ~500K imgs/hr | ~500K imgs/hr |
| GroundingDINO (objects) | ~7,200/hr | ~7,200/hr | ~28,800/hr (batch 4) |
| BLIP-2 (caption + VQA) | ~1,800/hr | ~4,500/hr | ~9,000/hr (batch) |
| CLIP embedding | ~10,800/hr | ~10,800/hr | ~43,200/hr (batch 16) |
| TrOCR (OCR) | ~3,600/hr | ~3,600/hr | ~7,200/hr |
| InsightFace (faces) | ~14,400/hr | ~14,400/hr | ~14,400/hr |
| Technical analysis | ~7,200/hr | ~14,400/hr | ~14,400/hr |
| **End-to-end per image** | **~2.0s/img** | **~1.0s/img** | **~0.3s/img** |
| **500K total time** | **~278 hrs** | **~139 hrs** | **~42 hrs** |

### SQLite Operation Counts

| Operation | Current (500K) | After Optimization |
|-----------|---------------|-------------------|
| Ingest commits | ~5,000,000 | ~5,000 (batched 1000) |
| Override checks | ~4,500,000 | ~0 (negative cache) |
| Search index queries | ~10,000,000+ | ~2,000,000 (single JOIN) |
| Queue polling queries | ~10,000,000 | ~200,000 (reduced frequency) |
| Claim table scans | O(n) per claim | O(log n) with index |

---

## Appendix: Bugs Found During Analysis

1. **Duplicate IPC handler registration** (`imganalyzer-app/src/main/index.ts:45-49`): On macOS, if all windows are closed and the app is re-activated from the dock, `registerSearchHandlers()` is called again, which calls `ipcMain.handle('search:run', ...)` a second time. Electron throws on duplicate handle registration, crashing the app.

2. **TOCTOU race in queue claim** (`db/queue.py:92-128`): The SELECT + UPDATE in `claim()` is not wrapped in an explicit exclusive lock. With WAL mode and concurrent worker threads, two threads could claim the same jobs. The code comments acknowledge this but the mitigation is incomplete.

3. **Dead code** (`analysis/ai/ocr.py:246-301`): The `_estimate_font_size()` function is defined but never called. `_compute_target_size()` uses a simple 1920px cap instead.

4. **Legacy face_db.py still loaded** (`analysis/ai/face_db.py`): Stores embeddings as JSON arrays of 512 floats (extremely wasteful). The database-backed face storage in `repository.py` has superseded this, but the old code is still imported and referenced.
