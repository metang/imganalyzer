# Performance Analysis Cross-Comparison

**Sources compared:**
- `perf_claude_sonnet_4.6.md` — Claude Sonnet 4.6 (16 issues, 991 lines)
- `perf_clause_opus_4.6.md` — Claude Opus 4.6 (20 issues, 449 lines)
- `perf_clause_codex5.2.md` — Codex 5.2 (10 issues, 736 lines)

**Date:** 2026-02-27
**Target workload:** 500K+ images, single GPU

---

## 1. Top 5 Issues All Three Models Agree On

These are the highest-confidence findings — every independent analysis flagged them.

### 1.1 Redundant Image Decoding Across Modules

| Model | Priority | Est. Impact |
|-------|----------|-------------|
| Sonnet | Critical (P0) | 4–6x throughput; 81 CPU-days saved at 500K RAW |
| Opus | Critical (P0) | 7x reduction in image I/O |
| Codex | Critical | ~7x reduction in decode CPU time |

**Consensus:** Each of the 7–10 analysis modules independently calls `_read_image(path)`, fully decoding the image from disk every time. For a 50MP RAW file this is ~2–5 seconds of CPU per decode. At 500K images this generates ~3.5M redundant file reads.

**Agreed fix:** Introduce a per-image decode cache in the worker so the image is decoded once and shared across all modules for that image. Each `_run_*` method should accept an optional pre-decoded `image_data` parameter. The CLIP embedder specifically needs a `embed_image_pil()` overload to accept a pre-loaded PIL image.

**Files:** `pipeline/modules.py:35–49`, `pipeline/worker.py`, `readers/raw.py:28–33`, `embeddings/clip_embedder.py:42–93`

---

### 1.2 Batch Database Operations During Ingest

| Model | Priority | Est. Impact |
|-------|----------|-------------|
| Sonnet | High (P1) | 10x ingest DB perf; 5M→5 queries |
| Opus | Critical (P0) | 5000x fewer commits; hours→minutes |
| Codex | Medium-High | 3–5x with batch FTS inserts |

**Consensus:** The ingest loop issues individual SQL operations per image per module — `register_image`, `is_analyzed`, `enqueue` — each with its own COMMIT. At 500K images × 10 modules this generates ~5–10M individual COMMITs. Even in WAL mode, each commit involves an fsync (~50–100μs), adding up to minutes–hours of pure commit overhead.

**Agreed fix:** Wrap ingest in batched transactions (BEGIN...COMMIT every 1000 images), replace `enqueue_batch` with a true bulk INSERT, batch `is_analyzed` checks, and use `executemany` for new image registration.

**Files:** `pipeline/batch.py:79–112`, `db/queue.py:25–88`, `db/repository.py:64–85`

---

### 1.3 O(N) Brute-Force CLIP Semantic Search

| Model | Priority | Est. Impact |
|-------|----------|-------------|
| Sonnet | Critical (P0) | 8–25s → <100ms per query; 6 GB → 50 MB memory |
| Opus | Critical (P0) | 10–60s → <100ms; 3 GB loaded per query |
| Codex | — (not covered as standalone item) | — |

**Note:** Codex did not identify this as a standalone issue, making it a consensus between Sonnet and Opus only. However, both rated it Critical.

**Consensus (Sonnet + Opus):** Every search query calls `get_all_embeddings()` which fetches ALL embeddings from SQLite into Python lists, then performs O(N) cosine similarity in a Python loop. At 500K images × 2 embedding types × 768 dims × 4 bytes = ~3 GB of raw BLOB data loaded per query. Since CLIP embeddings are L2-normalized, the `np.linalg.norm` calls are redundant — dot product suffices.

**Agreed fix:** Replace with vectorized numpy matrix multiply (short term) or FAISS ANN index (long term). Pre-load embeddings into a single `(N, 768)` numpy matrix at startup, cache in memory, and use `scores = matrix @ query_vec` (one BLAS call).

**Files:** `db/search.py:350–374`, `db/repository.py:540–546`, `embeddings/clip_embedder.py:149–153`

---

### 1.4 GPU Batch Inference (Single Image per Forward Pass)

| Model | Priority | Est. Impact |
|-------|----------|-------------|
| Sonnet | Medium (P2) | 2–3x GPU utilisation via CLIP micro-batching |
| Opus | Critical (P0) | 3–8x GPU throughput for CLIP/GroundingDINO |
| Codex | High | 2–4x wall-clock on GPU modules |

**Consensus:** All GPU modules process one image at a time. Modern GPUs can process 8–32 images per batch for CLIP/GroundingDINO with near-linear throughput scaling. GPU utilization is typically 20–40% due to Python overhead between forward passes.

**Agreed fix:** Claim batch_size=8–16 images from the queue per GPU module pass, preprocess all images to tensors on CPU threads, run a single batched forward pass, write results back. CLIP benefits most (batch 16–32), GroundingDINO supports batch 4–8. InsightFace (ONNX) cannot batch.

**Priority disagreement:** Sonnet rates this Medium; Opus rates it P0; Codex rates it High.

**Files:** `pipeline/worker.py:280–346`, `analysis/ai/objects.py:77–90`, `embeddings/clip_embedder.py:81–90`

---

### 1.5 BLIP-2 VQA Question Batching (5 ViT-L Passes → 1)

| Model | Priority | Est. Impact |
|-------|----------|-------------|
| Sonnet | Medium (implied, under GPU micro-batching) | Part of GPU batching item |
| Opus | High (P1) | 2–3x faster per-image BLIP-2 inference |
| Codex | High | ~4x reduction in BLIP-2 GPU compute |

**Consensus:** BLIP-2 runs 5 sequential forward passes per image (1 caption + 4 VQA). The ViT-L image encoder runs identically all 5 times — only the text prompt differs. Additionally, `torch.cuda.empty_cache()` is called between every VQA question, introducing unnecessary CUDA synchronization stalls.

**Agreed fix:** Cache the visual features from the first forward pass and reuse for VQA. Batch the 4 VQA questions into a single `processor(images=[img]*4, text=[q1,q2,q3,q4])` + single `model.generate()`. Remove per-question `empty_cache()` calls.

**Files:** `analysis/ai/local.py:65–94`

---

## 2. Issues Covered by Two of Three Models

### 2.1 FTS5 Search Index Rebuild Storm

| Model | Priority |
|-------|----------|
| Sonnet | Critical (P0) — 3M FTS5 DELETE+INSERT at 500K |
| Opus | High (P1) — 10M+ queries for search index maintenance |
| Codex | Medium-High — ~1.5M multi-table read + FTS5 write cycles |

All three models identify this, but with different severity ratings. Sonnet considers it the #2 most critical issue; Opus and Codex rate it lower.

**Consensus fix:** Defer FTS rebuild to post-processing (mark images as "FTS-dirty", batch rebuild at end), and only call `update_search_index` once after all modules for an image complete rather than after each module.

---

### 2.2 SHA-256 File Hashing at Ingest

| Model | Priority |
|-------|----------|
| Sonnet | High | 12.5 TB I/O at 500K; 7 hours just for hashing |
| Opus | Medium (P2) | Same estimate; suggests xxhash as 10-30x faster alternative |
| Codex | Medium | 20 TB estimate (using 40 MB average); suggests mtime+size fingerprint |

**Consensus fix:** Use `(path, size, mtime)` as primary dedup key; only fall back to hashing on collision. Alternatively, switch from SHA-256 to xxhash for 10–30x speedup.

---

### 2.3 RAW Half-Size Demosaic

| Model | Priority |
|-------|----------|
| Sonnet | Medium | 1/4 decode time and memory with `half_size=True` |
| Opus | High (P1) | 3–5x faster RAW decode, 4x less memory |
| Codex | High | Covered in both OPP-1 and OPP-7 |

**Consensus fix:** Default to `half_size=True` for rawpy when the downstream module doesn't need full resolution (all AI models resize to ≤1920px anyway).

---

### 2.4 `find_face_by_alias` Full Table Scan

| Model | Priority |
|-------|----------|
| Sonnet | Medium | O(N) Python JSON scan → O(log N) with indexed alias table |
| Opus | Medium (P1, under #9) | Full table scan + JSON parsing per detected face |
| Codex | Medium-High (under OPP-5) | Add index on canonical_name and display_name |

**Consensus fix:** Replace JSON aliases column with an indexed `face_aliases` table, or at minimum add B-tree indexes on `canonical_name` and `display_name`.

---

### 2.5 Technical Analysis Double Resize + Patch Loop

| Model | Priority |
|-------|----------|
| Sonnet | Medium | Two downsamples (3000px then 1200px); Python for-loop over 216 patches |
| Opus | Medium (P2) | 9 intermediate copies; ~300-400 MB peak per image |
| Codex | — (not standalone) | Mentioned tangentially in OPP-9 (worker parallelism) |

**Consensus fix:** Single downsample to 1200px upfront. Vectorize `_patch_sharpness` with strided array views (one Laplacian on full image, then reshape + var per patch).

---

## 3. Notable Unique Findings (One Model Only)

### 3.1 Sonnet-Only Findings

**Reverse Geocoding — Blocking HTTP With No Cache** (High priority)
- `metadata.py:41–60`: synchronous HTTP call to Nominatim per geotagged image
- At 1 req/sec rate limit × 400K geotagged images = **~111 hours** just for geocoding
- No two images at the same location reuse the result
- Fix: LRU cache with GPS coordinates rounded to 4 decimal places (~11m resolution)
- Neither Opus nor Codex mention this at all

**`asyncio.run()` Creates New Event Loop Per Call** (Low)
- Cloud AI backend creates/destroys an event loop per image
- Fix: Thread-local event loop reuse

**`iter_image_ids()` Loads All IDs Into Memory** (Low)
- Fix: Cursor-based chunked iteration

---

### 3.2 Opus-Only Findings

**52x Directory Traversal During Ingest** (High, P1)
- `batch.py:51–70`: 26 extensions × 2 cases (upper/lower) = 52 `rglob` calls
- Fix: Single `rglob("*")` with suffix filter — 5-line change
- Neither Sonnet nor Codex mention this

**Override Check Negative Cache** (Medium, P2)
- `repository.py:359–366`: 4.5M override queries at 500K; ~99.99% return empty
- Fix: Pre-fetch overridden image IDs into a Set at startup

**Electron GUI Analysis (5 items)**
- `listImages()` synchronous `statSync` — blocks main process for minutes at 500K
- Unbounded thumbnail cache (`Map<string, string>`, no eviction) — OOM at 500K
- Persistent Python process — eliminate 1–3s conda overhead per operation
- Batch thumbnail IPC — subprocess-per-thumbnail bottleneck
- Status polling via file/shared memory — eliminate 1 subprocess/second

**Bugs Found:**
1. Duplicate IPC handler registration crash on macOS re-activate (`index.ts:45–49`)
2. TOCTOU race in queue claim (`queue.py:92–128`) — two threads can claim same jobs
3. Dead code: `_estimate_font_size()` in OCR never called (`ocr.py:246–301`)
4. Legacy `face_db.py` JSON embedding storage still loaded (superseded by DB)

---

### 3.3 Codex-Only Findings

**GPU Inference Pipelining (3-Stage Double-Buffer)** (High)
- 3-stage pipeline: CPU decode (N+1) → GPU inference (N) → DB write (N-1)
- Use CUDA streams for async H2D transfers
- Expected 2–3x pipeline throughput improvement
- Neither Sonnet nor Opus propose this specific architecture

**GroundingDINO bfloat16/Quantization** (Low-Medium)
- Test mixed precision: visual backbone in bfloat16, text encoder in float32
- Evaluate `torch.compile()` and INT8 post-training quantization
- ~350 MB VRAM reduction, 10–30% faster inference

**Increase Default Worker Parallelism** (Medium)
- `workers=1` default for CPU-bound modules (metadata, technical)
- Fix: `workers = min(os.cpu_count(), 8)` — linear speedup on multi-core
- Simple change, significant impact for technical analysis

**Metadata Module — Skip Pixel Decode Entirely** (Medium)
- `_run_metadata()` doesn't need pixel data (EXIF only)
- Create `read_headers_only(path)` to avoid full decode for metadata pass
- Saves ~300–600 CPU-hours at 500K images

**Measurement Plan Methodology**
- Per-module timing aggregation (mean, p50, p95, p99)
- GPU utilization monitoring via `nvidia-smi dmon`
- Memory profiling via `tracemalloc`
- I/O profiling via `iostat`
- DB write throughput measurement

---

## 4. Priority Disagreements

| Issue | Sonnet | Opus | Codex |
|-------|--------|------|-------|
| FTS5 rebuilds | **Critical (P0)** | P1 | Medium-High |
| GPU batching | Medium (P2) | **P0** | High |
| SHA-256 hashing | **High (P1)** | P2 | Medium |
| RAW half-size | Medium (P2) | **P1** | High |
| CLIP search | **Critical** | **Critical** | Not covered |
| Technical analysis | Medium | P2 | Not standalone |

Notable patterns:
- **Sonnet** is most aggressive on database/I/O issues (FTS, hashing, CLIP search)
- **Opus** is most aggressive on GPU utilization (batching, BLIP-2 VQA)
- **Codex** takes a balanced middle ground but misses CLIP search entirely
- Only **Opus** covers the Electron/GUI layer; Sonnet and Codex are Python-only

---

## 5. Coverage Comparison

| Area | Sonnet | Opus | Codex |
|------|--------|------|-------|
| Python backend | Yes (16 items) | Yes (12 items) | Yes (10 items) |
| Electron GUI | No | Yes (5 items) | No |
| Database/SQLite | Deep (6 items) | Deep (5 items) | Moderate (2 items) |
| GPU optimization | Moderate | Deep | Deep |
| Implementation roadmap | Yes (4-phase) | Yes (4-phase) | Yes (4-phase) |
| Throughput estimates | Yes (quantitative) | Yes (with table) | Yes (with table) |
| Measurement plan | No | No | **Yes (detailed)** |
| Bugs found | No | **Yes (4 bugs)** | No |
| Code examples | Extensive | Moderate | Moderate |
| Effort estimates | Relative (Low/Med/High) | Relative | **Absolute (days/weeks)** |

---

## 6. Unified Implementation Plan

Combining all three analyses into a single prioritized plan:

### Phase 1 — Quick Wins (1–2 days, low risk)

| # | Item | Source | Effort | Status |
|---|------|--------|--------|--------|
| 1 | Single directory traversal (52x → 1x rglob) | Opus #6 | Very Low | ✅ DONE |
| 2 | RAW half-size demosaic default | All three | Very Low | ✅ DONE |
| 3 | Queue claim composite index | Opus #14 | Very Low | ✅ DONE |
| 4 | Override negative cache | Opus #16 | Very Low | ✅ DONE |
| 5 | Increase default worker parallelism | Codex OPP-9 | Trivial | ✅ DONE |
| 6 | Metadata header-only read (skip pixel decode) | Codex OPP-8 | Low | ✅ DONE |

### Phase 2 — Core Pipeline Optimization (3–5 days, medium risk)

| # | Item | Source | Effort | Status |
|---|------|--------|--------|--------|
| 7 | Per-image decode cache across modules | All three | Medium | ✅ DONE |
| 8 | Batch DB ingest operations | All three | Low-Medium | ✅ DONE |
| 9 | Batch queue commits | Sonnet #6, Opus #10 | Low-Medium | ✅ DONE |
| 10 | Pre-resize images for AI modules | Opus #7, Codex OPP-7 | Low | ✅ DONE |
| 11 | BLIP-2 VQA batching (5 passes → 1–2) | All three | Low-Medium | ✅ DONE |
| 12 | Reverse geocoding cache | Sonnet #4 | Low | ✅ DONE |
| 13 | SHA-256 → mtime/size fingerprint | All three | Low | ✅ DONE |

### Phase 3 — Search & UI (2–3 days)

| # | Item | Source | Effort | Status |
|---|------|--------|--------|--------|
| 14 | Vectorized / FAISS semantic search | Sonnet #3, Opus #3 | Medium | ✅ DONE |
| 15 | Deferred/batched FTS5 search index updates | All three | Low-Medium | ✅ DONE |
| 16 | Face alias indexed table | Sonnet #11, Opus #9 | Medium | ✅ DONE |
| 17 | Fix `listImages()` synchronous stats | Opus #5 | Medium | ✅ DONE |
| 18 | Bounded LRU thumbnail cache | Opus #12 | Low-Medium | ✅ DONE |
| 19 | Missing DB indexes (date, iso, camera) | Sonnet #14 | Low | ✅ DONE |

### Phase 4 — GPU & Architecture (5–7 days, high risk)

| # | Item | Source | Effort |
|---|------|--------|--------|
| 20 | GPU batch inference (CLIP, GroundingDINO) | All three | Medium-High |
| 21 | GPU inference pipelining (3-stage) | Codex OPP-4 | High |
| 22 | Persistent Python process for Electron | Opus #13 | High |
| 23 | Batch thumbnail IPC | Opus #19 | Medium |
| 24 | Status polling optimization | Opus #20 | Low |
| 25 | GroundingDINO bfloat16/quantization | Codex OPP-10 | Medium |
| 26 | Multi-GPU architecture | Sonnet #16 | High |

### Estimated Cumulative Impact

| After Phase | Est. Total Time (500K images) | Speedup vs Baseline |
|-------------|-------------------------------|---------------------|
| Baseline (current) | 30–60 days | 1x |
| Phase 1 | 20–40 days | ~1.5x |
| Phase 2 | 5–10 days | ~5–8x |
| Phase 3 | 3–5 days | ~8–12x |
| Phase 4 | 1–2 days | ~15–30x |

---

## 7. Bugs to Fix (Found by Opus)

These should be fixed regardless of performance work:

1. **Duplicate IPC handler registration** (`imganalyzer-app/src/main/index.ts:45–49`): On macOS dock re-activate, `registerSearchHandlers()` is called again, calling `ipcMain.handle('search:run', ...)` a second time. Electron throws on duplicate handle registration, crashing the app.

2. **TOCTOU race in queue claim** (`db/queue.py:92–128`): SELECT + UPDATE in `claim()` is not wrapped in an explicit exclusive lock. With WAL mode and concurrent worker threads, two threads could claim the same jobs.

3. **Dead code** (`analysis/ai/ocr.py:246–301`): `_estimate_font_size()` is defined but never called. `_compute_target_size()` uses a simple 1920px cap instead.

4. **Legacy face_db.py still loaded** (`analysis/ai/face_db.py`): Stores embeddings as JSON arrays of 512 floats (extremely wasteful). The database-backed face storage in `repository.py` has superseded this, but the old code is still imported and referenced.

---

## 8. Key Files Reference

| File | Issues |
|------|--------|
| `imganalyzer/pipeline/modules.py` | Image cache, pre-resize, metadata skip, FTS rebuild |
| `imganalyzer/pipeline/worker.py` | GPU batching, pipelining, worker parallelism |
| `imganalyzer/pipeline/batch.py` | Directory traversal, ingest batching, file hashing |
| `imganalyzer/readers/raw.py` | Half-size demosaic, header-only read |
| `imganalyzer/readers/standard.py` | Pre-downscale, header-only read |
| `imganalyzer/db/queue.py` | Batch commits, claim index, TOCTOU race |
| `imganalyzer/db/repository.py` | Search index, face aliases, override cache |
| `imganalyzer/db/search.py` | CLIP brute-force search, FTS5 |
| `imganalyzer/db/schema.py` | Missing indexes |
| `imganalyzer/analysis/ai/local.py` | BLIP-2 VQA batching |
| `imganalyzer/analysis/ai/objects.py` | GroundingDINO batch/quantization |
| `imganalyzer/analysis/ai/ocr.py` | Tiling overhead, dead code |
| `imganalyzer/analysis/ai/faces.py` | Full-res BGR copy |
| `imganalyzer/analysis/metadata.py` | Reverse geocoding cache |
| `imganalyzer/analysis/technical.py` | Double resize, patch loop vectorization |
| `imganalyzer/embeddings/clip_embedder.py` | Batch embed, PIL overload |
| `imganalyzer-app/src/main/images.ts` | Sync statSync, thumbnail cache |
| `imganalyzer-app/src/main/batch.ts` | Status polling |
| `imganalyzer-app/src/main/index.ts` | Duplicate IPC handler bug |

---

*Cross-comparison generated from three independent performance analyses by Claude Sonnet 4.6, Claude Opus 4.6, and Codex 5.2. No code was modified.*
