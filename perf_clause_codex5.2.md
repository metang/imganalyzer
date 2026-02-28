# Performance Analysis & Improvement Plan — imganalyzer
## Codex 5.2 | Targeting 500K+ Image Workloads

---

## 1. Scope & Methodology

This document captures a static analysis of the imganalyzer codebase as of the current
commit, focusing on CPU, GPU, and memory bottlenecks that compound at scale (≥500K images).
All file references use repository-relative paths. No code was modified; this is analysis
and a ranked improvement plan only.

Workload model used throughout:
- 500,000–1,000,000 images, mixture of JPEG, HEIC, RAW (ARW/CR3/NEF)
- Single GPU (NVIDIA, 16–24 GB VRAM), 16–32 CPU cores
- Pipeline runs `objects → blip2 → ocr → faces → embedding` (split-pass mode)

---

## 2. Architecture Summary (Relevant to Performance)

```
BatchProcessor.ingest()
  └── rglob scan + SHA-256 hash per file → DB register + job_queue enqueue

Worker.run()
  Phase 1: objects (GPU, serial)
           metadata/technical (CPU, ThreadPoolExecutor, workers=1 default)
  Phase 2: blip2/ocr/faces/embedding (GPU, serial)
           cloud_ai/aesthetic (cloud/CPU, ThreadPoolExecutor, cloud_workers=4)

Per job → ModuleRunner.run()
  └── _read_image(path)   ← decode from disk EVERY time
  └── <module analysis>
  └── DB upsert + optional search_index rebuild
```

Each module independently calls `_read_image()`, which goes through the full
decode chain (rawpy postprocess for RAW, Pillow for standard) on the original file.

---

## 3. Identified Hotspots

### 3.1 Repeated Full-Resolution Decode (CPU + Memory) — CRITICAL

**Files:** `imganalyzer/pipeline/modules.py:35–49`, `imganalyzer/readers/raw.py:17–73`

Every split-pass module (`metadata`, `technical`, `objects`, `blip2`, `ocr`, `faces`,
`cloud_ai`, `aesthetic`, `embedding`) independently calls `_read_image()` which:

- Opens and fully decodes the image from disk.
- For RAW (ARW/NEF/CR3): runs `rawpy.postprocess()` which invokes LibRaw's demosaicing,
  white-balance, and gamma pipeline. This produces a **full-resolution float32 → uint8
  H×W×3 array** (e.g., 50 MP Sony ARW → ~150 MB in-process).
- For JPEG/HEIC: opens via Pillow, allocates full numpy array.

For a single 50 MP RAW image processed by 8 modules, this means:

| Module     | Calls `_read_image` | RAW decode time (est.) | Peak memory per call |
|------------|---------------------|------------------------|----------------------|
| metadata   | yes                 | ~2–4 s                 | ~150 MB              |
| technical  | yes                 | ~2–4 s                 | ~150 MB              |
| objects    | yes                 | ~2–4 s                 | ~150 MB              |
| blip2      | yes                 | ~2–4 s                 | ~150 MB              |
| ocr        | yes                 | ~2–4 s                 | ~150 MB              |
| faces      | yes                 | ~2–4 s                 | ~150 MB              |
| cloud_ai   | yes                 | ~2–4 s                 | ~150 MB              |
| embedding  | yes (re-reads file) | ~2–4 s                 | ~150 MB              |

**Total redundant CPU time per RAW image: ~16–32 seconds of pure decode work.**
At 500K RAW images, this is ~2,000–4,000 CPU-hours wasted on redundant decoding.

**Root cause:** `ModuleRunner.run()` does not pass decoded image data between module
calls; each module re-reads from disk independently. Jobs are dispatched by module,
not by image, making cross-module sharing harder with the current DB-queue model.

---

### 3.2 GPU Strict Serialization (GPU Underutilization) — HIGH

**Files:** `imganalyzer/pipeline/worker.py:57–58`, `worker.py:325–365`

```python
GPU_MODULES = {"local_ai", "embedding", "blip2", "objects", "ocr", "faces"}
```

All GPU modules are processed **one job at a time on the main thread**. There is no
pipelining or overlap between:
- Data loading/preprocessing (CPU-bound) for image N+1
- Model inference on GPU for image N
- Result post-processing/DB write for image N-1

On a modern GPU, the actual forward-pass utilization is typically 20–40% of wall time
per image; the rest is Python overhead, IO, and CPU preprocessing. Strict serialization
means the GPU is idle during all that overhead.

At 500K images with 6 GPU modules, the GPU queue is serialized through **3 million
individual GPU calls** with no pipelining.

---

### 3.3 RAW Decode Resolution Mismatch (CPU + Memory) — HIGH

**Files:** `imganalyzer/readers/raw.py:28–33`, `imganalyzer/analysis/ai/objects.py:64–69`,
`imganalyzer/embeddings/clip_embedder.py:64–67`

Each AI model immediately downscales after receiving the full-res array:

| Module/model   | Internal resize target | Input fed to it           |
|----------------|------------------------|---------------------------|
| GroundingDINO  | 800 px long edge       | Full-res after `_read_image` |
| BLIP-2         | 224 px (processor)     | Full-res after `_read_image` |
| CLIP           | 224 px (model)         | 1280 px after `thumbnail()` |
| TrOCR          | 384 px per tile        | Resized to 1920 px max    |
| InsightFace    | 640×640 px det_size    | Full-res BGR copy          |

For RAW files, `raw.py` does not use `half_size=True` even when full resolution is
unnecessary. A 50 MP Sony ARW at half-size is ~12 MP — still far more than any AI
model needs — and decodes in ~0.5 s versus ~3 s for full resolution.

The `CLIPEmbedder` correctly pre-downsizes to 1280 px before the model. However, it
still receives a full-res image passed in via `_read_image`, allocating the full array
before downsizing.

---

### 3.4 BLIP-2 Serial VQA Loop (GPU Inefficiency) — HIGH

**Files:** `imganalyzer/analysis/ai/local.py:77–93`

```python
for question, key in [
    ("What type of scene is this?...", "scene_type"),
    ("What is the main subject?...",   "main_subject"),
    ("What is the lighting?...",       "lighting"),
    ("What is the mood?...",           "mood"),
]:
    with torch.inference_mode():
        inputs = processor(pil_img, question, return_tensors="pt").to(device)
        out = model.generate(...)
```

Each of the 4 VQA questions is a **separate forward pass** through the 8 GB BLIP-2
model. The image encoder (ViT-L) runs **5 times total** per image (1 captioning + 4 VQA).
The visual features (from the ViT encoder) are identical across all 5 passes — only
the text prompt differs. There is no mechanism to cache or reuse the visual features
between calls.

At 500K images this is ~2.5 million redundant ViT-L forward passes.

Additionally, `torch.cuda.empty_cache()` is called between every VQA question
(`local.py:92`), introducing unnecessary Python↔CUDA synchronization points.

---

### 3.5 Search Index Rebuilt Per Module Write (DB I/O) — MEDIUM-HIGH

**Files:** `imganalyzer/db/repository.py:550–676`, `imganalyzer/pipeline/modules.py:143`, 
`imganalyzer/pipeline/modules.py:179`

`update_search_index()` is called after each of `metadata`, `local_ai`, and `cloud_ai`
writes. Each call:
1. Reads back from 6 tables: `analysis_local_ai`, `analysis_blip2`, `analysis_faces`,
   `analysis_cloud_ai`, `analysis_metadata`.
2. Queries `find_face_by_alias()` which does a full `SELECT * FROM face_identities` scan
   (no index) for each detected face name (`repository.py:501`).
3. Deletes + re-inserts an FTS5 row.

For 500K images with 3 modules triggering rebuilds each:
- ~1.5 million FTS5 delete+insert cycles.
- ~1.5 million multi-table reads per image.
- Face alias lookup does full table scans on every call if unindexed.

---

### 3.6 Batch Ingest: SHA-256 Hashing Every File (I/O) — MEDIUM

**Files:** `imganalyzer/pipeline/batch.py:87`, `imganalyzer/pipeline/modules.py:23–32`

```python
file_hash = compute_file_hash(path) if compute_hash else None
```

`compute_file_hash` reads the entire file in 64 KB chunks and computes SHA-256. For
a 40 MB RAW file this means reading 40 MB per image just for deduplication during
ingest. At 500K images × 40 MB average = **20 TB of read I/O for hashing alone**.

This is especially costly on spinning disks or NAS-mounted network storage, which is
common for large photo libraries.

The hash is used only for deduplication. A cheaper fingerprint (inode + size + mtime)
would catch >99% of duplicates without reading file content.

---

### 3.7 GroundingDINO: float32 Only, No Batch Support (GPU) — MEDIUM

**Files:** `imganalyzer/analysis/ai/objects.py:145–149`

```python
cls._model = AutoModelForZeroShotObjectDetection.from_pretrained(
    model_id,
    torch_dtype=torch.float32,   # cannot use float16
    ...
)
```

The comment notes float16 causes dtype errors. However, float32 doubles VRAM usage
compared to bfloat16/float16. At 700 MB model size, this is ~350 MB extra VRAM that
could hold additional batch slots.

Additionally, `objects.py` processes images one at a time (no batch inference), though
GroundingDINO does support batch input in the `transformers` implementation.

---

### 3.8 InsightFace: Full-Resolution BGR Input (CPU + Memory) — MEDIUM

**Files:** `imganalyzer/analysis/ai/faces.py:44–46`

```python
rgb: np.ndarray = image_data["rgb_array"]
bgr = rgb[:, :, ::-1].copy()   # full-res BGR copy
```

InsightFace `app.get(bgr)` receives the full-resolution array regardless of image size.
For a 50 MP image (7952×5304), the detection network (`RetinaFace`) internally scales
to 640×640, but:
- The `.copy()` allocates another ~150 MB full-res array (RGB→BGR copy).
- All this data is passed to ONNX Runtime which allocates tensors and then immediately
  downscales to 640×640 at its first stage.

Pre-downscaling to ~1500 px (sufficient for face detection) before passing to InsightFace
would avoid the 150 MB intermediate copy and reduce ONNX tensor allocation.

---

### 3.9 OCR Tiling: Excessive Strips for Large Images (GPU) — MEDIUM

**Files:** `imganalyzer/analysis/ai/ocr.py:398–460`

Up to `_MAX_STRIPS = 40` tiles are processed per image in document mode. Each tile runs
`trocr-large-printed` with `num_beams=5`. For a full-page document at 4K resolution:

- Tiling overhead: up to 40 × beam-search runs.
- `_estimate_font_size()` (`ocr.py:246`) calls `cv2.Canny` on an 800-px downscale.
- `_crop_to_content()` per strip does grayscale array operations per strip.

For large volumes where most images do NOT contain text (typical photo library), the
`objects` pass already gates OCR via `has_text`. However, the tiling cost is still
paid for the subset that does have text, and the per-strip overhead is high.

---

### 3.10 Worker Default Parallelism: `workers=1` (CPU Underutilization) — MEDIUM

**Files:** `imganalyzer/pipeline/worker.py:94–99`

```python
def __init__(self, ..., workers: int = 1, cloud_workers: int = 4, ...):
    self.workers = max(1, workers)
```

The default for the local IO thread pool (`metadata`, `technical`) is 1 worker. These
modules are entirely CPU-bound (EXIF parsing, numpy operations) and could safely run
with `workers = min(cpu_count, 8)` without GPU interference. With the current default,
`technical` analysis is single-threaded even on 32-core machines.

At 500K images, `technical` analysis (sharpness, exposure, K-means) is a significant
fraction of total CPU time.

---

### 3.11 DB Connection Per Thread — Acceptable But Suboptimal at Scale — LOW-MEDIUM

**Files:** `imganalyzer/pipeline/worker.py:132–168`

Each worker thread opens its own SQLite connection with WAL mode. WAL mode is correct.
However, at high worker counts the WAL file can grow large before a checkpoint, and
`PRAGMA synchronous=NORMAL` means occasional write stalls. For 500K images with many
concurrent writers, a proper write-coalescing layer (or migration to PostgreSQL/DuckDB
for the analysis tables) would improve throughput.

---

## 4. Ranked Improvement Opportunities

The following opportunities are ranked by **expected throughput improvement per unit
of implementation complexity** at 500K+ scale.

---

### OPP-1: Per-Image Decode Cache Across Modules

**Impact:** Critical | **Complexity:** Medium | **Effort:** ~5 days

**Problem:** `_read_image()` is called once per module per image. 8 modules × 500K
images = 4 million decode calls. For RAW, each is ~2–4 s of CPU time.

**Solution:**
- Introduce an `ImageDecodeCache` (LRU, max 2–4 images by memory budget) keyed by
  `(path, mtime)`.
- In `ModuleRunner.run()`, check the cache before calling `_read_image()`. On cache
  miss, decode and insert.
- When all modules for an image complete (detectable via DB job count), evict the entry.
- For RAW: implement a two-tier decode — a "light" variant (`half_size=True`,
  ~0.4 s, sufficient for all AI modules and embeddings) and a "full" variant
  (needed only for `technical` analysis at its 3000 px downsample).

**Expected gain at 500K RAW images:**
- Before: ~4M rawpy decodes × ~3 s = ~3.3M CPU-seconds (~900 CPU-hours).
- After: ~500K decodes × ~3 s + ~500K half-size decodes × ~0.4 s = ~470K CPU-seconds.
- **~7× reduction in decode CPU time.**

**Implementation notes:**
- Cache lives in the `Worker` instance (not DB), shared across threads via lock.
- Eviction must be thread-safe; use `threading.Lock` around LRU operations.
- The `image_data` dict is currently passed in-memory between `_read_image()` and the
  module function — no DB schema change needed.
- Consider a shared memory buffer (Python `multiprocessing.shared_memory`) if workers
  scale beyond a single process in future.

**Files to modify:**
- `imganalyzer/pipeline/modules.py` — add cache to `ModuleRunner`
- `imganalyzer/readers/raw.py` — add `half_size` parameter
- `imganalyzer/pipeline/worker.py` — pass cache instance to `ModuleRunner`

---

### OPP-2: BLIP-2 Visual Feature Caching (Encode Once, Query N Times)

**Impact:** High | **Complexity:** Medium | **Effort:** ~3 days

**Problem:** BLIP-2's ViT-L encoder runs 5× per image (1 caption + 4 VQA). Visual
features are identical; only the text prompt differs.

**Solution:**
- After the captioning forward pass, extract and cache the image features
  (`model.vision_model()` output or the Q-Former output).
- For the 4 VQA passes, inject the cached features directly, bypassing the ViT-L encoder.
- This requires calling the model components at a lower level than `model.generate()`.
  The Blip2 model exposes `model.get_image_features()` and a two-step forward method.

**Expected gain at 500K images:**
- Before: 5 ViT-L forward passes × 500K = 2.5M passes.
- After: 1 ViT-L forward pass × 500K = 500K passes (Q-Former + decoder only for VQA).
- **~4× reduction in BLIP-2 GPU compute per image** (ViT-L is the dominant cost).

**Implementation notes:**
- Requires calling HuggingFace internals: `model.vision_model(pixel_values)` to get
  `image_embeds`, then using `model.language_projection`, `model.qformer`, and
  `model.language_model.generate()` with pre-computed vision conditioning.
- Add a validation check that cached features match the image being processed.
- Remove the intermediate `torch.cuda.empty_cache()` calls between VQA questions
  (they add sync points with no benefit since VRAM usage is steady).

**Files to modify:**
- `imganalyzer/analysis/ai/local.py` — refactor VQA loop to reuse visual features

---

### OPP-3: GPU Inference Batching for Objects, CLIP, and OCR

**Impact:** High | **Complexity:** Medium-High | **Effort:** ~4 days per module

**Problem:** All GPU modules process one image at a time, leaving the GPU idle during
Python overhead, data loading, and DB writes.

**Solution — Micro-batching in the worker:**
- Claim `batch_size=8–16` images from the queue per GPU module pass.
- Preprocess all images in the batch to tensors first (on CPU threads).
- Run a single batched forward pass.
- Write results back in parallel.

**Per-module feasibility:**

| Module        | Batching support           | VRAM cost per extra image | Recommended batch size |
|---------------|---------------------------|--------------------------|------------------------|
| GroundingDINO | Yes (transformers)        | ~0.5 GB                  | 4–8                    |
| CLIP          | Yes (open_clip)           | ~0.1 GB                  | 16–32                  |
| TrOCR         | Yes (already batch=4)     | ~0.3 GB per strip        | 8–12                   |
| BLIP-2        | Limited (KV-cache memory) | ~1.5 GB                  | 2–4                    |
| InsightFace   | No (ONNX serial)          | N/A                      | 1 (ONNX limitation)    |

**Expected gain:**
- GPU utilization: from ~25% to ~70–80% on GroundingDINO and CLIP passes.
- Wall-clock throughput: 2–4× on those modules.

**Implementation notes:**
- Worker phase 2 must claim a batch before dispatching GPU work.
- Need to split batch results back to per-image DB writes.
- VRAM budget must be checked before increasing batch size.

**Files to modify:**
- `imganalyzer/pipeline/worker.py` — batch claim and dispatch
- `imganalyzer/analysis/ai/objects.py` — accept list of images
- `imganalyzer/embeddings/clip_embedder.py` — accept list of images/paths
- `imganalyzer/analysis/ai/ocr.py` — outer batch already exists; expose to worker

---

### OPP-4: GPU Inference Pipelining (Prefetch + Overlap)

**Impact:** High | **Complexity:** High | **Effort:** ~1 week

**Problem:** GPU pipeline is fully serial — decode → GPU inference → DB write →
next image. GPU is idle during decode and DB write phases.

**Solution — 3-stage pipeline with double-buffering:**
```
Stage A (CPU thread): Decode image N+1, preprocess to tensor, pin to GPU memory
Stage B (GPU main):   Run model on image N
Stage C (CPU thread): Write results for image N-1 to DB
```

- A and C run on CPU threads (`ThreadPoolExecutor`).
- B runs on the GPU main thread.
- Use CUDA streams to overlap data transfer with computation.

**Expected gain:**
- GPU utilization: from ~25% to ~85%+.
- Overall pipeline throughput: 2–3× on GPU-bound modules.

**Implementation notes:**
- Requires refactoring `Worker._run_loop()` to use a queue-based pipeline
  rather than the current sequential dispatch.
- CUDA stream management: `torch.cuda.Stream` for async H2D transfers.
- Error propagation across pipeline stages needs careful handling.

**Files to modify:**
- `imganalyzer/pipeline/worker.py` — major refactor of processing loop
- All GPU module `analyze()` methods — accept pre-allocated tensor inputs

---

### OPP-5: Deferred / Batched Search Index Updates

**Impact:** Medium-High | **Complexity:** Low | **Effort:** ~1 day

**Problem:** `update_search_index()` is called after `metadata`, `local_ai`, and `cloud_ai`
writes. Each call reads from 6 DB tables, scans face identities, and does an FTS5
delete+insert. At 500K images this generates ~1.5M multi-table read + FTS5 write cycles.

**Solution:**
1. **Defer per-module updates:** Remove `update_search_index()` calls from individual
   module `_run_*` methods. Instead, call it once per image after all modules complete
   (end of `_write_pending_xmps` phase or at a post-processing step).
2. **Batch FTS5 inserts:** Collect index data for 1000 images and write in a single
   transaction (`BEGIN IMMEDIATE ... multiple inserts ... COMMIT`).
3. **Index the `face_identities` table:** Add an index on `canonical_name` and
   `display_name` to eliminate the full-table scan in `find_face_by_alias()`.

**Expected gain:**
- Reduce DB write load by ~2× (fewer redundant intermediate writes).
- FTS5 write throughput: 3–5× with batch inserts.
- Face lookup: O(log n) instead of O(n) per call.

**Files to modify:**
- `imganalyzer/db/repository.py` — add face identity indexes, add batch index method
- `imganalyzer/pipeline/modules.py` — remove mid-pipeline `update_search_index` calls
- `imganalyzer/pipeline/worker.py` — add final index rebuild step

---

### OPP-6: Fast Ingest Fingerprinting (Skip SHA-256 for Known Files)

**Impact:** Medium | **Complexity:** Low | **Effort:** ~0.5 days

**Problem:** Every ingest run reads every file in full for SHA-256 hashing. At 500K
images × 40 MB average = 20 TB of read I/O, even on re-runs where files haven't changed.

**Solution:**
- Use `(file_path, file_size, mtime)` as the primary deduplication key.
- Only fall back to full SHA-256 when two files share the same path+size+mtime.
- Add `mtime` column to the `images` table (migration).
- On re-ingest: if path + size + mtime match an existing row, skip hashing and
  skip registration entirely.
- Expose `--hash` flag to force full SHA-256 for archival deduplication use cases.

**Expected gain:**
- Re-ingest of a 500K unchanged library: IO reduced from ~20 TB to ~0 (only stat() calls).
- First ingest: optional, falls back to current behavior.

**Files to modify:**
- `imganalyzer/pipeline/batch.py` — change fingerprint logic
- `imganalyzer/pipeline/modules.py` — update `compute_file_hash` call site
- DB schema migration: add `mtime` to `images` table

---

### OPP-7: Pre-downscale Before AI Modules (Avoid Full-Res Array in GPU Pipeline)

**Impact:** Medium | **Complexity:** Low | **Effort:** ~1 day

**Problem:** Even though AI models resize internally, the full-resolution array is
decoded and allocated before passing to each module, wasting ~150 MB per RAW image
in the Python heap.

**Solution:**
- In `_read_image()`, add a `max_px` parameter. For AI-only modules (objects, blip2,
  ocr, faces, embedding), decode at `max_px=2048` (or use `half_size=True` for RAW).
- Only `technical` analysis needs the higher-resolution array (it downsizes to 3000 px).
- The module runner can select the appropriate resolution tier based on the module name.

**Resolution tiers:**

| Module         | Required resolution        | Recommended max_px |
|----------------|----------------------------|--------------------|
| metadata       | Not needed (EXIF only)     | None (no decode)   |
| technical      | High detail                | 3000 px long edge  |
| objects        | 800 px (internal limit)    | 1024 px            |
| blip2          | 224 px (internal)          | 512 px             |
| ocr            | 1920 px (already limited)  | 1920 px            |
| faces          | 640 px (det_size)          | 1280 px            |
| embedding      | 1280 px (pre-truncated)    | 1280 px            |
| cloud_ai       | 1568 px (API limit)        | 1568 px            |

**Special case — metadata:** EXIF is read directly from the file headers; the full
decoded array is not needed at all. `_run_metadata()` should be refactored to
skip `_read_image()` for the pixel array and use header-only reads.

**Expected gain:**
- Peak memory per RAW image: ~150 MB → ~8–40 MB depending on module.
- Faster Python GC cycles; less pressure on allocator.
- Improved cache utilization (smaller arrays fit in L3 cache).

**Files to modify:**
- `imganalyzer/readers/raw.py` — add `max_dim` / `half_size` parameters
- `imganalyzer/readers/standard.py` — add `max_dim` parameter
- `imganalyzer/pipeline/modules.py` — pass resolution tier per module

---

### OPP-8: Metadata Module — Skip Pixel Decode Entirely

**Impact:** Medium | **Complexity:** Low | **Effort:** ~0.5 days

**Problem:** `_run_metadata()` calls `_read_image()` which decodes the full pixel array,
but `MetadataExtractor` only needs EXIF data from the file headers (not pixel data).
The pixel array width/height are later used to `update_image(width=..., height=...)`,
but these can be read from EXIF tags without decoding pixels.

**Solution:**
- Create a `read_headers_only(path)` function that:
  - Opens the file without loading pixel data (Pillow `Image.open()` with lazy loading).
  - Reads EXIF via `exifread` from the raw file bytes.
  - Returns `{width, height, format, exif_bytes, is_raw}` without `rgb_array`.
- `_run_metadata()` uses `read_headers_only()`.
- This saves ~2–4 s decode time × 500K images = ~300–600 CPU-hours for the metadata pass alone.

**Files to modify:**
- `imganalyzer/readers/standard.py` — add header-only path
- `imganalyzer/readers/raw.py` — add header-only path (rawpy allows reading metadata without postprocess)
- `imganalyzer/pipeline/modules.py:126–147` — use header-only reader for metadata module

---

### OPP-9: Increase Default Worker Parallelism for CPU-Bound Modules

**Impact:** Medium | **Complexity:** Trivial | **Effort:** ~0.5 days

**Problem:** `workers=1` default means metadata and technical analysis run single-threaded,
even on 32-core machines.

**Solution:**
- Auto-detect CPU count: `workers = min(os.cpu_count() or 4, 8)` as default.
- Cap at 8 to avoid SQLite connection contention (each thread gets its own connection).
- Expose `--workers` CLI flag (may already exist; verify in `cli.py`).

**Expected gain:**
- Technical analysis (numpy K-means, sharpness) on 500K images: linear speedup to min(cores, 8).
- On a 16-core machine: ~7× speedup for the `technical` module.

**Files to modify:**
- `imganalyzer/pipeline/worker.py:120` — change default
- `imganalyzer/cli.py` — expose `--workers` flag if not already present

---

### OPP-10: GroundingDINO bfloat16 / Quantization

**Impact:** Low-Medium | **Complexity:** Medium | **Effort:** ~2 days

**Problem:** GroundingDINO is loaded in float32 only due to dtype errors in the text
enhancer layers with float16.

**Solution:**
- Test loading only the visual backbone in float16/bfloat16 while keeping the text
  encoder in float32 (mixed precision at the module level).
- Alternatively, apply `torch.compile()` (PyTorch 2.x) to the model for automatic
  kernel fusion and reduced memory bandwidth.
- Evaluate `INT8` post-training quantization (via `torch.quantization`) for the
  visual backbone, which has well-understood numerical properties.

**Expected gain:**
- VRAM: ~350 MB reduction (float32 → bfloat16 for the visual backbone).
- Throughput: 10–30% faster inference due to reduced memory bandwidth.

**Files to modify:**
- `imganalyzer/analysis/ai/objects.py:145–149` — experiment with bfloat16 and `torch.compile`

---

## 5. Cumulative Impact Estimate (500K Images, Single GPU)

Assumptions:
- Mix: 40% RAW (20 MP avg), 50% JPEG (10 MP avg), 10% HEIC
- All 8 split-pass modules enabled
- Single 24 GB GPU, 16-core CPU, fast NVMe storage

| Opportunity | Metric affected               | Estimated gain |
|-------------|-------------------------------|----------------|
| OPP-1       | Decode CPU time               | ~7× reduction  |
| OPP-2       | BLIP-2 GPU time               | ~4× reduction  |
| OPP-3       | GPU inference throughput       | 2–4× increase  |
| OPP-4       | End-to-end pipeline wall time | 2–3× increase  |
| OPP-5       | DB write throughput           | 3–5× increase  |
| OPP-6       | Ingest IO (re-runs)           | ~100× reduction|
| OPP-7       | Peak memory per image         | ~4–10× reduction|
| OPP-8       | Metadata pass CPU time        | eliminates decode|
| OPP-9       | Technical module CPU time     | ~7× (16-core)  |
| OPP-10      | GroundingDINO VRAM            | ~350 MB freed  |

Implementing OPP-1 + OPP-2 + OPP-3 alone is estimated to reduce total end-to-end
processing time for 500K mixed images from ~500 hours (current, estimated) to
**~70–100 hours** — a ~5–7× overall improvement before OPP-4 pipelining.

With OPP-4 added: **~25–40 hours**, a ~12–20× overall improvement.

---

## 6. Measurement Plan

Before implementing any optimization, establish baselines:

### 6.1 Per-Module Timing

The pipeline already emits `[RESULT]` JSON lines with `ms` per module:

```json
[RESULT] {"path": "...", "module": "objects", "status": "done", "ms": 1450}
```

**Action:** Aggregate these across a 1000-image test batch. Compute:
- Mean, p50, p95, p99 per module.
- Ratio of decode time to inference time (add timing inside `_read_image()`).

### 6.2 GPU Utilization

```bash
# During a batch run:
nvidia-smi dmon -s u -d 1 -f gpu_util.csv
```

Target: confirm GPU utilization is ~20–30% currently (expected due to serialization).
Post-OPP-4 target: >75%.

### 6.3 Memory Profiling

```bash
# Add to _read_image() temporarily:
import tracemalloc
tracemalloc.start()
# ... decode ...
current, peak = tracemalloc.get_traced_memory()
```

Measure peak per-module memory consumption to validate OPP-7 estimates.

### 6.4 Ingest I/O Profiling

```bash
# On Linux/WSL:
iostat -x 1 during ingest run
```

Measure MB/s read during `compute_file_hash()` vs. during analysis to quantify OPP-6.

### 6.5 DB Write Throughput

Enable SQLite query timing (`PRAGMA query_log`) or wrap transactions with wall-clock
timing to measure `update_search_index()` cost per call.

---

## 7. Recommended Implementation Order

Given the complexity/impact matrix, the recommended rollout order is:

```
Phase A (1–2 weeks, low risk, high gain):
  OPP-9  — Worker default parallelism           (trivial)
  OPP-8  — Metadata header-only read            (low risk)
  OPP-6  — Fast ingest fingerprinting            (low risk)
  OPP-5  — Deferred search index updates         (low risk)

Phase B (2–4 weeks, medium risk, high gain):
  OPP-7  — Pre-downscale before AI modules       (moderate)
  OPP-1  — Per-image decode cache                (moderate, high payoff)

Phase C (4–8 weeks, medium-high risk, very high gain):
  OPP-2  — BLIP-2 visual feature caching         (requires HF internals)
  OPP-3  — GPU batch inference (CLIP first)      (per-module work)

Phase D (8+ weeks, high risk, very high gain):
  OPP-4  — Full GPU pipeline pipelining          (major refactor)
  OPP-10 — GroundingDINO quantization            (requires validation)
```

Each phase should be validated against the measurement plan before proceeding to the next.

---

## 8. File Reference Index

| File | Relevant Sections |
|------|-------------------|
| `imganalyzer/pipeline/worker.py` | OPP-3, OPP-4, OPP-9 |
| `imganalyzer/pipeline/batch.py` | OPP-6 |
| `imganalyzer/pipeline/modules.py` | OPP-1, OPP-7, OPP-8 |
| `imganalyzer/readers/raw.py` | OPP-1, OPP-7, OPP-8 |
| `imganalyzer/readers/standard.py` | OPP-7, OPP-8 |
| `imganalyzer/analysis/metadata.py` | OPP-8 |
| `imganalyzer/analysis/technical.py` | OPP-7, OPP-9 |
| `imganalyzer/analysis/ai/local.py` | OPP-2 |
| `imganalyzer/analysis/ai/objects.py` | OPP-3, OPP-7, OPP-10 |
| `imganalyzer/analysis/ai/ocr.py` | OPP-3, OPP-7 |
| `imganalyzer/analysis/ai/faces.py` | OPP-3, OPP-7 |
| `imganalyzer/analysis/ai/cloud.py` | OPP-7 |
| `imganalyzer/embeddings/clip_embedder.py` | OPP-3, OPP-7 |
| `imganalyzer/db/repository.py` | OPP-5 |

---

*Generated by static codebase analysis — Codex 5.2 — no code changes made.*
