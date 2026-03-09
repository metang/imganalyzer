# imganalyzer — Architecture

## Overview

imganalyzer is a two-component system:

1. **`imganalyzer/`** — A Python package providing both a Typer CLI and a JSON-RPC 2.0 stdio server. Reads images, runs local and cloud AI models, performs technical quality analysis, manages a SQLite database of results, and writes Adobe Lightroom-compatible XMP sidecar files.
2. **`imganalyzer-app/`** — An Electron 31 + React 18 desktop GUI with five tabs (Gallery, Batch, Running, Search, Faces). Communicates with the Python backend via a single persistent child process using JSON-RPC over stdin/stdout.

The GUI spawns `python -m imganalyzer.server` once at startup. All commands (thumbnails, analysis, search, face management) go through this persistent connection — there is no per-call subprocess overhead.

---

## Repository Layout

```
imganalyzer/
├── imganalyzer/                       # Python package
│   ├── __init__.py                    # Version string
│   ├── cli.py                         # Typer CLI (20+ commands)
│   ├── server.py                      # JSON-RPC 2.0 stdio server (25+ methods)
│   ├── analyzer.py                    # Orchestrator: AnalysisResult dataclass + pipeline dispatch
│   ├── readers/
│   │   ├── raw.py                     # RAW decoder via rawpy (LibRaw), header-only mode
│   │   └── standard.py               # Standard formats via Pillow, header-only mode
│   ├── analysis/
│   │   ├── metadata.py                # EXIF/IPTC extraction + reverse geocoding
│   │   ├── technical.py               # Sharpness, exposure, noise, color (NumPy/OpenCV)
│   │   └── ai/
│   │       ├── local.py               # BLIP-2 captioning + VQA (batched)
│   │       ├── local_full.py          # 4-stage local AI orchestrator (legacy single-image mode)
│   │       ├── objects.py             # GroundingDINO object detection (via transformers, batched)
│   │       ├── ocr.py                 # TrOCR text recognition with document tiling
│   │       ├── faces.py               # InsightFace detection + recognition (buffalo_l)
│   │       ├── face_db.py             # Legacy JSON-file face database
│   │       └── cloud.py               # OpenAI / Anthropic / Google / Copilot backends
│   ├── db/
│   │   ├── connection.py              # SQLite singleton, WAL mode, thread-safe
│   │   ├── schema.py                  # 16 migrations, 13+ tables, FTS5
│   │   ├── repository.py             # CRUD, overrides, face management, FTS5 index
│   │   ├── queue.py                   # Job queue with atomic claim + priority ordering
│   │   └── search.py                  # Hybrid FTS5 + CLIP search with RRF scoring
│   ├── pipeline/
│   │   ├── batch.py                   # Folder scanning, image registration, job enqueue
│   │   ├── worker.py                  # Three-phase batch worker, GPU model management
│   │   ├── scheduler.py              # Three-phase GPU resource scheduler with VRAM budget
│   │   ├── vram_budget.py            # GPU memory tracking (CUDA + MPS)
│   │   ├── distributed_worker.py     # Remote worker for distributed batch processing
│   │   ├── modules.py                 # Per-module dispatch, override guard, XMP from DB
│   │   └── passes/                    # Individual GPU pass modules
│   │       ├── blip2.py               # BLIP-2 batch pass
│   │       ├── objects.py             # GroundingDINO batch pass
│   │       ├── ocr.py                 # TrOCR batch pass
│   │       └── faces.py               # InsightFace batch pass
│   ├── embeddings/
│   │   └── clip_embedder.py           # OpenCLIP ViT-L/14, 768-d vectors, batched encoding
│   └── output/
│       └── xmp.py                     # XMP sidecar writer (10 namespaces)
│
├── imganalyzer-app/                   # Electron desktop GUI
│   ├── src/
│   │   ├── main/                      # Electron main process (Node.js)
│   │   │   ├── index.ts               # App bootstrap, IPC registration, quit cleanup
│   │   │   ├── python-rpc.ts          # Persistent JSON-RPC 2.0 client over stdin/stdout
│   │   │   ├── images.ts              # Thumbnail (LRU 1000) + full-res (LRU 2) via RPC
│   │   │   ├── analyzer.ts            # Single-image analysis via RPC + progress stage map
│   │   │   ├── batch.ts               # Batch IPC handlers (ingest/start/pause/resume/stop)
│   │   │   ├── coordinator.ts         # HTTP coordinator lifecycle management
│   │   │   ├── search.ts              # Search IPC handler + filter types
│   │   │   ├── faces.ts               # Face management IPC (list/cluster/crop/alias)
│   │   │   ├── xmp.ts                 # XMP sidecar parser (fast-xml-parser)
│   │   │   └── copilot-analyzer.ts    # Cloud AI via GitHub Copilot SDK (gpt-4.1)
│   │   ├── preload/
│   │   │   └── index.ts               # contextBridge: window.api for all IPC methods
│   │   └── renderer/                  # React UI (Vite + Tailwind CSS)
│   │       ├── App.tsx                # Root: 5-tab layout, batch state orchestration
│   │       ├── global.d.ts            # Shared types + Window.api declaration
│   │       ├── components/
│   │       │   ├── FolderPicker.tsx    # Header bar with folder open button
│   │       │   ├── Gallery.tsx         # Responsive image grid
│   │       │   ├── Thumbnail.tsx       # Lazy-load thumbnail via IPC, XMP/RAW badges
│   │       │   ├── Lightbox.tsx        # Full-screen viewer: zoom/pan, blur placeholder
│   │       │   ├── Sidebar.tsx         # Right panel: local analysis results
│   │       │   ├── CloudSidebar.tsx    # Left panel: Copilot cloud AI results
│   │       │   ├── BatchView.tsx       # BatchConfigView + BatchRunView
│   │       │   ├── PassSelector.tsx    # 9 analysis pass checkboxes + worker config
│   │       │   ├── ProgressDashboard.tsx  # Per-module progress bars, stats, controls
│   │       │   ├── LiveResultsFeed.tsx    # Scrollable feed, last 200 results
│   │       │   ├── ConfirmStopDialog.tsx  # Modal requiring "STOP" to confirm
│   │       │   ├── SearchView.tsx      # Search layout: sidebar + results grid
│   │       │   ├── SearchBar.tsx       # Query input + filter fields + mode selector
│   │       │   ├── VirtualGrid.tsx     # Virtualized grid (ResizeObserver + IntersectionObserver)
│   │       │   ├── SearchLightbox.tsx  # Lightbox for search results with analysis sidebar
│   │       │   └── FacesView.tsx       # Face clusters + alias editing
│   │       └── hooks/
│   │           ├── useAnalysis.ts      # Single-image: XMP cache → auto-analyze → progress
│   │           ├── useCloudAnalysis.ts # Cloud AI via Copilot SDK, manual trigger
│   │           └── useBatchProcess.ts  # Batch lifecycle + event subscriptions
│   ├── electron.vite.config.ts        # Active build config with Copilot SDK externals
│   ├── tailwind.config.js
│   └── package.json
│
├── tests/                             # Python unit tests (pytest, 165 tests across 4 files)
├── scripts/                           # Diagnostic utilities + worker setup
├── pyproject.toml                     # Python project config + dependencies
├── ARCHITECTURE.md                    # This file
├── AGENTS.md                          # Agent coding rules
└── spec.md                            # Product specification
```

---

## Python Backend

### Entry Points

**CLI** (`cli.py` via Typer):
```
# Core analysis
imganalyzer analyze <image(s)> --ai local --overwrite --verbose
imganalyzer info <image> --format table|json|yaml

# Batch pipeline
imganalyzer ingest <folder> --modules metadata,technical,objects,blip2,...
imganalyzer run --workers 1 --cloud-workers 4 --cloud-provider copilot
imganalyzer status
imganalyzer queue-clear --status failed|pending|all
imganalyzer rebuild --module objects

# Search
imganalyzer search "sunset over mountains"
imganalyzer search-json "score>7 has:faces" --mode hybrid --limit 50

# Face management
imganalyzer register-face NAME image.jpg
imganalyzer list-faces
imganalyzer remove-face NAME
imganalyzer alias-face NAME "Display Name"
imganalyzer rename-face OLD_NAME NEW_NAME
imganalyzer merge-face SOURCE_NAME TARGET_NAME

# Utilities
imganalyzer override <image> <field> <value>
imganalyzer purge-missing

# Distributed processing
imganalyzer run-distributed-worker --coordinator http://host:8787

# Copilot session management
imganalyzer cleanup-sessions --dry-run

# Performance profiling
imganalyzer profile-report
```

**JSON-RPC Server** (`server.py`):
Spawned by Electron as `python -m imganalyzer.server`. Reads JSON-RPC requests from stdin, writes responses/notifications to stdout. All other output goes to stderr.

| Method | Description |
|---|---|
| `status` | Queue stats (pending/running/done/failed per module) |
| `ingest` | Scan folder, register images, enqueue jobs |
| `run` | Start batch workers (GPU + cloud threads) |
| `cancel_run` | Stop batch workers, release jobs |
| `analyze` | Single-image analysis (all modules) |
| `cancel_analyze` | Cancel in-progress single analysis |
| `thumbnail` | Generate 400x300 JPEG thumbnail |
| `fullimage` | Decode full-res image (RAW/HEIC via Python, standard via direct read) |
| `search` | Text/semantic/hybrid search with filters |
| `queue_clear` | Clear jobs by status |
| `rebuild` | Re-enqueue failed jobs by module |
| `faces/list` | List registered face identities |
| `faces/images` | Get images containing a face identity |
| `faces/set-alias` | Set display name for a face |
| `faces/clusters` | List face clusters |
| `faces/cluster-images` | Get face occurrences in a cluster |
| `faces/crop` | Get cropped face thumbnail (base64 JPEG) |
| `faces/run-clustering` | Run agglomerative clustering |
| `shutdown` | Graceful server shutdown |
| `workers/register` | Register a distributed worker |
| `workers/heartbeat` | Refresh worker heartbeat, recover expired leases |
| `workers/list` | List connected workers |
| `jobs/claim` | Atomic lease of pending jobs with TTL |
| `jobs/complete` | Mark leased job as done |
| `jobs/fail` | Mark leased job as failed |
| `jobs/release-expired` | Requeue expired leases |

**Notifications** (server → client, unsolicited):
| Notification | Payload |
|---|---|
| `run/result` | Per-image completion during batch |
| `run/done` | Batch pass completed |
| `run/error` | Batch error |
| `ingest/progress` | Folder scan progress |
| `analyze/progress` | Single-image analysis progress |

### Database Layer (`db/`)

SQLite with WAL mode at `~/.cache/imganalyzer/imganalyzer.db`. Schema managed by 16 sequential migrations.

**Core tables:**
| Table | Purpose |
|---|---|
| `images` | Registered images (path, fingerprint, dimensions, file size) |
| `analysis_metadata` | EXIF/IPTC extraction results |
| `analysis_technical` | Technical quality scores |
| `analysis_blip2` | BLIP-2 captioning results |
| `analysis_objects` | GroundingDINO detection results + `has_person`/`has_text` flags |
| `analysis_ocr` | TrOCR text recognition results |
| `analysis_faces` | InsightFace detection + recognition results |
| `analysis_local_ai` | Legacy combined local AI results |
| `analysis_cloud_ai` | Cloud AI results (description, keywords, mood, species) |
| `analysis_aesthetic` | Aesthetic scores (1-10, from cloud AI) |
| `overrides` | User manual overrides (protected from re-analysis) |
| `job_queue` | Batch processing job queue (UNIQUE per image+module) |
| `embeddings` | CLIP 768-d float32 vectors (image + description) |
| `search_index` | FTS5 virtual table for full-text search |
| `face_identities` | Named face identities |
| `face_embeddings` | Per-identity reference embeddings |
| `face_occurrences` | Per-image face detections with bounding boxes |
| `face_aliases` | Display names for face identities |

**Key patterns:**
- `get_db()` returns a thread-local singleton — safe for the main thread only
- Background threads must create their own connections with `check_same_thread=False`
- Override mask: fields in `overrides` table are excluded from `UPDATE` statements during re-analysis
- Negative cache for override lookups avoids millions of empty SELECTs at scale
- Column-name filtering in queries survives schema drift across migrations

### Job Queue (`db/queue.py`)

Priority-ordered queue with atomic claim (SELECT + UPDATE in a transaction). Jobs have states: `pending` → `running` → `done` | `failed`. UNIQUE(image_id, module) prevents duplicate jobs.

**Module priorities** (higher = processed first):
```
metadata(100) > technical(90) > objects(85) > blip2(80) > local_ai(80)
  > ocr(78) > faces(77) > cloud_ai(70) > aesthetic(60) > embedding(50)
```

**Prerequisites:**
- `cloud_ai`, `aesthetic`, `ocr`, and `faces` all depend on `objects` completing first
- `objects` determines `has_person` (privacy gate for cloud/aesthetic) and `has_text` (gate for OCR)

**Job deferral:** When a job's prerequisite isn't met, it is deferred via `defer()` which bumps `queued_at` forward by 30 seconds. This prevents the same ineligible jobs from being repeatedly claimed and starving eligible jobs further back in the queue. Contrast with `mark_pending()` which preserves the original `queued_at` (used for shutdown recovery).

### Batch Pipeline (`pipeline/`)

**Ingest** (`batch.py`):
- Scans folder (optionally recursive) for supported image files
- Registers images in `images` table with batched transactions (500 per COMMIT)
- Pre-fetches existing paths in bulk to skip duplicates
- Enqueues jobs per selected module, respecting priorities
- File fingerprinting uses path+size+mtime (not SHA-256) for speed

**Worker** (`worker.py`):
Three-phase GPU processing loop designed for GPU memory efficiency:

```
Phase 0: Drain ALL 'objects' jobs (GroundingDINO, batch=4)
  └─ Sets has_person / has_text flags, unlocking privacy gates
  └─ Unload GroundingDINO from GPU

Phase 1: BLIP-2 captioning (exclusive GPU, batch=1)
  └─ Requires sole GPU access (~6 GB VRAM)
  └─ Unload BLIP-2

Phase 2: Co-resident GPU models (~2.75 GB total)
  ├─ faces (batch=8)     → InsightFace (only if has_person)
  ├─ ocr (batch=4)       → TrOCR (only if has_text)
  └─ embedding (batch=16)→ CLIP ViT-L/14

[Concurrent throughout] cloud_ai + aesthetic (ThreadPoolExecutor)
  └─ Skipped for images with has_person=True (privacy)
  └─ Cloud futures capped at 2× cloud_workers to prevent phase starvation

IO drain: After all GPU phases, remaining cloud/IO jobs drain with boosted thread pool
```

**VRAM Budget** (`vram_budget.py`):
Thread-safe GPU memory tracker. Auto-detects CUDA or MPS GPU, applies 70% VRAM cap. Tracks per-module reservations and supports exclusive modules (blip2 cannot share GPU). Falls back to physical RAM for MPS (Apple Silicon).

**Resource Scheduler** (`scheduler.py`):
Orchestrates the three GPU phases with VRAM-aware model loading. Single-module phases run on the main thread; multi-module phases (Phase 2) run each module in its own thread with image prefetching to hide disk I/O. Cancels uncompleted cloud futures at phase boundaries to prevent blocking.

Peak VRAM: ~6 GB during BLIP-2 phase; ~2.75 GB during Phase 2 (models loaded/unloaded between phases).

**Module Runner** (`modules.py`):
- Dispatches to per-module analysis functions
- Override guard: skips fields that have user overrides
- Single-entry image decode cache: one decoded image kept in memory, avoiding redundant reads
- `_pre_resize()`: shrinks images to 1920px long edge once before all AI modules
- `write_xmp_from_db()`: reconstructs AnalysisResult from all DB tables, writes XMP sidecar
- Deferred FTS5 index rebuild + XMP flush: every 60 seconds, not per-job

### Analysis Pipeline

**Metadata** (`analysis/metadata.py`):
EXIF/IPTC extraction via exifread + piexif. GPS coordinates with reverse geocoding to city/state/country.

**Technical** (`analysis/technical.py`):
All metrics computed from raw pixels (NumPy/OpenCV):

| Field | Method |
|---|---|
| Sharpness | Laplacian variance with saliency-weighted patch selection |
| Exposure EV | Log2(mean luminance / mid-grey) |
| Noise level | Median absolute deviation of high-frequency residual |
| SNR (dB) | 20 * log10(signal / noise) |
| Dynamic range | Log2(99th pct / 1st pct) |
| Highlight/shadow clipping | Pixels above 252 / below 3 |
| Average saturation | Mean HSV saturation |
| Warm/cool ratio | (R - B) / (R + G + B) mean |
| Dominant colors | K-means (k=5) centroids as hex |

**Local AI** (`analysis/ai/`):

| Module | Model | Library | Batch Size |
|---|---|---|---|
| Captioning | BLIP-2 (flan-t5-xl) | `transformers` | 1 |
| Object detection | GroundingDINO (grounding-dino-base) | `transformers` (`AutoModelForZeroShotObjectDetection`) | 4 |
| OCR | TrOCR (trocr-large-printed) | `transformers` | 4 |
| Face detection | buffalo_l (RetinaFace + ArcFace) | `insightface` | 8 |
| Embeddings | OpenCLIP ViT-L/14 | `open_clip_torch` | 16 |

All models are lazy-loaded singletons — weights cached in `~/.cache/huggingface/` and `~/.insightface/`.

OCR uses document tiling with ink detection, adaptive strip sizing, content cropping, and square padding. Gated on `has_text` from GroundingDINO.

Face recognition compares detected embeddings against registered identities using cosine similarity (default threshold 0.40). Greedy agglomerative clustering groups unknown faces (threshold 0.55).

**Cloud AI** (`analysis/ai/cloud.py`):
Backends: OpenAI GPT-4o, Anthropic Claude 3.5 Sonnet, Google Vision, GitHub Copilot (gpt-4.1). Cloud AI and aesthetic scoring are combined into a single API call. Privacy guard excludes images with detected people.

### Search Engine (`db/search.py`)

Hybrid search combining three strategies via Reciprocal Rank Fusion (RRF):

1. **FTS5 text search**: Full-text search over descriptions, keywords, objects, faces, OCR text, camera/lens model, location
2. **CLIP semantic search (image embeddings)**: Vectorized cosine similarity between query text embedding and stored image embeddings. Tier 1 uses image CLIP with optional description CLIP boost (weight 0.25). Tier 2 is description-only fallback with z-score gating.
3. **Hybrid**: RRF fusion of FTS5 + CLIP scores

Vectorized scoring via numpy BLAS (`matrix @ query_vec`) achieves sub-10ms at 500K images. Cached embedding matrix with row-count staleness detection. Description quality gating at 100 chars filters noisy BLIP-2 captions.

### XMP Output (`output/xmp.py`)

Writes `<rdf:Description>` with namespaces:

| Prefix | Namespace | Usage |
|---|---|---|
| `dc:` | Dublin Core | Caption, keywords |
| `xmp:` | XMP Basic | CreateDate, CreatorTool |
| `tiff:` | TIFF/EP | Camera make/model, dimensions |
| `exif:` | EXIF | FNumber, shutter, ISO, GPS |
| `Iptc4xmpCore:` | IPTC | Location fields |
| `photoshop:` | Photoshop | Lens model |
| `crs:` | Camera Raw | Lightroom profile hints |
| `imganalyzer:` | Custom (`http://ns.imganalyzer.io/1.0/`) | All analysis scores, AI results, OCR, faces |

Importable by Adobe Lightroom Classic, Lightroom CC, Bridge, and Camera Raw without plugins.

---

## Electron GUI

### Process Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Main Process (Node.js)                                          │
│  src/main/index.ts                                               │
│  • Creates BrowserWindow                                         │
│  • Registers ipcMain.handle() for all API calls                  │
│  • Manages persistent Python JSON-RPC server (python-rpc.ts)     │
│  • Handles batch lifecycle, search, face management              │
│  • Copilot SDK cloud analysis (copilot-analyzer.ts)              │
└───────────────┬──────────────────────────────────────────────────┘
                │ contextBridge (IPC)
┌───────────────▼──────────────────────────────────────────────────┐
│  Preload Script                                                  │
│  src/preload/index.ts                                            │
│  • Bridges Node IPC to window.api                                │
│  • Only surface exposed to renderer                              │
└───────────────┬──────────────────────────────────────────────────┘
                │ window.api
┌───────────────▼──────────────────────────────────────────────────┐
│  Renderer Process (React, sandboxed)                             │
│  src/renderer/                                                   │
│  • No Node.js access                                             │
│  • All I/O via window.api calls                                  │
│  • 5 tabs: Gallery, Batch, Running, Search, Faces                │
└──────────────────────────────────────────────────────────────────┘
                │
                │ JSON-RPC 2.0 over stdin/stdout
┌───────────────▼──────────────────────────────────────────────────┐
│  Python Server (persistent child process)                        │
│  python -m imganalyzer.server                                    │
│  • Spawned once at app start                                     │
│  • All image/analysis/search/face operations via RPC             │
│  • stdout reserved for JSON-RPC; logging to stderr               │
│  • Killed on app quit                                            │
└──────────────────────────────────────────────────────────────────┘
```

### IPC API (`window.api`)

**Invoke channels** (renderer → main → response):

| Method | IPC Channel | Description |
|---|---|---|
| `openFolder()` | `dialog:openFolder` | Native folder picker |
| `listImages(path)` | `fs:listImages` | Returns `ImageFile[]` for a directory |
| `getThumbnail(path)` | `fs:getThumbnail` | 400x300 JPEG via RPC, LRU cached (1000) |
| `getFullImage(path)` | `fs:getFullImage` | Full-res via RPC (RAW/HEIC) or direct read, LRU (2) |
| `galleryListFolders()` | `gallery:list-folders` | Folder tree nodes (processed DB images only) |
| `galleryListImagesChunk(params)` | `gallery:list-images-chunk` | Progressive gallery chunk with cursor |
| `getThumbnailCacheConfig()` | `cache:thumbnail:getConfig` | Read thumbnail cache dir/size config |
| `setThumbnailCacheConfig(config)` | `cache:thumbnail:setConfig` | Update thumbnail cache dir/size config |
| `readXmp(path)` | `fs:readXmp` | Parses XMP sidecar → `XmpData` or null |
| `runAnalysis(path, backend)` | `analyze:run` | Single-image analysis via RPC (5min timeout) |
| `cancelAnalysis(path)` | `analyze:cancel` | Cancel single-image analysis |
| `runCopilotAnalysis(path)` | `analyze:copilot` | Cloud AI via GitHub Copilot SDK |
| `batchIngest(folder, opts)` | `batch:ingest` | Scan folder, register, enqueue jobs |
| `batchStart(config)` | `batch:start` | Start batch workers |
| `batchPause()` | `batch:pause` | Cancel run, preserve queue |
| `batchResume()` | `batch:resume` | Re-start with saved config |
| `batchStop()` | `batch:stop` | Cancel + clear pending/running jobs |
| `batchCheckPending()` | `batch:check-pending` | Crash recovery: check for orphaned jobs |
| `batchResumePending()` | `batch:resume-pending` | Resume orphaned jobs |
| `batchRetryFailed(module)` | `batch:retry-failed` | Re-enqueue failed jobs |
| `batchQueueClearAll()` | `batch:queue-clear-all` | Clear all jobs |
| `searchRun(query, filters)` | `search:run` | Search with filters → SearchResponse |
| `facesList()` | `faces:list` | List face identities |
| `facesImages(name)` | `faces:images` | Images for a face identity |
| `facesSetAlias(name, alias)` | `faces:setAlias` | Set display name |
| `facesClusters()` | `faces:clusters` | List face clusters |
| `facesClusterImages(id)` | `faces:clusterImages` | Occurrences in a cluster |
| `facesCrop(imageId, idx)` | `faces:crop` | Cropped face thumbnail (base64) |
| `facesRunClustering()` | `faces:runClustering` | Run clustering algorithm |

**Event channels** (main → renderer):

| Channel | Description |
|---|---|
| `analyze:progress` | Progress events during single-image analysis |
| `batch:tick` | Polled queue stats every 1s (BatchStats) |
| `batch:result` | Per-job completion notification (BatchResult) |
| `batch:ingest-line` | Raw ingest stdout lines |
| `batch:ingest-progress` | Structured ingest progress |

### React Component Tree

```
App (5-tab layout)
├── TabButton (Gallery / Batch / Running / Search / Faces)
│
├── [Gallery tab]
│   └── DbGalleryView
│       ├── FolderSidebar (folder tree + cache settings)
│       ├── VirtualGrid (processed DB images, progressive chunk loading)
│       └── SearchLightbox (zoom/pan + analysis sidebar)
│
├── [Batch tab]
│   └── BatchConfigView
│       ├── ConfigPanel (folder, PassSelector, start button)
│       │   └── PassSelector (9 module checkboxes, workers, cloud provider)
│       └── IngestPanel (progress, counters)
│
├── [Running tab]
│   └── BatchRunView
│       ├── ProgressDashboard (per-module bars, stats, controls)
│       ├── LiveResultsFeed (last 200 results, color-coded)
│       └── ConfirmStopDialog (requires typing "STOP")
│
├── [Search tab] (always mounted for state persistence)
│   └── SearchView
│       ├── SearchBar (query, filters, mode: text/semantic/hybrid)
│       ├── VirtualGrid (virtualized, ResizeObserver + IntersectionObserver)
│       │   └── GridCell (aesthetic badge, face count, RAW badge)
│       └── SearchLightbox (zoom/pan, analysis sidebar)
│
└── [Faces tab]
    └── FacesView (cluster mode + legacy identity mode)
        ├── FaceCropThumbnail (lazy-loaded crops)
        └── Inline alias editing (Enter/Esc save)
```

### Image Loading Strategy

**Thumbnails** (`getThumbnail` → `rpc.call('thumbnail')`):
- Python server decodes and resizes to 400x300 JPEG
- LRU cache in main process: 1000 entries, evicts oldest on overflow
- Persistent on-disk cache in `~/.cache/imganalyzer/thumbs` (configurable dir/size)
- Concurrency limited to 4 simultaneous RPC calls (queue in `images.ts`)
- In-flight deduplication prevents duplicate requests for the same path

**Full-resolution** (`getFullImage` → `rpc.call('fullimage')` or direct read):
- JPEG/PNG/TIFF/WebP/BMP: read directly from disk in main process
- RAW/HEIC: decoded by Python server, output capped at 3840px, 92% JPEG quality
- LRU cache: 2 entries
- Blur placeholder shown while full-res loads

### Lightbox Zoom / Pan

State: `zoom` (float, 1 = fit), `offset` ({x, y} px from center).

| Input | Action |
|---|---|
| Scroll wheel | Zoom toward cursor position |
| `+` / `=` | Zoom in (center) |
| `-` | Zoom out (center) |
| `0` | Reset to fit |
| Double-click | Toggle fit ↔ 2x at click point |
| Drag | Pan (only when zoomed) |
| Arrow keys | Navigate images at fit; pan when zoomed |
| `Esc` | Reset zoom if zoomed; close if at fit |

### Batch Processing Flow

```
User configures batch (folder, modules, workers)
    │
    ▼
batch:ingest → rpc.call('ingest', { folder, modules, ... })
    │  ← ingest/progress notifications (scanned/registered/enqueued)
    ▼
batch:start → rpc.call('run', { workers, cloud_workers, ... })
    │
    ├─ batch:tick (1s poll) → rpc.call('status') → BatchStats
    │     └─ Per-module progress bars in ProgressDashboard
    │
    ├─ run/result notifications → batch:result IPC → LiveResultsFeed
    │
    ├─ batch:pause → rpc.call('cancel_run') (preserves queue)
    ├─ batch:resume → rpc.call('run') (re-starts with saved config)
    └─ batch:stop → rpc.call('cancel_run') + rpc.call('queue_clear')
          └─ ConfirmStopDialog (requires typing "STOP")
```

### Single-Image Analysis Flow (`useAnalysis` hook)

```
Image selected
    │
    ▼
readXmp(path)
    │
 cached? ──yes──► setState({ status: 'cached', xmp })
    │
    no
    ▼
setState({ status: 'analyzing', stage: 'Starting…', pct: 0 })
    │
    ▼
runAnalysis(path, 'local')  ←── rpc.call('analyze', { imagePath })
    │                             analyze:progress notifications → setState
    │
 success? ──yes──► setState({ status: 'done', xmp })
    │
    no
    ▼
setState({ status: 'error', message })
```

Stale results discarded via epoch counter (bumped on image change / cancel).

### Search Flow

```
User types query + selects filters + mode (text/semantic/hybrid)
    │
    ▼
Inline query parsing extracts structured filters from natural language:
  "score>7 camera:Sony has:faces sunset" →
    { aestheticMin: 7, camera: "Sony", hasFaces: true, query: "sunset" }
    │
    ▼
search:run → rpc.call('search', { query, mode, filters, limit })
    │
    ▼
SearchResponse { results: [ { path, score, aesthetic_score, face_count, ... } ] }
    │
    ▼
VirtualGrid renders results (virtualized for 500K+ scale)
```

---

## Notification Pipeline

Result notifications flow through 5 stages from Python worker to React state:

```
worker.py _emit_result()
  → callback _result_notify(payload)
  → server.py _send_notification("run/result", payload)
  → _send() → _real_stdout (JSON-RPC line)
  → python-rpc.ts handleLine() → globalNotificationCb()
  → batch.ts notification handler → emitResult() → IPC batch:result
  → renderer onBatchResult → React state → LiveResultsFeed
```

**Progress vs Results are independent systems:**
- **Progress** (bars, counts): DB polling via `rpc.call('status')` every 1s
- **Results** (per-image): JSON-RPC notifications pushed through the pipeline above

---

## Distributed Processing

imganalyzer supports distributing batch analysis across multiple machines:

```
┌──────────────────────────────────────────────────────────┐
│  Coordinator (HTTP JSON-RPC server)                      │
│  python -m imganalyzer.server --transport http            │
│  • Manages job_queue in SQLite                           │
│  • Lease-based job distribution with TTL                 │
│  • Worker registration, heartbeats, capability tracking  │
│  • Optional bearer token authentication                  │
│  • Launched automatically by Electron app                │
└────────┬──────────────────────────┬──────────────────────┘
         │ HTTP JSON-RPC            │ HTTP JSON-RPC
┌────────▼──────────┐    ┌─────────▼─────────┐
│  Worker 1 (GPU)   │    │  Worker 2 (GPU)   │
│  run-distributed  │    │  run-distributed  │
│  -worker          │    │  -worker          │
│  • Claims leases  │    │  • Claims leases  │
│  • Runs analysis  │    │  • Runs analysis  │
│  • Reports results│    │  • Reports results│
└───────────────────┘    └───────────────────┘
```

Workers probe their capabilities at startup (available GPU models, cloud providers) and only claim jobs they can process. Leases include a TTL; if a worker crashes, expired leases are automatically re-queued.

---

## Runtime Dependencies

### Python (Python 3.10+, CUDA GPU recommended)

| Package | Purpose |
|---|---|
| `torch` 2.x + CUDA | GPU inference |
| `transformers` 4.40+ | BLIP-2, GroundingDINO, TrOCR |
| `open_clip_torch` 2.24+ | CLIP embeddings (ViT-L/14) |
| `insightface` 0.7+ | Face detection + recognition (buffalo_l) |
| `onnxruntime-gpu` 1.18+ | ONNX inference for InsightFace |
| `rawpy` 0.21+ | RAW file decoding (LibRaw wrapper) |
| `Pillow` 10.0+ | Standard image I/O |
| `pillow-heif` | HEIC/HEIF support |
| `numpy` 1.24+ | Array operations |
| `scikit-image` 0.22+ | Image processing utilities |
| `exifread` 3.0+ | EXIF metadata extraction |
| `piexif` 1.1+ | EXIF writing |
| `lxml` 5.0+ | XMP XML handling |
| `typer` 0.12+ | CLI framework |
| `rich` 13.0+ | Terminal output formatting |
| `httpx` 0.27+ | HTTP client (reverse geocoding) |
| `python-dotenv` 1.0+ | Environment variable loading |
| `github-copilot-sdk` | Copilot cloud AI backend (optional) |

Optional cloud backends: `openai`, `anthropic`, `google-cloud-vision`.

### Node.js / Electron

| Package | Purpose |
|---|---|
| `electron` 31 | Desktop app shell |
| `electron-vite` | Build tooling (Vite-based) |
| `react` 18 | UI framework |
| `tailwindcss` | Utility CSS |
| `fast-xml-parser` | XMP sidecar parsing |
| `@github/copilot-sdk` | Cloud AI via GitHub Copilot |

---

## Key Design Decisions

**Persistent JSON-RPC server instead of per-call subprocesses.** The original architecture spawned `conda run python <script>` for every thumbnail, full-res decode, and analysis call. This added 1-3 seconds of subprocess startup overhead per call. The current architecture spawns a single persistent Python process at app start, keeping all models loaded in memory and eliminating per-call overhead.

**Three-phase GPU processing.** Objects (GroundingDINO) runs first for all images before any other GPU module (Phase 0). This is required because `has_person` and `has_text` flags from object detection gate downstream modules: cloud AI and aesthetic scoring skip images with people (privacy), and OCR only runs on images with detected text. BLIP-2 then runs exclusively in Phase 1 (~6 GB VRAM). Phase 2 runs faces, OCR, and embedding co-resident (~2.75 GB). Processing objects first avoids wasted work.

**Sequential model loading with VRAM budget.** GPU models are loaded according to a VRAM budget tracker that caps usage at 70% of available GPU memory. Exclusive modules (BLIP-2) require sole GPU access; co-resident modules share within budget. This keeps the system usable on consumer GPUs (8 GB VRAM) while maximizing throughput in Phase 2.

**SQLite as the central data store.** All analysis results are stored in SQLite, not just in XMP files. This enables search, face clustering, progress tracking, and crash recovery. XMP files are regenerated from the database as an export format, maintaining Lightroom compatibility without making XMP the source of truth.

**Hybrid search with RRF.** Combining FTS5 text search with CLIP semantic search via Reciprocal Rank Fusion gives both keyword precision and semantic understanding. A user searching "golden hour landscape" gets results matching those exact words AND visually similar images that were never described with those terms.

**Path+size+mtime fingerprinting.** File fingerprinting uses path, size, and modification time instead of SHA-256 hashing. At 500K images, SHA-256 would add ~7 hours of I/O; mtime-based fingerprinting is instant.

**XMP as the export format.** The CLI writes analysis results to XMP sidecar files readable by Adobe Lightroom, Bridge, and Camera Raw. This means imganalyzer's output integrates directly into existing photography workflows without any Lightroom plugin.

**Privacy guard for people.** Images where GroundingDINO detects a person are automatically excluded from cloud AI analysis and aesthetic scoring. This is enforced at the module runner level, the worker level, and the cloud AI function level to prevent accidental leakage of photos containing people to third-party APIs.

**Override protection.** Users can manually override any analysis field (e.g., correcting an AI caption). Overridden fields are stored in a separate `overrides` table and excluded from UPDATE statements during re-analysis, ensuring manual corrections are never overwritten.

**Three-phase GPU scheduling with VRAM budget.** The original two-phase approach (objects first, then everything else) was replaced with a three-phase scheduler backed by a VRAM budget tracker. Phase 0 runs objects (prerequisite). Phase 1 runs BLIP-2 exclusively (~6 GB). Phase 2 runs faces+OCR+embedding co-resident (~2.75 GB). This maximizes GPU utilization while respecting memory constraints.

**Cloud future capping at phase boundaries.** Cloud API futures submitted during GPU phases are capped at 2× the cloud worker count. Uncompleted cloud futures are cancelled at GPU phase boundaries and returned to the pending queue. This prevents the ThreadPoolExecutor shutdown from blocking for hours when thousands of slow cloud API calls have been queued.

**Job deferral to prevent queue starvation.** When a job's prerequisite isn't met, `defer()` bumps its `queued_at` timestamp forward by 30 seconds instead of preserving the original timestamp. This prevents the same ineligible jobs from being repeatedly claimed at the front of the queue, starving eligible jobs with later timestamps.

**Distributed processing via HTTP coordinator.** The JSON-RPC server supports an HTTP transport mode for distributing batch work across multiple machines. Workers claim job leases with TTLs and report completions. This enables scaling GPU-intensive analysis across a fleet of machines without shared filesystem requirements (workers access images via their own mounts).
