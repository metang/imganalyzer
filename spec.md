# imganalyzer — Product Specification

## What Is It

imganalyzer is a desktop application for photographers that automatically analyzes image libraries using local AI models and cloud vision APIs. It extracts EXIF metadata, computes technical quality scores (sharpness, exposure, noise, dynamic range), generates natural-language captions, detects objects and faces, performs OCR on text in images, and produces aesthetic ratings. All results are written to Adobe Lightroom-compatible XMP sidecar files, so analysis data flows directly into existing photography workflows.

The system has two parts: a Python backend (usable standalone as a CLI) and an Electron + React desktop GUI that drives the backend as a persistent child process.

---

## Who Is It For

Photographers with large image libraries (tested up to 500K+ images) who want to:
- Auto-tag, caption, and rate thousands of images without manual effort
- Search their library by natural language ("sunset over mountains") or technical criteria ("sharpness > 80, ISO < 400, has faces")
- Identify and group people across their library using face recognition
- Get results directly in Adobe Lightroom via standard XMP sidecar files
- Run analysis locally (no cloud dependency) or use cloud APIs for higher quality

---

## Core Capabilities

### 1. Image Format Support

| Category | Formats |
|---|---|
| Standard | JPEG, PNG, TIFF, WebP, BMP, GIF, HEIC/HEIF |
| Canon RAW | CR2, CR3, CRW |
| Nikon RAW | NEF, NRW |
| Sony RAW | ARW, SR2, SRF |
| Universal RAW | DNG |
| Other RAW | RAF (Fuji), ORF (Olympus), RW2 (Panasonic), PEF (Pentax), 3FR/FFF (Hasselblad), IIQ (Phase One), X3F (Sigma), MRW (Minolta), RWL (Leica) |

RAW files are decoded via LibRaw (rawpy). HEIC/HEIF via pillow-heif. Standard formats via Pillow.

### 2. Metadata Extraction

Reads EXIF and IPTC metadata from image files:
- Camera make/model, lens model
- ISO, aperture (f-number), shutter speed, focal length
- Capture date/time
- Image dimensions, orientation
- GPS coordinates with reverse geocoding to human-readable location

### 3. Technical Quality Analysis

Computed from the raw pixel data (NumPy/OpenCV), not from metadata:

| Metric | Method |
|---|---|
| Sharpness score | Laplacian variance with saliency-weighted patch selection |
| Exposure EV | Log2 of mean luminance relative to mid-grey |
| Noise level | Median absolute deviation of high-frequency residual |
| SNR (dB) | 20 * log10(signal / noise) |
| Dynamic range (stops) | Log2(highlight 99th percentile / shadow 1st percentile) |
| Highlight clipping % | Pixels above 252 |
| Shadow clipping % | Pixels below 3 |
| Average saturation | Mean HSV saturation |
| Warm/cool ratio | (R - B) / (R + G + B) mean |
| Dominant colors | K-means (k=5) on RGB, reported as hex codes |

### 4. Local AI Analysis (No Internet Required)

Four GPU-accelerated models run on the user's machine:

| Module | Model | What It Does |
|---|---|---|
| Captioning | BLIP-2 (flan-t5-xl) | Natural-language image description + VQA for scene/mood/subject |
| Object detection | GroundingDINO (grounding-dino-base) | Zero-shot object detection with bounding boxes |
| Face detection | InsightFace (buffalo_l: RetinaFace + ArcFace) | Face detection, embedding extraction, recognition against registered identities |
| OCR | TrOCR (trocr-large-printed) | Text recognition with document tiling, gated on text detection from GroundingDINO |

Models are lazy-loaded singletons cached in `~/.cache/huggingface/` and `~/.insightface/`. Total VRAM requirement: ~4.7 GB peak (models are loaded and unloaded sequentially, not co-resident).

### 5. Cloud AI Analysis

Cloud vision APIs provide higher-quality captions and aesthetic scores. Supported backends:

| Backend | Model | Required Config |
|---|---|---|
| OpenAI | GPT-4o | `OPENAI_API_KEY` |
| Anthropic | Claude 3.5 Sonnet | `ANTHROPIC_API_KEY` |
| Google | Cloud Vision API | `GOOGLE_APPLICATION_CREDENTIALS` |
| GitHub Copilot | GPT-4.1 (via Copilot SDK) | GitHub Copilot subscription |

Cloud AI extracts: detailed description, keywords, scene type, mood, aesthetic score (1-10), composition notes, and species identification for animals. Cloud analysis also handles aesthetic scoring, avoiding a redundant API call.

**Privacy guard**: Images where GroundingDINO detects a person are automatically excluded from cloud AI and aesthetic scoring to protect privacy.

### 6. CLIP Embeddings & Semantic Search

- OpenCLIP ViT-L/14 generates 768-dimensional embeddings for every image
- Text descriptions from BLIP-2 and cloud AI are also embedded
- Hybrid search engine combines FTS5 full-text search with CLIP semantic similarity using Reciprocal Rank Fusion (RRF)
- Sub-10ms search at 500K images via numpy BLAS vectorized scoring

### 7. Face Recognition & Clustering

- Face embeddings are stored per-occurrence with bounding boxes
- Greedy agglomerative clustering groups faces by cosine similarity (threshold 0.55)
- Users can register known faces, set display aliases, merge/rename identities
- Face crops served as thumbnails in the GUI

### 8. XMP Sidecar Output

Every analyzed image gets a `.xmp` sidecar file written alongside it, readable by Adobe Lightroom Classic, Lightroom, Bridge, and Camera Raw. The XMP includes:

| XMP Namespace | Content |
|---|---|
| Dublin Core (`dc:`) | Caption, keywords |
| XMP Basic (`xmp:`) | CreateDate, CreatorTool |
| TIFF/EP (`tiff:`) | Camera make/model, dimensions |
| EXIF (`exif:`) | FNumber, shutter, ISO, GPS |
| IPTC (`Iptc4xmpCore:`) | Location fields |
| Photoshop (`photoshop:`) | Lens model |
| Camera Raw (`crs:`) | Lightroom profile hints |
| Custom (`imganalyzer:`) | All analysis scores, AI results, face data, OCR text |

### 9. Override System

Users can manually override any analysis field (e.g., correcting an AI-generated caption). Overridden fields are protected from being overwritten on re-analysis.

---

## Desktop GUI

The Electron app provides a five-tab interface:

### Gallery Tab
- Open a folder to browse images in a responsive grid
- Thumbnails generated via Python (400x300 JPEG), cached in LRU cache (1000 entries)
- Click an image to open a full-screen lightbox with zoom/pan (scroll wheel, double-click, keyboard shortcuts)
- Right sidebar shows local analysis results (metadata, technical scores, AI captions, detected objects, faces)
- Left sidebar provides cloud AI analysis via GitHub Copilot SDK
- Navigate between images with arrow keys

### Batch Tab
- Configure and launch batch analysis for an entire folder
- Select which analysis passes to run (metadata, technical, blip2, objects, ocr, faces, cloud_ai, aesthetic, embedding)
- Configure number of GPU workers, cloud provider, recursive scanning
- Folder ingestion scans and registers images, enqueues jobs by module priority

### Running Tab
- Real-time progress dashboard during batch processing
- Per-module progress bars and completion counts
- Live results feed showing the last 200 completed analyses with color-coded status
- Pause/resume/stop controls; stop requires typing "STOP" to confirm
- Session-average processing speed display

### Search Tab
- Natural-language search across the entire analyzed library
- Three search modes: text (FTS5), semantic (CLIP), hybrid (RRF fusion of both)
- Filter sidebar with fields for: aesthetic score, sharpness, ISO, faces, camera, lens, location, date range, people detection
- Inline query parsing: `score>7 camera:Sony has:faces sunset` extracts filters from the query string
- Virtualized result grid (handles 500K+ results) with aesthetic score and face count badges
- Click a result to open a search-specific lightbox with full analysis sidebar

### Faces Tab
- Browse detected faces grouped by cluster
- View face crops as thumbnails
- Edit display aliases inline
- Run clustering to group unknown faces by embedding similarity

---

## Batch Processing Architecture

The batch pipeline is designed for large libraries (500K+ images):

1. **Ingest**: Scan folder, register images in SQLite, enqueue jobs per module. Batched transactions (500 images per commit). File fingerprinting uses path+size+mtime instead of SHA-256.

2. **Two-phase worker**: Phase 1 drains all `objects` jobs first (GroundingDINO, serial GPU) to determine which images contain people (unlocking the privacy gate for cloud/aesthetic). Phase 2 runs remaining GPU passes sequentially (blip2 -> ocr -> faces -> embedding) with a concurrent cloud thread pool.

3. **GPU memory management**: Models are loaded and unloaded between passes, keeping peak VRAM at ~4.7 GB instead of ~9.5 GB if all models were co-resident.

4. **Batched inference**: objects (batch=8), blip2 (batch=2), embedding (batch=32).

5. **Deferred housekeeping**: FTS5 search index and XMP sidecar writes are flushed every 60 seconds, not per-job.

6. **Session recovery**: If the app crashes mid-batch, leftover jobs are detected on next launch and can be resumed.

---

## Communication Protocol

The Python backend runs as a persistent child process, communicating with Electron via JSON-RPC 2.0 over stdin/stdout:

- Electron spawns `python -m imganalyzer.server` once at startup
- All calls go through a single persistent connection (no per-call subprocess overhead)
- `stdout` is reserved exclusively for JSON-RPC messages; all Python logging goes to `stderr`
- Thread-safe writes via `_send_lock`
- Notifications push real-time results and progress to the GUI without polling

---

## Data Storage

All analysis results are stored in a SQLite database at `~/.cache/imganalyzer/imganalyzer.db` (configurable via `IMGANALYZER_DB_PATH`). The database uses WAL mode for concurrent read/write access across threads. Schema includes 13+ tables across 7 migration versions, with FTS5 virtual tables for full-text search and a dedicated table for CLIP embeddings.

XMP sidecar files are the export format — they are regenerated from the database on demand or periodically during batch processing.

---

## CLI Usage

The Python backend is fully usable without the GUI:

```bash
# Single image analysis
imganalyzer analyze photo.cr2 --ai local

# Batch pipeline (ingest + run)
imganalyzer ingest /photos --modules metadata,technical,objects,blip2
imganalyzer run --workers 1

# Monitor progress
imganalyzer status

# Search
imganalyzer search "sunset over mountains"
imganalyzer search-json "score>7 has:faces" --mode hybrid --limit 50

# Face management
imganalyzer register-face "John" reference.jpg
imganalyzer list-faces
imganalyzer alias-face "John" "John Smith"
imganalyzer merge-face "John" "Jonathan"

# Utilities
imganalyzer info photo.nef
imganalyzer override photo.jpg caption "My custom caption"
imganalyzer purge-missing
imganalyzer queue-clear --status failed
imganalyzer rebuild --module objects
```

---

## Configuration

Environment variables (set in `.env` or shell):

| Variable | Purpose | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key | — |
| `ANTHROPIC_API_KEY` | Anthropic API key | — |
| `GOOGLE_APPLICATION_CREDENTIALS` | Google Vision credentials path | — |
| `IMGANALYZER_DB_PATH` | SQLite database location | `~/.cache/imganalyzer/imganalyzer.db` |
| `IMGANALYZER_MODEL_CACHE` | Model download cache directory | `~/.cache/imganalyzer` |
| `IMGANALYZER_DEFAULT_AI` | Default AI backend | `none` |
| `IMGANALYZER_DETECTION_PROMPT` | GroundingDINO detection categories | `person . animal . vehicle ...` |
| `IMGANALYZER_DETECTION_THRESHOLD` | Detection confidence threshold | `0.30` |
| `IMGANALYZER_FACE_DB_THRESHOLD` | Face recognition similarity threshold | `0.40` |
| `IMGANALYZER_OCR_NUM_BEAMS` | TrOCR beam search width | — |
| `IMGANALYZER_FACE_DB` | Legacy face database path | `~/.cache/imganalyzer/faces.json` |

---

## Runtime Requirements

- **Python**: 3.10+ with CUDA-capable GPU for local AI (torch + CUDA)
- **Node.js**: Electron 31 for the desktop GUI
- **GPU VRAM**: ~4.7 GB peak (models loaded sequentially)
- **Disk**: ~3 GB for model weights (downloaded on first use)
- **OS**: Windows (primary target), Linux/macOS should work but less tested
