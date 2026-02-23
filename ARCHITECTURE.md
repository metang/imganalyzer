# imganalyzer — Architecture

## Overview

imganalyzer is a two-component system:

1. **`imganalyzer/`** — a Python CLI that reads images, runs local and cloud AI models, performs technical quality analysis, and writes Adobe Lightroom-compatible XMP sidecar files.
2. **`imganalyzer-app/`** — an Electron 31 + React 18 desktop GUI that wraps the CLI, providing a gallery browser, zoomable lightbox, and live analysis sidebar.

The two components share no code at runtime. The GUI drives the CLI as a child subprocess and communicates through stdout/stderr parsing and XMP file I/O.

---

## Repository Layout

```
imganalyzer/
├── imganalyzer/                  # Python package (CLI + analysis engine)
│   ├── cli.py                    # Typer CLI entry point
│   ├── analyzer.py               # Orchestrator: reads image → runs pipeline → AnalysisResult
│   ├── readers/
│   │   ├── raw.py                # RAW decoder via rawpy (LibRaw)
│   │   └── standard.py           # Standard formats via Pillow
│   ├── analysis/
│   │   ├── metadata.py           # EXIF / IPTC extraction (exifread + piexif)
│   │   ├── technical.py          # Sharpness, exposure, noise, color analysis (NumPy/OpenCV)
│   │   └── ai/
│   │       ├── local_full.py     # Local AI orchestrator (4-stage pipeline)
│   │       ├── local.py          # BLIP-2 captioning + scene/mood/subject (transformers)
│   │       ├── aesthetic.py      # LAION aesthetic scorer (transformers)
│   │       ├── objects.py        # GroundingDINO object detection (groundingdino)
│   │       ├── faces.py          # InsightFace detection + recognition
│   │       ├── face_db.py        # Face embedding database (numpy .npz)
│   │       └── cloud.py          # OpenAI / Anthropic / Google Vision backends
│   └── output/
│       └── xmp.py                # XMP sidecar writer (xml.etree + custom namespace)
│
├── imganalyzer-app/              # Electron desktop GUI
│   ├── src/
│   │   ├── main/                 # Electron main process (Node.js)
│   │   │   ├── index.ts          # App bootstrap + IPC handler registration
│   │   │   ├── images.ts         # Folder scanning, thumbnail generation, full-res loading
│   │   │   ├── analyzer.ts       # CLI subprocess management + progress parsing
│   │   │   └── xmp.ts            # XMP sidecar reader (fast-xml-parser)
│   │   ├── preload/
│   │   │   └── index.ts          # contextBridge: exposes window.api to renderer
│   │   └── renderer/             # React UI (Vite + Tailwind CSS)
│   │       ├── App.tsx           # Root: folder state, gallery, lightbox orchestration
│   │       ├── global.d.ts       # Shared type declarations (XmpData, ImageFile, etc.)
│   │       ├── components/
│   │       │   ├── FolderPicker.tsx   # Header bar with open-folder button
│   │       │   ├── Gallery.tsx        # Responsive image grid
│   │       │   ├── Thumbnail.tsx      # Single tile: loads thumbnail via IPC, badges
│   │       │   ├── Lightbox.tsx       # Full-screen viewer: zoom/pan, blur placeholder
│   │       │   └── Sidebar.tsx        # Analysis results panel
│   │       └── hooks/
│   │           └── useAnalysis.ts     # XMP cache check → auto-analyze → live progress
│   ├── electron-vite.config.ts   # Build config (main + preload as SSR, renderer as SPA)
│   ├── tailwind.config.js
│   └── package.json
│
├── tests/                        # Python unit tests (pytest)
├── .gitignore
└── ARCHITECTURE.md               # This file
```

---

## Python CLI

### Entry Points

```
python -m imganalyzer.cli analyze <image(s)> --ai local --overwrite --verbose
python -m imganalyzer.cli register-face NAME image.jpg
python -m imganalyzer.cli list-faces
python -m imganalyzer.cli remove-face NAME
python -m imganalyzer.cli info <image>
```

### Analysis Pipeline

`Analyzer.analyze()` runs these stages in order:

```
Image file
    │
    ├─ readers/raw.py          rawpy (LibRaw) → numpy RGB array
    └─ readers/standard.py     Pillow → numpy RGB array
         │
         ├─ analysis/metadata.py      EXIF/IPTC → dict
         ├─ analysis/technical.py     NumPy/OpenCV metrics → dict
         └─ analysis/ai/
              └─ local_full.py        4-stage local AI pipeline
                   ├─ [1/4] local.py + aesthetic.py   (parallel, ThreadPoolExecutor)
                   ├─ [2/4] objects.py                (GroundingDINO)
                   ├─ [3/4] faces.py                  (InsightFace, gated on has_person)
                   └─ [4/4] merge results
         │
         └─ output/xmp.py      AnalysisResult → XMP sidecar
```

### Technical Analysis (`analysis/technical.py`)

Operates entirely on the numpy RGB array. Computes:

| Field | Method |
|---|---|
| Sharpness score | Laplacian variance |
| Exposure EV | Log₂ of mean luminance relative to mid-grey |
| Noise level | Median absolute deviation of high-frequency residual |
| SNR (dB) | 20·log₁₀(signal/noise) |
| Dynamic range | Log₂(highlight 99th pct / shadow 1st pct) |
| Highlight / shadow clipping % | Pixels above 252 / below 3 |
| Average saturation | Mean HSV saturation |
| Warm/cool ratio | (R−B) / (R+G+B) mean |
| Dominant colors | K-means (k=5) on RGB, centroids as hex |

### Local AI Pipeline (`analysis/ai/`)

| Stage | Model | Library |
|---|---|---|
| Captioning | BLIP-2 (flan-t5-xl) | `transformers` |
| Aesthetic scoring | LAION aesthetic predictor | `transformers` + CLIP |
| Object detection | GroundingDINO | `groundingdino` |
| Face detection | buffalo_l (RetinaFace + ArcFace) | `insightface` |

All models are lazy-loaded singletons — model weights are downloaded once and cached in `~/.cache/huggingface/` or `~/.insightface/`. Subsequent runs in the same process reuse loaded models.

Face recognition compares detected face embeddings against a registered database stored at `~/.imganalyzer/face_db.npz` using cosine similarity (default threshold: 0.40).

### XMP Output (`output/xmp.py`)

Writes a single `<rdf:Description>` block with namespaces:

| Prefix | Namespace | Usage |
|---|---|---|
| `dc:` | Dublin Core | Caption, keywords |
| `xmp:` | XMP Basic | CreateDate, CreatorTool |
| `tiff:` | TIFF/EP | Camera make/model, dimensions |
| `exif:` | EXIF | FNumber, shutter, ISO, GPS |
| `Iptc4xmpCore:` | IPTC | Location fields |
| `photoshop:` | Photoshop | Lens model |
| `crs:` | Camera Raw | Lightroom profile hints |
| `imganalyzer:` | Custom (`http://ns.imganalyzer.io/1.0/`) | All analysis scores |

The output is importable by Adobe Lightroom Classic and Lightroom without any plugin.

---

## Electron GUI

### Process Architecture

Electron runs three separate JS contexts that communicate strictly through defined channels:

```
┌─────────────────────────────────────────────────────┐
│  Main Process (Node.js)                             │
│  src/main/index.ts                                  │
│  • Creates BrowserWindow                            │
│  • Registers ipcMain.handle() for all API calls     │
│  • Spawns conda subprocesses for thumbnails/analysis│
└───────────────┬─────────────────────────────────────┘
                │ contextBridge (IPC)
┌───────────────▼─────────────────────────────────────┐
│  Preload Script                                     │
│  src/preload/index.ts                               │
│  • Bridges Node IPC to window.api                   │
│  • Only surface exposed to renderer                 │
└───────────────┬─────────────────────────────────────┘
                │ window.api
┌───────────────▼─────────────────────────────────────┐
│  Renderer Process (React, isolated)                 │
│  src/renderer/                                      │
│  • No Node.js access                                │
│  • All I/O via window.api calls                     │
└─────────────────────────────────────────────────────┘
```

### IPC API (`window.api`)

| Method | IPC channel | Description |
|---|---|---|
| `openFolder()` | `dialog:openFolder` | Shows native folder picker |
| `listImages(path)` | `fs:listImages` | Returns `ImageFile[]` for a directory |
| `getThumbnail(path)` | `fs:getThumbnail` | Returns 400×300 JPEG as data URL |
| `getFullImage(path)` | `fs:getFullImage` | Returns original file as data URL |
| `readXmp(path)` | `fs:readXmp` | Parses XMP sidecar → `XmpData` or null |
| `runAnalysis(path, backend)` | `analyze:run` | Spawns CLI subprocess → `{xmp, error}` |
| `cancelAnalysis(path)` | `analyze:cancel` | Sends SIGTERM to running subprocess |
| `onAnalysisProgress(cb)` | `analyze:progress` (event) | Live progress from CLI stdout/stderr |

### Image Loading Strategy

**Gallery thumbnails** (`getThumbnail`):
- Python script written once to `%TEMP%\imganalyzer_thumb.py` at startup
- Each call: `conda run -n imganalyzer python <script> <path>`
- Outputs 400×300 JPEG to stdout, read as binary buffer
- Concurrency limited to 4 simultaneous subprocesses (queue in `images.ts`)
- In-flight deduplication prevents duplicate spawns for the same path
- Results cached in-memory for the session

**Lightbox full-resolution** (`getFullImage`):
- JPEG/PNG/TIFF/WebP/BMP: read directly from disk, no re-encoding
- RAW/HEIC: Python script (`imganalyzer_fullres.py`) decodes with rawpy at full resolution, output capped at 3840px max dimension, 92% quality JPEG
- Only the 2 most recently viewed full-res images are kept in memory
- While full-res loads, the low-res thumbnail is shown blurred as a placeholder

### Analysis Flow (`useAnalysis` hook)

```
Image selected
      │
      ▼
readXmp(path)
      │
   cached? ──yes──► setState({ status: 'cached', xmp })
      │
      no
      │
      ▼
setState({ status: 'analyzing', stage: 'Starting…', pct: 0 })
      │
      ▼
runAnalysis(path, 'local')  ←── IPC (blocks until subprocess exits)
      │                          onAnalysisProgress events → setState({ analyzing })
      │
   success? ──yes──► setState({ status: 'done', xmp })
      │
      no
      ▼
setState({ status: 'error', message })
```

Progress events from the CLI are matched against a stage map:

| CLI output pattern | Progress | Label |
|---|---|---|
| `[1/4]` | 5% | Caption + aesthetic scoring |
| `Loading BLIP-2` | 8% | Loading BLIP-2 model… |
| `[2/4]` | 40% | Object detection |
| `Loading.*GroundingDINO` | 42% | Loading GroundingDINO… |
| `[3/4]` | 65% | Face detection & recognition |
| `buffalo_l` | 67% | Loading InsightFace… |
| `[4/4]` | 90% | Merging results |
| `XMP written` / `Done.` | 100% | Done |

Stale results are discarded via an epoch counter — bumped on every new image selection and on cancel.

### Lightbox Zoom / Pan

State: `zoom` (float, 1 = fit), `offset` ({x, y} px from center).

| Input | Action |
|---|---|
| Scroll wheel | Zoom toward cursor position |
| `+` / `=` | Zoom in (center) |
| `-` | Zoom out (center) |
| `0` | Reset to fit |
| Double-click | Toggle fit ↔ 2× at click point |
| Drag | Pan (only when zoomed) |
| Arrow keys | Navigate images at fit; pan when zoomed |
| `Esc` | Reset zoom if zoomed; close if at fit |

---

## Runtime Dependencies

### Python (conda env `imganalyzer`, Python 3.12)

| Package | Purpose |
|---|---|
| `torch` 2.10 + cu128 | GPU inference |
| `transformers` 5.x | BLIP-2, LAION aesthetic model |
| `groundingdino` | Object detection |
| `insightface` | Face detection + recognition (buffalo_l) |
| `rawpy` | RAW file decoding (LibRaw wrapper) |
| `Pillow` | Standard image I/O |
| `pillow-heif` | HEIC/HEIF support |
| `numpy`, `opencv-python` | Technical analysis |
| `exifread`, `piexif` | EXIF/IPTC metadata extraction |
| `typer`, `rich` | CLI framework + terminal output |

### Node.js / Electron

| Package | Purpose |
|---|---|
| `electron` 31 | Desktop app shell |
| `electron-vite` | Build tooling (Vite-based) |
| `react` 18 | UI framework |
| `tailwindcss` | Utility CSS |
| `fast-xml-parser` | XMP sidecar parsing in main process |

---

## Key Design Decisions

**No native Node modules for images.** `sharp` was evaluated but caused bundling issues with Rollup. All image decoding (including standard formats) is handled by Python/Pillow via `conda run`, keeping the Node side dependency-free.

**`conda run` instead of direct Python path.** Ensures the correct conda environment is activated and all GPU-dependent packages (torch, insightface) are available without requiring the user to manually activate the environment.

**Temp file for Python scripts.** `conda run` rejects multi-line `-c` arguments on Windows (`AssertionError: Support for scripts where arguments contain newlines not implemented`). Scripts are written to temp files (`%TEMP%\imganalyzer_thumb.py`, `imganalyzer_fullres.py`) once at startup and reused.

**XMP as the data exchange format.** Rather than inventing a custom API between GUI and CLI, the CLI writes its output to an XMP sidecar (the same file Lightroom reads). The GUI reads this file after the subprocess exits. This means the CLI is a standalone tool that works independently of the GUI.

**Epoch-based stale result prevention.** A `useRef` counter is incremented on every image change and cancel. Any async result that resolves after the epoch has moved on is silently discarded, preventing state corruption from out-of-order IPC responses.
