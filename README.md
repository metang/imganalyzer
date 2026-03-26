# imganalyzer

> Analyze and catalog large image libraries with local AI. Extracts EXIF metadata, computes technical quality scores, detects objects and faces, generates captions, and scores aesthetics — all on your own hardware. Outputs **Adobe Lightroom-compatible XMP sidecar files**. Includes an Electron desktop app and a CLI.

---

## Features

- 📷 **All major formats** — JPEG, PNG, TIFF, WEBP, HEIC/HEIF, BMP, GIF, and more
- 🎞️ **Camera RAW files** — CR2, CR3 (Canon), NEF (Nikon), ARW/SR2 (Sony), DNG (Adobe/Leica/etc.), ORF (Olympus), RW2 (Panasonic), PEF (Pentax), RAF (Fujifilm), and more via LibRaw
- 🗂️ **EXIF metadata** — Camera model, lens, ISO, aperture, shutter speed, focal length, GPS location (with reverse geocoding)
- 📊 **Technical analysis** — Sharpness, exposure, noise, SNR, dynamic range, dominant colors, saturation
- 🤖 **Local AI pipeline** (GPU, no API keys needed):
  - **Captioning** — Qwen 3.5 9B via [Ollama](https://ollama.com)
  - **Object detection** — GroundingDINO (zero-shot, batched)
  - **Face recognition** — InsightFace buffalo_l (detection + ArcFace embeddings)
  - **Aesthetic scoring** — UniPercept (IAA, IQA, ISTA metrics; CUDA only)
  - **Semantic embeddings** — OpenCLIP ViT-L/14 (768-d vectors for search)
- ☁️ **Cloud AI** (optional) — OpenAI GPT-4o, Anthropic Claude 3.5 Sonnet, Google Vision, GitHub Copilot
- 🔍 **Search** — Hybrid FTS5 full-text + CLIP semantic search with Reciprocal Rank Fusion
- 🖥️ **Desktop app** — Electron + React GUI with Gallery, Batch, Running, Search, and Faces tabs
- 🌐 **Distributed processing** — HTTP JSON-RPC coordinator with remote workers for multi-machine batch analysis
- 📄 **XMP output** — Lightroom-compatible `.xmp` sidecar files you can import directly

---

## Prerequisites

- **Python 3.10+** (3.10, 3.11, or 3.12)
- **[Ollama](https://ollama.com)** — required for AI captioning. Install it, then pull the model:
  ```bash
  ollama pull qwen3.5:9b
  ```
- **CUDA GPU** (recommended) — needed for object detection, faces, embeddings, and perception scoring. CPU-only works for metadata + technical analysis.
- **Node.js 18+** — only needed if running the desktop app

---

## Installation

```bash
# Clone and install in editable mode
git clone <repo-url> && cd imganalyzer
pip install -e .

# With local AI models (GroundingDINO, InsightFace, OpenCLIP, UniPercept)
pip install -e ".[local-ai]"

# With cloud AI support
pip install -e ".[openai]"
pip install -e ".[anthropic]"
pip install -e ".[google]"
pip install -e ".[copilot]"

# Everything
pip install -e ".[all-ai]"

# Dev tools (pytest, ruff, mypy)
pip install -e ".[dev]"
```

> **Conda recommended for GPU dependencies.** See [Worker Bootstrap](#quick-worker-bootstrap) for automated Conda setup scripts that handle PyTorch CUDA/MPS installation.

---

## Quick Start

### Single-file analysis

```bash
# Metadata + technical only (no GPU needed)
imganalyzer analyze photo.cr2

# Full analysis with local AI (Ollama must be running)
imganalyzer analyze photo.nef --ai local

# Cloud AI (requires API key in .env)
imganalyzer analyze photo.arw --ai openai

# Recursive directory scan
imganalyzer analyze ./photos/ --recursive --ai local

# Skip XMP sidecar generation (DB-only)
imganalyzer analyze photo.jpg --ai local --no-xmp
```

### Batch processing (large libraries)

```bash
# 1. Scan and register images into the database
imganalyzer ingest /path/to/photos

# 2. Process the job queue (all AI modules)
imganalyzer run

# 3. Check progress
imganalyzer status

# Ctrl+C pauses gracefully — resume with:
imganalyzer run

# Retry previously failed jobs
imganalyzer run --retry-failed
```

### Quick info

```bash
# EXIF info in terminal (no XMP, no AI)
imganalyzer info photo.dng
imganalyzer info photo.dng --format json
```

---

## Desktop App

The Electron GUI provides a visual interface with five tabs: **Gallery**, **Batch**, **Running**, **Search**, and **Faces**.

```bash
cd imganalyzer-app
npm install
npm run dev      # Dev mode with hot reload
npm run build    # Production build
```

The app spawns `python -m imganalyzer.server` as a persistent child process at startup — all communication uses JSON-RPC 2.0 over stdin/stdout.

---

## Distributed Processing

The JSON-RPC server doubles as an HTTP coordinator. Remote workers connect, claim jobs, and return results — the coordinator handles all DB writes and XMP generation.

The desktop app can manage this from **Settings** (gear icon): enable the distributed job server, set host/port/auth, and optionally auto-start with the app.

### Start the coordinator

```bash
python -m imganalyzer.server --transport http --host 127.0.0.1 --port 8765
```

For LAN access, add a bearer token:

```bash
python -m imganalyzer.server --transport http --host 0.0.0.0 --port 8765 --auth-token YOUR_TOKEN
```

### Start a worker

```bash
imganalyzer run-distributed-worker \
  --coordinator http://127.0.0.1:8765/jsonrpc \
  --worker-id worker-01
```

### Quick worker bootstrap

**macOS / Linux** (Bash + Conda):

```bash
bash scripts/setup_worker_env.sh ~/imganalyzer-worker
```

**Windows** (PowerShell + Conda):

```powershell
.\scripts\setup_worker_env.ps1 -RepoDir D:\Code\imganalyzer
```

Both scripts create/reuse a Conda env (`imganalyzer` on Windows, `imganalyzer312`
on macOS/Linux), install `imganalyzer[local-ai]` from source, and verify local
AI runtime dependencies (`torch`, `insightface`, `onnxruntime`).
`unipercept_reward` is verified on CUDA-capable hosts (non-Darwin).
Override defaults with environment variables:
- `IMGANALYZER_ENV_NAME`
- `IMGANALYZER_PYTHON_VERSION`
- `IMGANALYZER_REPO_URL`

Platform notes:
- **Windows**: PyTorch is installed from the official PyTorch CUDA index
  (`https://download.pytorch.org/whl/cu128`) for GPU support.
- **macOS**: PyTorch is installed via `conda install -c pytorch` (not pip) because
  PyPI only hosts older macOS wheels incompatible with numpy ≥2.
  **Never mix pip and conda torch packages** — their native libraries conflict.
- macOS uses CPU `onnxruntime` (`conda-forge`); Windows/Linux use `onnxruntime-gpu`.

### Worker options

| Flag | Description |
|---|---|
| `--module metadata` | Dedicate a worker to a single module |
| `--batch-size 1` | Tune lease claim granularity |
| `--poll-interval 5` | How often empty queues are re-polled (seconds) |
| `--lease-ttl 300` | Request longer job leases |
| `--heartbeat-interval 15` | Refresh worker and lease liveness more often |
| `--path-mapping "SRC=LOCAL"` | Remap shared-NAS paths for a different mount root |
| `--auth-token TOKEN` | Authenticate when coordinator requires HTTP auth |
| `--no-xmp` | Skip coordinator-side XMP writes for this worker |

### Firewall (Windows coordinator, remote workers)

```powershell
# Allow coordinator port
New-NetFirewallRule -DisplayName "imganalyzer Coordinator TCP 8765" `
  -Direction Inbound -Protocol TCP -LocalPort 8765 -Action Allow

# Allow the Python process
New-NetFirewallRule -DisplayName "imganalyzer Python Inbound" `
  -Direction Inbound -Program "C:\Users\<you>\miniconda3\envs\imganalyzer\python.exe" `
  -Protocol TCP -Action Allow
```

---

## Configuration

Copy `.env.example` to `.env` and fill in your API keys (only needed for cloud AI):

```bash
cp .env.example .env
```

```env
# Cloud AI (optional — local AI via Ollama needs no keys)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json

# Default AI backend (default: none)
IMGANALYZER_DEFAULT_AI=none

# Object detection prompt (categories separated by ' . ')
IMGANALYZER_DETECTION_PROMPT=person . animal . vehicle . building .

# Object detection confidence threshold (0.0–1.0, default: 0.30)
IMGANALYZER_DETECTION_THRESHOLD=0.30

# Face recognition cosine similarity threshold (0.0–1.0, default: 0.40)
IMGANALYZER_FACE_DB_THRESHOLD=0.40
```

---

## Pipeline Modules

The batch worker processes images through a multi-phase GPU pipeline:

| Phase | Module | Model | VRAM |
|---|---|---|---|
| 0 | `caption` | Qwen 3.5 9B (Ollama) | ~8.7 GB |
| 1 | `objects` | GroundingDINO | ~2.4 GB |
| 2 | `faces` | InsightFace buffalo_l | ~1.0 GB |
| 2 | `embedding` | OpenCLIP ViT-L/14 | ~0.95 GB |
| 3 | `perception` | UniPercept (CUDA only) | ~13.8 GB |

`metadata` and `technical` run on CPU with no GPU requirements.

Phases load/unload models sequentially for GPU memory efficiency. Within a phase, co-resident modules (e.g., `faces` + `embedding`) share VRAM.

---

## XMP Output

The generated `.xmp` file is a standard XML sidecar compatible with Adobe Lightroom, Bridge, and Camera Raw.

### Lightroom Import

1. Place the `.xmp` file alongside the image file (same name, `.xmp` extension)
2. In Lightroom: **Metadata → Read Metadata from Files**
3. AI-generated keywords appear under **Metadata → Keywords**
4. AI description appears in the **Caption** field

---

## Supported RAW Formats

| Manufacturer | Extensions |
|---|---|
| Canon | `.cr2`, `.cr3`, `.crw` |
| Nikon | `.nef`, `.nrw` |
| Sony | `.arw`, `.sr2`, `.srf` |
| Adobe/Universal | `.dng` |
| Fujifilm | `.raf` |
| Olympus/OM | `.orf` |
| Panasonic | `.rw2`, `.raw` |
| Pentax/Ricoh | `.pef`, `.ptx` |
| Leica | `.dng`, `.rwl` |
| Hasselblad | `.3fr`, `.fff` |
| Phase One | `.iiq` |
| Sigma | `.x3f` |
| Minolta | `.mrw` |

---

## CLI Reference

### Image analysis

| Command | Description |
|---|---|
| `analyze IMAGE...` | Analyze images and generate XMP sidecars |
| `info IMAGE` | Display EXIF metadata in the terminal |

### Batch processing

| Command | Description |
|---|---|
| `ingest PATH` | Scan a folder and register images into the database |
| `run` | Process the job queue (local worker) |
| `run-distributed-worker` | Run as a remote worker connecting to an HTTP coordinator |
| `status` | Display queue status and worker metrics |
| `queue-clear` | Clear all jobs from the queue |
| `rebuild` | Re-enqueue module jobs for reprocessing |
| `purge-missing` | Remove database entries for files that no longer exist |

### Face management

| Command | Description |
|---|---|
| `register-face` | Register a face identity with a reference image |
| `list-faces` | List registered face identities |
| `remove-face` | Remove a registered face identity |
| `alias-face` | Create an alias for a face identity |
| `rename-face` | Rename a face identity |
| `merge-face` | Merge two face identities into one |

### Search

| Command | Description |
|---|---|
| `search QUERY` | Search the image library (text output) |
| `search-json QUERY` | Search with JSON output (FTS5, CLIP, or hybrid mode) |

### Maintenance

| Command | Description |
|---|---|
| `override` | Set manual overrides on analysis fields |
| `cleanup-sessions` | Delete GitHub Copilot cloud AI sessions |
| `profile-report` | Generate performance profiling reports |

---

## Development

```bash
# Install with dev tools
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check imganalyzer/

# Type check
mypy imganalyzer/
```

---

## License

MIT
