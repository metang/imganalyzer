# imganalyzer

> A powerful CLI tool for analyzing images and camera RAW files. Extracts EXIF metadata, computes technical quality scores, and runs AI-powered content analysis. Outputs **Adobe Lightroom-compatible XMP sidecar files**.

---

## Features

- 📷 **All major formats** — JPEG, PNG, TIFF, WEBP, HEIC/HEIF, BMP, GIF, and more
- 🎞️ **Camera RAW files** — CR2, CR3 (Canon), NEF (Nikon), ARW/SR2 (Sony), DNG (Adobe/Leica/etc.), ORF (Olympus), RW2 (Panasonic), PEF (Pentax), RAF (Fujifilm), and more via LibRaw
- 🗂️ **EXIF metadata** — Camera model, lens, ISO, aperture, shutter speed, focal length, GPS location (with reverse geocoding)
- 📊 **Technical analysis** — Sharpness score, exposure statistics, noise level, histogram, color profile, dynamic range estimate
- 🤖 **AI analysis** — Scene description, detected objects, dominant colors, aesthetic/mood scoring
  - **Local**: BLIP-2 via HuggingFace Transformers (offline, no API key needed)
  - **Cloud**: OpenAI GPT-4o Vision, Anthropic Claude 3.5 Sonnet, Google Vision API
- 📄 **XMP output** — Lightroom-compatible `.xmp` sidecar files you can import directly into Lightroom

---

## Installation

```bash
pip install imganalyzer

# With local AI support (downloads ~3GB BLIP-2 model on first use)
pip install "imganalyzer[local-ai]"

# With cloud AI support
pip install "imganalyzer[openai]"
pip install "imganalyzer[anthropic]"
pip install "imganalyzer[google]"

# Everything
pip install "imganalyzer[all-ai]"
```

---

## Quick Start

```bash
# Analyze a RAW file (metadata + technical only)
imganalyzer analyze photo.cr2

# Full analysis with OpenAI Vision
imganalyzer analyze photo.nef --ai openai

# Use local BLIP-2 model (no internet/API key needed)
imganalyzer analyze photo.arw --ai local

# Batch analyze a folder
imganalyzer analyze ./photos/*.jpg --ai openai

# Quick EXIF info in terminal (no XMP file)
imganalyzer info photo.dng

# Specify output path
imganalyzer analyze photo.cr3 --output ./sidecars/photo.xmp

# Skip AI, metadata + technical only
imganalyzer analyze photo.jpg --no-ai
```

---

## Distributed Coordinator + Workers

The distributed batch prototype uses the existing JSON-RPC server in HTTP mode as
the coordinator and one or more `run-distributed-worker` agents as workers.

If you use `imganalyzer-app`, you can now manage this from the desktop UI: open
the new **Settings** page from the gear icon, enable the distributed job server,
set host/port/auth options, and optionally have the coordinator start
automatically when the app launches. That page also shows the worker command to
run on other machines.

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
on macOS/Linux), install `imganalyzer[local-ai,<provider>]` from source, and
verify local AI and cloud-provider imports. Default provider is `copilot`.
Override defaults with environment variables:
- `IMGANALYZER_ENV_NAME`
- `IMGANALYZER_PYTHON_VERSION`
- `IMGANALYZER_REPO_URL`
- `IMGANALYZER_WORKER_CLOUD_PROVIDER` (`copilot`, `openai`, `anthropic`, `google`)

Platform notes:
- **Windows**: PyTorch is installed from the official PyTorch CUDA index
  (`https://download.pytorch.org/whl/cu128`) for GPU support. The setup
  script handles this automatically.
- **macOS**: PyTorch is installed via `conda install -c pytorch` (not pip) because
  PyPI only hosts older macOS wheels that are incompatible with numpy ≥2. The
  setup script handles this automatically. If you see errors like
  *"A module compiled using NumPy 1.x cannot be run in NumPy 2.x"* or
  *"Symbol not found"* in `libtorch_cpu`, clean-reinstall torch from one source:
  ```bash
  pip uninstall torch torchvision torchaudio -y
  conda install -n imganalyzer312 pytorch torchvision torchaudio -c pytorch --force-reinstall
  ```
  **Never mix pip and conda torch packages** — their native libraries conflict.
- macOS uses CPU `onnxruntime` (no `onnxruntime-gpu` wheels available); the
  setup script pre-installs it from `conda-forge`.
- Windows/Linux use `onnxruntime-gpu`.

### Firewall setup for remote workers (Windows coordinator)

If workers are on another machine/LAN, you need both a port rule and a Python
program rule on the coordinator host:

```powershell
# 1) Allow coordinator port
New-NetFirewallRule -DisplayName "imganalyzer Coordinator TCP 8765" `
  -Direction Inbound -Protocol TCP -LocalPort 8765 -Action Allow

# 2) Allow Python process that hosts the coordinator
New-NetFirewallRule -DisplayName "imganalyzer Python Inbound" `
  -Direction Inbound -Program "C:\Users\<you>\miniconda3\envs\imganalyzer\python.exe" `
  -Protocol TCP -Action Allow
```

If connectivity still times out, check for an auto-created inbound **block**
rule like `TCP Query User ... python.exe` and disable it:

```powershell
Get-NetFirewallRule -Enabled True -Direction Inbound -Action Block |
  Where-Object { $_.DisplayName -like "*python.exe*" } |
  Set-NetFirewallRule -Enabled False
```

If `netsh advfirewall show currentprofile` reports
`LocalFirewallRules N/A (GPO-store only)`, local rules are ignored and the same
allow rules must be added in domain Group Policy.

Useful worker options:

- `--module metadata` to dedicate a worker to a single module
- `--cloud copilot` to process `cloud_ai` and `aesthetic` with Copilot backend (match your batch provider)
- `--lease-ttl 300` to request longer job leases
- `--heartbeat-interval 15` to refresh worker and lease liveness more often
- `--path-mapping "SOURCE_PREFIX=LOCAL_PREFIX"` to remap shared-NAS paths on a worker with a different mount root
- `--auth-token YOUR_TOKEN` when the coordinator requires HTTP auth

### Current assumptions

- The coordinator is the only SQLite reader/writer; workers return structured
  results over HTTP and do not need direct DB access.
- Workers need read-only access to the shared image files and must either read
  the stored paths directly or provide `--path-mapping` rules when their NAS
  mount root differs.
- XMP sidecar generation, when enabled, happens on the coordinator after the
  last queued job for an image completes.
- The current implementation has been validated with local HTTP coordinator/worker
  smoke tests, including two concurrent workers claiming jobs without duplicate
  execution.

---

## Configuration

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

```env
# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Google Vision (path to service account JSON)
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

---

## XMP Output

The generated `.xmp` file is a standard XML sidecar compatible with Adobe Lightroom, Bridge, and Camera Raw. Example:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description
      xmlns:dc="http://purl.org/dc/elements/1.1/"
      xmlns:exif="http://ns.adobe.com/exif/1.0/"
      xmlns:xmp="http://ns.adobe.com/xap/1.0/"
      dc:description="A golden hour landscape with mountains..."
      exif:ISOSpeedRatings="400"
      exif:FNumber="2.8"
      ...
    />
  </rdf:RDF>
</x:xmpmeta>
```

### Lightroom Import

1. Place the `.xmp` file alongside the image file (same name, `.xmp` extension)
2. In Lightroom: **Metadata → Read Metadata from Files**
3. AI-generated keywords appear under **Metadata → Keywords**
4. AI description appears in **Caption** field

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

```
imganalyzer analyze [OPTIONS] IMAGE...

  Analyze one or more image files and generate XMP sidecar files.

Options:
  --ai [openai|anthropic|google|local|none]  AI backend for content analysis [default: none]
  --output PATH                              Output XMP file path (single file only)
  --no-ai                                    Skip AI analysis
  --no-technical                             Skip technical analysis
  --overwrite                                Overwrite existing XMP files
  --verbose / --quiet                        Verbosity
  --help                                     Show this message and exit.

imganalyzer info [OPTIONS] IMAGE

  Display EXIF and metadata in the terminal (no XMP output).

Options:
  --format [table|json|yaml]  Output format [default: table]
  --help                      Show this message and exit.
```

---

## License

MIT
