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
