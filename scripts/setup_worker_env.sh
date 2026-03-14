#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "$SCRIPT_DIR/../.git" ]]; then
  DEFAULT_REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
else
  DEFAULT_REPO_DIR="$HOME/imganalyzer"
fi

ENV_NAME="${IMGANALYZER_ENV_NAME:-imganalyzer312}"
PYTHON_VERSION="${IMGANALYZER_PYTHON_VERSION:-3.12}"
REPO_URL="${IMGANALYZER_REPO_URL:-https://github.com/metang/imganalyzer.git}"
REPO_DIR="${1:-$DEFAULT_REPO_DIR}"
OS_NAME="$(uname -s)"
WORKER_CLOUD_PROVIDER="${IMGANALYZER_WORKER_CLOUD_PROVIDER:-copilot}"

require_cmd() {
  local name="$1"
  if ! command -v "$name" >/dev/null 2>&1; then
    echo "Error: '$name' is required but not installed." >&2
    exit 1
  fi
}

require_cmd conda
require_cmd git

echo "==> Using worker env: $ENV_NAME (python $PYTHON_VERSION)"
if conda env list | awk 'NR > 2 {print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "==> Conda env '$ENV_NAME' already exists."
else
  echo "==> Creating conda env '$ENV_NAME'..."
  conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi

if [[ -d "$REPO_DIR/.git" ]]; then
  echo "==> Updating existing repo at $REPO_DIR..."
  git -C "$REPO_DIR" fetch --quiet origin
  git -C "$REPO_DIR" pull --ff-only
else
  echo "==> Cloning repo into $REPO_DIR..."
  mkdir -p "$(dirname "$REPO_DIR")"
  git clone "$REPO_URL" "$REPO_DIR"
fi

# ── Ensure cache directories are writable ────────────────────────────────────
# On macOS the system may create ~/.cache owned by root, which blocks
# HuggingFace model downloads and the imganalyzer model cache.
echo "==> Checking cache directory permissions..."
CACHE_BASE="$HOME/.cache"
if [[ -d "$CACHE_BASE" ]] && [[ ! -w "$CACHE_BASE" ]]; then
  echo "  ⚠ $CACHE_BASE exists but is not writable by $(whoami)."
  echo "  Attempting to fix ownership (may require sudo)..."
  sudo chown "$(whoami)" "$CACHE_BASE" || {
    echo "  Could not fix $CACHE_BASE ownership."
    echo "  Falling back to ~/Library/Caches (macOS) or ~/var/cache."
    if [[ "$OS_NAME" == "Darwin" ]]; then
      CACHE_BASE="$HOME/Library/Caches"
    else
      CACHE_BASE="$HOME/var/cache"
    fi
  }
fi
mkdir -p "$CACHE_BASE/huggingface" "$CACHE_BASE/imganalyzer"

# Export for downstream pip/torch downloads and the final instructions
export HF_HOME="$CACHE_BASE/huggingface"
export IMGANALYZER_MODEL_CACHE="$CACHE_BASE/imganalyzer"
echo "  HF_HOME=$HF_HOME"
echo "  IMGANALYZER_MODEL_CACHE=$IMGANALYZER_MODEL_CACHE"

echo "==> Installing worker dependencies in env '$ENV_NAME'..."
pushd "$REPO_DIR" >/dev/null
conda run -n "$ENV_NAME" python -m pip install -U pip setuptools wheel

if [[ "$OS_NAME" == "Darwin" ]]; then
  # Remove any conda-installed PyTorch first. Mixing conda-torch and pip-torch
  # leaves stale libtorch_*.dylib files in $CONDA_PREFIX/lib/ that the pip
  # package symlinks to, causing symbol-not-found crashes at import time.
  CONDA_PREFIX_ENV="$(conda info --base)/envs/$ENV_NAME"
  STALE_LIBS=("$CONDA_PREFIX_ENV"/lib/libtorch*.dylib)
  if [[ -e "${STALE_LIBS[0]}" ]]; then
    echo "==> Removing stale conda-installed libtorch libs to avoid conflicts..."
    conda run -n "$ENV_NAME" conda remove --force pytorch torchvision torchaudio -y 2>/dev/null || true
    rm -f "$CONDA_PREFIX_ENV"/lib/libtorch*.dylib
  fi

  # PyPI ships arm64 macOS wheels for torch ≥ 2.5, so pip handles this fine
  # now. Installing via pip keeps a single consistent torch installation and
  # avoids the conda/pip libtorch conflict.
  echo "==> Installing PyTorch via pip (macOS)..."
  conda run -n "$ENV_NAME" python -m pip install "torch>=2.5" torchvision torchaudio

  # onnxruntime: still install from conda-forge for macOS compatibility
  conda install -n "$ENV_NAME" -c conda-forge onnxruntime -y
fi

case "$WORKER_CLOUD_PROVIDER" in
  copilot|openai|anthropic|google)
    ;;
  *)
    echo "Error: Unsupported cloud provider '$WORKER_CLOUD_PROVIDER'." >&2
    echo "Use one of: copilot, openai, anthropic, google" >&2
    exit 1
    ;;
esac

EXTRAS="local-ai,$WORKER_CLOUD_PROVIDER"
echo "==> Installing editable package with extras: [$EXTRAS]"
conda run -n "$ENV_NAME" python -m pip install -e ".[${EXTRAS}]"

# ── Post-install: detect broken libtorch symlinks ────────────────────────────
# If conda torch was ever installed and later replaced by pip torch, broken
# symlinks can be left behind. Detect and fix before the verification step.
CONDA_PREFIX_ENV="$(conda info --base)/envs/$ENV_NAME"
TORCH_LIB_DIR="$CONDA_PREFIX_ENV/lib/python$PYTHON_VERSION/site-packages/torch/lib"
if [[ -d "$TORCH_LIB_DIR" ]]; then
  BROKEN_LINKS=()
  for f in "$TORCH_LIB_DIR"/libtorch*.dylib "$TORCH_LIB_DIR"/libtorch*.so; do
    if [[ -L "$f" ]] && [[ ! -e "$f" ]]; then
      BROKEN_LINKS+=("$f")
    fi
  done
  if [[ ${#BROKEN_LINKS[@]} -gt 0 ]]; then
    echo "⚠ Found broken torch symlinks — reinstalling torch cleanly..."
    rm -f "$CONDA_PREFIX_ENV"/lib/libtorch*.dylib "$CONDA_PREFIX_ENV"/lib/libtorch*.so
    conda run -n "$ENV_NAME" python -m pip uninstall torch -y
    rm -rf "$TORCH_LIB_DIR/../"  # remove leftover torch dir
    conda run -n "$ENV_NAME" python -m pip install --no-cache-dir "torch>=2.5" torchvision torchaudio
  fi
fi

if [[ "$OS_NAME" == "Darwin" ]]; then
  echo "==> Verifying local AI imports (torch + insightface + onnxruntime; UniPercept skipped on Apple Silicon)..."
  conda run -n "$ENV_NAME" python -c "
import torch, numpy as np
print('torch', torch.__version__, '/ numpy', np.__version__)
# Smoke-test: catch numpy ABI mismatches that only surface at runtime
_ = torch.tensor([1.0])
import transformers
print('transformers', transformers.__version__)
import open_clip
print('open_clip ok')
import insightface, onnxruntime as ort
print('insightface', insightface.__version__)
print('onnxruntime', ort.__version__)
"
else
  echo "==> Verifying local AI imports (torch + unipercept + insightface + onnxruntime)..."
  conda run -n "$ENV_NAME" python -c "
import torch, numpy as np
print('torch', torch.__version__, '/ numpy', np.__version__)
# Smoke-test: catch numpy ABI mismatches that only surface at runtime
_ = torch.tensor([1.0])
import transformers
print('transformers', transformers.__version__)
import unipercept_reward
print('unipercept_reward ok')
import open_clip
print('open_clip ok')
import insightface, onnxruntime as ort
print('insightface', insightface.__version__)
print('onnxruntime', ort.__version__)
"
fi

echo "==> Running capability probe..."
conda run -n "$ENV_NAME" python -c "
from imganalyzer.pipeline.distributed_worker import _probe_available_modules
modules = _probe_available_modules()
print('Supported modules:', ', '.join(modules))
from imganalyzer.db.repository import ALL_MODULES
missing = sorted(set(ALL_MODULES) - set(modules))
if missing:
    print('WARNING: Unavailable modules:', ', '.join(missing))
"

echo "==> Verifying cloud provider import ($WORKER_CLOUD_PROVIDER)..."
case "$WORKER_CLOUD_PROVIDER" in
  copilot)
    conda run -n "$ENV_NAME" python -c "import copilot; print('copilot sdk ok')"
    ;;
  openai)
    conda run -n "$ENV_NAME" python -c "import openai; print('openai ok')"
    ;;
  anthropic)
    conda run -n "$ENV_NAME" python -c "import anthropic; print('anthropic ok')"
    ;;
  google)
    conda run -n "$ENV_NAME" python -c "from google.cloud import vision; print('google vision ok')"
    ;;
esac

popd >/dev/null

# ── Write a convenience shell snippet ────────────────────────────────────────
CONDA_PREFIX="$(conda info --base)/envs/$ENV_NAME"

cat <<EOF

Setup complete.

Start worker with:
  conda activate $ENV_NAME
  export HF_HOME=$HF_HOME
  export IMGANALYZER_MODEL_CACHE=$IMGANALYZER_MODEL_CACHE
  imganalyzer run-distributed-worker \\
    --coordinator http://<COORDINATOR_IP>:8765/jsonrpc \\
    --worker-id worker-01 \\
    --cloud $WORKER_CLOUD_PROVIDER

Or run directly without activating:
  HF_HOME=$HF_HOME \\
  IMGANALYZER_MODEL_CACHE=$IMGANALYZER_MODEL_CACHE \\
  $CONDA_PREFIX/bin/python -m imganalyzer.cli run-distributed-worker \\
    --coordinator http://<COORDINATOR_IP>:8765/jsonrpc \\
    --worker-id worker-01 \\
    --cloud $WORKER_CLOUD_PROVIDER

Then requeue failed jobs on the coordinator:
  - UI: click "Retry failed"
  - CLI: imganalyzer run --retry-failed
EOF
