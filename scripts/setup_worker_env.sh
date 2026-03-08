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
  # On macOS, PyPI only hosts older PyTorch wheels (≤2.2.x) which are
  # incompatible with numpy 2.x.  Install PyTorch from conda's pytorch
  # channel first so pip won't try to pull the old wheels.
  echo "==> Installing PyTorch + ONNX runtime from conda channels (macOS)..."
  conda install -n "$ENV_NAME" -c pytorch pytorch torchvision torchaudio -y
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

echo "==> Verifying local AI imports (torch + insightface + onnxruntime)..."
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

echo "==> Running capability probe..."
conda run -n "$ENV_NAME" python -c "
from imganalyzer.pipeline.distributed_worker import _probe_available_modules
modules = _probe_available_modules('$WORKER_CLOUD_PROVIDER')
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
