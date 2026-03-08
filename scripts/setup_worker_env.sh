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

echo "==> Installing worker dependencies in env '$ENV_NAME'..."
pushd "$REPO_DIR" >/dev/null
conda run -n "$ENV_NAME" python -m pip install -U pip setuptools wheel

if [[ "$OS_NAME" == "Darwin" ]]; then
  echo "==> Installing macOS ONNX runtime from conda-forge..."
  conda install -n "$ENV_NAME" -c conda-forge onnxruntime -y
fi

conda run -n "$ENV_NAME" python -m pip install -e ".[local-ai]"

echo "==> Verifying local AI imports (torch + insightface + onnxruntime)..."
conda run -n "$ENV_NAME" python -c "import insightface, onnxruntime as ort, torch; print('torch', torch.__version__); print('insightface', insightface.__version__); print('onnxruntime', ort.__version__)"
popd >/dev/null

cat <<EOF

Setup complete.

Start worker with:
  conda run -n $ENV_NAME imganalyzer run-distributed-worker \\
    --coordinator http://<COORDINATOR_IP>:8765/jsonrpc \\
    --worker-id worker-01

Then requeue failed jobs on the coordinator:
  - UI: click "Retry failed"
  - CLI: imganalyzer run --retry-failed
EOF
