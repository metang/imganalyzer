<#
.SYNOPSIS
    Bootstrap a Windows distributed-worker environment for imganalyzer.

.DESCRIPTION
    Creates (or reuses) a Conda environment, clones/updates the repo, and
    installs local-AI + cloud-provider dependencies with CUDA GPU support.

.PARAMETER RepoDir
    Path to the imganalyzer repo clone.  Defaults to the parent of this
    script's directory (if inside a checkout) or $HOME\imganalyzer.

.EXAMPLE
    .\setup_worker_env.ps1
    .\setup_worker_env.ps1 -RepoDir D:\Code\imganalyzer
    $env:IMGANALYZER_WORKER_CLOUD_PROVIDER = 'openai'; .\setup_worker_env.ps1
#>
[CmdletBinding()]
param(
    [string]$RepoDir
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ── Defaults (override via env vars) ─────────────────────────────────────────
$EnvName         = if ($env:IMGANALYZER_ENV_NAME)            { $env:IMGANALYZER_ENV_NAME }            else { 'imganalyzer' }
$PythonVersion   = if ($env:IMGANALYZER_PYTHON_VERSION)      { $env:IMGANALYZER_PYTHON_VERSION }      else { '3.12' }
$RepoUrl         = if ($env:IMGANALYZER_REPO_URL)            { $env:IMGANALYZER_REPO_URL }            else { 'https://github.com/metang/imganalyzer.git' }
$CloudProvider   = if ($env:IMGANALYZER_WORKER_CLOUD_PROVIDER) { $env:IMGANALYZER_WORKER_CLOUD_PROVIDER } else { 'copilot' }

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
if (-not $RepoDir) {
    $ParentDir = Split-Path -Parent $ScriptDir
    if (Test-Path (Join-Path $ParentDir '.git')) {
        $RepoDir = $ParentDir
    } else {
        $RepoDir = Join-Path $HOME 'imganalyzer'
    }
}

# ── Pre-flight checks ───────────────────────────────────────────────────────
function Require-Command($Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        Write-Error "Error: '$Name' is required but not installed."
    }
}
Require-Command 'conda'
Require-Command 'git'

$ValidProviders = @('copilot', 'openai', 'anthropic', 'google')
if ($CloudProvider -notin $ValidProviders) {
    Write-Error "Error: Unsupported cloud provider '$CloudProvider'. Use one of: $($ValidProviders -join ', ')"
}

# ── Conda environment ───────────────────────────────────────────────────────
Write-Host "==> Using worker env: $EnvName (python $PythonVersion)"
$EnvList = conda env list 2>&1 | ForEach-Object { ($_ -split '\s+')[0] }
if ($EnvList -contains $EnvName) {
    Write-Host "==> Conda env '$EnvName' already exists."
} else {
    Write-Host "==> Creating conda env '$EnvName'..."
    conda create -n $EnvName "python=$PythonVersion" -y
}

# ── Repository ───────────────────────────────────────────────────────────────
if (Test-Path (Join-Path $RepoDir '.git')) {
    Write-Host "==> Updating existing repo at $RepoDir..."
    git -C $RepoDir fetch --quiet origin
    git -C $RepoDir pull --ff-only
} else {
    Write-Host "==> Cloning repo into $RepoDir..."
    $Parent = Split-Path -Parent $RepoDir
    if ($Parent -and -not (Test-Path $Parent)) { New-Item -ItemType Directory -Path $Parent -Force | Out-Null }
    git clone $RepoUrl $RepoDir
}

# ── Install dependencies ────────────────────────────────────────────────────
Write-Host "==> Installing worker dependencies in env '$EnvName'..."
Push-Location $RepoDir

conda run -n $EnvName python -m pip install -U pip setuptools wheel

# On Windows with an NVIDIA GPU, install PyTorch with CUDA from the official
# PyTorch index to get the latest GPU-enabled wheels.
Write-Host "==> Installing PyTorch with CUDA support..."
conda run -n $EnvName python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

$Extras = "local-ai,$CloudProvider"
Write-Host "==> Installing editable package with extras: [$Extras]"
conda run -n $EnvName python -m pip install -e ".[$Extras]"

# ── Verify imports ───────────────────────────────────────────────────────────
Write-Host "==> Verifying local AI imports (torch + insightface + onnxruntime)..."
conda run -n $EnvName python -c @"
import torch, numpy as np
print('torch', torch.__version__, '/ numpy', np.__version__)
_ = torch.tensor([1.0])
if torch.cuda.is_available():
    print('CUDA', torch.version.cuda, '/', torch.cuda.get_device_name(0))
else:
    print('WARNING: CUDA not available - GPU acceleration disabled')
import insightface, onnxruntime as ort
print('insightface', insightface.__version__)
print('onnxruntime', ort.__version__)
"@

Write-Host "==> Verifying cloud provider import ($CloudProvider)..."
switch ($CloudProvider) {
    'copilot'   { conda run -n $EnvName python -c "import copilot; print('copilot sdk ok')" }
    'openai'    { conda run -n $EnvName python -c "import openai; print('openai ok')" }
    'anthropic' { conda run -n $EnvName python -c "import anthropic; print('anthropic ok')" }
    'google'    { conda run -n $EnvName python -c "from google.cloud import vision; print('google vision ok')" }
}

Pop-Location

# ── Done ─────────────────────────────────────────────────────────────────────
Write-Host @"

Setup complete.

Start worker with:
  conda run -n $EnvName imganalyzer run-distributed-worker `
    --coordinator http://<COORDINATOR_IP>:8765/jsonrpc `
    --worker-id worker-01 `
    --cloud $CloudProvider

Then requeue failed jobs on the coordinator:
  - UI: click "Retry failed"
  - CLI: imganalyzer run --retry-failed
"@
