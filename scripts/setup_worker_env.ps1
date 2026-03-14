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
function Assert-LastExit([string]$Step) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE."
    }
}
Require-Command 'conda'
Require-Command 'git'

# ── Ensure cache directories are writable ────────────────────────────────────
Write-Host "==> Checking cache directory permissions..."
$CacheBase = Join-Path $HOME '.cache'
if (-not (Test-Path $CacheBase)) {
    New-Item -ItemType Directory -Path $CacheBase -Force | Out-Null
}
$HfHome = Join-Path $CacheBase 'huggingface'
$ModelCache = Join-Path $CacheBase 'imganalyzer'
New-Item -ItemType Directory -Path $HfHome -Force | Out-Null
New-Item -ItemType Directory -Path $ModelCache -Force | Out-Null

$env:HF_HOME = $HfHome
$env:IMGANALYZER_MODEL_CACHE = $ModelCache
Write-Host "  HF_HOME=$HfHome"
Write-Host "  IMGANALYZER_MODEL_CACHE=$ModelCache"

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
Assert-LastExit "Upgrade pip/setuptools/wheel"

# On Windows with an NVIDIA GPU, install PyTorch with CUDA from the official
# PyTorch index to get the latest GPU-enabled wheels (2.5+).
# First, remove any conda-installed torch to avoid stale libtorch DLL conflicts.
$CondaPrefixEnv = (conda info --base) + "\envs\$EnvName"
$StaleLibs = Get-ChildItem -Path "$CondaPrefixEnv\Library\lib" -Filter "torch*.lib" -ErrorAction SilentlyContinue
if ($StaleLibs) {
    Write-Host "==> Removing stale conda-installed torch libs to avoid conflicts..."
    conda run -n $EnvName conda remove --force pytorch torchvision torchaudio -y 2>$null
    Remove-Item -Force "$CondaPrefixEnv\Library\lib\torch*.lib" -ErrorAction SilentlyContinue
    Remove-Item -Force "$CondaPrefixEnv\Library\bin\torch*.dll" -ErrorAction SilentlyContinue
}
Write-Host "==> Installing PyTorch with CUDA support..."
conda run -n $EnvName python -m pip install "torch>=2.5" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
Assert-LastExit "Install PyTorch CUDA wheels"

$Extras = "local-ai"
Write-Host "==> Installing editable package with extras: [$Extras]"
conda run -n $EnvName python -m pip install -e ".[$Extras]"
Assert-LastExit "Install imganalyzer editable package with extras"

# ── Verify imports ───────────────────────────────────────────────────────────
Write-Host "==> Verifying local AI imports (torch + unipercept + insightface + onnxruntime)..."
conda run -n $EnvName python -c @"
import torch, numpy as np
print('torch', torch.__version__, '/ numpy', np.__version__)
_ = torch.tensor([1.0])
if torch.cuda.is_available():
    print('CUDA', torch.version.cuda, '/', torch.cuda.get_device_name(0))
else:
    print('WARNING: CUDA not available - GPU acceleration disabled')
import transformers
print('transformers', transformers.__version__)
import unipercept_reward
print('unipercept_reward ok')
import open_clip
print('open_clip ok')
import insightface, onnxruntime as ort
print('insightface', insightface.__version__)
print('onnxruntime', ort.__version__)
"@
Assert-LastExit "Verify local AI imports"

Write-Host "==> Running capability probe..."
conda run -n $EnvName python -c @"
from imganalyzer.pipeline.distributed_worker import _probe_available_modules
modules = _probe_available_modules()
print('Supported modules:', ', '.join(modules))
from imganalyzer.db.repository import ALL_MODULES
missing = sorted(set(ALL_MODULES) - set(modules))
if missing:
    print('WARNING: Unavailable modules:', ', '.join(missing))
"@
Assert-LastExit "Run capability probe"

Pop-Location

# ── Done ─────────────────────────────────────────────────────────────────────
$CondaPrefix = (conda info --base) + "\envs\$EnvName"
$Hostname = [System.Net.Dns]::GetHostName()

Write-Host @"

Setup complete.

Start worker with:
  conda activate $EnvName
  `$env:HF_HOME = '$HfHome'
  `$env:IMGANALYZER_MODEL_CACHE = '$ModelCache'
  imganalyzer run-distributed-worker ``
    --coordinator http://<COORDINATOR_IP>:8765/jsonrpc ``
    --auto-update

Or run directly without activating:
  `$env:HF_HOME = '$HfHome'
  `$env:IMGANALYZER_MODEL_CACHE = '$ModelCache'
  & "$CondaPrefix\python.exe" -m imganalyzer.cli run-distributed-worker ``
    --coordinator http://<COORDINATOR_IP>:8765/jsonrpc ``
    --auto-update

Worker ID defaults to hostname ($Hostname).
Override with --worker-id <name> if needed.
"@
