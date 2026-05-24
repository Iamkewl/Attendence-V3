<#
.SYNOPSIS
    One-shot bootstrap for Attendence-V3's Kaggle nightly inference resources.

.DESCRIPTION
    Handles the manual steps that come AFTER you've added KAGGLE_USERNAME +
    KAGGLE_KEY repo secrets and dropped kaggle.json into ~/.kaggle/:

      1. Stages local model .onnx files into infra/kaggle_dataset/
      2. Materializes both Kaggle Dataset payloads with your username
         substituted, in temp dirs (your repo files stay clean)
      3. Creates the `attendence-v3-models` Kaggle Dataset (or skips if it
         already exists)
      4. Creates the `attendence-v3-baselines` Kaggle Dataset with the
         placeholder JSON (or skips if it exists)
      5. Pushes the kernel for the first time, polls until done, downloads
         the output to .kaggle-output\inference_current.json
      6. Prints the exact next command to commit the new baseline

    Idempotent: re-running after a partial failure is safe. Each Kaggle
    operation checks for prior existence first.

.PARAMETER KaggleUsername
    Your Kaggle username (lowercase). Defaults to $env:KAGGLE_USERNAME.

.PARAMETER SkipDatasets
    Skip dataset creation. Use when you've already uploaded both datasets
    and just want to trigger the kernel push + bootstrap baseline.

.PARAMETER SkipKernel
    Skip the kernel push. Use when you just want to upload/refresh the
    datasets without running an inference.

.EXAMPLE
    $env:KAGGLE_USERNAME = 'iamkewl'
    .\scripts\Initialize-KaggleResources.ps1

.EXAMPLE
    # Re-run after fixing a single dataset that failed to upload
    .\scripts\Initialize-KaggleResources.ps1 -KaggleUsername iamkewl
#>

[CmdletBinding()]
param(
    [Parameter()]
    [string]$KaggleUsername = $env:KAGGLE_USERNAME,

    [Parameter()]
    [switch]$SkipDatasets,

    [Parameter()]
    [switch]$SkipKernel
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-Sub {
    param([string]$Message)
    Write-Host "    $Message" -ForegroundColor DarkGray
}

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------

if ([string]::IsNullOrWhiteSpace($KaggleUsername)) {
    throw "KaggleUsername not provided. Set -KaggleUsername or `$env:KAGGLE_USERNAME."
}
if (-not (Get-Command kaggle -ErrorAction SilentlyContinue)) {
    throw "kaggle CLI not found. Install with: pip install 'kaggle>=1.6.0,<2.0.0'"
}

# Try a cheap auth check (config view just reads the kaggle.json).
& kaggle config view *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Kaggle auth check failed. Make sure ~/.kaggle/kaggle.json exists with correct permissions."
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$modelsSourceDir = Join-Path $repoRoot 'infra\models'
$yoloSrc = Join-Path $modelsSourceDir 'yolo12n.onnx'
$lvfaceSrc = Join-Path $repoRoot 'infra\triton\model_repository\lvface\1\model.onnx'

$kaggleDatasetDir = Join-Path $repoRoot 'infra\kaggle_dataset'
$kaggleBaselinesDir = Join-Path $repoRoot 'infra\kaggle_baselines'

Write-Step "Bootstrapping Kaggle resources for username '$KaggleUsername'"

# ---------------------------------------------------------------------------
# Helper: substitute KAGGLE_USERNAME into a metadata file -> temp file
# ---------------------------------------------------------------------------

function Copy-WithSubstitution {
    param(
        [Parameter(Mandatory)][string]$Source,
        [Parameter(Mandatory)][string]$Destination,
        [Parameter(Mandatory)][string]$Username
    )
    $content = Get-Content -LiteralPath $Source -Raw
    $content = $content -replace 'KAGGLE_USERNAME', $Username
    Set-Content -LiteralPath $Destination -Value $content -Encoding utf8 -NoNewline
}

# ---------------------------------------------------------------------------
# Helper: check if a Kaggle dataset already exists for this user
# ---------------------------------------------------------------------------

function Test-KaggleDatasetExists {
    param([Parameter(Mandatory)][string]$Slug)
    & kaggle datasets status $Slug *> $null
    return ($LASTEXITCODE -eq 0)
}

# ---------------------------------------------------------------------------
# Stage models into infra/kaggle_dataset/
# ---------------------------------------------------------------------------

if (-not $SkipDatasets) {
    Write-Step "Staging model files into infra\kaggle_dataset\"

    if (-not (Test-Path $yoloSrc)) {
        throw "YOLO source not found: $yoloSrc. Run scripts\setup_models.py first."
    }
    if (-not (Test-Path $lvfaceSrc)) {
        throw "LVFace source not found: $lvfaceSrc. Run scripts\setup_models.py first."
    }

    Copy-Item -LiteralPath $yoloSrc -Destination (Join-Path $kaggleDatasetDir 'yolo12n.onnx') -Force
    Write-Sub "yolo12n.onnx -> infra\kaggle_dataset\yolo12n.onnx"
    Copy-Item -LiteralPath $lvfaceSrc -Destination (Join-Path $kaggleDatasetDir 'lvface_model.onnx') -Force
    Write-Sub "lvface model.onnx -> infra\kaggle_dataset\lvface_model.onnx"

    # ---------------------------------------------------------------------------
    # Upload models dataset (skip if already exists)
    # ---------------------------------------------------------------------------

    $modelsSlug = "$KaggleUsername/attendence-v3-models"
    Write-Step "Creating dataset '$modelsSlug' (~446 MB, takes 5-15 min)"

    if (Test-KaggleDatasetExists -Slug $modelsSlug) {
        Write-Sub "Dataset already exists; skipping create. Use 'kaggle datasets version' to update."
    } else {
        $tmp = New-Item -ItemType Directory -Path ([System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), "kaggle-models-$([Guid]::NewGuid().ToString('N'))")) -Force
        try {
            Copy-WithSubstitution `
                -Source (Join-Path $kaggleDatasetDir 'dataset-metadata.json') `
                -Destination (Join-Path $tmp.FullName 'dataset-metadata.json') `
                -Username $KaggleUsername
            Copy-Item -LiteralPath (Join-Path $kaggleDatasetDir 'yolo12n.onnx') -Destination $tmp.FullName
            Copy-Item -LiteralPath (Join-Path $kaggleDatasetDir 'lvface_model.onnx') -Destination $tmp.FullName

            & kaggle datasets create -p $tmp.FullName --dir-mode tar
            if ($LASTEXITCODE -ne 0) {
                throw "kaggle datasets create failed for $modelsSlug (exit $LASTEXITCODE)"
            }
            Write-Sub "Created."
        } finally {
            Remove-Item -Recurse -Force -LiteralPath $tmp.FullName -ErrorAction SilentlyContinue
        }
    }

    # ---------------------------------------------------------------------------
    # Upload baselines dataset (skip if already exists). Initial payload is the
    # placeholder; the workflow will sync the real baseline on each run.
    # ---------------------------------------------------------------------------

    $baselinesSlug = "$KaggleUsername/attendence-v3-baselines"
    Write-Step "Creating dataset '$baselinesSlug' (tiny placeholder)"

    if (Test-KaggleDatasetExists -Slug $baselinesSlug) {
        Write-Sub "Dataset already exists; skipping create."
    } else {
        $tmp = New-Item -ItemType Directory -Path ([System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), "kaggle-baselines-$([Guid]::NewGuid().ToString('N'))")) -Force
        try {
            Copy-WithSubstitution `
                -Source (Join-Path $kaggleBaselinesDir 'dataset-metadata.json') `
                -Destination (Join-Path $tmp.FullName 'dataset-metadata.json') `
                -Username $KaggleUsername
            Copy-Item -LiteralPath (Join-Path $repoRoot 'tests\inference_baseline.json') -Destination (Join-Path $tmp.FullName 'inference_baseline.json')

            & kaggle datasets create -p $tmp.FullName
            if ($LASTEXITCODE -ne 0) {
                throw "kaggle datasets create failed for $baselinesSlug (exit $LASTEXITCODE)"
            }
            Write-Sub "Created."
        } finally {
            Remove-Item -Recurse -Force -LiteralPath $tmp.FullName -ErrorAction SilentlyContinue
        }
    }
} else {
    Write-Step "Skipping dataset creation (-SkipDatasets set)"
}

# ---------------------------------------------------------------------------
# Push kernel for the first time and wait for outputs
# ---------------------------------------------------------------------------

if (-not $SkipKernel) {
    Write-Step "Pushing kernel and waiting for first run (5-10 min)"

    $pushScript = Join-Path $PSScriptRoot 'Push-KaggleKernel.ps1'
    if (-not (Test-Path $pushScript)) {
        throw "Push-KaggleKernel.ps1 missing at $pushScript"
    }

    & $pushScript -KaggleUsername $KaggleUsername -Watch -SkipBaselineSync
    if ($LASTEXITCODE -ne 0) {
        throw "Kernel push/poll failed."
    }

    $outputJson = Join-Path $repoRoot '.kaggle-output\inference_current.json'
    if (Test-Path $outputJson) {
        Write-Step "Bootstrap complete"
        Write-Host ""
        Write-Host "  Output landed at:" -ForegroundColor White
        Write-Host "    $outputJson"
        Write-Host ""
        Write-Host "  Next step: commit it as the new baseline." -ForegroundColor White
        Write-Host ""
        Write-Host "    Copy-Item .\.kaggle-output\inference_current.json .\tests\inference_baseline.json -Force"
        Write-Host "    powershell -File C:\Users\DELL\.agents\scripts\New-AgentPR.ps1 ``"
        Write-Host "        -Slug bootstrap-inference-baseline ``"
        Write-Host "        -Title `"tests: bootstrap nightly inference baseline`" ``"
        Write-Host "        -AgentName Claude -ModelName `"Claude Opus 4.7`""
        Write-Host ""
    } else {
        Write-Warning "Expected output not found at $outputJson."
    }
} else {
    Write-Step "Skipping kernel push (-SkipKernel set)"
}
