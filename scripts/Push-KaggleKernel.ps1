<#
.SYNOPSIS
    Push the nightly inference kernel to Kaggle and optionally wait for completion.

.DESCRIPTION
    Local equivalent of what the GitHub Actions nightly workflow does:
    materializes a temp directory from notebooks/ with KAGGLE_USERNAME
    substitutions, syncs the baseline dataset from tests/inference_baseline.json,
    pushes the kernel, and (if -Watch) polls until done + downloads outputs.

    Use this for iterating on the kernel script without burning GitHub Actions
    minutes or waiting for the cron.

.PARAMETER KaggleUsername
    Your Kaggle username (lowercase). Defaults to $env:KAGGLE_USERNAME.

.PARAMETER Watch
    Poll kernel status every 30s up to 30 min, then download outputs to
    .kaggle-output/ in the repo root.

.PARAMETER SkipBaselineSync
    Don't bump the baselines dataset. Useful if you know the in-repo
    baseline already matches what's on Kaggle, or for first-time pushes
    where the baselines dataset doesn't exist yet.

.EXAMPLE
    $env:KAGGLE_USERNAME = 'iamkewl'
    .\scripts\Push-KaggleKernel.ps1 -Watch

.EXAMPLE
    .\scripts\Push-KaggleKernel.ps1 -KaggleUsername iamkewl -SkipBaselineSync
#>

[CmdletBinding()]
param(
    [Parameter()]
    [string]$KaggleUsername = $env:KAGGLE_USERNAME,

    [Parameter()]
    [switch]$Watch,

    [Parameter()]
    [switch]$SkipBaselineSync
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($KaggleUsername)) {
    throw "KaggleUsername not provided. Set -KaggleUsername or $env:KAGGLE_USERNAME."
}

if (-not (Get-Command kaggle -ErrorAction SilentlyContinue)) {
    throw "kaggle CLI not found. Install with: pip install kaggle"
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$kernelSourceDir = Join-Path $repoRoot 'notebooks'
$baselineSourcePath = Join-Path $repoRoot 'tests\inference_baseline.json'
$baselineDatasetDir = Join-Path $repoRoot 'infra\kaggle_baselines'

# ---------------------------------------------------------------------------
# Helper: copy a file replacing KAGGLE_USERNAME with the real username
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
# Sync baselines dataset (unless skipped)
# ---------------------------------------------------------------------------

if (-not $SkipBaselineSync) {
    Write-Host "[+] Syncing baselines dataset from $baselineSourcePath ..."

    $tmpBaselineDir = New-Item -ItemType Directory -Path ([System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), "kaggle-baselines-$([Guid]::NewGuid().ToString('N'))")) -Force
    try {
        Copy-WithSubstitution -Source (Join-Path $baselineDatasetDir 'dataset-metadata.json') `
                              -Destination (Join-Path $tmpBaselineDir.FullName 'dataset-metadata.json') `
                              -Username $KaggleUsername
        Copy-Item -LiteralPath $baselineSourcePath -Destination (Join-Path $tmpBaselineDir.FullName 'inference_baseline.json')

        Write-Host "    kaggle datasets version -p $($tmpBaselineDir.FullName)"
        # If no changes since last version, Kaggle errors with "no changes" - that's harmless.
        & kaggle datasets version -p $tmpBaselineDir.FullName -m "Sync from repo $(Get-Date -Format 'yyyy-MM-dd HH:mm:ssZ')" 2>&1 | Tee-Object -Variable datasetOutput | Out-Host
        if ($LASTEXITCODE -ne 0) {
            $msgText = ($datasetOutput | Out-String)
            if ($msgText -match 'no changes' -or $msgText -match 'No files to upload') {
                Write-Host "    (baseline already up to date)"
            } elseif ($msgText -match 'not found' -or $msgText -match '404') {
                Write-Host "    Dataset doesn't exist yet -- creating ..."
                & kaggle datasets create -p $tmpBaselineDir.FullName
                if ($LASTEXITCODE -ne 0) {
                    throw "kaggle datasets create failed (exit $LASTEXITCODE)"
                }
            } else {
                throw "kaggle datasets version failed (exit $LASTEXITCODE)"
            }
        }
    } finally {
        Remove-Item -Recurse -Force -LiteralPath $tmpBaselineDir.FullName -ErrorAction SilentlyContinue
    }
}

# ---------------------------------------------------------------------------
# Materialize kernel dir with username substitution and push
# ---------------------------------------------------------------------------

Write-Host "[+] Preparing kernel push directory ..."
$tmpKernelDir = New-Item -ItemType Directory -Path ([System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), "kaggle-kernel-$([Guid]::NewGuid().ToString('N'))")) -Force
try {
    Copy-WithSubstitution -Source (Join-Path $kernelSourceDir 'kernel-metadata.json') `
                          -Destination (Join-Path $tmpKernelDir.FullName 'kernel-metadata.json') `
                          -Username $KaggleUsername
    Copy-Item -LiteralPath (Join-Path $kernelSourceDir 'nightly_inference.py') `
              -Destination (Join-Path $tmpKernelDir.FullName 'nightly_inference.py')

    Write-Host "[+] kaggle kernels push -p $($tmpKernelDir.FullName)"
    & kaggle kernels push -p $tmpKernelDir.FullName
    if ($LASTEXITCODE -ne 0) {
        throw "kaggle kernels push failed (exit $LASTEXITCODE)"
    }
} finally {
    Remove-Item -Recurse -Force -LiteralPath $tmpKernelDir.FullName -ErrorAction SilentlyContinue
}

$kernelSlug = "$KaggleUsername/attendence-v3-nightly-inference"
Write-Host ""
Write-Host "[OK] Kernel pushed: https://www.kaggle.com/code/$KaggleUsername/attendence-v3-nightly-inference"

if (-not $Watch) {
    Write-Host ""
    Write-Host "Skipping wait (-Watch not set)."
    Write-Host "Check status later: kaggle kernels status $kernelSlug"
    return
}

# ---------------------------------------------------------------------------
# Poll status until done
# ---------------------------------------------------------------------------

Write-Host ""
Write-Host "[+] Polling kernel status (max 30 min) ..."

$maxIterations = 60   # 60 * 30s = 30 min
$iteration = 0
$terminal = $null
while ($iteration -lt $maxIterations) {
    Start-Sleep -Seconds 30
    $iteration += 1
    $statusOutput = & kaggle kernels status $kernelSlug 2>&1 | Out-String
    Write-Host "    [$iteration/$maxIterations] $($statusOutput.Trim())"
    if ($statusOutput -match 'has status "complete"') {
        $terminal = 'complete'
        break
    }
    if ($statusOutput -match 'has status "error"') {
        $terminal = 'error'
        break
    }
    if ($statusOutput -match 'has status "cancelAcknowledged"' -or $statusOutput -match 'has status "cancelRequested"') {
        $terminal = 'cancelled'
        break
    }
}

if ($null -eq $terminal) {
    throw "Timed out after $($maxIterations * 30 / 60) minutes waiting for kernel."
}

if ($terminal -ne 'complete') {
    throw "Kernel terminated with status: $terminal"
}

# ---------------------------------------------------------------------------
# Download outputs
# ---------------------------------------------------------------------------

$outputDir = Join-Path $repoRoot '.kaggle-output'
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
Write-Host ""
Write-Host "[+] Downloading outputs to $outputDir ..."
& kaggle kernels output $kernelSlug -p $outputDir
if ($LASTEXITCODE -ne 0) {
    throw "kaggle kernels output failed (exit $LASTEXITCODE)"
}

$reportPath = Join-Path $outputDir 'inference_current.json'
if (Test-Path $reportPath) {
    $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
    Write-Host ""
    Write-Host "==============================================="
    Write-Host "  FINAL_STATUS: $($report.status)" -ForegroundColor $(if ($report.status -eq 'PASS' -or $report.status -eq 'BASELINE_GENERATED') { 'Green' } else { 'Red' })
    Write-Host "==============================================="
    if ($report.status -eq 'FAIL') {
        exit 1
    }
} else {
    Write-Warning "Output file not found at $reportPath"
}
