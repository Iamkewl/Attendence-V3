[CmdletBinding()]
param(
    [Parameter()]
    [switch]$SkipInstall,

    [Parameter()]
    [switch]$UseExistingEnvironment
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Assert-CommandExists {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    if (-not (Get-Command -Name $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' is not available in PATH."
    }
}

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,

        [Parameter()]
        [string[]]$ArgumentList = @(),

        [Parameter(Mandatory = $true)]
        [string]$FailureMessage
    )

    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "$FailureMessage (exit code: $LASTEXITCODE)."
    }
}

function Get-ComposeInvocation {
    $dockerCompose = Get-Command -Name 'docker-compose' -ErrorAction SilentlyContinue
    if ($dockerCompose) {
        return @{
            FilePath = 'docker-compose'
            Prefix = @()
        }
    }

    $docker = Get-Command -Name 'docker' -ErrorAction SilentlyContinue
    if (-not $docker) {
        throw "Neither 'docker-compose' nor 'docker' is available in PATH."
    }

    & docker compose version *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker is available but 'docker compose' is not. Install Docker Compose or the Docker Compose plugin."
    }

    return @{
        FilePath = 'docker'
        Prefix = @('compose')
    }
}

function Resolve-LauncherShell {
    $pwsh = Get-Command -Name 'pwsh' -ErrorAction SilentlyContinue
    if ($pwsh) {
        return 'pwsh'
    }

    return 'powershell'
}

function Convert-ToSingleQuotedLiteral {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    return "'$($Value -replace "'", "''")'"
}

function Set-DefaultEnvVar {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    $existing = [System.Environment]::GetEnvironmentVariable($Name, 'Process')
    if ([string]::IsNullOrWhiteSpace($existing)) {
        [System.Environment]::SetEnvironmentVariable($Name, $Value, 'Process')
        Write-Host "Using default $Name=$Value"
    }
}

function Set-ProcessEnvVar {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    [System.Environment]::SetEnvironmentVariable($Name, $Value, 'Process')
    Write-Host "Using local $Name=$Value"
}

Assert-CommandExists -Name 'python'

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = (Resolve-Path -Path (Join-Path -Path $scriptRoot -ChildPath '..')).Path
$backendDir = (Resolve-Path -Path (Join-Path -Path $projectRoot -ChildPath 'backend')).Path
$composeFile = Join-Path -Path $projectRoot -ChildPath 'docker-compose.yml'

if (-not (Test-Path -Path $composeFile -PathType Leaf)) {
    throw "Compose file not found at: $composeFile"
}

if (-not (Test-Path -Path (Join-Path -Path $backendDir -ChildPath 'alembic.ini') -PathType Leaf)) {
    throw "Alembic configuration not found in backend directory: $backendDir"
}

if ($UseExistingEnvironment) {
    Set-DefaultEnvVar -Name 'ATTENDANCE_DATABASE_URL' -Value 'postgresql+asyncpg://attendance:attendance@localhost:5432/attendance'
    Set-DefaultEnvVar -Name 'ATTENDANCE_REDIS_URL' -Value 'redis://localhost:6379/0'
    Set-DefaultEnvVar -Name 'ATTENDANCE_JWT_SECRET' -Value 'dev-only-change-me'
    Set-DefaultEnvVar -Name 'ATTENDANCE_ALLOWED_ORIGINS' -Value 'http://localhost:5173,http://localhost:3000,http://localhost:8000'
} else {
    Set-ProcessEnvVar -Name 'ATTENDANCE_DATABASE_URL' -Value 'postgresql+asyncpg://attendance:attendance@localhost:5432/attendance'
    Set-ProcessEnvVar -Name 'ATTENDANCE_REDIS_URL' -Value 'redis://localhost:6379/0'
    Set-ProcessEnvVar -Name 'ATTENDANCE_JWT_SECRET' -Value 'dev-only-change-me'
    Set-ProcessEnvVar -Name 'ATTENDANCE_ALLOWED_ORIGINS' -Value 'http://localhost:5173,http://localhost:3000,http://localhost:8000'
}

$compose = Get-ComposeInvocation
$composeArgs = @() + $compose.Prefix + @('-f', $composeFile, 'up', '-d', 'postgres', 'redis')

Write-Host '[1/5] Starting postgres and redis containers...'
# If you switched from a standard postgres image to pgvector/pgvector and reused
# an old local database volume, PostgreSQL can fail with a data-directory mismatch.
Write-Warning "If PostgreSQL reports a data-directory version mismatch after moving to pgvector, remove old volumes with 'docker compose down -v' (or 'docker-compose down -v') and start again."
Invoke-NativeChecked -FilePath $compose.FilePath -ArgumentList $composeArgs -FailureMessage 'Failed to start postgres and redis containers'

if ($SkipInstall) {
    Write-Host '[2/5] Skipping dependency install (-SkipInstall was set).'
} else {
    Write-Host '[2/5] Installing project dependencies with editable install...'
    Push-Location -Path $projectRoot
    try {
        Invoke-NativeChecked -FilePath 'python' -ArgumentList @('-m', 'pip', 'install', '-e', '.') -FailureMessage 'Dependency installation failed'
    } finally {
        Pop-Location
    }
}

Write-Host '[3/5] Running Alembic migrations...'
Push-Location -Path $backendDir
try {
    Invoke-NativeChecked -FilePath 'python' -ArgumentList @('-m', 'alembic', 'upgrade', 'head') -FailureMessage 'Alembic migration failed'
} finally {
    Pop-Location
}

$shellExe = Resolve-LauncherShell
$backendLiteral = Convert-ToSingleQuotedLiteral -Value $backendDir

Write-Host '[4/5] Launching FastAPI (uvicorn) in a new terminal window...'
$uvicornCommand = "Set-Location -LiteralPath $backendLiteral; python -m uvicorn app.main:app --reload --port 8000"
Start-Process -FilePath $shellExe -ArgumentList @('-NoExit', '-Command', $uvicornCommand) -WorkingDirectory $backendDir | Out-Null

Write-Host '[5/5] Launching Celery worker in a new terminal window...'
$celeryCommand = "Set-Location -LiteralPath $backendLiteral; python -m celery -A app.worker.celery_app worker --loglevel=info --pool=solo"
Start-Process -FilePath $shellExe -ArgumentList @('-NoExit', '-Command', $celeryCommand) -WorkingDirectory $backendDir | Out-Null

Write-Host 'Local development startup completed. API and worker terminals are running separately.'