<#
.SYNOPSIS
Synchronize the local Triton model repository to a remote GPU server over SSH.

.DESCRIPTION
Uploads infra/triton/model_repository from this repository to a remote Linux host.
Supports:
- scp (native Windows OpenSSH)
- rsync (via WSL)

The script creates the remote destination directory if it does not exist.

.PARAMETER RemoteUser
Remote SSH user. Defaults to ATTENDANCE_GPU_REMOTE_USER or 'ubuntu'.

.PARAMETER RemoteHost
Remote SSH host or IP. Defaults to ATTENDANCE_GPU_REMOTE_HOST or 'a10-gpu-server.local'.

.PARAMETER RemotePath
Destination path on remote Linux host.
Defaults to ATTENDANCE_GPU_MODEL_REPO_PATH or '/opt/attendance/triton/model_repository'.

.PARAMETER SshPort
SSH port. Default is 22.

.PARAMETER TransferTool
Transfer engine: scp or rsync. Default is scp.

.PARAMETER IdentityFile
Optional private key file path.

.PARAMETER LocalRepositoryPath
Optional local model repository path.
Default: repository-local path resolved from this script location at '..\\infra\\triton\\model_repository'.

.PARAMETER NoCompression
Disable OpenSSH/scp compression.

.PARAMETER SkipHostKeyCheck
Disable strict host key checking. Use only in controlled environments.

.PARAMETER AllowPasswordAuth
Allow interactive SSH password authentication by disabling BatchMode.

.EXAMPLE
.\scripts\Deploy-Models.ps1 -RemoteUser mlops -RemoteHost 10.50.0.12

.EXAMPLE
.\scripts\Deploy-Models.ps1 -RemoteUser mlops -RemoteHost 10.50.0.12 -TransferTool rsync -IdentityFile C:\Users\DELL\.ssh\attendance_a10
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter()]
    [ValidateNotNullOrEmpty()]
    [string]$RemoteUser = $(if ($env:ATTENDANCE_GPU_REMOTE_USER) { $env:ATTENDANCE_GPU_REMOTE_USER } else { 'ubuntu' }),

    [Parameter()]
    [ValidateNotNullOrEmpty()]
    [string]$RemoteHost = $(if ($env:ATTENDANCE_GPU_REMOTE_HOST) { $env:ATTENDANCE_GPU_REMOTE_HOST } else { 'a10-gpu-server.local' }),

    [Parameter()]
    [ValidateNotNullOrEmpty()]
    [string]$RemotePath = $(if ($env:ATTENDANCE_GPU_MODEL_REPO_PATH) { $env:ATTENDANCE_GPU_MODEL_REPO_PATH } else { '/opt/attendance/triton/model_repository' }),

    [Parameter()]
    [ValidateRange(1, 65535)]
    [int]$SshPort = 22,

    [Parameter()]
    [ValidateSet('scp', 'rsync')]
    [string]$TransferTool = 'scp',

    [Parameter()]
    [string]$IdentityFile,

    [Parameter()]
    [ValidateNotNullOrEmpty()]
    [string]$LocalRepositoryPath = (Join-Path (Split-Path -Parent $PSScriptRoot) 'infra\triton\model_repository'),

    [Parameter()]
    [switch]$NoCompression,

    [Parameter()]
    [switch]$SkipHostKeyCheck,

    [Parameter()]
    [switch]$AllowPasswordAuth
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

function Resolve-AbsolutePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    return (Resolve-Path -Path $Path -ErrorAction Stop).Path
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

function Convert-ToPosixSingleQuoted {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    $singleQuoteEscape = [string]::Concat("'", '"', "'", '"', "'")
    return "'" + ($Value -replace "'", $singleQuoteEscape) + "'"
}

function Convert-ToWslPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if ($Path -match '^[A-Za-z]:\\') {
        $drive = $Path.Substring(0, 1).ToLowerInvariant()
        $suffix = $Path.Substring(2).Replace('\\', '/')
        return "/mnt/$drive$suffix"
    }

    if ($Path.StartsWith('/')) {
        return $Path
    }

    throw "Cannot convert path '$Path' to WSL path format."
}

Assert-CommandExists -Name ssh

if ($TransferTool -eq 'scp') {
    Assert-CommandExists -Name scp
}
else {
    Assert-CommandExists -Name wsl
}

$resolvedLocalRepositoryPath = Resolve-AbsolutePath -Path $LocalRepositoryPath
if (-not (Test-Path -LiteralPath $resolvedLocalRepositoryPath -PathType Container)) {
    throw "Local repository path '$resolvedLocalRepositoryPath' does not exist or is not a directory."
}

$resolvedIdentityFile = $null
if ($IdentityFile) {
    $resolvedIdentityFile = Resolve-AbsolutePath -Path $IdentityFile
}

$remoteTarget = "$RemoteUser@$RemoteHost"
$normalizedRemotePath = $RemotePath.TrimEnd('/')

$sshCommonArgs = @(
    '-p', $SshPort.ToString(),
    '-o', 'ServerAliveInterval=30',
    '-o', 'ServerAliveCountMax=3'
)

if (-not $AllowPasswordAuth) {
    $sshCommonArgs += @('-o', 'BatchMode=yes')
}

if ($SkipHostKeyCheck) {
    $sshCommonArgs += @(
        '-o', 'StrictHostKeyChecking=no',
        '-o', 'UserKnownHostsFile=/dev/null'
    )
}

if ($resolvedIdentityFile) {
    $sshCommonArgs += @('-i', $resolvedIdentityFile)
}

$mkdirCommand = "mkdir -p " + (Convert-ToPosixSingleQuoted -Value $normalizedRemotePath)

if (-not $PSCmdlet.ShouldProcess("${remoteTarget}:$normalizedRemotePath", "Synchronize model repository using $TransferTool")) {
    return
}

Invoke-NativeChecked -FilePath 'ssh' -ArgumentList ($sshCommonArgs + @($remoteTarget, $mkdirCommand)) -FailureMessage 'Failed to create remote destination path'

if ($TransferTool -eq 'scp') {
    $itemsToCopy = @(Get-ChildItem -LiteralPath $resolvedLocalRepositoryPath -Force)
    if ($itemsToCopy.Count -eq 0) {
        throw "Local repository '$resolvedLocalRepositoryPath' is empty."
    }

    $scpBaseArgs = @('-P', $SshPort.ToString())
    if (-not $NoCompression) {
        $scpBaseArgs += '-C'
    }

    if ($SkipHostKeyCheck) {
        $scpBaseArgs += @(
            '-o', 'StrictHostKeyChecking=no',
            '-o', 'UserKnownHostsFile=/dev/null'
        )
    }

    if ($resolvedIdentityFile) {
        $scpBaseArgs += @('-i', $resolvedIdentityFile)
    }

    foreach ($item in $itemsToCopy) {
        $scpArgs = $scpBaseArgs + @(
            '-r',
            $item.FullName,
            "${remoteTarget}:$normalizedRemotePath/"
        )

        Invoke-NativeChecked -FilePath 'scp' -ArgumentList $scpArgs -FailureMessage "Failed to copy '$($item.FullName)'"
    }
}
else {
    Invoke-NativeChecked -FilePath 'wsl' -ArgumentList @('bash', '-lc', 'command -v rsync >/dev/null 2>&1') -FailureMessage 'rsync is not installed in WSL'

    $localRepoWslPath = Convert-ToWslPath -Path $resolvedLocalRepositoryPath

    $identityWslPath = $null
    if ($resolvedIdentityFile) {
        $identityWslPath = Convert-ToWslPath -Path $resolvedIdentityFile
    }

    $rsyncSshTransport = "ssh -p $SshPort"
    if ($identityWslPath) {
        $rsyncSshTransport += " -i $identityWslPath"
    }

    if ($SkipHostKeyCheck) {
        $rsyncSshTransport += ' -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null'
    }

    $rsyncCommand = @(
        'rsync -az --delete --info=progress2',
        '-e',
        (Convert-ToPosixSingleQuoted -Value $rsyncSshTransport),
        (Convert-ToPosixSingleQuoted -Value ($localRepoWslPath.TrimEnd('/') + '/')),
        (Convert-ToPosixSingleQuoted -Value ("${remoteTarget}:$normalizedRemotePath/"))
    ) -join ' '

    Invoke-NativeChecked -FilePath 'wsl' -ArgumentList @('bash', '-lc', $rsyncCommand) -FailureMessage 'rsync transfer failed'
}

Write-Output ("Model repository synchronized to {0}:{1} using {2}." -f $remoteTarget, $normalizedRemotePath, $TransferTool)
