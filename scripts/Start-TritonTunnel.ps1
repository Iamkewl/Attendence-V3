<#
.SYNOPSIS
Start a resilient SSH tunnel from local Windows to remote Triton endpoints.

.DESCRIPTION
Creates local forwards:
- local gRPC port -> remote localhost:8001
- local HTTP port -> remote localhost:8000

By default this script starts ssh.exe in the background and returns process details.

.PARAMETER RemoteUser
Remote SSH user. Defaults to ATTENDANCE_GPU_REMOTE_USER or 'ubuntu'.

.PARAMETER RemoteHost
Remote SSH host or IP. Defaults to ATTENDANCE_GPU_REMOTE_HOST or 'a10-gpu-server.local'.

.PARAMETER SshPort
SSH port. Default is 22.

.PARAMETER IdentityFile
Optional private key file path.

.PARAMETER LocalGrpcPort
Local gRPC listen port. Default is 8001.

.PARAMETER RemoteGrpcPort
Remote gRPC target port. Default is 8001.

.PARAMETER LocalHttpPort
Local HTTP listen port. Default is 8000.

.PARAMETER RemoteHttpPort
Remote HTTP target port. Default is 8000.

.PARAMETER ServerAliveIntervalSeconds
SSH keepalive interval in seconds. Default is 30.

.PARAMETER ServerAliveCountMax
Max missed keepalive probes before disconnect. Default is 3.

.PARAMETER Foreground
Run tunnel in current shell instead of background process.

.PARAMETER Force
Kill existing matching tunnel process(es) before starting a new one.

.PARAMETER SkipHostKeyCheck
Disable strict host key checking. Use only in controlled environments.

.EXAMPLE
.\scripts\Start-TritonTunnel.ps1 -RemoteUser mlops -RemoteHost 10.50.0.12

.EXAMPLE
.\scripts\Start-TritonTunnel.ps1 -RemoteUser mlops -RemoteHost 10.50.0.12 -Foreground
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
    [ValidateRange(1, 65535)]
    [int]$SshPort = 22,

    [Parameter()]
    [string]$IdentityFile,

    [Parameter()]
    [ValidateRange(1, 65535)]
    [int]$LocalGrpcPort = 8001,

    [Parameter()]
    [ValidateRange(1, 65535)]
    [int]$RemoteGrpcPort = 8001,

    [Parameter()]
    [ValidateRange(1, 65535)]
    [int]$LocalHttpPort = 8000,

    [Parameter()]
    [ValidateRange(1, 65535)]
    [int]$RemoteHttpPort = 8000,

    [Parameter()]
    [ValidateRange(5, 3600)]
    [int]$ServerAliveIntervalSeconds = 30,

    [Parameter()]
    [ValidateRange(1, 20)]
    [int]$ServerAliveCountMax = 3,

    [Parameter()]
    [switch]$Foreground,

    [Parameter()]
    [switch]$Force,

    [Parameter()]
    [switch]$SkipHostKeyCheck
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

function Get-MatchingTunnelProcess {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RemoteTarget,

        [Parameter(Mandatory = $true)]
        [string]$GrpcSpec,

        [Parameter(Mandatory = $true)]
        [string]$HttpSpec,

        [Parameter(Mandatory = $true)]
        [int]$Port
    )

    $sshProcesses = Get-CimInstance Win32_Process -Filter "Name = 'ssh.exe'"
    return @(
        $sshProcesses | Where-Object {
            $_.CommandLine -and
            $_.CommandLine -like "*$RemoteTarget*" -and
            $_.CommandLine -like "*-L $GrpcSpec*" -and
            $_.CommandLine -like "*-L $HttpSpec*" -and
            $_.CommandLine -like "*-p $Port*"
        }
    )
}

function Test-PortFree {
    param(
        [Parameter(Mandatory = $true)]
        [int]$Port
    )

    $listeners = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
    return $null -eq $listeners
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

Assert-CommandExists -Name ssh

$resolvedIdentityFile = $null
if ($IdentityFile) {
    $resolvedIdentityFile = Resolve-AbsolutePath -Path $IdentityFile
}

$remoteTarget = "$RemoteUser@$RemoteHost"
$grpcForwardSpec = "${LocalGrpcPort}:localhost:${RemoteGrpcPort}"
$httpForwardSpec = "${LocalHttpPort}:localhost:${RemoteHttpPort}"

$existingTunnelProcesses = @(Get-MatchingTunnelProcess -RemoteTarget $remoteTarget -GrpcSpec $grpcForwardSpec -HttpSpec $httpForwardSpec -Port $SshPort)

if ($existingTunnelProcesses.Count -gt 0) {
    if (-not $Force) {
        $existingIds = ($existingTunnelProcesses | Select-Object -ExpandProperty ProcessId) -join ', '
        Write-Output "Matching tunnel is already active (PID: $existingIds). Use -Force to restart it."
        return
    }

    foreach ($process in $existingTunnelProcesses) {
        Stop-Process -Id $process.ProcessId -Force
    }
}

if (-not (Test-PortFree -Port $LocalGrpcPort)) {
    throw "Local gRPC port $LocalGrpcPort is already in use."
}

if (-not (Test-PortFree -Port $LocalHttpPort)) {
    throw "Local HTTP port $LocalHttpPort is already in use."
}

$sshArgs = @(
    '-N',
    '-p', $SshPort.ToString(),
    '-o', 'BatchMode=yes',
    '-o', 'ExitOnForwardFailure=yes',
    '-o', "ServerAliveInterval=$ServerAliveIntervalSeconds",
    '-o', "ServerAliveCountMax=$ServerAliveCountMax",
    '-L', $grpcForwardSpec,
    '-L', $httpForwardSpec
)

if ($SkipHostKeyCheck) {
    $sshArgs += @(
        '-o', 'StrictHostKeyChecking=no',
        '-o', 'UserKnownHostsFile=/dev/null'
    )
}

if ($resolvedIdentityFile) {
    $sshArgs += @('-i', $resolvedIdentityFile)
}

$sshArgs += $remoteTarget

if (-not $PSCmdlet.ShouldProcess($remoteTarget, 'Start SSH tunnel for Triton')) {
    return
}

if ($Foreground) {
    Invoke-NativeChecked -FilePath 'ssh' -ArgumentList $sshArgs -FailureMessage 'Foreground SSH tunnel failed'
    return
}

$process = Start-Process -FilePath 'ssh' -ArgumentList $sshArgs -WindowStyle Hidden -PassThru
$null = Wait-Process -Id $process.Id -Timeout 2 -ErrorAction SilentlyContinue
$process.Refresh()

if ($process.HasExited) {
    throw 'Background SSH process terminated immediately. Verify SSH connectivity, key auth, and host trust configuration.'
}

[PSCustomObject]@{
    TunnelProcessId = $process.Id
    RemoteTarget = $remoteTarget
    LocalGrpcEndpoint = "127.0.0.1:$LocalGrpcPort"
    LocalHttpEndpoint = "127.0.0.1:$LocalHttpPort"
    RemoteGrpcEndpoint = "127.0.0.1:$RemoteGrpcPort"
    RemoteHttpEndpoint = "127.0.0.1:$RemoteHttpPort"
}
