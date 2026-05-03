# GPU Deployment and Secure Local Bridging

This document describes the Phase 3.5 production workflow for running Triton on the remote A10 server while keeping local Windows workers connected through SSH tunnels.

## Architecture

- Local machine (Windows): Celery workers and backend services.
- Remote machine (Linux + NVIDIA A10): Triton Inference Server in Docker.
- Connectivity:
  - Triton runs on remote ports 8000 (HTTP), 8001 (gRPC), 8002 (metrics).
  - SSH tunnel maps remote 8000/8001 to local 8000/8001.
  - Local workers use ATTENDANCE_TRITON_URL=127.0.0.1:8001.

## Prerequisites

### Windows

- OpenSSH client available in PATH (`ssh`, `scp`).
- Optional for delta sync mode: WSL with `rsync` installed.

### Remote Linux A10 server

- Docker Engine and Docker Compose plugin.
- NVIDIA drivers and `nvidia-container-toolkit` configured for Docker.

Recommended remote validation commands:

```bash
docker --version
docker compose version
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## 1) Deploy the Triton model repository to the remote server

From the project root on Windows:

```powershell
.\scripts\Deploy-Models.ps1 -RemoteUser mlops -RemoteHost 10.50.0.12
```

Default destination on remote host:

- `/opt/attendance/triton/model_repository`

Optional rsync mode (faster for repeated deployments):

```powershell
.\scripts\Deploy-Models.ps1 -RemoteUser mlops -RemoteHost 10.50.0.12 -TransferTool rsync
```

## 2) Copy and run the remote Triton compose stack

Copy compose file to remote host:

```powershell
scp -P 22 .\infra\docker-compose.gpu.yml mlops@10.50.0.12:/opt/attendance/triton/docker-compose.gpu.yml
```

Start Triton on remote host:

```powershell
ssh -p 22 mlops@10.50.0.12 "cd /opt/attendance/triton && docker compose -f docker-compose.gpu.yml pull && docker compose -f docker-compose.gpu.yml up -d"
```

Check remote status:

```powershell
ssh -p 22 mlops@10.50.0.12 "docker compose -f /opt/attendance/triton/docker-compose.gpu.yml ps"
ssh -p 22 mlops@10.50.0.12 "docker logs --tail=200 attendance-triton"
```

## 3) Start the local SSH tunnel for Triton

From the project root on Windows:

```powershell
.\scripts\Start-TritonTunnel.ps1 -RemoteUser mlops -RemoteHost 10.50.0.12
```

This starts a background SSH process with forwards:

- `127.0.0.1:8001 -> remote localhost:8001` (gRPC)
- `127.0.0.1:8000 -> remote localhost:8000` (HTTP)

To run in the foreground (debugging mode):

```powershell
.\scripts\Start-TritonTunnel.ps1 -RemoteUser mlops -RemoteHost 10.50.0.12 -Foreground
```

## 4) Point local workers at tunneled Triton

Set runtime environment for local backend/worker shell:

```powershell
$env:ATTENDANCE_TRITON_URL = "127.0.0.1:8001"
$env:ATTENDANCE_TRITON_SSL_ENABLED = "false"
$env:ATTENDANCE_TRITON_YOLO_MODEL_NAME = "yolov12"
$env:ATTENDANCE_TRITON_YOLO_INPUT_NAME = "INPUT__0"
$env:ATTENDANCE_TRITON_YOLO_OUTPUT_NAME = "OUTPUT__0"
$env:ATTENDANCE_TRITON_LVFACE_MODEL_NAME = "lvface"
$env:ATTENDANCE_TRITON_LVFACE_INPUT_NAME = "INPUT__0"
$env:ATTENDANCE_TRITON_LVFACE_LIVENESS_OUTPUT_NAME = "LIVENESS__0"
$env:ATTENDANCE_TRITON_LVFACE_EMBEDDING_OUTPUT_NAME = "EMBEDDING__1"
```

## 5) Validate local bridge end-to-end

After tunnel startup:

```powershell
curl.exe -s http://127.0.0.1:8000/v2/health/live
curl.exe -s http://127.0.0.1:8000/v2/health/ready
```

Expected response body for each endpoint: `OK`

## Operational Notes

- Remote compose file binds Triton ports to loopback only (`127.0.0.1`) to reduce exposure.
- Keep key-based SSH auth enabled for unattended tunnel and deployment usage.
- Use `-TransferTool rsync` for frequent model updates and exact mirror behavior.
