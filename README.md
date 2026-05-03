# Attendance V3 (Project Workspace)

This folder contains the shareable software stack for the Attendance V3 system.
It is intentionally separated from the agent control plane.

## What This Project Contains

- Backend API: FastAPI application with authentication, attendance domain services, and realtime endpoints.
- Worker: Celery worker for asynchronous inference and attendance aggregation jobs.
- Database and cache: PostgreSQL (pgvector image) and Redis via Docker Compose.
- Frontend: React + Vite single-page app.
- Inference tooling: model setup and remote Triton deployment/tunnel scripts.

## Repository Layout

```text
project/
|- backend/                # FastAPI app, Alembic migrations, Celery worker code
|- frontend/               # React + Vite frontend
|- scripts/                # Local dev, model setup, tunnel, and deploy scripts
|- infra/                  # Triton GPU compose stack and model repository
|- docker-compose.yml      # Local postgres + redis services
|- pyproject.toml          # Python package metadata and test/dev configuration
```

## Prerequisites

- Python 3.12+
- Node.js 20+
- Docker Desktop (or Docker Engine + Compose)
- PowerShell (for the provided automation scripts on Windows)

Optional for GPU deployment flow:

- OpenSSH client (`ssh`, `scp`)
- WSL with `rsync` if you want delta model sync mode

## Quick Start (Windows, Recommended)

From the project root:

```powershell
.\scripts\Start-LocalDev.ps1
```

What this does:

1. Starts Postgres and Redis containers.
2. Installs Python dependencies (`python -m pip install -e .`) unless `-SkipInstall` is used.
3. Runs Alembic migrations.
4. Launches backend API (`uvicorn`) in a new terminal.
5. Launches Celery worker in a new terminal.

Then start the frontend in another terminal:

```powershell
Set-Location .\frontend
npm install
npm run dev
```

Default local URLs:

- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:8000`
- OpenAPI docs: `http://localhost:8000/docs`
- Health endpoint: `http://localhost:8000/healthz`

## Manual Setup (If Not Using Start-LocalDev.ps1)

From the project root:

```powershell
docker compose -f .\docker-compose.yml up -d postgres redis
python -m pip install -e .
```

Set required backend environment variables in your shell:

```powershell
$env:ATTENDANCE_DATABASE_URL = "postgresql+asyncpg://attendance:attendance@localhost:5432/attendance"
$env:ATTENDANCE_REDIS_URL = "redis://localhost:6379/0"
$env:ATTENDANCE_JWT_SECRET = "dev-only-change-me"
$env:ATTENDANCE_ALLOWED_ORIGINS = "http://localhost:5173,http://localhost:3000,http://localhost:8000"
```

Run migrations:

```powershell
Set-Location .\backend
python -m alembic upgrade head
```

Run backend API:

```powershell
Set-Location .\backend
python -m uvicorn app.main:app --reload --port 8000
```

Run Celery worker:

```powershell
Set-Location .\backend
python -m celery -A app.worker.celery_app worker --loglevel=info --pool=solo
```

Run frontend:

```powershell
Set-Location .\frontend
npm install
npm run dev
```

## Command Reference

### Local Development

```powershell
# Start local stack and launch backend + worker in new terminals
.\scripts\Start-LocalDev.ps1

# Reuse existing shell variables instead of overriding process env vars
.\scripts\Start-LocalDev.ps1 -UseExistingEnvironment

# Skip editable pip install (faster if already installed)
.\scripts\Start-LocalDev.ps1 -SkipInstall
```

### Model Setup and Inference Assets

```powershell
# Export/setup YOLOv12 + LVFace ONNX models into infra/triton/model_repository
python .\scripts\setup_models.py

# Keep temporary workspace for debugging setup_models internals
python .\scripts\setup_models.py --keep-temp
```

### Remote GPU and Triton Operations

```powershell
# Deploy local model repository to remote host (scp mode)
.\scripts\Deploy-Models.ps1 -RemoteUser mlops -RemoteHost 10.50.0.12

# Deploy using rsync mode (via WSL)
.\scripts\Deploy-Models.ps1 -RemoteUser mlops -RemoteHost 10.50.0.12 -TransferTool rsync

# Start resilient local SSH tunnel to remote Triton HTTP/gRPC ports
.\scripts\Start-TritonTunnel.ps1 -RemoteUser mlops -RemoteHost 10.50.0.12

# Run tunnel in foreground for debugging
.\scripts\Start-TritonTunnel.ps1 -RemoteUser mlops -RemoteHost 10.50.0.12 -Foreground
```

### Frontend Build and Lint

```powershell
Set-Location .\frontend
npm run lint
npm run build
npm run preview
```

### Container Lifecycle

```powershell
# Stop local infra containers
docker compose -f .\docker-compose.yml down

# Stop and remove local volumes (useful when resetting local DB state)
docker compose -f .\docker-compose.yml down -v
```

## Environment Variables

Required for backend startup:

- `ATTENDANCE_DATABASE_URL`
- `ATTENDANCE_REDIS_URL`
- `ATTENDANCE_JWT_SECRET`
- `ATTENDANCE_ALLOWED_ORIGINS`

Common optional backend/worker runtime variables:

- `ATTENDANCE_JWT_ALGORITHM`
- `ATTENDANCE_JWT_ISSUER`
- `ATTENDANCE_JWT_AUDIENCE`
- `ATTENDANCE_ACCESS_TOKEN_TTL_MINUTES`
- `ATTENDANCE_REFRESH_TOKEN_TTL_DAYS`
- `ATTENDANCE_CELERY_BROKER_URL`
- `ATTENDANCE_CELERY_RESULT_BACKEND`

Triton/inference variables (needed for inference tasks):

- `ATTENDANCE_TRITON_URL`
- `ATTENDANCE_TRITON_SSL_ENABLED`
- `ATTENDANCE_TRITON_REQUEST_TIMEOUT_SECONDS`
- `ATTENDANCE_TRITON_MAX_RETRIES`
- `ATTENDANCE_TRITON_RETRY_BACKOFF_SECONDS`
- `ATTENDANCE_TRITON_YOLO_MODEL_NAME`
- `ATTENDANCE_TRITON_YOLO_INPUT_NAME`
- `ATTENDANCE_TRITON_YOLO_OUTPUT_NAME`
- `ATTENDANCE_TRITON_LVFACE_MODEL_NAME`
- `ATTENDANCE_TRITON_LVFACE_INPUT_NAME`
- `ATTENDANCE_TRITON_LVFACE_LIVENESS_OUTPUT_NAME`
- `ATTENDANCE_TRITON_LVFACE_EMBEDDING_OUTPUT_NAME`

Remote deployment script variables:

- `ATTENDANCE_GPU_REMOTE_USER`
- `ATTENDANCE_GPU_REMOTE_HOST`
- `ATTENDANCE_GPU_MODEL_REPO_PATH`

## How To Test

### 1) Basic Service Smoke Tests

After starting local services:

```powershell
curl.exe -s http://localhost:8000/healthz
```

Expected response:

```json
{"status":"ok"}
```

Open API docs in a browser:

- `http://localhost:8000/docs`

Open frontend in a browser:

- `http://localhost:5173`

### 2) Frontend Quality Checks

```powershell
Set-Location .\frontend
npm run lint
npm run build
```

### 3) Python/Backend Test Runner

Pytest is configured in `pyproject.toml` to use `backend/tests`.

```powershell
python -m pytest
```

If no tests exist yet in `backend/tests`, pytest may report no tests collected.

### 4) Inference Pipeline Local Validation (Optional)

If backend auth and Triton connectivity are already configured, run the webcam upload tester:

```powershell
python .\scripts\run_webcam_test.py --endpoint http://localhost:8000/api/v1/inference/stream --interval-seconds 5
```

Notes:

- The `/api/v1/inference/stream` endpoint requires authentication.
- Set `ATTENDANCE_BEARER_TOKEN` in your shell to attach an `Authorization` header for this script.
- Press `q` in the webcam preview window to stop.

### 5) Triton Bridge Validation (Optional, Remote GPU Flow)

After starting the SSH tunnel:

```powershell
curl.exe -s http://127.0.0.1:8000/v2/health/live
curl.exe -s http://127.0.0.1:8000/v2/health/ready
```

Expected response for both: `OK`

## Troubleshooting

- If migrations fail due stale shell env vars, explicitly reset `ATTENDANCE_DATABASE_URL` and `ATTENDANCE_REDIS_URL` before running Alembic.
- If PostgreSQL fails after image/version changes, clear local volumes with `docker compose down -v` and start again.
- If frontend API calls fail locally, ensure backend is running on port `8000` (Vite proxy is configured for `/api` and `/ws` to `localhost:8000`).
