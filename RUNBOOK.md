# Runbook — Attendance V3

Operational guide for local development, testing, Triton operations, and
common troubleshooting. See [README.md](./README.md) for the full environment
variable reference and command listing.

---

## 1. Bringing the Stack Up

### 1.1 Recommended: Start-LocalDev.ps1

From the project root in a PowerShell terminal:

```powershell
.\scripts\Start-LocalDev.ps1
```

This script performs five steps in order:

1. Starts the `postgres` and `redis` Docker containers from `docker-compose.dev.yml`
   (or `docker-compose.yml` if that is the active file — the script resolves both).
2. Installs Python dependencies with `python -m pip install -e .` from the project
   root (skip with `-SkipInstall` if already installed).
3. Runs Alembic migrations (`python -m alembic upgrade head`) from the `backend/`
   directory.
4. Opens a new PowerShell terminal window running `uvicorn app.main:app --reload --port 8000`.
5. Opens a new PowerShell terminal window running the Celery worker with `--pool=solo`.

Optional flags:

```powershell
# Skip pip install (faster when dependencies have not changed)
.\scripts\Start-LocalDev.ps1 -SkipInstall

# Keep whatever ATTENDANCE_* env vars are already in the shell instead of
# overriding them with the script defaults
.\scripts\Start-LocalDev.ps1 -UseExistingEnvironment
```

Then start the frontend in a separate terminal:

```powershell
Set-Location .\frontend
npm install
npm run dev
```

Default local endpoints after startup:

| Service | URL |
|---|---|
| Frontend | `http://localhost:5173` |
| Backend API | `http://localhost:8000` |
| OpenAPI docs | `http://localhost:8000/docs` |
| Health probe | `http://localhost:8000/healthz` |

### 1.2 Manual Startup (Individual Commands)

Start infrastructure containers:

```powershell
docker compose -f .\docker-compose.dev.yml up -d postgres redis
```

Set required environment variables. Postgres is exposed on host port **15432**
(not 5432) to avoid conflicts with any local Postgres installation:

```powershell
$env:ATTENDANCE_DATABASE_URL = "postgresql+asyncpg://attendance:attendance@localhost:15432/attendance"
$env:ATTENDANCE_REDIS_URL = "redis://localhost:6379/0"
$env:ATTENDANCE_JWT_SECRET = "dev-only-change-me-min-32-chars-needed"
$env:ATTENDANCE_ALLOWED_ORIGINS = "http://localhost:5173,http://localhost:3000,http://localhost:8000"
```

Install dependencies (once, or when `pyproject.toml` changes):

```powershell
python -m pip install -e .
```

Run Alembic migrations (must run from or with `backend/` as the working path so
`alembic.ini` is found):

```powershell
Set-Location .\backend
python -m alembic upgrade head
```

Start the API server:

```powershell
Set-Location .\backend
python -m uvicorn app.main:app --reload --port 8000
```

Start the Celery worker in a separate terminal (from `backend/`):

```powershell
Set-Location .\backend
python -m celery -A app.worker.celery_app worker --loglevel=info --pool=solo
```

Start the Celery Beat scheduler (needed for hourly attendance aggregation and demo
mode; run in a third terminal from `backend/`):

```powershell
Set-Location .\backend
python -m celery -A app.worker.celery_app beat --loglevel=info
```

---

## 2. Bringing the Stack Down

Stop containers but preserve data volumes (Postgres data survives):

```powershell
docker compose -f .\docker-compose.dev.yml down
```

Stop containers and delete all volumes (complete reset — Postgres data is lost):

```powershell
docker compose -f .\docker-compose.dev.yml down -v
```

Use `down -v` after image or schema changes that leave the volume in an
incompatible state. The `attendance_pg_data` named volume is the only persistent
asset.

---

## 3. Running Tests

Prerequisites: Postgres and Redis must be running (the containers, not necessarily
the full API server). Set these environment variables in the PowerShell session
before calling pytest:

```powershell
docker compose -f .\docker-compose.dev.yml up -d postgres redis

$env:ATTENDANCE_DATABASE_URL = "postgresql+asyncpg://attendance:attendance@localhost:15432/attendance"
$env:ATTENDANCE_DATABASE_URL_TEST = $env:ATTENDANCE_DATABASE_URL
$env:ATTENDANCE_REDIS_URL = "redis://localhost:6379/0"
$env:ATTENDANCE_JWT_SECRET = "test-secret-32chars-minimum-needed"
$env:ATTENDANCE_ALLOWED_ORIGINS = "http://localhost:3000"
$env:ATTENDANCE_TRITON_URL = "fake-host:8001"

python -m pytest
```

Expected result: `7 passed`. No GPU or real Triton server is needed; Triton is
replaced in every test by `FakeTritonGrpcClient` via the
`set_triton_client_override()` seam (`backend/tests/conftest.py:163-170`).

The `truncate_tables` autouse fixture runs after every test and truncates all
domain tables with `RESTART IDENTITY CASCADE`. This means test isolation is
automatic — you do not need to clean up data manually between test runs.

The test configuration is in `pyproject.toml` (`[tool.pytest.ini_options]`). The
`testpaths` setting points to `backend/tests`.

---

## 4. Common Troubleshooting

### 4.1 "Future attached to a different loop"

**Symptom:** An async test or `TestClient`-based test raises
`RuntimeError: Task/Future attached to a different loop`.

**Cause:** `asyncio.Redis` and the asyncpg engine bind to the event loop that
created them. The pytest-asyncio session loop and `starlette.testclient.TestClient`
use different loops. Cached module-level singletons carry the old loop reference.

**Fix:** Reset the two singleton references before using `TestClient`:

```python
import app.core.security as security
import app.api.v1.websockets as ws_module

security._redis_client = None
ws_module._pubsub_manager._redis_client = None
```

Do not enter `TestClient` as a context manager (`with TestClient(app) as client`)
when testing WebSocket or SSE routes; this triggers the FastAPI lifespan which
tries to close asyncpg connections on the wrong loop. See
`backend/tests/test_smoke_realtime.py:15-64` for the canonical pattern.

### 4.2 "Cannot connect to Postgres"

**Symptom:** Alembic migration fails, pytest skips DB fixtures, or the API crashes
on startup with a connection refused error.

**Checklist:**

1. Confirm the container is running: `docker compose -f .\docker-compose.dev.yml ps`
2. Confirm the container is healthy (health check retries up to 10 times with 5-second
   intervals): `docker inspect <container_name> --format "{{.State.Health.Status}}"`
3. Confirm the host port is 15432, not 5432:
   `$env:ATTENDANCE_DATABASE_URL` must contain `@localhost:15432/attendance`.
4. If the container is running but failing its health check after a version change,
   the data directory may be incompatible: run `docker compose -f .\docker-compose.dev.yml down -v`
   and restart.

### 4.3 "Token too short" / InsecureKeyLengthWarning

**Symptom:** The API emits a `InsecureKeyLengthWarning` about the JWT secret, or
token validation raises an error.

**Cause:** `ATTENDANCE_JWT_SECRET` must be at least 32 bytes for HS256. Shorter
secrets are rejected by the PyJWT/passlib stack.

**Fix:** Use a secret of 32 characters or more. The test suite uses
`"test-secret-32chars-minimum-needed"` (34 chars). The warning is suppressed in
the test config via `filterwarnings` in `pyproject.toml`, but it will appear in a
dev server that uses a short key.

### 4.4 Email Validator Rejects `.local` TLD

**Symptom:** User creation fails with a validation error about an invalid email
address when using addresses ending in `.local` (e.g. `user@company.local`).

**Cause:** `email-validator` (used by Pydantic's `EmailStr`) rejects `.local` as a
non-deliverable TLD by default.

**Fix:** Use `.example` for synthetic test emails (e.g. `admin@test.example`). The
`.example` TLD is reserved by IANA for documentation and testing and is accepted by
the validator. All fixtures in `backend/tests/conftest.py` use this convention.

### 4.5 Frontend Webcam Preview is Black

**Symptom:** The webcam capture component in the frontend shows a black preview
rectangle instead of the camera feed.

**Cause:** The `WebcamCapture` React component attaches `srcObject` to the video
element inside a `useEffect`. If the effect runs before the DOM element is mounted,
the assignment is a no-op and the preview stays black.

**Fix:** Verify that the effect in `WebcamCapture.jsx` uses a ref guard:
`if (videoRef.current) videoRef.current.srcObject = stream`. A page refresh after
granting camera permissions usually resolves the issue in development builds.

---

## 5. Triton Operations

### 5.1 Model Setup (Local / ONNX Export)

Exports YOLOv12 and LVFace ONNX models and populates `infra/triton/model_repository`:

```powershell
python .\scripts\setup_models.py
```

Keep the temporary workspace for debugging the export process:

```powershell
python .\scripts\setup_models.py --keep-temp
```

Environment variables read by this script: none required beyond a working Python
environment with the model source packages installed.

### 5.2 Deploying Models to a Remote GPU Server

```powershell
# SCP mode (default)
.\scripts\Deploy-Models.ps1 -RemoteUser mlops -RemoteHost 10.50.0.12

# rsync mode via WSL (delta sync, faster for large model updates)
.\scripts\Deploy-Models.ps1 -RemoteUser mlops -RemoteHost 10.50.0.12 -TransferTool rsync
```

Environment variables that `Deploy-Models.ps1` reads (can substitute for parameters):

- `ATTENDANCE_GPU_REMOTE_USER`
- `ATTENDANCE_GPU_REMOTE_HOST`
- `ATTENDANCE_GPU_MODEL_REPO_PATH` — target path on the remote host

### 5.3 SSH Tunnel to Remote Triton

Forwards local ports to the remote Triton gRPC (8001) and HTTP (8000) ports:

```powershell
# Background (returns process details)
.\scripts\Start-TritonTunnel.ps1 -RemoteUser mlops -RemoteHost 10.50.0.12

# Foreground (useful for debugging tunnel connectivity)
.\scripts\Start-TritonTunnel.ps1 -RemoteUser mlops -RemoteHost 10.50.0.12 -Foreground
```

Environment variables read by the tunnel script:

- `ATTENDANCE_GPU_REMOTE_USER` (default: `ubuntu`)
- `ATTENDANCE_GPU_REMOTE_HOST` (default: `a10-gpu-server.local`)

After the tunnel is running, set:

```powershell
$env:ATTENDANCE_TRITON_URL = "localhost:8001"
```

Verify tunnel health:

```powershell
curl.exe -s http://127.0.0.1:8000/v2/health/live
curl.exe -s http://127.0.0.1:8000/v2/health/ready
```

Expected: `OK` for both.

---

## 6. Demo Mode

Demo mode emits synthetic Sighting rows at a configurable interval so the
realtime dashboard shows activity without a real camera or GPU.

Enable it by setting two environment variables before starting the worker and
beat processes:

```powershell
$env:ATTENDANCE_DEMO_MODE = "true"
$env:ATTENDANCE_TRITON_DEMO_MODE = "true"
$env:ATTENDANCE_DEMO_COURSE_ID = "00000000-0000-4000-a000-000000000001"
$env:ATTENDANCE_DEMO_SIGHTING_INTERVAL_SECONDS = "5"
```

What happens:

1. `celery_app.py` reads `ATTENDANCE_DEMO_MODE` at startup and, when enabled, adds
   a `demo-synthetic-sighting` entry to the beat schedule
   (`backend/app/worker/celery_app.py:67-75`).
2. Every `ATTENDANCE_DEMO_SIGHTING_INTERVAL_SECONDS` seconds (default: 5), Celery
   Beat enqueues `demo_emit_sighting`.
3. The task calls `demo_emitter.emit_one_synthetic_sighting()`, which picks a
   random active student from the demo course and inserts a Sighting row via
   `AttendanceService.log_sighting()`.
4. The task is gated on both `ATTENDANCE_DEMO_MODE=true` AND
   `ATTENDANCE_TRITON_DEMO_MODE=true`; if either is false the task returns
   `{"task_state": "SKIPPED"}` and no row is written.

Verify demo mode is active by watching Celery Beat logs for
`demo-synthetic-sighting` schedule entries, or by querying the `sightings` table
after a few intervals:

```powershell
# From a psql session against localhost:15432
SELECT COUNT(*) FROM sightings WHERE camera_id = 'demo-camera-overhead-01';
```

The demo course must already exist in the database (seed it via the admin API or a
migration fixture) and must have at least one enrolled student, otherwise
`emit_one_synthetic_sighting()` returns 0 and logs a warning.

---

## 7. Database Operations

### 7.1 Applying Migrations

Run from the `backend/` directory (Alembic reads `alembic.ini` from the cwd):

```powershell
Set-Location .\backend
python -m alembic upgrade head
```

The `sqlalchemy.url` in `alembic.ini` is overridden at runtime by the
`ATTENDANCE_DATABASE_URL` environment variable (see `backend/alembic/env.py`).

### 7.2 Creating a New Migration

After modifying a model in `backend/app/domain/models/`:

```powershell
Set-Location .\backend
python -m alembic revision --autogenerate -m "short description of change"
```

Review the generated file in `backend/alembic/versions/` before applying. Auto-
generated migrations may miss server defaults, custom types, or index changes;
always inspect and adjust before committing.

Apply the new migration:

```powershell
python -m alembic upgrade head
```

Roll back one step:

```powershell
python -m alembic downgrade -1
```

### 7.3 Truncating Tables Between Tests

The `truncate_tables` autouse fixture in `backend/tests/conftest.py:122-129` runs
after every test and issues:

```sql
TRUNCATE governance_logs, class_session_records, sightings,
         template_audit_logs, student_embeddings, students,
         courses, rooms, users
RESTART IDENTITY CASCADE
```

This is automatic. Do not truncate manually in individual tests.

---

## 8. Logs

All log output goes to **stdout** (no log files by default).

**Access logs:** `configure_json_access_log()` (`backend/app/core/middleware.py`)
installs a structlog-compatible JSON formatter on the `uvicorn.access` logger.
Each request line is emitted as a JSON object containing `method`, `path`,
`status_code`, `duration_ms`, and `request_id` (from `X-Request-ID` header injected
by `RequestIDMiddleware`).

**Application logs:** Python's standard `logging` module, propagated to the root
logger. Log level is controlled by the uvicorn `--log-level` flag (default `info`)
and the Celery `--loglevel` flag.

**Celery task logs:** the `LOGGER = logging.getLogger(__name__)` in
`backend/app/worker/tasks.py` emits warnings for skipped sightings and exceptions
for unexpected failures. These appear in the Celery worker terminal.

To capture structured logs in production, attach a log aggregator (e.g.
Fluentd, Loki) to the container stdout stream.
