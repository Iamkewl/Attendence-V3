# Attendance v3 — Local Demo Stack

This document describes the one-command developer/reviewer environment powered by
`docker-compose.dev.yml`. Everything runs in Docker; only Docker (and `make`) are
required on the host machine.

---

## What the stack contains

| Service    | Image / Build              | Port  | Purpose                                   |
|------------|----------------------------|-------|-------------------------------------------|
| `postgres`  | `pgvector/pgvector:pg16`  | 5432  | Primary database with pgvector extension  |
| `redis`     | `redis:7-alpine`          | 6379  | Celery broker and JWT blocklist           |
| `api`       | `project/backend/Dockerfile.dev` | 8000 | FastAPI app with uvicorn hot-reload  |
| `worker`    | same as api               | —     | Celery worker (inference + aggregation)   |
| `beat`      | same as api               | —     | Celery beat scheduler                     |
| `frontend`  | `node:20-alpine`          | 5173  | Vite dev server                           |

The stack runs in **demo mode** (`ATTENDANCE_TRITON_DEMO_MODE=true`). No GPU or
Triton server is needed. The `docker-compose.gpu.yml` file (remote Triton over SSH
tunnel) is untouched and not started by this configuration.

---

## Quick start

```bash
git clone <repo-url>
cd Attendence-v3
make demo
```

After roughly 30 seconds the terminal prints:

```
Open http://localhost:5173
Login: admin@demo.local / DemoAdmin1!
API docs: http://localhost:8000/docs
```

---

## Demo credentials

| Role  | Email               | Password     |
|-------|---------------------|--------------|
| Admin | admin@demo.local    | DemoAdmin1!  |

Five demo students (`student01@demo.local` … `student05@demo.local`) are also
seeded with passwords `DemoStudent01!` … `DemoStudent05!`.

---

## Available make targets

```
make demo     Build, start, migrate, seed, and print the access URL.
make up       Start all services in detached mode (build if needed).
make down     Stop all services (data is preserved).
make clean    Stop services AND delete the Postgres volume (full reset).
make migrate  Run alembic upgrade head inside the api container.
make seed     Re-run the idempotent demo seed script.
make logs     Tail logs for every service (Ctrl-C to stop).
make test     Run the pytest suite inside the api container.
```

---

## Resetting to a clean state

```bash
make clean   # removes the attendance_pg_data volume
make demo    # re-creates everything from scratch
```

---

## Viewing logs for a single service

```bash
docker compose -f docker-compose.dev.yml logs -f api
docker compose -f docker-compose.dev.yml logs -f worker
docker compose -f docker-compose.dev.yml logs -f frontend
```

---

## API documentation

- OpenAPI (Swagger UI): http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health probe: http://localhost:8000/healthz

---

## Notes

- Backend source (`project/backend`) is bind-mounted into the `api` and `worker`
  containers, so edits are reflected immediately via uvicorn `--reload`.
- Frontend source (`project/frontend`) is bind-mounted into `frontend`; Vite HMR
  handles live updates.
- The named volume `attendance_pg_data` persists between `make up/down` cycles.
  Use `make clean` to wipe it.
- All four Alembic revisions (up to `20260519_0004`) are applied by `make migrate`
  or automatically by `make demo`.
