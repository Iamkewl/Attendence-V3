# Smoke Tests — Local Run Guide

## Required Environment Variables

| Variable | Example | Purpose |
|---|---|---|
| `ATTENDANCE_DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/attendance_test` | PostgreSQL DSN (asyncpg dialect auto-detected) |
| `ATTENDANCE_DATABASE_URL_TEST` | same as above (optional override) | If set, takes priority over `ATTENDANCE_DATABASE_URL` |
| `ATTENDANCE_REDIS_URL` | `redis://localhost:6379/0` | Redis URL for token blocklist + pub/sub |
| `ATTENDANCE_REDIS_URL_TEST` | same (optional override) | If set, takes priority |
| `ATTENDANCE_JWT_SECRET` | `local-dev-secret` | JWT signing key (any string ≥ 16 chars) |
| `ATTENDANCE_ALLOWED_ORIGINS` | `http://localhost:3000` | CORS allow-list (no wildcard) |
| `ATTENDANCE_TRITON_URL` | `fake-host:8001` | Intercepted by FakeTritonGrpcClient — set to any string |

All variables are defaulted inside `conftest.py` so collection works without them set.
Tests that require Postgres or Redis are **skipped** (not failed) when the services are unreachable.

---

## Spin Up Services Locally

### PostgreSQL with pgvector

```bash
docker run -d \
  --name attendance-postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=attendance_test \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### Redis

```bash
docker run -d \
  --name attendance-redis \
  -p 6379:6379 \
  redis:7-alpine
```

---

## Install Dev Dependencies

From the `project/` directory:

```bash
pip install -e ".[dev]"
# or from the backend directory:
pip install -e "backend/[dev]"
```

---

## Run the Smoke Tests

From `project/`:

```bash
export ATTENDANCE_DATABASE_URL="postgresql://postgres:postgres@localhost:5432/attendance_test"
export ATTENDANCE_REDIS_URL="redis://localhost:6379/0"
export ATTENDANCE_JWT_SECRET="local-dev-secret-do-not-use-in-prod"
export ATTENDANCE_ALLOWED_ORIGINS="http://localhost:3000"
export ATTENDANCE_TRITON_URL="fake-host:8001"

pytest -v backend/tests/
```

### Collection-only check (no services required):

```bash
pytest --collect-only backend/tests/ -q
```

### Syntax check:

```bash
python -c "import ast, glob; [ast.parse(open(p).read()) for p in glob.glob('backend/tests/**/*.py', recursive=True)]; print('OK')"
```

---

## Test Files

| File | Flows |
|---|---|
| `test_smoke_auth.py` | 1 — Login sets cookies |
| `test_smoke_inference.py` | 2 — Stream accepts raw tensor; 3 — Eager pipeline persists Sighting; 4 — Task status strips embeddings |
| `test_smoke_realtime.py` | 5 — WS ticket gate; 6 — SSE real-newline regression |
| `test_smoke_aggregation.py` | 7 — Daily aggregation marks PRESENT |

---

## Notes

- Tests run against **real Postgres + Redis**; Triton is fully faked.
- Alembic `upgrade head` runs once per session against the test DB.
- Tables are `TRUNCATE ... RESTART IDENTITY CASCADE` between each test.
- Redis is `FLUSHDB` after each test.
- `ATTENDANCE_TRITON_URL=fake-host:8001` is intercepted by the `fake_triton` fixture; no real Triton needed.
