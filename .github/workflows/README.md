# GitHub Actions Workflows

## backend-tests.yml — Backend Smoke Tests

Runs the Attendance v3 backend test suite (unit + API integration + Celery task flows)
on every pull request and every push to `main`/`master`. Target wall clock: ~75 s.

### What it does

1. Spins up ephemeral `pgvector/pgvector:pg16` and `redis:7-alpine` service containers.
2. Installs the project in editable mode (`pip install -e "project/[dev]"`).
3. Applies Alembic migrations against the ephemeral test database.
4. Runs pytest with coverage (gate: `--cov-fail-under=70`).
5. Uploads `coverage.xml` as the `coverage-report` artifact (always, even on failure).

### Repo settings required

Nothing. Service containers are ephemeral and fully self-contained. The JWT secret
in the workflow file (`ci-test-secret-not-for-production-use-only`) is test-only and
is not a production credential — no repository secret configuration is needed.

### Reproducing the run locally

Start Postgres and Redis via Docker:

```bash
docker compose -f project/docker-compose.yml up -d db redis
```

Export the same environment variables used by the workflow, then run:

```bash
pip install -e "project/[dev]"
cd project/backend && alembic upgrade head && cd ../..
cd project && pytest -v --tb=short -m "not slow" \
  --cov=project/backend/app --cov-report=term --cov-fail-under=70
```

The `FakeTritonGrpcClient` fixture intercepts all calls to `ATTENDANCE_TRITON_URL`;
no real Triton server is contacted during the test run.
