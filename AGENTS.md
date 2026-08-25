# AGENTS.md

Orientation for AI agents working in this repo. Prose docs drift — where they
disagree with config, workflows, or scripts, trust the executable source.

## Instruction sources (read before editing)

- `CLAUDE.md` — branch/PR protocol and commit style. Applies to any agent, not just Claude.
- `.opencode/skills/*/SKILL.md` — the deepest repo context. Load the relevant skill first:
  - `attendance-v3-context` — **before editing any backend file** (architecture invariants below).
  - `issue-fix-workflow` / `scope-discipline` / `verification-gate` / `git-pr-hygiene` — for backlog-fix work.
- `ARCHITECTURE.md` (design), `RUNBOOK.md` (ops), `.env.example` (every `ATTENDANCE_*` var).
- Active fix runs live in `fixes/`: `BATCHES.md` is the single source of truth (edited **only**
  by `python3 fixes/bin/next_batch.py`); `NOTES.md` is an append-only outcome log; `issues.json`
  holds per-issue records. `[x]` means PR opened, not merged — confirm against `git log`.

## Stack & layout

- Backend: FastAPI + SQLAlchemy 2 async + Celery under `backend/app/` (Python ≥3.12).
  Layers: `api/` → `services/` → `infrastructure/` → `worker/`. Entrypoint `app.main:app`.
- Frontend: React 19 + Vite + Tailwind v4 in `frontend/` (`src/pages/*.jsx`). No JS test runner;
  verification = `npm run lint && npm run build`.
- Dev infra: Postgres (pgvector) + Redis via `docker-compose.dev.yml`. Local automation scripts
  are PowerShell-first; on Linux use the `Makefile`.

## Commands

```bash
pip install -e ".[dev]" -c constraints.txt      # constraints.txt pins the resolved set — always use -c
docker compose -f docker-compose.dev.yml up -d postgres redis   # Postgres on host port 15432 (not 5432)
cd backend && alembic upgrade head              # tests require migrated schema
python -m pytest -m "not slow"                  # from repo root; testpaths=backend/tests
ruff check --select=F backend                   # exactly what CI lint runs
(cd frontend && npm run lint && npm run build)
make demo                                       # full stack + seed (seeder prints the per-run admin password)
```

## Testing quirks

- Needs live Postgres + Redis and env vars: `ATTENDANCE_DATABASE_URL(_TEST)`,
  `ATTENDANCE_REDIS_URL(_TEST)`, `ATTENDANCE_JWT_SECRET`, `ATTENDANCE_ALLOWED_ORIGINS`,
  `ATTENDANCE_TRITON_URL=fake-host:8001`. See CLAUDE.md for a copy-paste block.
- Triton/GPU is faked via `FakeTritonGrpcClient` through `set_triton_client_override()` — never
  bypass that seam.
- `filterwarnings = ["error"]` in pyproject: any new upstream warning fails the suite. That is why
  `constraints.txt` exists; regenerate per the header comment, don't loosen filters.
- Both `httpx` and `httpx2` are dev deps on purpose (TestClient prefers httpx2; conftest fixtures
  use httpx). Don't deduplicate.
- Doc-stated pass counts are stale ("7"/"8 passed"); the suite has grown. Run it; don't quote old counts.
- Any ORM model change needs a matching Alembic migration; CI round-trips `upgrade head` →
  `downgrade base` → re-upgrade, so downgrades must stay valid.

## PRs & the auto-merge trap

- `main` is protected: PR + squash-merge only, linear history. Branch from fresh `main`, never
  from another fix branch; rebase on `origin/main` right before pushing; never force-push or merge.
- `.github/workflows/auto-merge.yml`: a PR labeled `agent-pr` is **squash-merged automatically**
  once CI passes — unless labeled `do-not-merge` or touching a protected path. Protected paths now
  include `backend/app/**` and `backend/tests/**` (CLAUDE.md's older, shorter list is stale), so most
  code changes need manual merge anyway. If you do not want unreviewed auto-merge, apply
  `do-not-merge` yourself; never edit the auto-merge workflow to change this.
- Commits: conventional style (`fix(ATT-nnn): ...`), subject ≤72 chars, body explains why,
  ends with `Refs: ATT-nnn`. One batch = one branch = one PR; batch branches are named `fix/Bnn-slug`.

## Load-bearing backend rules (details in attendance-v3-context skill)

- Import pipeline via facade `app.services.pipeline_service`, never its submodules.
- Import Triton from `app.infrastructure.triton` (not legacy `app.worker.triton_client`) and call
  `get_triton_client()` at point of use — never cache the client module-level; it defeats the test seam.
- Celery tasks wrap exactly one `asyncio.run()`; loop-bound singletons (Redis clients, asyncpg
  engine) must not cross loops; worker code must not use the FastAPI-loop Redis singleton; no
  blocking calls on the event loop.
- Face embeddings are biometric data of possibly-minors: never log/serialize/return them or widen
  exposure; task-status endpoint strips them — keep it that way. Never put embeddings in commits/PRs.
- Realtime tickets: single Redis `EVAL` doing GET+DEL atomically; auth fixes fail closed; never
  delete/weaken a test to make a fix pass.

## Post-autonomy-run additions (2026-08-24)

- New backend features on `integration/wave1` (awaiting push): bulk enrollment
  importer, course-scoped authz (`ATTENDANCE_COURSE_SCOPED_AUTHZ`), audit
  service + `/api/v1/governance/events`, consent gate
  (`ATTENDANCE_ENFORCE_BIOMETRIC_CONSENT`), retention sweep
  (`ATTENDANCE_EMBEDDING_RETENTION_DAYS`), manual overrides,
  `/api/v1/admin/enrollment-coverage`. Migrations chain 0005→0008.
- tritonclient 2.70 gRPC: never pass `binary_data=` to
  `set_data_from_numpy`, and never `request_id=None`.
- Design specs + owner decision record live in `docs/design/`.
