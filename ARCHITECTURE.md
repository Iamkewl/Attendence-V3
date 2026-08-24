# Architecture — Attendance V3

See [README.md](./README.md) for quick-start, environment variables, and command reference.

---

## 1. System Overview

Attendance V3 is a camera-driven facial recognition attendance platform. A React
frontend streams webcam frames to a FastAPI backend, which enqueues them as Celery
tasks. A worker process runs a two-model AI pipeline (person detection via YOLOv12,
then liveness scoring and 512-D face embedding via LVFace), persists recognition
events as Sighting rows in Postgres, and publishes results to a Redis pub/sub channel.
The frontend receives live updates over WebSocket or SSE.

```
                         +---------------------+
                         |  React + Vite SPA   |
                         |  (localhost:5173)   |
                         +---------+-----------+
                                   |  HTTP/WS
                         +---------v-----------+
                         |   FastAPI (uvicorn) |
                         |  /api/v1/*          |  <-- REST routes
                         |  /ws/live           |  <-- WebSocket
                         |  /sse/live          |  <-- Server-Sent Events
                         +--+------+--------+--+
                            |      |        |
              enqueue task  |  SQL |  Redis |
                            |      |        |
               +------------+  +---+  +-----+-------+
               |               |      |             |
     +---------v-------+  +----v----+ +v-----------+|
     |  Celery Worker  |  |Postgres | | Redis       ||
     |  (inference,    |  |(pgvector| | pub/sub +   ||
     |   aggregation)  |  | :15432) | | token store ||
     +--------+--------+  +---------+ +-------------+|
              |                                       |
              |  gRPC (via SSH tunnel in dev)         |
     +--------v---------+                            |
     |  Triton Inference |  (remote GPU server)      |
     |  Server           |                           |
     |  - yolov12 model  |                           |
     |  - lvface model   |                           |
     +------------------+                            |
              |                                      |
              +-- pipeline result published to Redis-+
```

Celery Beat (a separate process) fires the `task_evaluate_daily_attendance` task
hourly and, when demo mode is on, `demo_emit_sighting` on a configurable interval.

---

## 2. Subsystems

### 2.1 API Layer

**Entry point:** `backend/app/main.py`

`create_app()` builds the FastAPI instance and wires:

- `CORSMiddleware` — explicit origin allowlist (no wildcards), credentials enabled.
- `RequestIDMiddleware` / `AccessLogMiddleware` — request ID injection and JSON
  structured access logs configured via `configure_json_access_log()`.
- HTTP security header middleware (`inject_security_headers`) — CSP, HSTS, frame
  denial, referrer policy, and permissions policy applied to every response.

Routers mounted:

| Prefix | Module | Purpose |
|---|---|---|
| `/api/v1/auth` | `api/v1/auth.py` | Login, refresh, logout, WS ticket issuance |
| `/api/v1/inference` | `api/v1/inference.py` | Frame stream enqueue, task status, photo recognition |
| `/api/v1/students` | `api/v1/students.py` | Student and embedding CRUD |
| `/api/v1/attendance` | `api/v1/attendance.py` | Course, room, session, sighting queries |
| `/api/v1/users` | `api/v1/users.py` | User management |
| `/ws/live`, `/sse/live` | `api/v1/websockets.py` | Realtime transports |
| `/operations/*` | `api/operations.py` | Internal health and admin ops |
| `/healthz` | `main.py` (inline) | Lightweight liveness probe |

> **Note (ATT-059):** the operations router is currently mounted
> **unprefixed at the app root** in `main.py` (i.e.
> `app.include_router(operations_router)` with no `prefix=...`), so the
> operational routes actually live at `/health`, `/ready`, `/version` —
> *not* under `/operations/*`. The table row above documents the
> *design intent* (a discrete URL prefix grouping internal ops away
> from the public API surface); a follow-up is needed in `main.py` to
> add `prefix="/operations"` and update the README/RUNBOOK/healthcheck
> URLs to match. Until then, treat the `/operations/*` cells above as
> the planned layout, and use `/health`, `/ready`, `/version` directly.

Application lifespan (`_lifespan`) runs `initialize_redis()` on startup and
`close_redis()` + `dispose_engine()` on shutdown.

### 2.2 Domain — Models and Schemas

**Models:** `backend/app/domain/models/`

Each entity lives in its own file. `__init__.py` re-exports all public symbols so
existing `from app.domain.models import X` imports continue to work without change.
All entities inherit from `UUIDPrimaryKeyMixin` (UUID PK) and `TimestampMixin`
(`created_at`, `updated_at`).

**Schemas:** `backend/app/domain/schemas/`

Pydantic v2 request/response schemas split by domain area (user, student, course,
attendance, inference, common). `__init__.py` re-exports everything for backward
compatibility.

### 2.3 Services — Pipeline Subpackage

**Location:** `backend/app/services/pipeline/`

Eight focused modules:

| Module | Responsibility |
|---|---|
| `settings.py` | `PipelineSettings` (Pydantic settings from env vars), `get_pipeline_settings()` |
| `frame.py` | Frame tensor decoding, normalization, face crop extraction, batch preparation |
| `detection.py` | YOLO output parsing, bounding-box decoding, `Detection` dataclass |
| `tracking.py` | Centroid-distance track linking across frames, `TrackedDetection` dataclass |
| `liveness.py` | LVFace liveness output decoding to per-face scores |
| `embedding.py` | LVFace embedding output decoding, L2 normalization, identity hash |
| `matching.py` | pgvector nearest-neighbour search, cosine similarity threshold gate |
| `orchestrator.py` | `process_inference_batch()` and `extract_enrollment_embedding()` — the two public entry points |

The orchestrator calls `get_triton_client()` on each invocation (not cached at
module level) so the test override is always respected.

### 2.4 Services — Facade

**Location:** `backend/app/services/pipeline_service.py`

A thin import-forwarding module. It re-exports every public symbol from the
pipeline subpackage. Call sites outside the subpackage import from
`pipeline_service` rather than reaching into individual submodules. This gives the
subpackage a stable, single-import-path public surface without duplicating logic.
See section 4 for why this matters.

### 2.5 Worker — Celery

**Location:** `backend/app/worker/`

| File | Content |
|---|---|
| `celery_app.py` | `get_celery_app()` (lru_cache); broker/backend from Redis URL; three queues: `inference`, `inference_priority`, `attendance_aggregation`; beat schedule; worker_process_init primes Triton readiness |
| `tasks.py` | `run_inference_pipeline` (Celery task), `task_evaluate_daily_attendance` (Celery task), `demo_emit_sighting` (Celery task); private async helpers `_run_pipeline_and_log_sightings` and `_evaluate_daily_attendance` |
| `demo_emitter.py` | `emit_one_synthetic_sighting()` — inserts a random Sighting for a real student in the demo course |

Celery tasks are sync functions that call `asyncio.run()` over a single private
async function. All DB and Triton I/O happen inside that one `asyncio.run()` call
so the asyncpg engine is not shared across event loops. See section 4.

### 2.6 Infrastructure — Triton Client

**Location:** `backend/app/infrastructure/triton/`

`TritonGrpcClient` wraps `tritonclient.grpc` with:

- Retry policy with exponential backoff and a total-budget ceiling.
- Per-instance readiness caching (`_server_ready`, `_ready_models`); cache is
  invalidated on transient failures.
- Async wrappers (`infer_fp32_async`, etc.) using `asyncio.to_thread`.

`get_triton_client()` checks `_test_client_override` first, then falls back to the
`lru_cache`-backed `_build_triton_client()` singleton. The override seam is
described in detail in section 4.

### 2.7 Realtime Transport — WebSocket and SSE

**Location:** `backend/app/api/v1/websockets.py`

Two module-level singletons are created at import time:

```python
_pubsub_manager = RedisPubSubManager()   # lazy Redis init
_connection_manager = LiveConnectionManager(_pubsub_manager, DEFAULT_REALTIME_CHANNELS)
```

`LiveConnectionManager` maintains the set of connected WebSocket clients and a
single shared Redis broadcast task. When the first client connects the task starts;
when the last disconnects it stops cleanly via a shared `asyncio.Event`.

Both `/ws/live` and `/sse/live` require a one-time ticket (issued by
`POST /api/v1/auth/websocket-ticket`). The ticket is atomically consumed from Redis
via a Lua `GET`+`DEL` script before the connection is accepted.

### 2.8 Core — Auth, Pub/Sub, Security

**Location:** `backend/app/core/`

- `security.py` — JWT lifecycle (`create_access_token`, `create_refresh_token`,
  `validate_token`), Argon2 password hashing, Redis-backed token blocklist.
  Module-level `_redis_client` is lazily initialized on the first call to
  `get_redis_client()`. Important for tests: see section 6.
- `pubsub.py` — `RedisPubSubManager` (publish, subscribe, ticket lifecycle).
  `websocket_ticket_key()` builds the namespaced Redis key.
  `WS_TICKET_TTL_SECONDS = 30`.
- `database.py` — asyncpg engine factory (`get_session_factory()`), `get_async_session()` dependency.
- `middleware.py` — `RequestIDMiddleware`, `AccessLogMiddleware`, `configure_json_access_log()`.

---

## 3. Request Flows

### 3.1 Login → Cookie Issuance → Authenticated Request

```
Client                        FastAPI /api/v1/auth/login
  |                                     |
  |-- POST {email, password} ---------->|
  |                                     |-- hash_password verify (Argon2)
  |                                     |-- create_access_token (JWT, HS256)
  |                                     |-- create_refresh_token (JWT)
  |<-- Set-Cookie: access_token --------|
  |<-- Set-Cookie: refresh_token -------|
  |<-- SessionResponse (user, exp) -----|

Client                        FastAPI /api/v1/<protected>
  |-- GET / (cookie header) ----------->|
  |                                     |-- validate_token (decode JWT)
  |                                     |-- check Redis blocklist (jti)
  |                                     |-- CurrentUser dependency resolved
  |<-- 200 OK + resource ---------------|
```

Refresh (`POST /api/v1/auth/refresh`) reads the refresh token cookie, validates it,
blocklists the old JTI, and issues a new access + refresh pair. Logout blocklists
the current access token JTI.

### 3.2 /inference/stream POST → Celery → Pipeline → Sighting → Broadcast

```
Client                    FastAPI                       Celery Worker
  |                          |                               |
  |-- POST /inference/stream |                               |
  |   (multipart frame data) |                               |
  |                          |-- validate auth cookie        |
  |                          |-- decode frame payload        |
  |                          |-- run_inference_pipeline      |
  |                          |   .apply_async(queue=...)     |
  |<-- 202 {task_id} --------|                               |
  |                          |                               |
  |-- GET /inference/status/{task_id}                        |
  |                          |                               |-- asyncio.run(
  |                          |                               |     _run_pipeline_and_log_sightings)
  |                          |                               |
  |                          |                     [pipeline: YOLO detect]
  |                          |                     [pipeline: track faces]
  |                          |                     [pipeline: LVFace embed]
  |                          |                     [pipeline: pgvector match]
  |                          |                               |
  |                          |                               |-- INSERT Sighting rows
  |                          |                               |-- publish_json(LIVE_ATTENDANCE_CHANNEL)
  |                          |                               |   per sighting (fanout to WS+SSE)
  |<-- 200 {results} --------|-- celery_app.AsyncResult -----|
```

Note: the pipeline result is stored in the Celery result backend (Redis). The
client polls `GET /api/v1/inference/task/{task_id}` to retrieve it. The embedding
vector is stripped from the result before the status response is returned to the
client (`backend/app/api/v1/inference.py`).

Live broadcast happens inside `attendance_service.log_sighting()`
(`backend/app/services/attendance_service.py:94`): after the Sighting row is
committed, `_publish_live_sighting_event` serializes the row and calls
`publish_json(LIVE_ATTENDANCE_CHANNEL, ...)`. The publish is best-effort —
failures are logged but do not propagate to the worker, so persistence and
broadcast cannot interfere with each other. This means the inference pipeline,
the demo emitter, and any future caller of `log_sighting` all reach the realtime
dashboard through the same channel.

### 3.3 WebSocket Ticket Lifecycle

```
Client                    FastAPI /api/v1/auth            Redis
  |                              |                          |
  |-- POST /auth/websocket-ticket|                          |
  |   (requires auth cookie)     |                          |
  |                              |-- uuid4() ticket         |
  |                              |-- SET auth:ws_ticket:<t> |
  |                              |   EX=30 NX=true -------->|
  |<-- {ticket: "<uuid>"} -------|                          |

Client                    FastAPI /ws/live                 Redis
  |                              |                          |
  |-- WS CONNECT /ws/live?ticket=<t>                        |
  |                              |-- EVAL GET+DEL script -->|
  |                              |<-- ticket payload --------|
  |                              |   (or nil if expired)    |
  |   (if nil: close 1008)       |                          |
  |                              |-- _connection_manager.connect(ws)
  |<-- WS OPEN ------------------|                          |
  |<-- broadcast messages --------|<-- pubsub messages -----|
  |-- WS CLOSE ----------------->|                          |
  |                              |-- _connection_manager.disconnect(ws)
```

The Lua `GET`+`DEL` in `consume_ticket()` (`backend/app/core/pubsub.py:142`)
ensures a ticket can be used exactly once even under concurrent connection
attempts.

---

## 4. Key Abstractions and Design Decisions

### 4.1 The Triton Test Seam

**Files:** `backend/app/infrastructure/triton/client.py:314-333`

The problem: `_build_triton_client()` is decorated with `@lru_cache(maxsize=1)`.
Once it runs — which happens the instant any production code path imports and calls
`get_triton_client()` — the real `TritonGrpcClient` is cached. Replacing it via
`monkeypatch` after the fact is fragile because the cached object is already
embedded in local references inside running tasks and coroutines. Celery
eager-mode tasks (`CELERY_TASK_ALWAYS_EAGER=1`) call the same module in the same
process, so monkeypatching `sys.modules` would need to cover every call site.

The solution: a module-level `_test_client_override` variable (initially `None`).
`get_triton_client()` checks it on every call:

```python
_test_client_override: TritonGrpcClient | None = None

def set_triton_client_override(client: TritonGrpcClient | None) -> None:
    global _test_client_override
    _test_client_override = client

def get_triton_client() -> TritonGrpcClient:
    if _test_client_override is not None:
        return _test_client_override
    return _build_triton_client()
```

Why the check happens at every call, not once at startup: if the override were
cached into a local variable during module import, toggling it between tests would
have no effect. The per-call check costs one `is not None` comparison per inference
call — negligible in production, essential in tests.

Why it works for both FastAPI paths AND Celery eager-mode tasks: both use the same
Python process and the same module. The fixture sets the override before the test,
the task runs (in-process, eager), reads `_test_client_override`, uses the fake,
and the fixture restores `None` after `yield`. No monkeypatching or import hacks
required.

Usage in `backend/tests/conftest.py:165-170`:

```python
@pytest.fixture()
def fake_triton() -> FakeTritonGrpcClient:
    from app.infrastructure.triton.client import set_triton_client_override
    fake = FakeTritonGrpcClient()
    set_triton_client_override(fake)
    yield fake
    set_triton_client_override(None)
```

### 4.2 The Pipeline Facade Pattern

`backend/app/services/pipeline_service.py` contains no logic. It re-exports every
public symbol from the eight pipeline submodules using `from .pipeline.X import Y`.
This gives external callers (the inference API router, the Celery task, enrollment
endpoints) a single stable import:

```python
from app.services.pipeline_service import process_inference_batch
```

Without the facade, callers would need to know which internal submodule owns each
symbol. The facade makes the internal structure refactorable without touching call
sites. The `__init__.py` of the pipeline subpackage itself exports only the public
surface; the facade then re-exports that surface one more time for the legacy import
path.

### 4.3 The `_pubsub_manager` Singleton and Event-Loop Conflicts

`backend/app/api/v1/websockets.py:121`:

```python
_pubsub_manager = RedisPubSubManager()
```

`RedisPubSubManager.__init__` accepts an optional Redis client but defaults to
`None`. `_get_client()` lazily calls `get_redis_client()` the first time it is
needed, caching the result in `self._redis_client`. The problem in tests: the
pytest-asyncio session loop may have already resolved `get_redis_client()` for
earlier async tests, creating an `asyncio.Redis` client bound to that loop. When a
later test uses `starlette.testclient.TestClient` (which runs ASGI in its own anyio
loop), any awaited call on the old client raises:

```
Future attached to a different loop
```

The fix is to reset both singleton references before each `TestClient` interaction:

```python
import app.core.security as security
import app.api.v1.websockets as ws_module

security._redis_client = None
ws_module._pubsub_manager._redis_client = None
```

See `backend/tests/test_smoke_realtime.py:50-51` for the canonical example.

### 4.4 Private Async Task Helpers

`backend/app/worker/tasks.py` exposes `_run_pipeline_and_log_sightings` and
`_evaluate_daily_attendance` as module-level async functions (private by
convention). The Celery tasks call them via `asyncio.run()`. Tests can call them
directly with `await` inside an async test function, bypassing the sync Celery
task wrapper entirely. This avoids the `asyncio.run()` inside `asyncio.run()` error
that would occur if tests called the Celery task synchronously in eager mode inside
an already-running event loop.

### 4.5 The asyncpg `AsyncEngine` `lru_cache` and the Production Cross-Loop Hazard

§4.3 documents an event-loop-binding hazard for tests: `redis.asyncio`
clients cached on module-level singletons pin themselves to whichever
loop first touched them, and a subsequent `TestClient` on a different
loop raises `Future attached to a different loop`. **The same hazard
applies in production to the asyncpg `AsyncEngine`** and bit us as
ATT-011; this section pins the diagnosis and the canonical fix so a
future reviewer reading §4.3 does not mistakenly conclude loop-bound
singletons are a test-time-only concern.

**The cached singleton:** `backend/app/core/database.py` defines

```python
@lru_cache(maxsize=1)
def get_async_engine() -> AsyncEngine: ...
```

and a sibling `get_session_factory()` `lru_cache`. Both are module-level
and bound to a single `AsyncEngine`. asyncpg's ``AsyncEngine`` binds its
connection pool to the event loop that created it.

**Why it looks safe but isn't.** Under `celery -A app.worker.celery_app
worker --pool=prefork`, each child process has its own fresh Python
interpreter, so the `lru_cache` is *per child*. That part is fine. The
problem is the second half: **Celery tasks are sync wrappers that call
`asyncio.run(helper())` per invocation**, and `asyncio.run()` mints a
fresh event loop for every call. The first task on a child primes
`get_async_engine()` and binds its engine to *task #1's loop*. Task #1
exits, its loop closes — but the engine is still in `lru_cache`. Every
subsequent task on that child awaits sessions on a cached engine whose
loop is closed → `RuntimeError: Future attached to a different loop` (or
a `Event loop is closed` / `Another event loop is already being awaited`
variant, depending on asyncpg version).

**The fix (canonical pattern, ATT-011):** register a Celery `task_postrun`
signal that disposes the cached engine after each task completes, so the
next task mints a fresh engine bound to its own fresh loop. See
`backend/app/worker/celery_app.py` (`dispose_engine` + the
`@task_postrun.connect` receiver) shipped in PR #71. The same pattern
applies to any other cross-loop `lru_cache`'d async singleton added to
the codebase in the future.

**Do not** "fix" this by removing the `lru_cache`: a per-call engine
would defeat asyncpg's connection pool, dramatically raise latency, and
re-introduce the very cross-loop hazard under a different name (every
call would still bind a pool to a fresh loop). The right answer is the
`task_postrun` disposer, not caching removal.


---

## 5. Data Model

Core entities and key relations:

- **User** — platform identity; roles: ADMIN, INSTRUCTOR, OPERATOR, AUDITOR. One-to-one with Student.
- **Student** — academic record; linked to User via `user_id` (CASCADE delete). Holds program, enrollment year, active flag.
- **StudentEmbedding** — 512-D pgvector embedding per student face template. Many-to-one with Student. Includes quality score, pose label, and active flag. Nearest-neighbour search uses cosine distance (`<=>` operator).
- **Course** — academic course entity; has `is_active` flag.
- **Room** — physical classroom with optional capacity.
- **Session** — *not yet modeled.* A scheduled-class-meeting entity (Course + Room + scheduled time)
  does not currently exist in the ORM. The closest analog today is `ClassSessionRecord` (below),
  but that is the **output** of the daily aggregation job, not the planned-meeting entity itself.
  The class-timetable gap is tracked as ATT-040 (Roadmap P0). Do not confuse `Session` (which
  does not exist) with auth sessions, which are JWT/Redis-managed — see §2 (security).
- **Sighting** — one raw AI recognition event: `student_id` (nullable, SET NULL on student delete), `course_id` (RESTRICT), `room_id` (nullable), `timestamp`, `camera_id`, `confidence_score`, `embedding_reference`. Indexed on `(course_id, timestamp)`, `(student_id, timestamp)`, `(camera_id, timestamp)`.
- **ClassSessionRecord** — daily attendance summary per student per course, upserted by `task_evaluate_daily_attendance`.

---

## 6. Async and Event-Loop Concerns

This section documents the asymmetry between the pytest-asyncio session loop and
`TestClient`'s internal loop, which is the most common source of unexpected
test failures.

**The asymmetry:** pytest-asyncio (with `asyncio_mode = "auto"`) runs all `async`
test functions and session-scoped async fixtures on a single shared event loop for
the session. `starlette.testclient.TestClient` runs the ASGI application in its
own thread using anyio, which creates a separate event loop.

**Resources that cache their loop:** `asyncio.Redis` from `redis.asyncio` binds
to the loop that created it. SQLAlchemy's asyncpg engine also binds to its creation
loop. Both are cached as module-level singletons:

- `app.core.security._redis_client` — security module, auth blocklist and token ops.
- `app.api.v1.websockets._pubsub_manager._redis_client` — pub/sub manager used by
  `/ws/live` and `/sse/live`.

**Rule:** before using `TestClient` after any async test has run, reset these two
references to `None`. The fresh client will then be created on `TestClient`'s own
loop where it can be awaited without error.

**Do not enter `TestClient` as a context manager** when testing WebSocket or SSE
routes: the context manager triggers FastAPI's lifespan (`initialize_redis`,
`dispose_engine`), which attempts to close asyncpg connections created on a
different loop. Instantiate `TestClient` directly without `with`.

**Worked example:** `backend/tests/test_smoke_realtime.py:15-64`
(`test_websocket_accepts_valid_ticket`). The test is deliberately synchronous, uses
a sync `redis.Redis` client to issue the ticket, resets both singletons, and never
enters `TestClient` as a context manager.
