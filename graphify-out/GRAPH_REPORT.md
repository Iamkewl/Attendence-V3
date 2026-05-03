# Graph Report - .  (2026-05-03)

## Corpus Check
- 46 files · ~27,610 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 647 nodes · 1867 edges · 34 communities detected
- Extraction: 46% EXTRACTED · 54% INFERRED · 0% AMBIGUOUS · INFERRED: 1005 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]

## God Nodes (most connected - your core abstractions)
1. `UserRole` - 133 edges
2. `AttendanceStatus` - 78 edges
3. `Student` - 75 edges
4. `RedisPubSubManager` - 57 edges
5. `InferenceBatchRequest` - 52 edges
6. `User` - 51 edges
7. `AsyncCRUDService` - 47 edges
8. `StudentEmbedding` - 45 edges
9. `TritonGrpcClient` - 43 edges
10. `ImageTensorPayload` - 42 edges

## Surprising Connections (you probably didn't know these)
- `Alembic environment configuration for asynchronous PostgreSQL migrations.` --uses--> `Base`  [INFERRED]
  backend\alembic\env.py → backend\app\domain\models.py
- `Resolve the database URL from environment or Alembic configuration.` --uses--> `Base`  [INFERRED]
  backend\alembic\env.py → backend\app\domain\models.py
- `Run migrations in offline mode using emitted SQL scripts.` --uses--> `Base`  [INFERRED]
  backend\alembic\env.py → backend\app\domain\models.py
- `Configure migration context and execute migrations for a live connection.` --uses--> `Base`  [INFERRED]
  backend\alembic\env.py → backend\app\domain\models.py
- `Run migrations in online mode using an asynchronous SQLAlchemy engine.` --uses--> `Base`  [INFERRED]
  backend\alembic\env.py → backend\app\domain\models.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (99): AuthUserRead, _clear_auth_cookies(), issue_websocket_ticket(), login(), LoginRequest, logout(), LogoutResponse, Authentication endpoints for login, refresh rotation, and logout workflows. (+91 more)

### Community 1 - "Community 1"
Cohesion: 0.05
Nodes (78): Enum, enqueue_batch_inference(), _enqueue_inference_batch(), enqueue_stream_inference(), get_inference_task_status(), Inference API endpoints for enqueueing and tracking asynchronous AI pipeline tas, Accept a validated multi-frame payload and enqueue the asynchronous inference pi, Return execution state for a task ID from the Celery result backend. (+70 more)

### Community 2 - "Community 2"
Cohesion: 0.1
Nodes (71): _as_utc(), AttendanceNotFoundError, AttendanceService, AttendanceServiceError, AttendanceValidationError, Attendance domain service for heartbeat logging and temporal aggregation workflo, Aggregate daily sightings into final class-session attendance records., Publish a serialized sighting payload to realtime subscribers. (+63 more)

### Community 3 - "Community 3"
Cohesion: 0.13
Nodes (63): Represents a student entity linked one-to-one with a platform user identity., Stores active and historical face template embeddings for a student across poses, Student, StudentEmbedding, _bbox_centroid(), _crop_face(), _decode_bbox(), _decode_embeddings() (+55 more)

### Community 4 - "Community 4"
Cohesion: 0.1
Nodes (40): Worker integration package exposing Celery and Triton client utilities., Tracks enrollment template lifecycle events such as create, archive, and updates, TemplateAuditLog, Input schema used to patch mutable student profile fields., Output schema representing a student profile from persistence., Output schema representing one persisted enrollment embedding template., Input schema used to create a student profile., StudentCreate (+32 more)

### Community 5 - "Community 5"
Cohesion: 0.07
Nodes (29): _normalize_channels(), _parse_pubsub_message(), PubSubMessage, Redis-backed pub/sub primitives and one-time ticket helpers for realtime channel, Store a one-time realtime ticket in Redis with strict expiration semantics., Atomically read and delete a one-time realtime ticket value from Redis., Return a cached Redis client instance, initializing lazily when required., Validate and normalize channel names before Redis subscription calls. (+21 more)

### Community 6 - "Community 6"
Cohesion: 0.09
Nodes (37): _clone_lvface_repo(), _copy_model(), _dims_to_pbtxt(), _download_lvface_weights(), _ensure_python_dependencies(), _export_lvface_model(), _export_yolov12_model(), _extract_onnx_signature() (+29 more)

### Community 7 - "Community 7"
Cohesion: 0.07
Nodes (26): RuntimeError, clear_triton_client_cache(), get_triton_client(), get_triton_client_settings(), _is_retryable_exception(), Robust Triton gRPC client wrapper with retry-aware inference helpers., Return cached Triton client settings sourced from environment variables., Close the underlying Triton client transport if supported by the client implemen (+18 more)

### Community 8 - "Community 8"
Cohesion: 0.12
Nodes (9): Generic asynchronous CRUD repository primitives for SQLAlchemy ORM models., Delete one entity by identifier and report whether a row was removed., Execute a read query and rollback the session if execution fails., Normalize Pydantic or mapping payloads into mutable dictionaries., Load one entity by primary key identifier., Load a paginated entity collection with optional filtering and ordering., Persist a new entity instance and return the inserted row., Patch an existing entity and return the updated row when found. (+1 more)

### Community 9 - "Community 9"
Cohesion: 0.17
Nodes (15): dispose_engine(), get_async_engine(), get_async_session(), get_session_factory(), _normalize_async_url(), Database engine and session lifecycle management for the attendance service., Read an integer environment variable and validate its type and bounds., Return the configured PostgreSQL DSN and fail fast when missing. (+7 more)

### Community 10 - "Community 10"
Cohesion: 0.24
Nodes (9): _database_url(), do_run_migrations(), Alembic environment configuration for asynchronous PostgreSQL migrations., Resolve the database URL from environment or Alembic configuration., Run migrations in offline mode using emitted SQL scripts., Configure migration context and execute migrations for a live connection., Run migrations in online mode using an asynchronous SQLAlchemy engine., run_migrations_offline() (+1 more)

### Community 11 - "Community 11"
Cohesion: 0.22
Nodes (9): _content_security_policy(), create_app(), _lifespan(), _parse_allowed_origins(), FastAPI application factory and runtime middleware wiring for Attendance V2., Parse and validate explicit CORS origins for credentialed requests., Return a strict default CSP unless explicitly overridden by environment variable, Initialize and tear down runtime resources used by auth and database layers. (+1 more)

### Community 12 - "Community 12"
Cohesion: 0.32
Nodes (7): get_celery_app(), Celery application configuration for asynchronous attendance inference workloads, Read a required environment variable and fail fast when missing., Read an integer environment variable with explicit lower-bound validation., Create and cache the Celery application used by inference workers., _read_int_env(), _read_required_env()

### Community 13 - "Community 13"
Cohesion: 0.36
Nodes (7): _build_headers(), _is_uuid(), main(), Return True when the input can be parsed as a UUID., Build optional request headers for authenticated local testing., Encode and upload one frame using multipart form data., _send_frame()

### Community 14 - "Community 14"
Cohesion: 0.38
Nodes (3): createEventId(), normalizeRealtimeMessage(), parseEventPayload()

### Community 15 - "Community 15"
Cohesion: 0.33
Nodes (5): downgrade(), Initial attendance database schema.  Revision ID: 20260501_0001 Revises: Cre, Drop all domain tables, indexes, constraints, and enum types., Create all domain tables, constraints, indexes, and enum types., upgrade()

### Community 16 - "Community 16"
Cohesion: 0.33
Nodes (5): downgrade(), Refactor attendance persistence for periodic heartbeat and temporal aggregation., Revert heartbeat and class-session tables back to turnstile attendance records., Migrate turnstile attendance events to class sessions plus raw heartbeat sightin, upgrade()

### Community 17 - "Community 17"
Cohesion: 0.33
Nodes (5): downgrade(), Add multi-pose student embeddings and template audit logs with pgvector search s, Drop pgvector template tables and restore strict sighting student requirement., Create pgvector extension, enrollment template tables, and unknown-sighting supp, upgrade()

### Community 18 - "Community 18"
Cohesion: 0.33
Nodes (0): 

### Community 19 - "Community 19"
Cohesion: 0.33
Nodes (0): 

### Community 20 - "Community 20"
Cohesion: 0.4
Nodes (0): 

### Community 21 - "Community 21"
Cohesion: 0.4
Nodes (0): 

### Community 22 - "Community 22"
Cohesion: 0.67
Nodes (0): 

### Community 23 - "Community 23"
Cohesion: 0.67
Nodes (0): 

### Community 24 - "Community 24"
Cohesion: 1.0
Nodes (0): 

### Community 25 - "Community 25"
Cohesion: 1.0
Nodes (0): 

### Community 26 - "Community 26"
Cohesion: 1.0
Nodes (0): 

### Community 27 - "Community 27"
Cohesion: 1.0
Nodes (1): Create a managed pub/sub subscription that always unsubscribes and closes cleanl

### Community 28 - "Community 28"
Cohesion: 1.0
Nodes (1): Wrap a write workflow in commit-or-rollback transaction handling.

### Community 29 - "Community 29"
Cohesion: 1.0
Nodes (1): Return whether an exception likely represents transient infrastructure failure.

### Community 30 - "Community 30"
Cohesion: 1.0
Nodes (1): Detect timeout-oriented Triton server exceptions by normalized error message.

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (0): 

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (0): 

### Community 33 - "Community 33"
Cohesion: 1.0
Nodes (0): 

## Knowledge Gaps
- **107 isolated node(s):** `Initial attendance database schema.  Revision ID: 20260501_0001 Revises: Cre`, `Create all domain tables, constraints, indexes, and enum types.`, `Drop all domain tables, indexes, constraints, and enum types.`, `Refactor attendance persistence for periodic heartbeat and temporal aggregation.`, `Migrate turnstile attendance events to class sessions plus raw heartbeat sightin` (+102 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 24`** (2 nodes): `useAuth()`, `AuthContext.jsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (2 nodes): `DashboardLayout()`, `DashboardLayout.jsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (2 nodes): `LoginPage.jsx`, `LoginPage()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (1 nodes): `Create a managed pub/sub subscription that always unsubscribes and closes cleanl`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 28`** (1 nodes): `Wrap a write workflow in commit-or-rollback transaction handling.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 29`** (1 nodes): `Return whether an exception likely represents transient infrastructure failure.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `Detect timeout-oriented Triton server exceptions by normalized error message.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): `eslint.config.js`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (1 nodes): `vite.config.js`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (1 nodes): `main.jsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `UserRole` connect `Community 0` to `Community 1`, `Community 2`, `Community 3`, `Community 4`?**
  _High betweenness centrality (0.217) - this node is a cross-community bridge._
- **Why does `RedisPubSubManager` connect `Community 0` to `Community 2`, `Community 5`?**
  _High betweenness centrality (0.111) - this node is a cross-community bridge._
- **Why does `Student` connect `Community 3` to `Community 2`, `Community 4`?**
  _High betweenness centrality (0.072) - this node is a cross-community bridge._
- **Are the 129 inferred relationships involving `UserRole` (e.g. with `Reusable FastAPI dependencies for authentication and authorization.` and `Resolve the currently authenticated and active user from bearer token or cookie.`) actually correct?**
  _`UserRole` has 129 INFERRED edges - model-reasoned connections that need verification._
- **Are the 74 inferred relationships involving `AttendanceStatus` (e.g. with `SchemaModel` and `UserCreate`) actually correct?**
  _`AttendanceStatus` has 74 INFERRED edges - model-reasoned connections that need verification._
- **Are the 70 inferred relationships involving `Student` (e.g. with `Student profile endpoints protected with instructor/admin RBAC policies.` and `Normalize pose labels for deterministic multi-template enrollment.`) actually correct?**
  _`Student` has 70 INFERRED edges - model-reasoned connections that need verification._
- **Are the 48 inferred relationships involving `RedisPubSubManager` (e.g. with `AuthUserRead` and `LoginRequest`) actually correct?**
  _`RedisPubSubManager` has 48 INFERRED edges - model-reasoned connections that need verification._