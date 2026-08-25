# audit_service + GovernanceLog writes — decision-ready design (ATT-006)

**Status:** design research, no production code changed. Unblocks ATT-044 (consent capture) and ATT-038 (manual overrides/appeals); interacts with ATT-045 (retention).
**Repo state verified against:** `main` @ current checkout, migration head `20260519_0004_pgvector_hnsw_index`.

---

## 0. What exists today (verified)

`backend/app/domain/models/governance.py:20` — `GovernanceLog(UUIDPrimaryKeyMixin, TimestampMixin, Base)`, table `governance_logs`. Columns:

| Column | Type | Null | Notes |
|---|---|---|---|
| `id` | UUID PK | no | uuid4 default |
| `actor_user_id` | UUID FK → `users.id` **ON DELETE SET NULL** | yes | survives actor deletion |
| `class_session_record_id` | UUID FK → `class_session_records.id` ON DELETE SET NULL | yes | renamed from `attendance_record_id` in migration 0002 |
| `action` | String(64) NOT NULL | no | CK `governance_action_not_blank` |
| `entity_type` | String(64) NOT NULL | no | CK `governance_entity_type_not_blank` |
| `entity_id` | UUID | yes | plain value, **no FK** — survives entity deletion by design |
| `reason` | Text | yes | free text, ≤4000 in schema |
| `change_summary` | JSONB NOT NULL | no | server_default `'{}'::jsonb` |
| `request_id` | UUID | yes | |
| `ip_address` | INET | yes | |
| `created_at`/`updated_at` | timestamptz | no | TimestampMixin (`updated_at` is vestigial on an append-only table) |

Indexes: `ix_governance_actor_created_at(actor_user_id, created_at)`, `ix_governance_entity_lookup(entity_type, entity_id)`, `ix_governance_class_session_record_id`, `ix_governance_request_id`, `ix_governance_created_at`.

Schemas already exist: `GovernanceLogCreate` / `GovernanceLogRead` at `backend/app/domain/schemas/attendance.py:103-131`, re-exported from `app.domain.schemas`. Their regexes dictate the vocabulary casing: **`action` must match `^[A-Z_]+$`**, **`entity_type` must match `^[a-z_]+$`**.

Confirmed dead code: zero `GovernanceLog(` instantiations, zero route usages of the schemas, no `/governance` route (matches B20 abandonment note in `fixes/NOTES.md:649-678`).

Precedent to mirror: `TemplateAuditLog` rows are written **inside the same transaction as the embedding write** at `backend/app/api/v1/students.py:350-383` (added via `session.add(...)`, flushed, committed once). GovernanceLog should follow exactly this pattern.

---

## 1. Event taxonomy

Design rules:
- One row per completed human-meaningful action; `change_summary` carries structured context per action type.
- **Never** put embedding vectors, `embedding_reference`, or raw frame data into any payload field (hard rule from `.opencode/skills/attendance-v3-context/SKILL.md`). Quality scores and pose labels are fine.
- Per-frame sightings are **not** governance events — they are high-volume (one row per recognized face per frame, written in the Celery hot loop at `backend/app/worker/tasks.py:_log_pipeline_sightings`) and are themselves an append-only domain record (`sightings` table). Doubling that write path for zero compliance gain would be wrong. The aggregation run that *derives* attendance verdicts IS logged (system actor).

### 1.1 Table: endpoint/service → actor → event → payload

| Trigger (file:line) | Actor | `action` | `entity_type` | `entity_id` | `change_summary` fields | Class |
|---|---|---|---|---|---|---|
| POST `/api/v1/users` (`users.py:29`) | ADMIN | `USER_CREATE` | `user` | new user id | `{target_role}` | mandatory |
| PATCH `/api/v1/users/{id}` (`users.py:95`) | ADMIN | `USER_UPDATE` | `user` | target id | `{fields_changed: [names], before/after for non-secret fields only}` | mandatory |
| DELETE `/api/v1/users/{id}` (`users.py:124`) | ADMIN | `USER_DELETE` | `user` | target id | `{target_role, target_email_domain_only?}` | mandatory |
| POST `/api/v1/students` (`students.py:153`) | INSTRUCTOR+ | `STUDENT_CREATE` | `student` | new id | `{student_number, enrollment_year}` | mandatory |
| PATCH `/api/v1/students/{id}` (`students.py:409`) | INSTRUCTOR+ | `STUDENT_UPDATE` | `student` | target id | `{fields_changed}` | mandatory |
| DELETE `/api/v1/students/{id}` (`students.py:438`) | ADMIN | `STUDENT_DELETE` | `student` | target id | `{was_active}` | mandatory |
| POST `/api/v1/students/{id}/enroll` (`students.py:225`) | INSTRUCTOR+ | `TEMPLATE_ENROLL` | `student_embedding` | new embedding id | `{pose_label, quality_score, replaced_count}` (never the vector) | mandatory |
| Celery `task_evaluate_daily_attendance` (`worker/tasks.py:340`) | system (NULL actor) | `ATTENDANCE_EVALUATE` | `class_session_record` | course-scoped: use `class_session_record_id`=NULL, `entity_id`=course id | `{session_date, records_upserted, threshold}` | mandatory |
| POST `/auth/login` success (`auth.py:272`) | self | `LOGIN_SUCCEEDED` | `auth_session` | user id | `{method:"password"}` | advisory |
| POST `/auth/logout` (`auth.py:446`) | self | `LOGOUT` | `auth_session` | user id | `{}` | advisory |
| POST `/auth/refresh` replay rejected (`auth.py:377-381`) | attacker/self | `REFRESH_REUSED` | `auth_session` | user id | `{jti_prefix}` (no token material) | mandatory (security signal) |
| POST `/inference/stream` + `/batch` (`inference.py:84,149`) | CurrentUser | `INFERENCE_ENQUEUED` | `inference_task` | NULL (Celery id not a UUID; put task id string later in summary if needed) | `{frame_count, course_id?}` | advisory |
| GET `/inference/tasks/{task_id}` (`inference.py:164`) | CurrentUser | `TASK_READ` | `inference_task` | NULL | `{task_id}` | advisory, **default OFF** (frontend polls this endpoint; volume risk) — Q7 |
| POST `/inference/photo` (`inference.py:217`) | INSTRUCTOR+ | `RECOGNITION_RUN` | `inference_task` | NULL | `{match_count}` | advisory |

### 1.2 Reserved actions (ship in enum now, wire later)

These exist so ATT-044 / ATT-038 / ATT-045 land without re-designing the vocabulary:

| Future trigger | `action` | `entity_type` | Notes |
|---|---|---|---|
| ATT-044 `POST /students/{id}/consent` | `CONSENT_GRANT` / `CONSENT_WITHDRAW` | `student` | `reason` = controller form version; `ip_address` captured here (Q4); guardian consent distinguished inside `change_summary {is_minor, guardian_signature_ref_present}` |
| ATT-038 `PATCH /attendance/sessions/{course}/{date}/students/{sid}` | `OVERRIDE_APPLY` | `class_session_record` | `class_session_record_id` column finally gets a writer; `reason` REQUIRED |
| ATT-045 retention sweep | `EMBED_HARD_DELETE` | `student_embedding` | system actor; `change_summary {deletion_reason, age_days}` |
| future data export | `EXPORT` | `student` | |

### 1.3 Enum definition (lives in audit_service, mirrored nowhere else)

```python
# backend/app/services/audit_service.py
class GovernanceAction(str, Enum):
    USER_CREATE = "USER_CREATE"
    USER_UPDATE = "USER_UPDATE"
    USER_DELETE = "USER_DELETE"
    STUDENT_CREATE = "STUDENT_CREATE"
    STUDENT_UPDATE = "STUDENT_UPDATE"
    STUDENT_DELETE = "STUDENT_DELETE"
    TEMPLATE_ENROLL = "TEMPLATE_ENROLL"
    ATTENDANCE_EVALUATE = "ATTENDANCE_EVALUATE"
    REFRESH_REUSED = "REFRESH_REUSED"
    LOGIN_SUCCEEDED = "LOGIN_SUCCEEDED"
    LOGOUT = "LOGOUT"
    INFERENCE_ENQUEUED = "INFERENCE_ENQUEUED"   # advisory
    TASK_READ = "TASK_READ"                     # advisory
    RECOGNITION_RUN = "RECOGNITION_RUN"         # advisory
    # reserved for parked efforts:
    CONSENT_GRANT = "CONSENT_GRANT"
    CONSENT_WITHDRAW = "CONSENT_WITHDRAW"
    OVERRIDE_APPLY = "OVERRIDE_APPLY"
    EMBED_HARD_DELETE = "EMBED_HARD_DELETE"
    EXPORT = "EXPORT"

ENTITY_TYPES = frozenset({"user", "student", "student_embedding",
                          "class_session_record", "inference_task", "auth_session"})
```

---

## 2. Write-path design

### 2.1 Facade

New module `backend/app/services/audit_service.py`, exported through `backend/app/services/__init__.py` (same convention as `StudentService`/`UserService`; mirrors the pipeline-facade rule "callers import the facade, never construct ORM rows ad hoc"):

```python
@dataclass(frozen=True)
class AuditEvent:
    action: GovernanceAction
    entity_type: str
    entity_id: UUID | None = None
    class_session_record_id: UUID | None = None
    actor_user_id: UUID | None = None          # None => system actor
    reason: str | None = None
    change_summary: Mapping[str, object] = field(default_factory=dict)
    request_id: UUID | None = None
    ip_address: str | None = None              # INET-castable string

MANDATORY_ACTIONS: frozenset[GovernanceAction] = frozenset({ ... })  # see §1.1

class AuditService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def emit(self, event: AuditEvent) -> GovernanceLog:
        """INSERT one governance row via flush() — joins the CALLER's transaction."""
        row = GovernanceLog(
            action=event.action.value,
            entity_type=event.entity_type,
            entity_id=event.entity_id,
            class_session_record_id=event.class_session_record_id,
            actor_user_id=event.actor_user_id,
            reason=event.reason,
            change_summary=dict(event.change_summary),
            request_id=event.request_id,
            ip_address=event.ip_address,
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def list_events(self, *, filters..., offset, limit) -> list[GovernanceLog]: ...

async def emit(session: AsyncSession, event: AuditEvent, *, strict: bool = True) -> None:
    """Module-level convenience. strict=True: raise on failure (mandatory).
    strict=False: log-and-continue, mirroring _publish_live_sighting_event's
    best-effort precedent (attendance_service.py:91-99)."""
```

The single load-bearing choice: **`flush()`, never `commit()`**. The row participates in whatever transaction the caller owns; commit stays where it is today.

### 2.2 Same-transaction vs fire-and-forget — justification

Same-tx (all-or-nothing) wins for compliance because both failure directions are forbidden:

1. **Mutation without trace** ("who rotated this child's biometric template?" — no answer): with a fire-and-forget queue/task, a crash between business commit and audit write silently loses the record. GDPR Art. 5(2) accountability and BIPA §15(b) consent-retention both require contemporaneous, complete records; an eventually-consistent audit trail has a window where the legally required evidence does not exist.
2. **Trace without mutation** (phantom rows): a queued write after rollback asserts things that never happened — worse than missing rows in an investigation, because it fabricates history.

Same-tx gives you both guarantees from Postgres atomicity for free, needs no outbox pattern, no retry worker, no new failure modes. Cost: audit insert latency (~single-digit ms, same connection) on mutating endpoints — acceptable; these are low-QPS admin/instructor routes, and the high-volume sighting path is deliberately *not* audited (§1).

### 2.3 Failure policy (framed as the decision question it is)

With same-tx semantics, a DB failure fails both writes by definition. The residual question is what happens when *constructing or flushing the audit row* fails while the business write could still succeed:

- **Option F1 — fail the business op (strict):** correct for mandatory events. If we cannot record `CONSENT_GRANT`, the consent-gated operation must not proceed — shipping biometric enrollment without its evidence recreates the exact ATT-044 violation. Recommended default for every action in `MANDATORY_ACTIONS`.
- **Option F2 — log-and-continue (non-strict):** right for events that are pure side-effects with no business write to protect (`LOGIN_SUCCEEDED`, `LOGOUT`, `TASK_READ`): failing a logout because the audit insert failed locks the user out of nothing but their own goodbye, and degrades availability for zero evidentiary gain. Recommended default for advisory events.

This maps 1:1 onto the existing codebase precedent: DB writes are transactional, while Redis publish is explicitly best-effort (`attendance_service.py:91-99`). Audit rows are DB state → transactional; they are not telemetry.

### 2.4 The structural wrinkle: who owns the commit

`AsyncCRUDService.create/update/delete` (`backend/app/services/base.py:68-112`) each open and commit **their own** `transaction()`. Emitting *after* calling them yields two transactions — violating §2.2. Two options:

- **Option A (recommended): extend `base.AsyncCRUDService` with an audit hook (~15 lines).** Add keyword-only `pending_audit: list[AuditEvent] | None = None` (or an internal `_drain_pending_audit()` invoked inside `transaction()` just before commit). Service subclasses append events before delegating to `self.create/update/delete`; base emits them inside the same tx. DRY, one file touched, all three CRUD services get same-tx for free.
- **Option B: stop using base CRUD from audited flows** — each service method inlines the `insert/update/delete` statement inside its own `async with self.transaction():` alongside `AuditService.emit(...)` (copying the 6-line body of base). No base-class change, but duplicates logic in `student_service.py` and `user_service.py` and invites drift.

Raw-session endpoints don't need either option: `POST /students/{id}/enroll` already does explicit `session.add(TemplateAuditLog(...))` + `await session.commit()` (`students.py:373-383`) — add one more `AuditService(session).emit(...)` immediately before that commit. This is literally the existing TemplateAuditLog pattern with a second row type.

### 2.5 Worker path

`run_inference_pipeline`: no governance emission (sightings excluded by design, §1).
`task_evaluate_daily_attendance`: emit `ATTENDANCE_EVALUATE` with `actor_user_id=None` and `change_summary {"source": "celery", "task_id": ..., ...}` inside `_evaluate_daily_attendance`'s existing per-course service call — respects the one-`asyncio.run()` rule since emission happens inside the helper's loop, using the same session factory (`get_session_factory()`), never the FastAPI-loop Redis singleton.

---

## 3. Actor resolution — what's missing and the blast radius

### 3.1 Today's propagation: there is none

Grep result: every service is constructed with a bare session — `StudentService(session)` (5 sites in `students.py`), `UserService(session)` (3 sites in `users.py`), `AttendanceService(session=session)` (`attendance.py:34`, `worker/tasks.py:156,203`, `demo_emitter.py:106`). Every API handler receives the authenticated `User` but discards it:

- `_: CurrentInstructorUser` — students.py:155, 227, 412; inference.py:218
- `_: CurrentUser` — inference.py:85, 151, 164
- `_current_user: CurrentUser` — auth.py:449
- exception: attendance.py:28 binds `current_user` but never uses it

`deps.py` resolves the full `User` ORM object (id available) — so plumbing is purely mechanical renaming + passing.

### 3.2 Required signature changes (exact)

```python
# backend/app/services/student_service.py
def __init__(self, session: AsyncSession, *, actor: User | None = None) -> None:
    super().__init__(session=session, model=Student)
    self._actor = actor

async def create_student(self, payload: StudentCreate) -> Student:      # unchanged signature;
                                                                        # uses self._actor internally
```
Same shape for `UserService.__init__` and `AttendanceService.__init__(session, *, pubsub_manager=None, actor=None)` (keyword-only additions are backward-compatible: the four non-API callers — worker tasks ×2, demo emitter, tests — keep compiling untouched).

Route handlers: rename `_` → `current_user` and pass `actor=current_user` (or set on service ctor):

```python
# backend/app/api/v1/users.py:29
async def create_user(payload: UserCreate, current_user: CurrentAdminUser,
                      session: Annotated[AsyncSession, Depends(get_async_session)]) -> UserRead:
    service = UserService(session, actor=current_user)
```

`request_id` resolution: `RequestIDMiddleware` stores `request.state.request_id` as a **str** (`core/middleware.py:51-52`); `GovernanceLog.request_id` is UUID. Helper in audit_service:

```python
def resolve_request_context(request: Request) -> tuple[UUID | None, str | None]:
    try:
        rid = UUID(request.state.request_id)
    except (AttributeError, ValueError, TypeError):
        rid = None
    ip = request.client.host if request.client else None   # honor X-Forwarded-For policy: Q10
    return rid, ip
```

Handlers needing IP capture add a `request: Request` parameter (FastAPI DI; non-breaking) — only auth.py login/logout and the future consent endpoint per Q4.

### 3.3 Blast radius (files touched for actor plumbing alone)

1. `app/api/deps.py` — no change needed for actors (CurrentUser suffices); only §4 adds the reader dep
2. `app/api/v1/students.py` (4 handlers)
3. `app/api/v1/users.py` (3 handlers)
4. `app/api/v1/auth.py` (3 handlers + `Request` param on login)
5. `app/api/v1/inference.py` (only if advisory events enabled — Q7)
6. `app/api/v1/attendance.py` (rename-only; read endpoint today, real writes arrive with ATT-038)
7. `app/services/student_service.py`
8. `app/services/user_service.py`
9. `app/services/attendance_service.py`
10. `app/services/base.py` (Option A hook)

≈10 files touched purely for actor propagation — this confirms why B20 was un-shippable as a drive-by fix.

---

## 4. RBAC / read surface

### 4.1 AUDITOR role semantics today

`UserRole.AUDITOR = "auditor"` (`domain/models/_base.py:59`) appears in **zero allowlists**: `deps.py` gates ADMIN (`{ADMIN}`), instructor-plus (`{ADMIN, INSTRUCTOR}`), worker-system (`{ADMIN, OPERATOR}`). AUDITOR exists only in denial tests (`test_smoke_rbac_denial.py`) as "a caller who is neither ADMIN nor INSTRUCTOR". Granting AUDITOR any access is therefore a deliberate widening of the role surface — it must be additive and narrow.

### 4.2 Recommended scope: AUDITOR = read-all, write-nothing

ATT-006's acceptance wording ("AUDITOR role can read its own audit trail") is incoherent — auditors perform none of the logged actions, so "their own trail" is empty by construction. Three candidate scopes:

| Scope | Meaning | Verdict |
|---|---|---|
| sees-all | AUDITOR reads every event | **Recommended.** Matches the universal meaning of auditor; separation-of-duties preserved because AUDITOR gains no write power anywhere else (denials stay). |
| sees-by-actor | only own events | useless (empty result set, see above) |
| sees-by-entity-type | configurable subset | premature configurability; revisit only if a regulator demands partitioning |

ADMIN inherits everything. INSTRUCTOR and OPERATOR: denied (403) — instructors see rosters, not the governance ledger; fail closed per skill invariant.

### 4.3 New dependency + endpoint sketch

```python
# backend/app/api/deps.py (append; export both names)
async def get_current_governance_reader(current_user: CurrentUser) -> User:
    return _ensure_user_role(
        current_user,
        allowed_roles={UserRole.ADMIN, UserRole.AUDITOR},
        detail="Auditor or administrator privileges are required for this operation.",
    )
CurrentGovernanceReader = Annotated[User, Depends(get_current_governance_reader)]
```

```python
# backend/app/api/v1/governance.py (new)
router = APIRouter(prefix="/governance", tags=["Governance"])

@router.get("/events", response_model=list[GovernanceLogRead])
async def list_governance_events(
    reader: CurrentGovernanceReader,
    session: Annotated[AsyncSession, Depends(get_async_session)],
    actor_user_id: UUID | None = Query(None),
    entity_type: str | None = Query(None),
    entity_id: UUID | None = Query(None),
    action: str | None = Query(None),
    class_session_record_id: UUID | None = Query(None),
    request_id: UUID | None = Query(None),
    since: datetime | None = Query(None),
    until: datetime | None = Query(None),
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
) -> list[GovernanceLogRead]:
    """Ordered created_at DESC; filters map 1:1 onto existing indexes."""
    return await AuditService(session).list_events(...)
```

Register in `backend/app/api/v1/__init__.py` alongside the other five routers. Response reuses the existing `GovernanceLogRead` schema — no new schema work except maybe a filter-model refactor if we want to move `GovernanceLog*` schemas out of `attendance.py` into `schemas/governance.py` (optional; keeping them in place minimizes diff).

No write endpoints. Ever. Purging is ops-only SQL (Q3).

---

## 5. Retention & tamper-evidence options

| Option | Mechanism | Guarantees | Tradeoffs |
|---|---|---|---|
| **A. App discipline** | Convention: no UPDATE/DELETE statements against `governance_logs` anywhere; reviewed diffs only | none enforceable | Cheapest; weakest; one rogue `session.execute(delete(...))` destroys evidence silently. Acceptable only as a complement. |
| **B. Append-only DB trigger (recommended)** | `BEFORE UPDATE OR DELETE FOR EACH ROW` plpgsql trigger raising an exception; break-glass = `ALTER TABLE ... DISABLE TRIGGER` requiring table-owner privileges the app role must not have | prevents in-place tampering by the application DB user, including via SQL injection in app context | Migration ships function+trigger; downgrade drops both. Deliberately **no `ON TRUNCATE` trigger** — `conftest.py:110` truncates `governance_logs` between tests and would break otherwise (documented in migration docstring). |
| **C. pgcrypto digest chain** | `digest` + `prev_digest` columns; hash = sha256(row ‖ prev); nightly verifier beat task | tamper-*evident* (detects, doesn't prevent); covers even privileged DBAs if anchored externally | 2 extra columns + verifier job + key/anchor management; **chain breaks at every retention purge boundary** (needs periodic anchor rows); pgcrypto extension enablement; significant added complexity for threats B already blocks at the app layer. Defer until a regulator demands cryptographic evidence. |

**Recommendation: A + B now, C deferred.** Ship the trigger in the same migration as the first writer; treat C as its own future issue.

### Retention interplay with ATT-045

- `entity_id` has **no FK** and `actor_user_id`/`class_session_record_id` are SET NULL — deleting embeddings, students, users, or sessions never cascades into `governance_logs`. Consent evidence therefore **outlives** the biometric template it authorized, which is exactly the required direction (BIPA §15 destruction of templates does not authorize destroying proof of consent).
- Governance retention must be ≥ biometric retention: propose `ATTENDANCE_GOVERNANCE_RETENTION_DAYS` (default **2555** ≈ 7y, covering BIPA's 3-year-post-collection horizon plus limitations buffer; final number is Q2).
- Because Option B blocks DELETEs, the eventual purge path cannot be a plain Celery `DELETE`. Provide a `SECURITY DEFINER` plpgsql function `purge_governance_before(cutoff timestamptz)` owned by an ops role which sets `session_replication_role = replica` locally, deletes, restores — callable by the future ATT-045 maintenance task's DB role, never exposed via API. This doubles as the answer to "who can purge".

---

## 6. Migration sketch

New revision `20260824_0005_governance_action_domain_and_append_only`, `down_revision = "20260519_0004"` (head; chain is linear: `0001 → 0002 → 0003 → 0004`). No table/column changes — constraints, index, trigger only:

```python
def upgrade() -> None:
    # 1. Vocabulary pinning: CHECK over implemented actions (extend via future
    #    migrations when reserved actions ship — deliberate friction).
    op.create_check_constraint(
        "governance_action_domain",
        "governance_logs",
        sa.text("action IN ('USER_CREATE','USER_UPDATE','USER_DELETE',"
                "'STUDENT_CREATE','STUDENT_UPDATE','STUDENT_DELETE',"
                "'TEMPLATE_ENROLL','ATTENDANCE_EVALUATE','REFRESH_REUSED')"),
    )
    # 2. Action-filtered listing support (GET /governance/events?action=...&since=...).
    op.create_index(
        "ix_governance_action_created_at", "governance_logs",
        ["action", sa.text("created_at DESC")],
    )
    # 3. Append-only guard (Option B). NOTE: intentionally no ON TRUNCATE
    #    component — test fixtures TRUNCATE this table (conftest._DOMAIN_TABLES).
    op.execute("""
        CREATE FUNCTION forbid_governance_mutation() RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION 'governance_logs is append-only'
                USING ERRCODE = 'restrict_violation';
        END $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER trg_governance_append_only
        BEFORE UPDATE OR DELETE ON governance_logs
        FOR EACH ROW EXECUTE FUNCTION forbid_governance_mutation();
    """)

def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_governance_append_only ON governance_logs")
    op.execute("DROP FUNCTION IF EXISTS forbid_governance_mutation()")
    op.drop_index("ix_governance_action_created_at", table_name="governance_logs")
    op.drop_constraint("governance_action_domain", "governance_logs", type_="check")
```

Round-trip safe for CI's `upgrade head → downgrade base → re-upgrade` (table is empty pre-existing, so the CHECK validates trivially; trigger drop precedes any downgrade table work). Alternative considered: native PG enum for `action` — rejected: `ALTER TYPE ADD VALUE` choreography across environments and messier downgrades vs. a CHECK that matches the existing constraint style (`governance_action_not_blank`).

ORM note: model change is optional (a CHECK lives happily outside SQLAlchemy metadata). If the team wants the model to reflect it, add to `governance.py __table_args__` — then CI round-trips cover it automatically. Recommend adding it to the model for documentation value.

---

## 7. Test plan + acceptance criteria

New file `backend/tests/test_smoke_governance.py` (existing fixtures `admin_user`, `instructor_user`, `auditor_user`, `operator_user`, `auth_cookie`, `async_client` in conftest suffice; `_DOMAIN_TABLES` already truncates `governance_logs` first — dependency order correct):

1. **Row-on-mutation (ATT-006 core acceptance):** instructor `POST /api/v1/students` → assert one `governance_logs` row with `action='STUDENT_CREATE'`, `entity_type='student'`, `entity_id == created.id`, `actor_user_id == instructor.id`.
2. **Atomicity:** duplicate `student_number` create → 409 → `SELECT count(*) FROM governance_logs` == 0. Proves same-tx (the fire-and-forget counterexample).
3. **Enrollment coexistence:** `POST /students/{id}/enroll` → `TEMPLATE_ENROLL` governance row AND ≥1 `template_audit_logs` row; assert `change_summary['quality_score']` present and response/summary contain no vector material (biometric-data rule regression guard).
4. **Actor anonymization on delete:** delete the acting user → prior rows survive with `actor_user_id IS NULL` (FK SET NULL contract).
5. **RBAC matrix on `GET /api/v1/governance/events`:** ADMIN 200, AUDITOR 200, INSTRUCTOR 403, OPERATOR 403, anonymous 401. Mirrors `test_smoke_rbac_denial.py` style.
6. **Filters:** seed rows; assert filtering by `action`, `entity_id`, `since/until`, pagination offsets.
7. **Append-only enforcement:** `session.execute(text("UPDATE governance_logs SET action='X'"))` and `DELETE FROM` both raise `DBAPIError`; TRUNCATE succeeds (fixture implicitly re-proves this every test).
8. **System actor:** call `AttendanceService.evaluate_class_attendance` directly → `ATTENDANCE_EVALUATE` row, `actor_user_id IS NULL`, `change_summary["source"] == "celery"` (direct-service variant; no Celery needed).
9. **Refresh-replay signal:** `POST /auth/refresh` twice with same cookie → second returns 401 AND `REFRESH_REUSED` row exists.
10. **Migration round-trip:** covered by existing CI alembic job; locally `alembic upgrade head && alembic downgrade -1 && alembic upgrade head`.

**Acceptance criteria (literal mapping to ATT-006):**
- AC1: every endpoint that mutates attendance/embedding/user state writes a GovernanceLog row → taxonomy §1.1 mandatory rows wired + tests 1–3, 8–9.
- AC2: AUDITOR can read the audit trail → endpoint §4.3 + test 5 (scope reinterpretation documented as Q5).
- AC3: integration test asserts a row is created on a state change → test 1/2 pair.

---

## 8. Implementation order + effort estimate

| Phase | Contents | Files |
|---|---|---|
| **P0** (human) | Sign off §9 open questions — blocks nothing below Q1/Q5 but blocks ATT-044/038 reuse | 0 files |
| **P1 foundation** | `services/audit_service.py` (enum, AuditEvent, AuditService.emit/list_events, helpers); `services/__init__.py` exports; migration 0005; model `__table_args__` CHECK mirror | 4 new/mod |
| **P2 read surface** | `api/deps.py` `CurrentGovernanceReader`; `api/v1/governance.py`; `api/v1/__init__.py` registration; RBAC + filter tests | 3 mod + test file |
| **P3 write wiring** | `services/base.py` audit hook (Option A); `student_service.py`, `user_service.py`, `attendance_service.py` actor kwarg + emissions; `students.py`, `users.py`, `auth.py` handler renames + passes; `worker/tasks.py` system event | 8 mod |
| **P4 hardening/docs** | `test_smoke_governance.py` completion; ARCHITECTURE.md §governance subsection (currently absent — grep shows no governance section); RUNBOOK purge-function note | 1 test + 2 docs |

**Full concrete file list:**

New (4): `backend/app/services/audit_service.py`, `backend/app/api/v1/governance.py`, `backend/alembic/versions/20260824_0005_governance_action_domain_and_append_only.py`, `backend/tests/test_smoke_governance.py`.

Modified core (11): `backend/app/domain/models/governance.py`, `backend/app/services/__init__.py`, `backend/app/services/base.py`, `backend/app/services/student_service.py`, `backend/app/services/user_service.py`, `backend/app/services/attendance_service.py`, `backend/app/api/deps.py`, `backend/app/api/v1/__init__.py`, `backend/app/api/v1/students.py`, `backend/app/api/v1/users.py`, `backend/app/api/v1/auth.py`, plus conditionally `backend/app/api/v1/inference.py` (Q7) and `backend/app/worker/tasks.py`.

Docs (2): `ARCHITECTURE.md`, `RUNBOOK.md`.

**Verdict on "~10–12 files":** slightly optimistic but in the right zip code. Realistic count: **13–16 code files + 2 docs** depending on whether advisory inference/auth events ship in v1. Effort: ~600–800 LOC including tests; roughly a focused 3–4 day sprint for one engineer, dominated by wiring (P3) and policy sign-off (P0), not by the service itself (~150 lines). The original 'L' rating in the issue stands.

Sequencing rationale: P1/P2 deliver observable value (readable ledger + enforced vocabulary) without touching any business path — reviewable independently; P3 is the wide-diff phase and should be one PR per concern if split (users.py+user_service vs students.py+student_service) to keep review tractable under the repo's one-batch-one-PR convention (this whole effort = one branch `fix/B20-audit-service`, or two stacked batches if reviewers prefer).

---

## 9. Open questions for the human (each with recommended default)

| # | Question | Recommended default |
|---|---|---|
| Q1 | Which events are mandatory (fail-the-op) vs advisory (log-and-continue)? | Mandatory: all student/template/user lifecycle + `ATTENDANCE_EVALUATE` + `REFRESH_REUSED` + future consent/override/export. Advisory: `LOGIN_SUCCEEDED`, `LOGOUT`, `INFERENCE_ENQUEUED`, `RECOGNITION_RUN`. `TASK_READ` advisory-disabled until Q7 answered. |
| Q2 | Retention duration for `governance_logs`? | 2555 days (7y) via `ATTENDANCE_GOVERNANCE_RETENTION_DAYS`; must exceed ATT-045's 3-year embedding horizon; purge only via ops-owned `SECURITY DEFINER` function, never an API. |
| Q3 | Who can purge? | No app/API principal ever. Ops role via SECURITY DEFINER `purge_governance_before(cutoff)`; Celery maintenance role may *call* it post-ATT-045; every invocation itself logs a governance row written before the trigger-disabled delete. |
| Q4 | Do consent events require IP capture given GDPR Art. 5(1)(c) minimization? | Yes for consent/auth events only — IP is the signature-evidence datum that makes non-repudiation meaningful for Art. 9 processing; purpose-bound so minimization is satisfied. Routine domain events (`STUDENT_UPDATE` etc.) capture no IP. Store nullable INET, never log it elsewhere. |
| Q5 | AUDITOR read scope (sees-all / by-actor / by-entity)? | Sees-all, read-only, plus nothing else anywhere (fail-closed denials unchanged). Rewrite ATT-006's "own audit trail" wording accordingly. |
| Q6 | Log failed login attempts? | No governance rows (PII + volume + brute-force amplification); aggregate counters/metrics instead. Exception already in scope: `REFRESH_REUSED` (rare, high-value). |
| Q7 | Enable `TASK_READ` auditing? | Off by default — frontend polling makes volume unpredictable; enable behind env flag when a data-access-report requirement materializes. |
| Q8 | Vocabulary governance process? | CHECK-constrained; adding a value requires a migration (deliberate friction). Reserved values documented in enum now. |
| Q9 | `change_summary` content bounds? | Field names + non-secret scalar values only; never password hashes, tokens, embedding vectors, `embedding_reference`; enforced by unit test asserting forbidden keys. |
| Q10 | Trust `X-Forwarded-For` for IP capture? | Not until deployment topology decides; capture `request.client.host` only (documented limitation behind reverse proxies). |

---

## Appendix: verification commands (for the implementing sprint)

```bash
pip install -e ".[dev]" -c constraints.txt
docker compose -f docker-compose.dev.yml up -d postgres redis   # host port 15432
cd backend && alembic upgrade head
python -m pytest -m "not slow" backend/tests/test_smoke_governance.py   # new suite
python -m pytest -m "not slow"                                          # full gate
ruff check --select=F backend                                           # CI lint parity
```
