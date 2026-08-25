# Design: Instructor↔Course Ownership for Course-Scoped Attendance Authz

**Repo:** /home/activetest/attendance-fix (Attendance V3)
**Scope:** design research only — no code modified. Unblocks ATT-016 (abandoned twice, `fixes/NOTES.md:1180,1199`) and the instructor half of ATT-037 (deferred, `fixes/ROADMAP.md:16`).
**Read evidence:** `backend/app/domain/models/{course,user,student,sighting,session,governance,_base}.py`, `backend/app/api/v1/attendance.py`, `backend/app/api/deps.py`, `backend/app/services/attendance_service.py`, `backend/alembic/versions/20260519_0004_pgvector_hnsw_index.py` (head), `fixes/BATCHES.md` B10, `backend/scripts/seed_demo_data.py`.

---

## 0. Current state (what the code actually does)

- `Course` (`domain/models/course.py:18`) has **no relationship to `User`** — no owner column, no association. Only `class_session_records` and `sightings`.
- `User.role` is one global enum value: `admin|instructor|auditor|operator` (`_base.py:54-60`). "Instructor" is a platform-wide role, not a course link.
- `GET /api/v1/attendance/sessions` (`api/v1/attendance.py:27-44`) requires **only `CurrentUser`** — not even `CurrentInstructorUser`. Any active authenticated user passes any `course_id` UUID and gets a student-name roster.
- `get_current_instructor_user` (`api/deps.py:83-89`) is a pure role check (`{ADMIN, INSTRUCTOR}`); it cannot express "this instructor, this course". Used by 5 routes in `students.py` and `recognize_photo` in `inference.py:218`.
- Roster data path: `AttendanceService.list_session_records` (`services/attendance_service.py:195-279`) reads only existing `class_session_records`, which are produced solely by `evaluate_class_attendance` aggregating **sightings**. There is no enrollment table anywhere; `Student` is a global row with `user_id` FK (`student.py:47-51`).
- Latent adjacent bug: `_require_course` raises `AttendanceValidationError("Course is inactive.")` (`attendance_service.py:336`) but `list_session_records` route catches only `AttendanceNotFoundError` → an inactive course yields **HTTP 500**, not 4xx.

---

## 1. Schema options

### Option A — M:N `course_instructors` association table (recommended)

DDL (names follow `MODEL_NAMING_CONVENTION`, `_base.py:15-23`; UUID PK + timestamps via existing mixins):

```sql
CREATE TABLE course_instructors (
    id             uuid        NOT NULL PRIMARY KEY,                      -- pk_course_instructors
    course_id      uuid        NOT NULL,
    user_id        uuid        NOT NULL,
    role_in_course varchar(32) NOT NULL DEFAULT 'owner',                  -- CHECK below
    created_at     timestamptz NOT NULL DEFAULT now(),
    updated_at     timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT pk_course_instructors PRIMARY KEY (id),
    CONSTRAINT uq_course_instructors_course_user UNIQUE (course_id, user_id),
    CONSTRAINT fk_course_instructors_course_id_courses
        FOREIGN KEY (course_id) REFERENCES courses(id) ON DELETE CASCADE,
    CONSTRAINT fk_course_instructors_user_id_users
        FOREIGN KEY (user_id)   REFERENCES users(id)   ON DELETE CASCADE,
    CONSTRAINT ck_course_instructor_role_valid
        CHECK (role_in_course IN ('owner', 'ta'))
);
CREATE INDEX ix_course_instructors_user_id ON course_instructors (user_id);
```

Why `varchar(32)`+CHECK and **not** the native `user_role` enum: extending a Postgres native enum needs `ALTER TYPE ... ADD VALUE`, which cannot run inside a transaction block (Alembic runs migrations transactionally by default) and makes the CI downgrade round-trip asymmetric. A CHECK constraint downgrades cleanly.

Alembic sketch — new revision `2026xxxx_0005_course_instructors.py`, `down_revision = "20260519_0004"`:

```python
def upgrade() -> None:
    op.create_table(
        "course_instructors",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("course_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("role_in_course", sa.String(32), nullable=False,
                  server_default=sa.text("'owner'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.PrimaryKeyConstraint("id", name="pk_course_instructors"),
        sa.UniqueConstraint("course_id", "user_id", name="uq_course_instructors_course_user"),
        sa.CheckConstraint("role_in_course IN ('owner','ta')", name="ck_course_instructor_role_valid"),
        sa.ForeignKeyConstraint(["course_id"], ["courses.id"], ondelete="CASCADE",
                                name="fk_course_instructors_course_id_courses"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE",
                                name="fk_course_instructors_user_id_users"),
    )
    op.create_index("ix_course_instructors_user_id", "course_instructors", ["user_id"])

def downgrade() -> None:
    op.drop_index("ix_course_instructors_user_id", table_name="course_instructors")
    op.drop_table("course_instructors")   # constraints/index drop with the table
```

ORM: new file `domain/models/course_instructor.py` mirroring `governance.py` style; export from `models/__init__.py`. Relationships: `Course.instructor_links`, `User.course_assignments` (`lazy="select"`, consistent with every other model here).

**Hot-path cost:** one extra point SELECT per authorized request:
`SELECT 1 FROM course_instructors WHERE user_id=? AND course_id=?` — PK-shaped via the unique `(course_id,user_id)` index, sub-millisecond. `list_session_records` already issues ~5 queries (`attendance_service.py:203-256`); this adds <10% to that route and nothing else. The check must be executed fresh per request — **never memoize the link** (same reasoning as the Triton-client seam rule in `.opencode/skills/attendance-v3-context/SKILL.md`: caching defeats mid-session revocation).

**Fit with real code:** matches the fix text recorded for ATT-037 (`fixes/issues.json:334`: "`course_instructors` association table (`course_id`,`user_id`)") and the abandonment note (`NOTES.md:1181`). CASCADE on both FKs is safe: deleting a User cascades their Student profile already (`student.py:49`); deleting a Course is blocked by `Sighting.course_id RESTRICT` (`sighting.py:46`) anyway, so the CASCADE branch is nearly unreachable but semantically right.

### Option B — FK on `courses` (single lead-instructor column)

```sql
ALTER TABLE courses
    ADD COLUMN lead_instructor_user_id uuid REFERENCES users(id) ON DELETE SET NULL;
```
Downgrade: `ALTER TABLE courses DROP COLUMN lead_instructor_user_id;` (trivial round-trip).

**Hot-path cost:** zero extra queries — `_require_course` (`attendance_service.py:324-336`) already loads the `Course` row; authz can read `course.lead_instructor_user_id` off it. Cheapest possible.

**Why rejected:** single owner only — no co-instructors, no TA path (the B10 notes explicitly require a TA policy); ownership transfer silently overwrites history with no audit trail (`GovernanceLog` exists precisely for such events, `governance.py:20`); `ON DELETE SET NULL` creates *ownerless* courses which under fail-closed rules become admin-only orphan data, while `RESTRICT` blocks legitimate staff deprovisioning. It also diverges from the documented ATT-037 fix intent.

### Option C — term-scoped assignments

```sql
CREATE TABLE course_instructor_assignments (
    id          uuid PRIMARY KEY,
    course_id   uuid NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    user_id     uuid NOT NULL REFERENCES users(id)   ON DELETE CASCADE,
    term_code   varchar(16) NOT NULL,
    valid_from  date,
    valid_to    date,
    assigned_by uuid REFERENCES users(id) ON DELETE SET NULL,
    CONSTRAINT uq_cia_course_user_term UNIQUE (course_id, user_id, term_code)
);
```
Authz predicate becomes `WHERE user_id=? AND course_id=? AND term_code=current_term()` — correct academically, but there is **no term/schedule entity anywhere in the schema** (timetable is itself deferred as ATT-040, `ROADMAP.md:20`). Every authz query would need a term source of truth that doesn't exist, and seeding/backfill would require inventing term codes by hand.

**Hot-path cost:** same shape as A plus temporal predicate (still index-covered), but the operational cost (term lifecycle management, rollover jobs) lands before any timetable system exists.

### Recommendation

**Option A, without term columns in v1.** It is the only option that satisfies all three blockers named in the B10 abandonment (`NOTES.md:1181`): seed binding, graceful 403, TA policy hook. Option C's semantics are reachable later without breaking anything: widen `course_instructors` into assignments by adding nullable `term_code` columns when ATT-040 lands; the v1 authz predicate (`user_id, course_id`) remains a valid subset query. Option B should be rejected even as a stopgap because tightening `CurrentInstructorUser` against it would have to be re-tightened again after migration to A.

---

## 2. Is a `course_students` roster required for attendance-read authz?

**No — not for the authz gate.** Read authorization is a property of the **caller↔course** edge (`course_instructors`), not of caller↔students. Adding enrollment first would delay ATT-016 behind another product decision for zero additional denial power: today's roster endpoint serves whatever rows exist in `class_session_records`, and access control on it changes not at all based on whether an `enrollments` table exists.

**But the served data is sightings-derived, and that leaks in two directions:**

1. **Over-inclusion (privacy leak):** `Sighting.course_id` is caller-supplied and never validated against an enrollment (`log_sighting`, `attendance_service.py:48-101`). A student sighted in room/camera X appears on course X's roster regardless of registration; `evaluate_class_attendance` then upserts them into `class_session_records` permanently. Without enrollment, cross-course presence disclosure is baked into the data — exactly the FERPA concern raised in `issues.json:337`.
2. **Under-inclusion (fairness gap):** enrolled students who were never sighted produce **no** record — `evaluate_class_attendance` iterates only over sighted/existing student_ids (`attendance_service.py:144`), so true absents are invisible and cannot be marked ABSENT or EXCUSED. This is what a roster fixes, not authz.

Also note `seed_demo_data.py:106`: demo "students" get `UserRole.INSTRUCTOR` platform accounts — under any future enrollment model these five users must **not** inherit instructor-grade read paths.

**Recommendation:** keep enrollment (`enrollments` table, ATT-037 part b) deferred as its own batch. Phase 1 ships `course_instructors` only; document explicitly that rosters are sighting-attributed until enrollment exists. Do not let the two land in one PR — combined they re-trigger scope-discipline abandon trigger #1 that killed B10.

---

## 3. Backfill script shape (human-gated)

New file `backend/scripts/backfill_course_instructors.py`, same execution pattern as `seed_demo_data.py` (async entrypoint, `ATTENDANCE_DATABASE_URL` from env, exit 0/1):

```
Usage:
  python scripts/backfill_course_instructors.py --mapping course_owners.csv --dry-run
  python scripts/backfill_course_instructors.py --mapping course_owners.csv --apply
```

- **Input:** a CSV produced by a human (`course_code,instructor_email,role_in_course`), committed nowhere — it is a product data decision (`NOTES.md:1181` says exactly this). Script ships inert without it.
- **Dry-run default:** prints planned `(course_id, user_id, role)` triples and a coverage report ("N of M active courses unassigned"); `--apply` additionally requires env confirmation (`ATTENDANCE_BACKFILL_CONFIRM=YES`) — two independent human actions, matching the repo's fail-closed posture.
- **Idempotent:** `INSERT ... ON CONFLICT (course_id, user_id) DO NOTHING` (deterministic like the seeder's fixed UUIDs).
- **Refuses to guess:** unknown email or course code → collect all errors, print them, exit 1 having written nothing. Never fuzzy-matches names.
- **Auditable:** each applied insert also writes a `GovernanceLog` row (`action='course_instructor.backfill'`, `entity_type='course_instructors'`, actor = executing admin), using the existing table (`governance.py:20`).
- **Demo binding:** extend `seed_demo_data.py` in-batch with one deterministic `course_instructors` row binding a dedicated demo instructor (or the existing admin) to `DEMO_COURSE_ID` (`00000000-…-0001`), so `make demo` keeps working once the flag flips. Do **not** bind the five `INSTRUCTOR`-roled student accounts (§2).

Gating: PR carries `do-not-merge` label until a human reviews the mapping procedure (per AGENTS.md auto-merge trap); production rollout order is §5-Q3.

---

## 4. Tightening `CurrentInstructorUser`

### Diff shape (phase 1: `/attendance/sessions` only)

`backend/app/api/deps.py` — add a course-scoped dependency alongside the existing ones (FastAPI injects the shared `course_id` query param into dependencies automatically):

```python
# deps.py (new, ~30 lines)
from app.domain.models import CourseInstructor

async def get_course_scoped_principal(
    course_id: UUID,
    current_user: CurrentUser,
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> User:
    """ADMIN bypasses; otherwise require a live course_instructors row."""
    settings = get_security_settings()
    if not settings.course_scoped_authz_enabled:      # rollout flag, see Q3
        return _ensure_user_role(current_user, allowed_roles={ADMIN, INSTRUCTOR},
                                 detail="...")          # legacy behavior
    if current_user.role == UserRole.ADMIN:
        return current_user
    if current_user.role != UserRole.INSTRUCTOR:
        raise HTTPException(403, "Not authorized for this course roster.")
    linked = await session.scalar(
        select(CourseInstructor.id)
        .where(CourseInstructor.user_id == current_user.id,
               CourseInstructor.course_id == course_id,
               CourseInstructor.role_in_course == "owner"))
    if linked is None:
        raise HTTPException(403, "You are not assigned to this course.")
    return current_user

CourseScopedPrincipal = Annotated[User, Depends(get_course_scoped_principal)]
```

`backend/app/api/v1/attendance.py` — one-line swap plus error mapping:

```diff
-from app.api.deps import CurrentUser
+from app.api.deps import CurrentScopedPrincipal as ...
 ...
 async def list_session_records(
-    current_user: CurrentUser,
+    _: CourseScopedPrincipal,
@@
     except AttendanceNotFoundError as exc:
         raise HTTPException(404, ...) from exc
+    except AttendanceValidationError as exc:      # fixes latent inactive-course 500
+        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc
```

No service-signature change: authorization stays in the API layer, preserving `api → services` layering; `AttendanceService.list_session_records` is untouched.

### 403 vs 404 policy (enumeration-safe ordering)

| Situation | Response | Why |
|---|---|---|
| Caller linked / ADMIN, course missing | **404** | existence revealed only to principals who'd otherwise pass |
| Caller unlinked (any role incl. OPERATOR/AUDITOR), course missing **or real** | **403 always** | checking the link *first* denies without confirming the UUID exists → no course-ID oracle |
| Link revoked mid-session (JWT still valid) | **403 on next request** | link is re-read from DB every request (no cache); in-flight request completes; frontend must render the 403 gracefully — this is the "graceful degradation" the B10 note demanded |
| Course deactivated mid-session | **409** after fix (was 500) | distinct signal from revocation |

Fail-closed rule honored: unknown state (flag on + backfill gap) denies; it never falls open.

### TA matrix (no `ta` enforcement in v1 — stored but denied)

| Caller | `owner` link | `ta` link | no link |
|---|---|---|---|
| ADMIN | allow (bypass) | allow (bypass) | allow (bypass) |
| INSTRUCTOR | **allow** | deny v1 (fail-closed until policy Q1) | 403 |
| OPERATOR | 403 | 403 | 403 |
| AUDITOR | flag-gated (Q5) | flag-gated (Q5) | flag-gated (Q5) |

### Affected routes

| Route | Today | Phase 1 change |
|---|---|---|
| `GET /api/v1/attendance/sessions` (`attendance.py:27`) | `CurrentUser` — **any** authenticated user | `CourseScopedPrincipal` (this design) |
| `POST /api/v1/students/*` ×5 (`students.py:153-438`) | `CurrentInstructorUser`, global | unchanged phase 1 — students are non-course-scoped entities; scoping them needs enrollment (§2) |
| `POST /api/v1/inference/stream`, `/batch` (`inference.py:84,149`) | `CurrentUser`, optional `course_id` | unchanged phase 1; phase 2 candidate (validate link when `course_id` supplied) |
| `POST /api/v1/inference/recognize-photo` (`inference.py:217`) | `CurrentInstructorUser`, optional `course_id` | cheap phase-1 add-on: if `course_id` present, apply same link check |
| `GET /api/v1/inference/tasks/{id}` (`inference.py:164`) | `CurrentUser`, strips embeddings | unchanged (biometric rule preserved) |

---

## 5. Open policy questions (each with recommended default)

| # | Question | Recommended default |
|---|---|---|
| Q1 | **TA scope** — no TA concept exists in `UserRole`; B10 demands a policy | Ship `role_in_course` column now with CHECK `('owner','ta')`; authorize **only** `'owner'` in v1; `'ta'` rows are stored-but-denied (fail-closed). Deciding TA read-vs-write later is a one-line predicate change, not a migration. |
| Q2 | **Cross-listing** — same meeting under two course codes | No automatic sharing. Require an explicit second `course_instructors` row per code. Cross-list resolution belongs to the deferred timetable work (ATT-040); guessing links now would encode unmade product decisions. |
| Q3 | **Fail-closed fallback / rollout** — flipping authz on with zero rows locks out every instructor incl. demo | Env flag `ATTENDANCE_COURSE_SCOPED_AUTHZ` (default **false**, documented in `.env.example`). Rollout: migrate → run reviewed backfill (coverage report shows 0 unassigned active courses) → flip flag. When on, missing link ⇒ 403 unconditionally; never auto-provision a fallback grant. |
| Q4 | **OPERATOR overlap** — operators feed cameras/pipelines; do they read rosters? | No. OPERATOR keeps exactly its current surface (`CurrentWorkerSystem`, `deps.py:92-98`) and loses nothing it has today; roster reads require ADMIN or an `owner` link. Camera-ops needing context can later receive explicit rows rather than role widening. |
| Q5 | **AUDITOR visibility** — issue body suggests a configurable limit | Phase 1: AUDITOR keeps global roster read (that is the role's charter and `test_smoke_rbac_denial.py` documents it), gated by the same flag discussion as Q3; revisit as a dedicated decision rather than smuggling a narrowing into ATT-016. |
| Q6 | Inactive-course currently 500s (`attendance_service.py:336` vs route catch) | Map `AttendanceValidationError` → 409 on this route in the same PR (one except-clause); it touches only lines the diff already owns. |

---

## 6. CI round-trip safety notes

- CI round-trips `upgrade head → downgrade base → re-upgrade` (AGENTS.md). The proposed downgrade is a bare `drop_table`/`drop_index` — symmetric by construction; no data-preservation promises needed for a pure association table.
- **Do not touch native enums.** Reusing `user_role` for `role_in_course` would need `ALTER TYPE ... ADD VALUE` (non-transactional, breaks the round-trip); the String+CHECK avoids it entirely (§1-A).
- Pass explicit constraint/index names matching `MODEL_NAMING_CONVENTION` (`fk_%(table)s_%(column_0_name)s_%(referred_table)s`, etc.) so ORM metadata and DB agree and future `alembic revision --autogenerate` emits empty diffs instead of spurious drops.
- Follow the house migration style (`20260501_0001`): `sa.text(...)` server defaults, `postgresql.UUID(as_uuid=True)`, `Sequence` typing imports — keeps `filterwarnings=["error"]` quiet.
- Tests require migrated schema (`conftest.py` applies alembic), so the new table is visible to pytest automatically once the revision lands; no conftest change needed beyond new fixtures.
- Keep `seed_demo_data.py` idempotency intact: the added `course_instructors` row uses the same fixed-UUID + existence-check pattern (`session.get`) as every other seeded entity.
- No pgvector/HNSW interaction (revision `20260519_0004` concerns `student_embeddings` only); no Celery/loop-bound-singleton surface touched.

---

## 7. Test plan & files-touched estimate

**Tests**

- New `backend/tests/test_smoke_att016_course_scoped_authz.py`:
  - matrix: unlinked INSTRUCTOR→403; linked INSTRUCTOR→200; ADMIN→200; OPERATOR→403; unauthenticated→401;
  - enumeration order: unlinked caller + nonexistent UUID → 403 (never 404);
  - ADMIN + nonexistent course → 404;
  - mid-session revoke: delete link row, replay same cookie → 403;
  - flag-off regression test asserting legacy behavior (protects rollout);
  - inactive course → 409 (Q6).
- Extend `test_smoke_rbac_denial.py` route registry with the tightened route (its header docstring demands every role-gated route appear there).
- Backfill script: dry-run test with tmp CSV against test DB (unknown-email refusal, idempotent re-run, coverage report).
- Migration: covered by CI round-trip; add assertion in operations smoke if one enumerates tables.

**Files touched (~11; 4 new)**

| File | Change |
|---|---|
| `backend/app/domain/models/course_instructor.py` | new (~50 LOC) |
| `backend/app/domain/models/__init__.py` | exports |
| `backend/alembic/versions/2026xxxx_0005_course_instructors.py` | new |
| `backend/app/api/deps.py` | +~35 LOC dependency |
| `backend/app/api/v1/attendance.py` | dependency swap + 409 clause |
| `backend/app/core/config` (settings class hosting the flag) | +1 bool |
| `.env.example` | flag documentation |
| `backend/scripts/seed_demo_data.py` | +demo link row |
| `backend/scripts/backfill_course_instructors.py` | new (~150 LOC) |
| `backend/tests/test_smoke_att016_course_scoped_authz.py` | new (~250 LOC) |
| `backend/tests/test_smoke_rbac_denial.py` (+conftest fixtures) | registry/fixtures |

Net ≈ +650/-25 LOC. Above the S ceiling; fits one batch only if the backfill script ships as its own follow-up batch — otherwise it re-triggers the scope-discipline abandon conditions that killed B10.

---

## Decisions needed from humans

1. Confirm Option A (M:N, no term columns) over B/C.
2. Approve flag-defaulted rollout (Q3) and who produces the instructor↔course mapping CSV.
3. Rule on AUDITOR global read retention (Q5) and OPERATOR exclusion (Q4).
4. Accept TA stored-but-denied v1 (Q1) and no-cross-listing (Q2) defaults.
