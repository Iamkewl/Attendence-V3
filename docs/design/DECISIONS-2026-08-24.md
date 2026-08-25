# Product decisions — recorded 2026-08-24

Owner sign-off for the three deferred design sprints. Source docs:
`/tmp/opencode/design/{audit-service,course-instructors,enrollment-importer}-design.md`.
Each kickoff PR must carry its spec + this record.

## Audit service (ATT-006 foundation)

| # | Decision |
|---|---|
| D1 | Lifecycle events (student/template/user, ATTENDANCE_EVALUATE, REFRESH_REUSED, future consent/override/export) are **mandatory** (fail-the-op); auth events (LOGIN_SUCCEEDED, LOGOUT, INFERENCE_ENQUEUED, RECOGNITION_RUN) are **advisory** (log-and-continue). TASK_READ stays disabled. |
| D2 | Retention: **2555 days (7y)** via `ATTENDANCE_GOVERNANCE_RETENTION_DAYS`. Must exceed biometric horizon (ATT-045). |
| D3 | Purge: **ops-only** `SECURITY DEFINER purge_governance_before(cutoff)`; never an API principal; every invocation logs a governance row first. |
| D4 | IP capture: **consent + auth events only**, nullable INET, never logged elsewhere. Routine domain events store none. |
| D5 | AUDITOR scope: **sees-all, read-only**, fail-closed denials elsewhere unchanged. Rewrite ATT-006 wording accordingly. |
| D6 | Failed logins: **aggregate metrics only** (no rows); REFRESH_REUSED remains row-logged. |
| D7 | TASK_READ auditing: **off by default**, env-flag-ready. |

## Course instructors (ATT-016 / ATT-037)

| # | Decision |
|---|---|
| D8 | Schema: **Option A** — M:N `course_instructors`, `(course_id,user_id)` unique, `role_in_course CHECK ('owner','ta')`, CASCADE FKs. Term scoping deferred to ATT-040. |
| D9 | Rollout: `ATTENDANCE_COURSE_SCOPED_AUTHZ` **defaults false**; backfill driven by human CSV (`course_code,instructor_email,role_in_course`), dry-run until `--apply`; script ships inert without the CSV. |
| D10 | Phase-1 edges: TA assignments **stored-but-denied**; cross-listed courses require an explicit second row (no guessed links). |

## Enrollment importer (ROADMAP §2 slice 1)

| # | Decision |
|---|---|
| D11 | Identifier: **`student_number`** binds manifest photos to students. |
| D12 | Overwrite default: **skip existing active (student,pose)**; opt-in `--overwrite` rotates API-style. One-active-per-pose invariant. |
| D13 | **Add `template_version` column now** (Alembic migration, round-trip-safe downgrade required by CI). |
| D14 | Photos referencing inactive/unenrolled refs get dedicated reason code **INACTIVE_STUDENT** (fail closed, dashboard-distinguishable). |
| D15 | Identical image bound to multiple refs: import later ones + **warn counter** in coverage report. |
| D16 | Consent metadata: **deferred to consent sprint**; no passthrough field in slice 1. |
