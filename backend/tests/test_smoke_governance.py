"""ATT-006 smoke suite — audit_service, GovernanceLog wiring, RBAC, append-only ledger.

Covers the design test-plan (audit-service-design.md §7):
same-transaction atomicity, mandatory-vs-advisory policy matrix, the
/governance/events RBAC matrix (D5), the append-only trigger, system-actor
worker-path events, actor propagation, IP capture on auth events only (D4),
no rows for failed logins (D6), TASK_READ gating (D7), the ops purge function
(D3), and the privacy rule that forbidden secret keys never reach
``change_summary`` (Q9).
"""

from __future__ import annotations

import base64
import uuid
from datetime import UTC, datetime, timedelta
from io import BytesIO

import pytest
from httpx import AsyncClient
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _governance_rows(session_factory) -> list:
    """Return every governance_logs row ordered newest-first."""
    from app.domain.models import GovernanceLog

    async with session_factory() as session:
        rows = list(
            (
                await session.execute(
                    select(GovernanceLog).order_by(GovernanceLog.created_at.desc())
                )
            ).scalars().all()
        )
    return rows


async def _seed_student(session_factory, *, number: str = "GOV0001", is_active: bool = True):
    """Insert one active user + student pair directly; return (user, student)."""
    from app.core.security import hash_password
    from app.domain.models import Student, User, UserRole

    async with session_factory() as session:
        link_user = User(
            id=uuid.uuid4(),
            email=f"{number.lower()}@test.example",
            full_name="Linked User",
            password_hash=hash_password("TestPass1!"),
            role=UserRole.AUDITOR,
            is_active=True,
        )
        session.add(link_user)
        await session.flush()
        student = Student(
            id=uuid.uuid4(),
            user_id=link_user.id,
            student_number=number,
            program="Test Program",
            enrollment_year=2024,
            is_active=is_active,
        )
        session.add(student)
        await session.commit()
        return link_user, student


def _student_payload(link_user_id) -> dict:
    return {
        "user_id": str(link_user_id),
        "student_number": f"API{uuid.uuid4().hex[:8].upper()}",
        "program": "Test Program",
        "enrollment_year": 2024,
    }


# ---------------------------------------------------------------------------
# Privacy: forbidden keys are stripped from change_summary (design Q9)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_change_summary_forbidden_keys_never_persist(_session_factory) -> None:
    from app.services.audit_service import AuditEvent, AuditService, GovernanceAction

    poisoned = {
        "embedding": [0.0] * 8,
        "embedding_reference": "sha256:deadbeef",
        "matched_embedding_id": "x",
        "identity": "y",
        "password": "hunter2000",
        "password_hash": "$argon2id$...",
        "access_token": "eyJhbGci",
        "refresh_token": "eyJhbGci",
        "secret": "s3cret",
        "ip_address": "10.0.0.1",
        "target_role": "instructor",  # benign scalar must survive
    }
    async with _session_factory() as session:
        await AuditService(session).emit(
            AuditEvent(
                action=GovernanceAction.USER_CREATE,
                entity_type="user",
                change_summary=poisoned,
            )
        )
        await session.commit()

    rows = await _governance_rows(_session_factory)
    assert len(rows) == 1
    stored = dict(rows[0].change_summary)
    assert stored == {"target_role": "instructor"}, stored


# ---------------------------------------------------------------------------
# Mandatory-vs-advisory policy matrix (D1)
# ---------------------------------------------------------------------------

def test_mandatory_vs_advisory_classification_matrix() -> None:
    from app.services.audit_service import GovernanceAction, MANDATORY_ACTIONS

    mandatory = {
        "USER_CREATE", "USER_UPDATE", "USER_DELETE",
        "STUDENT_CREATE", "STUDENT_UPDATE", "STUDENT_DELETE",
        "TEMPLATE_ENROLL", "ATTENDANCE_EVALUATE", "REFRESH_REUSED",
        # reserved future actions are mandatory by policy ahead of wiring;
        # ATT-044/038/045 (migration 20260824_0008) wired CONSENT_*/OVERRIDE_
        # APPLY/EMBED_HARD_DELETE, adding CONSENT_DENIED to the vocabulary.
        "CONSENT_GRANT", "CONSENT_WITHDRAW", "CONSENT_DENIED",
        "OVERRIDE_APPLY", "EMBED_HARD_DELETE",
    }
    advisory = {
        "LOGIN_SUCCEEDED", "LOGOUT", "INFERENCE_ENQUEUED",
        "TASK_READ", "RECOGNITION_RUN",
        # ATT-039 reclassified EXPORT as advisory when it wired the writer
        # (migration 20260824_0009): a ledger hiccup must never block a
        # roster the requester is already authorized to receive.
        "EXPORT",
    }
    assert {a.value for a in MANDATORY_ACTIONS} == mandatory
    assert not mandatory & advisory
    for name in advisory:
        assert GovernanceAction[name] not in MANDATORY_ACTIONS, name


class _ExplodingSession:
    """Session double whose flush always fails, for policy-matrix unit tests."""

    def __init__(self) -> None:
        self.rollback_calls = 0

    def add(self, _row) -> None:  # pragma: no cover - trivial
        return None

    async def flush(self) -> None:
        raise RuntimeError("simulated audit storage failure")

    async def rollback(self) -> None:
        self.rollback_calls += 1


@pytest.mark.asyncio
async def test_strict_emit_raises_and_non_strict_swallows() -> None:
    from app.services.audit_service import AuditEvent, GovernanceAction, emit

    event = AuditEvent(
        action=GovernanceAction.USER_CREATE,
        entity_type="user",
        change_summary={"target_role": "instructor"},
    )

    exploding = _ExplodingSession()
    with pytest.raises(RuntimeError, match="simulated audit storage failure"):
        await emit(exploding, event, strict=True)

    tolerant = _ExplodingSession()
    result = await emit(tolerant, event, strict=False)
    assert result is None
    assert tolerant.rollback_calls == 1, "non-strict failure must restore the session"


# ---------------------------------------------------------------------------
# Row-on-mutation + atomicity (AC1 / AC3)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_student_create_writes_row_with_actor(
    async_client: AsyncClient,
    instructor_user,
    auth_cookie,
    _session_factory,
) -> None:
    response = await async_client.post(
        "/api/v1/students",
        json=_student_payload(instructor_user.id),
        cookies=auth_cookie(instructor_user),
    )
    assert response.status_code == 201, response.text
    created_id = response.json()["id"]

    rows = await _governance_rows(_session_factory)
    assert len(rows) == 1
    row = rows[0]
    assert row.action == "STUDENT_CREATE"
    assert row.entity_type == "student"
    assert str(row.entity_id) == created_id
    assert row.actor_user_id == instructor_user.id  # actor propagation spot-check
    assert row.change_summary["enrollment_year"] == 2024
    assert row.change_summary["student_number"] == response.json()["student_number"]


@pytest.mark.asyncio
async def test_duplicate_student_create_fails_with_zero_governance_rows(
    async_client: AsyncClient,
    instructor_user,
    auth_cookie,
    _session_factory,
) -> None:
    _, existing = await _seed_student(_session_factory, number="DUP0001")

    payload = {
        "user_id": str(instructor_user.id),
        "student_number": "DUP0001",  # conflicts with the seeded row
        "program": "Test Program",
        "enrollment_year": 2024,
    }
    response = await async_client.post(
        "/api/v1/students", json=payload, cookies=auth_cookie(instructor_user)
    )
    assert response.status_code == 409, response.text

    # Atomicity proof: the business op failed => ZERO governance rows. A
    # fire-and-forget audit write would have left a phantom STUDENT_CREATE.
    assert await _governance_rows(_session_factory) == []


@pytest.mark.asyncio
async def test_user_lifecycle_rows_summarize_names_not_secrets(
    async_client: AsyncClient,
    admin_user,
    auth_cookie,
    _session_factory,
) -> None:
    created = await async_client.post(
        "/api/v1/users",
        json={
            "email": "gov-target@test.example",
            "full_name": "Gov Target",
            "password": "SuperSecret99!",
            "role": "instructor",
        },
        cookies=auth_cookie(admin_user),
    )
    assert created.status_code == 201, created.text
    target_id = created.json()["id"]

    patched = await async_client.patch(
        f"/api/v1/users/{target_id}",
        json={"role": "operator"},
        cookies=auth_cookie(admin_user),
    )
    assert patched.status_code == 200, patched.text

    deleted = await async_client.delete(
        f"/api/v1/users/{target_id}", cookies=auth_cookie(admin_user)
    )
    assert deleted.status_code == 204, deleted.text

    rows = await _governance_rows(_session_factory)
    assert [row.action for row in rows] == ["USER_DELETE", "USER_UPDATE", "USER_CREATE"]
    assert all(str(row.entity_id) == target_id for row in rows)
    assert all(row.actor_user_id == admin_user.id for row in rows)

    create_row, update_row, delete_row = rows[2], rows[1], rows[0]
    assert create_row.change_summary == {"target_role": "instructor"}
    assert update_row.change_summary == {"fields_changed": ["role"]}
    assert delete_row.change_summary["target_role"] == "operator"
    blob = repr([r.change_summary for r in rows])
    assert "password" not in blob.replace("password_changed", "")
    assert "SuperSecret99!" not in blob


@pytest.mark.asyncio
async def test_actor_rows_survive_actor_deletion_via_set_null(
    async_client: AsyncClient,
    admin_user,
    instructor_user,
    auth_cookie,
    _session_factory,
) -> None:
    from app.services.audit_service import GovernanceAction, GovernanceLog

    # Seed one row attributed to the instructor, then remove the actor.
    async with _session_factory() as session:
        session.add(
            GovernanceLog(
                action=GovernanceAction.STUDENT_CREATE.value,
                entity_type="student",
                entity_id=uuid.uuid4(),
                actor_user_id=instructor_user.id,
                change_summary={"seeded": True},
            )
        )
        await session.commit()

    response = await async_client.delete(
        f"/api/v1/users/{instructor_user.id}", cookies=auth_cookie(admin_user)
    )
    assert response.status_code == 204, response.text

    rows = await _governance_rows(_session_factory)
    seeded = [row for row in rows if row.change_summary.get("seeded")]
    assert len(seeded) == 1, "governance rows must survive actor deletion"
    assert seeded[0].actor_user_id is None


# ---------------------------------------------------------------------------
# TEMPLATE_ENROLL coexistence with template_audit_logs (biometric-data guard)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_enroll_writes_governance_and_template_audit_rows(
    async_client: AsyncClient,
    instructor_user,
    auth_cookie,
    fake_triton,
    monkeypatch: pytest.MonkeyPatch,
    _session_factory,
) -> None:
    from PIL import Image

    monkeypatch.setenv("ATTENDANCE_ENROLLMENT_MIN_QUALITY", "0.0")

    _, student = await _seed_student(_session_factory, number="ENR0001")

    buffer = BytesIO()
    Image.new("RGB", (64, 64), color=(120, 40, 200)).save(buffer, format="JPEG")
    response = await async_client.post(
        f"/api/v1/students/{student.id}/enroll",
        files={"image_file": ("face.jpg", buffer.getvalue(), "image/jpeg")},
        data={"pose_label": "front"},
        cookies=auth_cookie(instructor_user),
    )
    assert response.status_code == 201, response.text
    embedding_id = response.json()["id"]

    rows = await _governance_rows(_session_factory)
    enroll_rows = [row for row in rows if row.action == "TEMPLATE_ENROLL"]
    assert len(enroll_rows) == 1
    row = enroll_rows[0]
    assert str(row.entity_id) == embedding_id
    assert row.entity_type == "student_embedding"
    assert row.actor_user_id == instructor_user.id

    summary = dict(row.change_summary)
    assert set(summary) == {"pose_label", "quality_score", "replaced_count"}, summary
    assert summary["pose_label"] == "front"
    # Biometric-data regression guard: no vector material anywhere.
    serialized = repr(summary)
    assert "embedding" not in serialized
    assert len(serialized) < 300

    async with _session_factory() as session:
        audit_count = len(
            list(
                (
                    await session.execute(
                        text(
                            "SELECT id FROM template_audit_logs "
                            "WHERE student_embedding_id = :eid"
                        ),
                        {"eid": embedding_id},
                    )
                ).scalars()
            )
        )
    assert audit_count >= 1, "TemplateAuditLog coexistence broken"


# ---------------------------------------------------------------------------
# System actor from the worker path (direct-service variant, no Celery)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_attendance_evaluate_system_actor_event(test_engine: AsyncEngine) -> None:
    from datetime import date as date_type

    from app.domain.models import GovernanceLog
    from app.services.attendance_service import AttendanceService

    factory = async_sessionmaker(
        bind=test_engine, class_=AsyncSession, autoflush=False, autocommit=False,
        expire_on_commit=False,
    )
    course, _student = await _seed_course_student_sightings(factory, sighting_count=3)

    async with factory() as session:
        service = AttendanceService(session)  # no actor => system actor
        updated = await service.evaluate_class_attendance(
            course_id=course.id,
            date=date_type.today(),
            required_sightings_threshold=3,
        )
        assert len(updated) >= 1

        row = (
            await session.execute(
                select(GovernanceLog).where(GovernanceLog.action == "ATTENDANCE_EVALUATE")
            )
        ).scalar_one()
        assert row.actor_user_id is None, "system actor must persist as NULL"
        assert row.entity_id == course.id
        summary = dict(row.change_summary)
        assert summary["source"] == "celery"
        assert summary["records_upserted"] == len(updated)
        assert summary["threshold"] == 3
        assert "session_date" in summary


async def _seed_course_student_sightings(factory, *, sighting_count: int = 3):
    from app.domain.models import Course, Sighting, Student, User, UserRole
    from app.core.security import hash_password

    now = datetime.now(tz=UTC)
    async with factory() as session:
        user = User(
            id=uuid.uuid4(),
            email=f"sysactor-{uuid.uuid4().hex[:6]}@test.example",
            full_name="System Actor Seed",
            password_hash=hash_password("TestPass1!"),
            role=UserRole.AUDITOR,
            is_active=True,
        )
        session.add(user)
        await session.flush()

        course = Course(
            id=uuid.uuid4(),
            code=f"SYS{uuid.uuid4().hex[:4].upper()}",
            title="System Actor Course",
            credits=3,
            is_active=True,
        )
        session.add(course)
        await session.flush()

        student = Student(
            id=uuid.uuid4(),
            user_id=user.id,
            student_number=f"SYS{uuid.uuid4().hex[:6].upper()}",
            program="Test Program",
            enrollment_year=2024,
            is_active=True,
        )
        session.add(student)
        await session.flush()

        for i in range(sighting_count):
            session.add(
                Sighting(
                    id=uuid.uuid4(),
                    student_id=student.id,
                    course_id=course.id,
                    room_id=None,
                    timestamp=now,
                    camera_id=f"cam-sys-{i}",
                    confidence_score=0.9,
                )
            )
        await session.commit()
    return course, student


# ---------------------------------------------------------------------------
# Auth events: LOGIN_SUCCEEDED / LOGOUT with IP (D4); failed logins: none (D6);
# REFRESH_REUSED mandatory security signal
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_login_logout_capture_ip_and_failed_login_writes_nothing(
    async_client: AsyncClient,
    instructor_user,
    _session_factory,
) -> None:
    miss = await async_client.post(
        "/api/v1/auth/login",
        json={"email": instructor_user.email, "password": "WrongPass99!"},
    )
    assert miss.status_code == 401
    assert await _governance_rows(_session_factory) == []  # D6: no failed-login rows

    hit = await async_client.post(
        "/api/v1/auth/login",
        json={"email": instructor_user.email, "password": "TestPass1!"},
    )
    assert hit.status_code == 200, hit.text

    logged_in = await _governance_rows(_session_factory)
    assert len(logged_in) == 1
    login_row = logged_in[0]
    assert login_row.action == "LOGIN_SUCCEEDED"
    assert login_row.entity_id == instructor_user.id
    assert login_row.actor_user_id == instructor_user.id
    assert login_row.change_summary == {"method": "password"}
    assert login_row.ip_address is not None, "auth events capture client IP (D4)"

    out = await async_client.post("/api/v1/auth/logout", cookies=dict(hit.cookies))
    assert out.status_code == 200, out.text

    rows = await _governance_rows(_session_factory)
    assert [row.action for row in rows] == ["LOGOUT", "LOGIN_SUCCEEDED"]
    logout_row = rows[0]
    assert logout_row.ip_address is not None
    # D4 boundary: IP appears ONLY on consent/auth events — nothing else here.


@pytest.mark.asyncio
async def test_refresh_replay_writes_mandatory_row(
    async_client: AsyncClient,
    instructor_user,
    auth_cookie,
    _session_factory,
) -> None:
    from app.core.security import get_security_settings

    settings = get_security_settings()
    login = await async_client.post(
        "/api/v1/auth/login",
        json={"email": instructor_user.email, "password": "TestPass1!"},
    )
    assert login.status_code == 200, login.text
    refresh_token = login.cookies.get(settings.refresh_cookie_name)
    assert refresh_token, "login must set the refresh cookie"

    first = await async_client.post(
        "/api/v1/auth/refresh", cookies={settings.refresh_cookie_name: refresh_token}
    )
    assert first.status_code == 200, first.text

    second = await async_client.post(
        "/api/v1/auth/refresh", cookies={settings.refresh_cookie_name: refresh_token}
    )
    assert second.status_code == 401

    rows = [
        row for row in await _governance_rows(_session_factory)
        if row.action == "REFRESH_REUSED"
    ]
    assert len(rows) == 1
    assert rows[0].entity_id == instructor_user.id
    summary = dict(rows[0].change_summary)
    assert set(summary) == {"jti_prefix"}
    assert len(summary["jti_prefix"]) <= 9, "only an 8-char jti prefix, never token material"


# ---------------------------------------------------------------------------
# Inference advisory events + TASK_READ gate (D7)
# ---------------------------------------------------------------------------

def _mock_celery_enqueue(task_id: str):
    from unittest.mock import MagicMock, patch

    mock_task = MagicMock()
    mock_task.id = task_id
    return patch("app.api.v1.inference.run_inference_pipeline"), mock_task


@pytest.mark.asyncio
async def test_batch_enqueue_writes_advisory_row(
    async_client: AsyncClient,
    admin_user,
    auth_cookie,
    _session_factory,
) -> None:
    patcher, mock_task = _mock_celery_enqueue(f"gov-{uuid.uuid4().hex[:8]}")
    frames = [
        {
            "frame_id": "f",
            "data_base64": base64.b64encode(b"\x00" * (8 * 8 * 3)).decode("ascii"),
            "width": 8,
            "height": 8,
            "channels": 3,
            "dtype": "uint8",
        }
    ]
    with patcher as mock_fn:
        mock_fn.apply_async.return_value = mock_task
        response = await async_client.post(
            "/api/v1/inference/batch",
            json={"frames": frames},
            cookies=auth_cookie(admin_user),
        )
    assert response.status_code == 202, response.text

    rows = await _governance_rows(_session_factory)
    assert len(rows) == 1
    assert rows[0].action == "INFERENCE_ENQUEUED"
    assert rows[0].change_summary["frame_count"] == 1
    assert rows[0].actor_user_id == admin_user.id


@pytest.mark.asyncio
async def test_task_read_gated_off_by_default_then_on_with_env(
    async_client: AsyncClient,
    admin_user,
    auth_cookie,
    monkeypatch: pytest.MonkeyPatch,
    _session_factory,
) -> None:
    task_id = f"taskread-{uuid.uuid4().hex[:8]}"
    patcher, mock_task = _mock_celery_enqueue(task_id)
    with patcher as mock_fn:
        mock_fn.apply_async.return_value = mock_task
        accepted = await async_client.post(
            "/api/v1/inference/batch",
            json={
                "frames": [
                    {
                        "frame_id": "f",
                        "data_base64": base64.b64encode(b"\x00" * 192).decode("ascii"),
                        "width": 8,
                        "height": 8,
                        "channels": 3,
                        "dtype": "uint8",
                    }
                ]
            },
            cookies=auth_cookie(admin_user),
        )
    assert accepted.status_code == 202, accepted.text
    enqueued_id = accepted.json()["task_id"]

    status_response = await async_client.get(
        f"/api/v1/inference/tasks/{enqueued_id}", cookies=auth_cookie(admin_user)
    )
    assert status_response.status_code == 200, status_response.text
    # D7: default OFF — only the INFERENCE_ENQUEUED row exists so far.
    rows = await _governance_rows(_session_factory)
    assert [row.action for row in rows] == ["INFERENCE_ENQUEUED"]

    monkeypatch.setenv("ATTENDANCE_TASK_READ_AUDIT", "true")
    again = await async_client.get(
        f"/api/v1/inference/tasks/{enqueued_id}", cookies=auth_cookie(admin_user)
    )
    assert again.status_code == 200, again.text

    rows = await _governance_rows(_session_factory)
    reads = [row for row in rows if row.action == "TASK_READ"]
    assert len(reads) == 1
    assert reads[0].change_summary["task_id"] == enqueued_id
    assert reads[0].actor_user_id == admin_user.id


# ---------------------------------------------------------------------------
# RBAC matrix on GET /governance/events (D5)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_governance_events_rbac_matrix(
    async_client: AsyncClient,
    admin_user,
    auditor_user,
    instructor_user,
    operator_user,
    auth_cookie,
) -> None:
    url = "/api/v1/governance/events"
    assert (await async_client.get(url)).status_code == 401  # anonymous fails closed
    assert (
        await async_client.get(url, cookies=auth_cookie(admin_user))
    ).status_code == 200
    assert (
        await async_client.get(url, cookies=auth_cookie(auditor_user))
    ).status_code == 200
    assert (
        await async_client.get(url, cookies=auth_cookie(instructor_user))
    ).status_code == 403
    assert (
        await async_client.get(url, cookies=auth_cookie(operator_user))
    ).status_code == 403

    bad_vocab = await async_client.get(
        url + "?entity_type=nonsense", cookies=auth_cookie(admin_user)
    )
    assert bad_vocab.status_code == 422


@pytest.mark.asyncio
async def test_governance_events_filters_and_pagination(
    async_client: AsyncClient,
    admin_user,
    auth_cookie,
    _session_factory,
) -> None:
    entity_a, entity_b = uuid.uuid4(), uuid.uuid4()
    old_ts = datetime.now(tz=UTC) - timedelta(days=10)
    mid_ts = datetime.now(tz=UTC) - timedelta(days=5)
    new_ts = datetime.now(tz=UTC)

    seeds = [
        ("STUDENT_CREATE", "student", entity_a, old_ts),
        ("STUDENT_UPDATE", "student", entity_a, mid_ts),
        ("USER_CREATE", "user", entity_b, new_ts),
    ]
    # Direct ORM seeding (created_at pinned; bypasses the facade because the
    # facade does not carry timestamps — this is test data, not behavior).
    from app.domain.models import GovernanceLog

    async with _session_factory() as session:
        for action, entity_type, entity_id, ts in seeds:
            session.add(
                GovernanceLog(
                    action=action,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    change_summary={"seed": action},
                    created_at=ts,
                    updated_at=ts,
                )
            )
        await session.commit()

    cookie = auth_cookie(admin_user)

    everything = (await async_client.get("/api/v1/governance/events", cookies=cookie)).json()
    assert len(everything) == 3
    assert [row["action"] for row in everything] == [
        "USER_CREATE", "STUDENT_UPDATE", "STUDENT_CREATE",
    ]  # newest first

    by_action = (
        await async_client.get(
            "/api/v1/governance/events?action=STUDENT_UPDATE", cookies=cookie
        )
    ).json()
    assert len(by_action) == 1 and by_action[0]["action"] == "STUDENT_UPDATE"

    by_entity = (
        await async_client.get(
            f"/api/v1/governance/events?entity_id={entity_a}", cookies=cookie
        )
    ).json()
    assert len(by_entity) == 2

    # params= (not string interpolation): isoformat contains '+' which a raw
    # query string would decode as a space server-side.
    window = (
        await async_client.get(
            "/api/v1/governance/events",
            params={"since": mid_ts.isoformat(), "until": new_ts.isoformat()},
            cookies=cookie,
        )
    ).json()
    assert [row["action"] for row in window] == ["USER_CREATE", "STUDENT_UPDATE"]

    paged = (
        await async_client.get(
            "/api/v1/governance/events?offset=1&limit=1", cookies=cookie
        )
    ).json()
    assert [row["action"] for row in paged] == ["STUDENT_UPDATE"]

    by_actor = (
        await async_client.get(
            f"/api/v1/governance/events?actor_user_id={admin_user.id}", cookies=cookie
        )
    ).json()
    assert by_actor == []  # seeded rows have NULL actors


# ---------------------------------------------------------------------------
# Append-only enforcement at the database level
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_append_only_trigger_denies_update_delete_allows_truncate(
    test_engine: AsyncEngine,
    _session_factory,
) -> None:
    from sqlalchemy.exc import DBAPIError

    from app.domain.models import GovernanceLog

    async with _session_factory() as session:
        session.add(
            GovernanceLog(
                action="USER_CREATE",
                entity_type="user",
                change_summary={},
            )
        )
        await session.commit()

    # Each denied statement runs in its OWN transaction: a failed statement
    # aborts the surrounding asyncpg transaction, so sharing one would turn
    # the second attempt into InFailedSQLTransaction noise.
    async with test_engine.begin() as conn:
        with pytest.raises(DBAPIError, match="append-only"):
            await conn.execute(text("UPDATE governance_logs SET action = 'FORGERY'"))
    async with test_engine.begin() as conn:
        with pytest.raises(DBAPIError, match="append-only"):
            await conn.execute(text("DELETE FROM governance_logs"))
    # TRUNCATE deliberately stays possible: conftest relies on it between
    # tests (documented in the migration; no ON TRUNCATE guard shipped).
    async with test_engine.begin() as conn:
        await conn.execute(text("TRUNCATE governance_logs"))


# ---------------------------------------------------------------------------
# Ops purge path (D3): SECURITY DEFINER function logs its own invocation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_purge_governance_before_logs_invocation_and_deletes(
    test_engine: AsyncEngine,
    _session_factory,
) -> None:
    from app.domain.models import GovernanceLog

    stale_ts = datetime.now(tz=UTC) - timedelta(days=3000)
    async with _session_factory() as session:
        # TEMPLATE_ENROLL is an implemented action inside the DB CHECK
        # (reserved actions like EXPORT would be refused by the constraint).
        session.add(
            GovernanceLog(
                action="TEMPLATE_ENROLL",
                entity_type="student_embedding",
                change_summary={},
                created_at=stale_ts,
                updated_at=stale_ts,
            )
        )
        session.add(
            GovernanceLog(action="USER_CREATE", entity_type="user", change_summary={})
        )
        await session.commit()

    async with test_engine.begin() as conn:
        purged = (
            await conn.execute(
                text("SELECT public.purge_governance_before(now() - interval '2555 days')")
            )
        ).scalar_one()
    assert purged == 1, "only the stale row is inside the retention cutoff"

    rows = await _governance_rows(_session_factory)
    actions = [row.action for row in rows]
    assert "TEMPLATE_ENROLL" not in actions
    assert "USER_CREATE" in actions
    assert "GOVERNANCE_PURGE" in actions, "every purge logs its own invocation row"
