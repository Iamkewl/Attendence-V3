"""Wave-3 smoke suite: ATT-044 biometric consent, ATT-038 manual overrides,
ATT-045 embedding-retention sweep, and the admin enrollment-coverage report.

Anchors:
* Consent decisions update the student row AND emit the matching MANDATORY
  governance event (CONSENT_GRANT / CONSENT_DENIED / CONSENT_WITHDRAW) in the
  same transaction, with client IP captured per decision D4.
* The ATTENDANCE_ENFORCE_BIOMETRIC_CONSENT gate refuses template writes
  (403) unless consent is 'granted' — default-off keeps legacy behavior.
* POST /attendance/sessions/{id}/override upserts TODAY's roster row
  (last-write-wins) and emits OVERRIDE_APPLY carrying class_session_record_id.
* GET /api/v1/admin/enrollment-coverage aggregates per-student template +
  sighting + consent state (admin-only).
* task_purge_expired_embeddings hard-deletes expired AND withdrawn/denied-
  consent templates with batch-level EMBED_HARD_DELETE audit rows, never
  destroying a template whose evidence write failed.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from httpx import AsyncClient
from sqlalchemy import select


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------

async def _seed_student(session_factory, *, number: str | None = None):
    """Insert one active user + student pair; return (user, student)."""
    from app.core.security import hash_password
    from app.domain.models import Student, User, UserRole

    number = number or f"W3{uuid.uuid4().hex[:6].upper()}"
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
            is_active=True,
        )
        session.add(student)
        await session.commit()
        return link_user, student


async def _seed_student_with_consent(session_factory, *, consent: str) -> object:
    """Insert a user+student pair pinned to a given consent status."""
    from app.domain.models import Student

    _, student = await _seed_student(session_factory)
    async with session_factory() as session:
        row = await session.get(Student, student.id)
        row.biometric_consent_status = consent
        await session.commit()
        return row


async def _seed_course(session_factory) -> object:
    from app.domain.models import Course

    async with session_factory() as session:
        course = Course(
            id=uuid.uuid4(),
            code=f"W3C{uuid.uuid4().hex[:8].upper()}",
            title="Wave-3 override course",
            credits=3,
            is_active=True,
        )
        session.add(course)
        await session.commit()
        return course


async def _add_embedding(
    session_factory,
    student_id: uuid.UUID,
    *,
    pose_label: str = "front",
    created_at: datetime | None = None,
    is_active: bool = True,
) -> uuid.UUID:
    from app.domain.models import StudentEmbedding

    embedding_id = uuid.uuid4()
    async with session_factory() as session:
        session.add(
            StudentEmbedding(
                id=embedding_id,
                student_id=student_id,
                embedding=[0.01] * 512,
                pose_label=pose_label,
                quality_score=0.9,
                is_active=is_active,
                created_at=created_at or datetime.now(tz=UTC),
            )
        )
        await session.commit()
    return embedding_id


async def _add_sighting(
    session_factory,
    *,
    student_id: uuid.UUID,
    course_id: uuid.UUID,
    timestamp: datetime,
) -> None:
    from app.domain.models import Sighting

    async with session_factory() as session:
        session.add(
            Sighting(
                id=uuid.uuid4(),
                student_id=student_id,
                course_id=course_id,
                room_id=None,
                timestamp=timestamp,
                camera_id="cam-w3",
                confidence_score=0.9,
            )
        )
        await session.commit()


async def _governance_rows(session_factory) -> list:
    from app.domain.models import GovernanceLog

    async with session_factory() as session:
        return list(
            (
                await session.execute(
                    select(GovernanceLog).order_by(GovernanceLog.created_at.desc())
                )
            ).scalars().all()
        )


def _consent_fixture(monkeypatch: pytest.MonkeyPatch, *, enabled: bool):
    """Pin ATTENDANCE_ENFORCE_BIOMETRIC_CONSENT with lru_cache hygiene."""
    from app.core.security import get_security_settings

    if enabled:
        monkeypatch.setenv("ATTENDANCE_ENFORCE_BIOMETRIC_CONSENT", "true")
    else:
        monkeypatch.delenv("ATTENDANCE_ENFORCE_BIOMETRIC_CONSENT", raising=False)
    get_security_settings.cache_clear()
    yield
    get_security_settings.cache_clear()


@pytest.fixture()
def consent_gate_on(monkeypatch: pytest.MonkeyPatch):
    yield from _consent_fixture(monkeypatch, enabled=True)


@pytest.fixture()
def consent_gate_off(monkeypatch: pytest.MonkeyPatch):
    yield from _consent_fixture(monkeypatch, enabled=False)


@pytest.fixture()
def authz_on(monkeypatch: pytest.MonkeyPatch):
    from app.core.security import get_security_settings

    monkeypatch.setenv("ATTENDANCE_COURSE_SCOPED_AUTHZ", "true")
    get_security_settings.cache_clear()
    yield
    get_security_settings.cache_clear()


# ---------------------------------------------------------------------------
# SLICE A: consent endpoint + events
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_consent_grant_updates_columns_and_emits_mandatory_event(
    async_client: AsyncClient,
    instructor_user,
    auth_cookie,
    _session_factory,
) -> None:
    _, student = await _seed_student(_session_factory)

    response = await async_client.post(
        f"/api/v1/students/{student.id}/consent",
        json={"status": "granted", "reason": "form-v2"},
        cookies=auth_cookie(instructor_user),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["biometric_consent_status"] == "granted"
    assert body["biometric_consent_at"] is not None

    async with _session_factory() as session:
        refreshed = await session.get(type(student), student.id)
        assert refreshed.biometric_consent_status == "granted"
        assert refreshed.biometric_consent_at is not None

    rows = await _governance_rows(_session_factory)
    assert len(rows) == 1
    row = rows[0]
    assert row.action == "CONSENT_GRANT"
    assert row.entity_type == "student"
    assert row.entity_id == student.id
    assert row.actor_user_id == instructor_user.id
    summary = dict(row.change_summary)
    assert summary["from"] == "pending"
    assert summary["to"] == "granted"
    assert "is_minor" in summary
    # D4: consent events capture the client IP as signature evidence.
    assert row.ip_address is not None


@pytest.mark.asyncio
async def test_consent_denied_then_withdrawn_chain_from_to(
    async_client: AsyncClient,
    instructor_user,
    auth_cookie,
    _session_factory,
) -> None:
    _, student = await _seed_student(_session_factory)

    denied = await async_client.post(
        f"/api/v1/students/{student.id}/consent",
        json={"status": "denied"},
        cookies=auth_cookie(instructor_user),
    )
    assert denied.status_code == 200, denied.text
    withdrawn = await async_client.post(
        f"/api/v1/students/{student.id}/consent",
        json={"status": "withdrawn", "reason": "guardian request"},
        cookies=auth_cookie(instructor_user),
    )
    assert withdrawn.status_code == 200, withdrawn.text
    assert withdrawn.json()["biometric_consent_status"] == "withdrawn"

    rows = await _governance_rows(_session_factory)
    assert [row.action for row in rows] == ["CONSENT_WITHDRAW", "CONSENT_DENIED"]
    deny_row, withdraw_row = rows[1], rows[0]
    assert dict(deny_row.change_summary)["to"] == "denied"
    assert dict(withdraw_row.change_summary)["from"] == "denied"
    assert withdraw_row.reason == "guardian request"
    assert all(row.ip_address is not None for row in rows), (
        "every consent event captures IP (D4)"
    )


@pytest.mark.asyncio
async def test_consent_rbac_matrix_fails_closed(
    async_client: AsyncClient,
    admin_user,
    auditor_user,
    operator_user,
    auth_cookie,
    _session_factory,
) -> None:
    _, student = await _seed_student(_session_factory)
    url = f"/api/v1/students/{student.id}/consent"
    payload = {"status": "granted"}

    anonymous = await async_client.post(url, json=payload)
    assert anonymous.status_code == 401
    assert await _governance_rows(_session_factory) == []

    for role_user in (auditor_user, operator_user):
        denied = await async_client.post(
            url, json=payload, cookies=auth_cookie(role_user)
        )
        assert denied.status_code == 403, role_user.role
    assert await _governance_rows(_session_factory) == [], "403s must not write rows"

    ok = await async_client.post(url, json=payload, cookies=auth_cookie(admin_user))
    assert ok.status_code == 200, ok.text


@pytest.mark.asyncio
async def test_consent_unknown_student_404_and_invalid_status_422(
    async_client: AsyncClient,
    instructor_user,
    auth_cookie,
    _session_factory,
) -> None:
    missing = await async_client.post(
        f"/api/v1/students/{uuid.uuid4()}/consent",
        json={"status": "granted"},
        cookies=auth_cookie(instructor_user),
    )
    assert missing.status_code == 404

    _, student = await _seed_student(_session_factory)
    bad_vocab = await async_client.post(
        f"/api/v1/students/{student.id}/consent",
        json={"status": "pending"},  # 'pending' is only the initial state
        cookies=auth_cookie(instructor_user),
    )
    assert bad_vocab.status_code == 422
    garbage = await async_client.post(
        f"/api/v1/students/{student.id}/consent",
        json={"status": "GRANTED"},  # vocabulary is lowercase-exact
        cookies=auth_cookie(instructor_user),
    )
    assert garbage.status_code == 422
    assert await _governance_rows(_session_factory) == []


# ---------------------------------------------------------------------------
# SLICE A: enrollment consent gate
# ---------------------------------------------------------------------------

async def _enroll(async_client: AsyncClient, student_id: uuid.UUID, cookies) -> AsyncClient:
    from io import BytesIO

    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (64, 64), color=(90, 90, 200)).save(buffer, format="JPEG")
    return await async_client.post(
        f"/api/v1/students/{student_id}/enroll",
        files={"image_file": ("face.jpg", buffer.getvalue(), "image/jpeg")},
        data={"pose_label": "front"},
        cookies=cookies,
    )


@pytest.mark.asyncio
async def test_enroll_consent_gate_off_is_legacy_behavior(
    async_client: AsyncClient,
    instructor_user,
    auth_cookie,
    fake_triton,
    monkeypatch: pytest.MonkeyPatch,
    consent_gate_off,
    _session_factory,
) -> None:
    monkeypatch.setenv("ATTENDANCE_ENROLLMENT_MIN_QUALITY", "0.0")
    _, student = await _seed_student(_session_factory)
    assert student.biometric_consent_status == "pending"

    response = await _enroll(async_client, student.id, auth_cookie(instructor_user))
    assert response.status_code == 201, response.text


@pytest.mark.asyncio
async def test_enroll_consent_gate_refuses_until_granted(
    async_client: AsyncClient,
    instructor_user,
    auth_cookie,
    fake_triton,
    monkeypatch: pytest.MonkeyPatch,
    consent_gate_on,
    _session_factory,
) -> None:
    monkeypatch.setenv("ATTENDANCE_ENROLLMENT_MIN_QUALITY", "0.0")
    _, student = await _seed_student(_session_factory)

    refused = await _enroll(async_client, student.id, auth_cookie(instructor_user))
    assert refused.status_code == 403, refused.text
    assert "consent" in refused.json()["detail"].lower()

    granted = await async_client.post(
        f"/api/v1/students/{student.id}/consent",
        json={"status": "granted"},
        cookies=auth_cookie(instructor_user),
    )
    assert granted.status_code == 200, granted.text

    allowed = await _enroll(async_client, student.id, auth_cookie(instructor_user))
    assert allowed.status_code == 201, allowed.text

    withdrawn_again = await async_client.post(
        f"/api/v1/students/{student.id}/consent",
        json={"status": "withdrawn"},
        cookies=auth_cookie(instructor_user),
    )
    assert withdrawn_again.status_code == 200
    refused_again = await _enroll(async_client, student.id, auth_cookie(instructor_user))
    assert refused_again.status_code == 403, refused_again.text


# ---------------------------------------------------------------------------
# SLICE C: manual overrides (ATT-038)
# ---------------------------------------------------------------------------

async def _override(
    async_client: AsyncClient,
    course_id: uuid.UUID,
    payload: dict,
    cookies,
):
    return await async_client.post(
        f"/api/v1/attendance/sessions/{course_id}/override",
        json=payload,
        cookies=cookies,
    )


@pytest.mark.asyncio
async def test_override_creates_today_row_and_emits_override_apply(
    async_client: AsyncClient,
    instructor_user,
    auth_cookie,
    _session_factory,
) -> None:
    from app.domain.models import ClassSessionRecord

    course = await _seed_course(_session_factory)
    _, student = await _seed_student(_session_factory)

    response = await _override(
        async_client,
        course.id,
        {
            "student_id": str(student.id),
            "status": "present",
            "reason": "arrived late with pass",
        },
        auth_cookie(instructor_user),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["previous_status"] is None
    assert body["status"] == "present"
    assert body["course_id"] == str(course.id)

    async with _session_factory() as session:
        record = await session.get(ClassSessionRecord, uuid.UUID(body["id"]))
        assert record is not None
        assert record.status.value == "present"
        assert record.session_date == datetime.now(tz=UTC).date()
        assert record.student_id == student.id
        assert record.notes.startswith("manual_override:")

    rows = [
        row for row in await _governance_rows(_session_factory)
        if row.action == "OVERRIDE_APPLY"
    ]
    assert len(rows) == 1
    row = rows[0]
    assert row.class_session_record_id == record.id, (
        "OVERRIDE_APPLY must carry the overridden roster row id"
    )
    assert row.entity_id == record.id
    assert row.actor_user_id == instructor_user.id
    assert row.reason == "arrived late with pass"
    summary = dict(row.change_summary)
    assert summary["from"] is None
    assert summary["to"] == "present"
    assert summary["student_id"] == str(student.id)


@pytest.mark.asyncio
async def test_override_last_write_wins_reapplication(
    async_client: AsyncClient,
    instructor_user,
    auth_cookie,
    _session_factory,
) -> None:
    from sqlalchemy import func

    from app.domain.models import ClassSessionRecord

    course = await _seed_course(_session_factory)
    _, student = await _seed_student(_session_factory)

    first = await _override(
        async_client,
        course.id,
        {"student_id": str(student.id), "status": "present", "reason": "seen at door"},
        auth_cookie(instructor_user),
    )
    second = await _override(
        async_client,
        course.id,
        {"student_id": str(student.id), "status": "absent", "reason": "left before roll call"},
        auth_cookie(instructor_user),
    )
    assert first.status_code == 200 and second.status_code == 200
    first_body, second_body = first.json(), second.json()
    assert second_body["id"] == first_body["id"], "upsert reuses the same roster row"
    assert second_body["previous_status"] == "present"
    assert second_body["status"] == "absent"

    async with _session_factory() as session:
        count = await session.scalar(select(func.count()).select_from(ClassSessionRecord))
        assert count == 1, "no duplicate rows per (student, course, date)"
        record = await session.get(ClassSessionRecord, uuid.UUID(first_body["id"]))
        assert record.status.value == "absent"
        assert record.notes.endswith("left before roll call")

    rows = [
        row for row in await _governance_rows(_session_factory)
        if row.action == "OVERRIDE_APPLY"
    ]
    assert len(rows) == 2, "each application is audited"


@pytest.mark.asyncio
async def test_override_rbac_matrix_and_flagged_course_link(
    async_client: AsyncClient,
    admin_user,
    instructor_user,
    auditor_user,
    operator_user,
    auth_cookie,
    authz_on,
    _session_factory,
) -> None:
    from app.domain.models import CourseInstructor

    course = await _seed_course(_session_factory)
    _, student = await _seed_student(_session_factory)
    payload = {
        "student_id": str(student.id),
        "status": "absent",
        "reason": "never sighted today",
    }

    anonymous = await async_client.post(
        f"/api/v1/attendance/sessions/{course.id}/override", json=payload
    )
    assert anonymous.status_code == 401

    for role_user in (auditor_user, operator_user):
        forbidden = await _override(
            async_client, course.id, payload, auth_cookie(role_user)
        )
        assert forbidden.status_code == 403, role_user.role

    # Flag ON: unlinked instructor gets the existence-denying 404 (fail closed).
    unlinked = await _override(
        async_client, course.id, payload, auth_cookie(instructor_user)
    )
    assert unlinked.status_code == 404, unlinked.text

    async with _session_factory() as session:
        session.add(
            CourseInstructor(
                course_id=course.id,
                user_id=instructor_user.id,
                role_in_course="owner",
            )
        )
        await session.commit()

    linked = await _override(
        async_client, course.id, payload, auth_cookie(instructor_user)
    )
    assert linked.status_code == 200, linked.text

    other_course = await _seed_course(_session_factory)
    admin_ok = await _override(
        async_client,
        other_course.id,
        payload,
        auth_cookie(admin_user),
    )
    assert admin_ok.status_code == 200, (
        f"{admin_ok.text} — ADMIN bypasses the link check"
    )


@pytest.mark.asyncio
async def test_override_validation_errors(
    async_client: AsyncClient,
    instructor_user,
    auth_cookie,
    _session_factory,
) -> None:
    course = await _seed_course(_session_factory)
    _, student = await _seed_student(_session_factory)

    blank_reason = await _override(
        async_client,
        course.id,
        {"student_id": str(student.id), "status": "present", "reason": "  "},
        auth_cookie(instructor_user),
    )
    assert blank_reason.status_code == 422

    bad_status = await _override(
        async_client,
        course.id,
        {"student_id": str(student.id), "status": "late", "reason": "not hand-settable"},
        auth_cookie(instructor_user),
    )
    assert bad_status.status_code == 422

    unknown_course = await _override(
        async_client,
        uuid.uuid4(),
        {"student_id": str(student.id), "status": "present", "reason": "nowhere to write"},
        auth_cookie(instructor_user),
    )
    assert unknown_course.status_code == 404

    unknown_student = await _override(
        async_client,
        course.id,
        {"student_id": str(uuid.uuid4()), "status": "present", "reason": "nobody by that id"},
        auth_cookie(instructor_user),
    )
    assert unknown_student.status_code == 404

    assert await _governance_rows(_session_factory) == []


# ---------------------------------------------------------------------------
# SLICE C: enrollment coverage aggregate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_enrollment_coverage_aggregates_without_n_plus_one(
    async_client: AsyncClient,
    admin_user,
    auth_cookie,
    _session_factory,
) -> None:
    course = await _seed_course(_session_factory)
    _, covered = await _seed_student(_session_factory)
    _, bare = await _seed_student(_session_factory)

    old = datetime.now(tz=UTC) - timedelta(days=40)
    await _add_embedding(_session_factory, covered.id, pose_label="front", created_at=old)
    await _add_embedding(
        _session_factory, covered.id, pose_label="left", created_at=datetime.now(tz=UTC)
    )
    await _add_embedding(
        _session_factory,
        covered.id,
        pose_label="stale",
        created_at=old,
        is_active=False,
    )

    now = datetime.now(tz=UTC)
    await _add_sighting(
        _session_factory, student_id=covered.id, course_id=course.id, timestamp=now
    )
    await _add_sighting(
        _session_factory,
        student_id=covered.id,
        course_id=course.id,
        timestamp=now - timedelta(days=10),
    )

    response = await async_client.get(
        "/api/v1/admin/enrollment-coverage", cookies=auth_cookie(admin_user)
    )
    assert response.status_code == 200, response.text
    rows = {row["student_number"]: row for row in response.json()}
    assert len(rows) == 2

    covered_row = rows[covered.student_number]
    assert covered_row["active_template_count"] == 2, "inactive templates excluded"
    assert sorted(covered_row["poses"]) == ["front", "left"]
    assert covered_row["last_enrolled_at"] is not None
    assert covered_row["sightings_last_7d"] == 1, "only trailing-week sightings count"
    assert covered_row["biometric_consent_status"] == "pending"

    bare_row = rows[bare.student_number]
    assert bare_row["active_template_count"] == 0
    assert bare_row["poses"] == []
    assert bare_row["last_enrolled_at"] is None
    assert bare_row["sightings_last_7d"] == 0


@pytest.mark.asyncio
async def test_enrollment_coverage_rbac(
    async_client: AsyncClient,
    admin_user,
    instructor_user,
    auth_cookie,
) -> None:
    url = "/api/v1/admin/enrollment-coverage"
    assert (await async_client.get(url)).status_code == 401
    assert (
        await async_client.get(url, cookies=auth_cookie(instructor_user))
    ).status_code == 403
    assert (
        await async_client.get(url, cookies=auth_cookie(admin_user))
    ).status_code == 200


# ---------------------------------------------------------------------------
# SLICE B: embedding retention sweep (ATT-045)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retention_sweep_deletes_expired_and_unconsented_keeps_recent(
    _session_factory,
) -> None:
    from app.domain.models import StudentEmbedding

    _, expired_owner = await _seed_student(_session_factory)
    _, recent_owner = await _seed_student(_session_factory)
    withdrawn_row = await _seed_student_with_consent(_session_factory, consent="withdrawn")
    denied_row = await _seed_student_with_consent(_session_factory, consent="denied")

    stale = datetime.now(tz=UTC) - timedelta(days=4000)
    await _add_embedding(_session_factory, expired_owner.id, created_at=stale)
    kept_id = await _add_embedding(_session_factory, recent_owner.id)
    await _add_embedding(_session_factory, withdrawn_row.id, pose_label="front")
    await _add_embedding(
        _session_factory, withdrawn_row.id, pose_label="left", created_at=stale
    )
    await _add_embedding(_session_factory, denied_row.id)

    from app.worker.tasks import _purge_expired_embeddings

    summary = await _purge_expired_embeddings(1095)
    # The expired-horizon batch is consent-blind by design: it claims the
    # stale withdrawn-consent 'left' template too (2 rows). The consent
    # batch then removes whatever remains for withdrawn/denied students
    # regardless of age (2 rows).
    assert summary["retention_deleted"] == 2
    assert summary["consent_deleted"] == 2

    remaining = []
    async with _session_factory() as session:
        remaining = list((await session.execute(select(StudentEmbedding.id))).scalars().all())
    assert set(remaining) == {kept_id}, "only the recent granted-consent template survives"

    rows = [row for row in await _governance_rows(_session_factory)]
    deletes = {row.action for row in rows}
    assert deletes == {"EMBED_HARD_DELETE"}
    by_reason = {dict(row.change_summary)["deletion_reason"]: dict(row.change_summary) for row in rows}
    assert set(by_reason) == {"retention_expired", "consent_not_granted"}
    assert by_reason["retention_expired"]["deleted_count"] == 2
    assert by_reason["consent_not_granted"]["deleted_count"] == 2
    assert all(row.actor_user_id is None for row in rows), "system actor"
    serialized = repr([r.change_summary for r in rows])
    assert "[0.01" not in serialized and "embedding" not in serialized.replace(
        "EMBED_HARD_DELETE", ""
    ), "vectors never enter summaries"


@pytest.mark.asyncio
async def test_retention_sweep_aborts_when_audit_write_cannot_land(
    _session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.domain.models import StudentEmbedding
    from app.services.audit_service import AuditService

    stale = datetime.now(tz=UTC) - timedelta(days=4000)
    _, owner = await _seed_student(_session_factory)
    await _add_embedding(_session_factory, owner.id, created_at=stale)

    original_emit = AuditService.emit

    async def exploding_emit(self, event):  # noqa: ANN001
        raise RuntimeError("simulated ledger outage")

    monkeypatch.setattr(AuditService, "emit", exploding_emit)
    try:
        from app.worker.tasks import _purge_expired_embeddings

        with pytest.raises(RuntimeError, match="simulated ledger outage"):
            await _purge_expired_embeddings(1095)
    finally:
        AuditService.emit = original_emit

    async with _session_factory() as session:
        surviving = list(
            (await session.execute(select(StudentEmbedding.id))).scalars().all()
        )
    assert surviving, "templates are NEVER destroyed without their evidence row"


def test_retention_task_registered_daily_on_beat_schedule() -> None:
    from app.worker.celery_app import get_celery_app

    schedule = get_celery_app().conf.beat_schedule
    entry = schedule.get("embedding-retention-daily")
    assert entry is not None, "ATT-045 sweep must be on the beat schedule"
    assert entry["task"] == "app.worker.tasks.task_purge_expired_embeddings"
    assert entry["schedule"].total_seconds() >= 86_400, "daily cadence convention"


def test_retention_horizon_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.core.security import get_security_settings

    monkeypatch.delenv("ATTENDANCE_EMBEDDING_RETENTION_DAYS", raising=False)
    monkeypatch.delenv("ATTENDANCE_ENFORCE_BIOMETRIC_CONSENT", raising=False)
    get_security_settings.cache_clear()
    try:
        settings = get_security_settings()
        assert settings.embedding_retention_days == 1095
        assert settings.enforce_biometric_consent is False
        assert settings.embedding_retention_days < settings.governance_retention_days, (
            "D2: consent evidence must outlive the templates it authorized"
        )
    finally:
        get_security_settings.cache_clear()


def test_retention_horizon_env_overrides_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.core.security import get_security_settings

    monkeypatch.setenv("ATTENDANCE_EMBEDDING_RETENTION_DAYS", "365")
    get_security_settings.cache_clear()
    try:
        assert get_security_settings().embedding_retention_days == 365
    finally:
        get_security_settings.cache_clear()

    monkeypatch.setenv("ATTENDANCE_EMBEDDING_RETENTION_DAYS", "zero")
    get_security_settings.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="positive integer"):
            get_security_settings()
    finally:
        get_security_settings.cache_clear()

    monkeypatch.setenv("ATTENDANCE_EMBEDDING_RETENTION_DAYS", "-3")
    get_security_settings.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="positive integer"):
            get_security_settings()
    finally:
        get_security_settings.cache_clear()


def test_reserved_vocabulary_now_implemented() -> None:
    """Migration 0008 wired writers for the reserved actions (design Q8)."""
    from app.services.audit_service import IMPLEMENTED_ACTIONS, MANDATORY_ACTIONS, GovernanceAction

    for name in (
        "CONSENT_GRANT",
        "CONSENT_WITHDRAW",
        "CONSENT_DENIED",
        "OVERRIDE_APPLY",
        "EMBED_HARD_DELETE",
    ):
        assert name in IMPLEMENTED_ACTIONS, name
        assert GovernanceAction[name] in MANDATORY_ACTIONS, name
    # ATT-039 wired the EXPORT writer (migration 20260824_0009) and
    # classified it advisory — implemented, but deliberately not mandatory.
    assert "EXPORT" in IMPLEMENTED_ACTIONS, "wired by ATT-039"
    assert GovernanceAction["EXPORT"] not in MANDATORY_ACTIONS, "advisory"
