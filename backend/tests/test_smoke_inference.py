"""Flows 2, 3, 4 — inference stream, eager pipeline, and task-status embedding strip."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from tests._fakes import FakeTritonGrpcClient, _unit_norm_vector


# ---------------------------------------------------------------------------
# Flow 2 — stream endpoint returns 202 + task_id
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_accepts_raw_tensor(
    async_client: AsyncClient,
    admin_user,
    auth_cookie,
    fake_triton: FakeTritonGrpcClient,
    synthetic_frame_bytes: bytes,
    synthetic_frame_form: dict,
) -> None:
    # Mock apply_async so the Celery task is not actually executed — this
    # test only verifies the endpoint returns 202 with a task_id.
    mock_task = MagicMock()
    mock_task.id = "test-task-id-smoke-001"

    with patch("app.api.v1.inference.run_inference_pipeline") as mock_fn:
        mock_fn.apply_async.return_value = mock_task

        response = await async_client.post(
            "/api/v1/inference/stream",
            data=synthetic_frame_form,
            files={"frame_file": ("frame.bin", synthetic_frame_bytes, "application/octet-stream")},
            cookies=auth_cookie(admin_user),
        )

    assert response.status_code == 202, response.text
    body = response.json()
    assert body.get("task_id"), f"task_id missing in {body}"


# ---------------------------------------------------------------------------
# Flow 3 — eager pipeline persists a Sighting row
# ---------------------------------------------------------------------------

async def _seed_course_student_embedding(engine: AsyncEngine) -> tuple:
    """Insert a Course, Student (with linked User), and a matching StudentEmbedding."""
    from app.domain.models import Course, Student, StudentEmbedding, User, UserRole
    from app.core.security import hash_password

    factory = async_sessionmaker(bind=engine, class_=AsyncSession, autoflush=False, autocommit=False, expire_on_commit=False)

    async with factory() as session:
        user = User(
            id=uuid.uuid4(),
            email="enrolled@test.example",
            full_name="Enrolled Student",
            password_hash=hash_password("TestPass1!"),
            role=UserRole.AUDITOR,
            is_active=True,
        )
        session.add(user)
        await session.flush()

        course = Course(
            id=uuid.uuid4(),
            code="CS101",
            title="Intro to Computing",
            credits=3,
            is_active=True,
        )
        session.add(course)
        await session.flush()

        student = Student(
            id=uuid.uuid4(),
            user_id=user.id,
            student_number="STU0001",
            program="Computer Science",
            enrollment_year=2024,
            is_active=True,
        )
        session.add(student)
        await session.flush()

        # The embedding must match the deterministic fake output so cosine >= 0.85.
        embedding_vec = _unit_norm_vector(seed=42).tolist()
        emb = StudentEmbedding(
            id=uuid.uuid4(),
            student_id=student.id,
            embedding=embedding_vec,
            pose_label="front",
            quality_score=1.0,
            is_active=True,
        )
        session.add(emb)
        await session.commit()
        await session.refresh(course)
        await session.refresh(student)

    return course, student, emb


@pytest.mark.asyncio
async def test_eager_pipeline_persists_sighting(
    test_engine: AsyncEngine,
    fake_triton: FakeTritonGrpcClient,
) -> None:
    from app.domain.models import Sighting
    from app.domain.schemas import InferenceBatchRequest, ImageTensorPayload
    # Call the underlying async function directly — Celery tasks use asyncio.run()
    # which cannot be called from within an already-running event loop.
    from app.worker.tasks import _run_pipeline_and_log_sightings
    import base64

    course, student, _ = await _seed_course_student_embedding(test_engine)

    frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
    frame[120:200, 160:220] = 128
    raw_bytes = frame.tobytes()
    data_b64 = base64.b64encode(raw_bytes).decode("ascii")

    request = InferenceBatchRequest(
        request_id="smoke-test-001",
        course_id=course.id,
        camera_id="cam-smoke",
        frames=[
            ImageTensorPayload(
                frame_id="smoke-frame-001",
                data_base64=data_b64,
                width=640,
                height=480,
                channels=3,
                dtype="uint8",
                normalize=True,
                captured_at=datetime.now(tz=UTC),
            )
        ],
        confidence_threshold=0.25,
        liveness_threshold=0.5,
        include_embeddings=True,
    )

    pipeline_result = await _run_pipeline_and_log_sightings(request)

    assert pipeline_result.get("sightings_logged", 0) >= 1, (
        f"Expected sighting; got {pipeline_result}"
    )

    factory = async_sessionmaker(bind=test_engine, class_=AsyncSession, autoflush=False, autocommit=False, expire_on_commit=False)
    async with factory() as session:
        rows = (await session.execute(select(Sighting))).scalars().all()

    assert len(rows) >= 1, "No Sighting rows found after eager pipeline run"


# ---------------------------------------------------------------------------
# Flow 4 — task status strips embeddings from results
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_task_status_returns_success_and_strips_embeddings(
    async_client: AsyncClient,
    admin_user,
    auth_cookie,
) -> None:
    fake_task_id = "aaaaaaaa-0000-0000-0000-bbbbbbbbbbbb"

    mock_result = MagicMock()
    mock_result.state = "SUCCESS"
    mock_result.result = {
        "frame_count": 1,
        "results": [
            {
                "frame_id": "f1",
                "student_id": None,
                "embedding": [0.1, 0.2, 0.3],
            }
        ],
    }

    with patch("app.api.v1.inference.celery_app") as mock_celery:
        mock_celery.AsyncResult.return_value = mock_result

        response = await async_client.get(
            f"/api/v1/inference/tasks/{fake_task_id}",
            cookies=auth_cookie(admin_user),
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["state"] == "SUCCESS"
    results = body["result"]["results"]
    assert results[0]["embedding"] is None, (
        f"embedding should be stripped but got: {results[0]['embedding']}"
    )
