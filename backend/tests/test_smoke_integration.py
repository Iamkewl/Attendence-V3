"""Flow 8 — end-to-end inference pipeline broadcasts a sighting via Redis pub/sub.

This is the only smoke test that exercises the full publish→subscribe loop:
worker → attendance_service.log_sighting → publish_json(LIVE_ATTENDANCE_CHANNEL)
→ Redis → subscriber.  The individual smoke tests for inference, WS, and SSE
each verify a single leg; this one ties them together.
"""

from __future__ import annotations

import asyncio
import base64
import json
import uuid
from datetime import UTC, datetime

import numpy as np
import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from tests._fakes import FakeTritonGrpcClient, _unit_norm_vector


async def _seed_course_student_embedding(engine: AsyncEngine):
    """Insert a Course, Student (with linked User), and a matching StudentEmbedding."""
    from app.core.security import hash_password
    from app.domain.models import Course, Student, StudentEmbedding, User, UserRole

    factory = async_sessionmaker(
        bind=engine, class_=AsyncSession, autoflush=False, autocommit=False, expire_on_commit=False
    )

    async with factory() as session:
        user = User(
            id=uuid.uuid4(),
            email="integration@test.example",
            full_name="Integration Student",
            password_hash=hash_password("TestPass1!"),
            role=UserRole.AUDITOR,
            is_active=True,
        )
        session.add(user)
        await session.flush()

        course = Course(
            id=uuid.uuid4(),
            code="INT101",
            title="Integration Test Course",
            credits=3,
            is_active=True,
        )
        session.add(course)
        await session.flush()

        student = Student(
            id=uuid.uuid4(),
            user_id=user.id,
            student_number="INT0001",
            program="Integration Program",
            enrollment_year=2024,
            is_active=True,
        )
        session.add(student)
        await session.flush()

        emb = StudentEmbedding(
            id=uuid.uuid4(),
            student_id=student.id,
            embedding=_unit_norm_vector(seed=42).tolist(),
            pose_label="front",
            quality_score=1.0,
            is_active=True,
        )
        session.add(emb)
        await session.commit()
        await session.refresh(course)
        await session.refresh(student)

    return course, student


@pytest.mark.asyncio
async def test_pipeline_publishes_sighting_to_redis_channel(
    test_engine: AsyncEngine,
    fake_triton: FakeTritonGrpcClient,
    redis_client,
) -> None:
    from app.core.pubsub import LIVE_ATTENDANCE_CHANNEL, RedisPubSubManager
    from app.domain.schemas import ImageTensorPayload, InferenceBatchRequest
    from app.worker.tasks import _run_pipeline_and_log_sightings

    course, student = await _seed_course_student_embedding(test_engine)

    frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
    frame[120:200, 160:220] = 128
    data_b64 = base64.b64encode(frame.tobytes()).decode("ascii")

    request = InferenceBatchRequest(
        request_id="integration-smoke-001",
        course_id=course.id,
        camera_id="cam-integration",
        frames=[
            ImageTensorPayload(
                frame_id="integration-frame-001",
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

    received: list[dict] = []
    stop_event = asyncio.Event()
    subscriber_ready = asyncio.Event()
    manager = RedisPubSubManager(redis_client=redis_client)

    async def _subscriber() -> None:
        # Subscribe FIRST so the publish from log_sighting is not missed
        # (Redis pub/sub does not replay missed messages).
        async with manager.subscribe([LIVE_ATTENDANCE_CHANNEL]) as pubsub:
            subscriber_ready.set()
            while not stop_event.is_set():
                msg = await pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=0.2,
                )
                if msg is None:
                    continue
                received.append(json.loads(msg["data"]))
                if received:
                    return

    sub_task = asyncio.create_task(_subscriber())

    # Wait for the subscriber to actually be registered with Redis before
    # publishing, otherwise the SUBSCRIBE may land after the PUBLISH.
    await asyncio.wait_for(subscriber_ready.wait(), timeout=2.0)

    pipeline_result = await _run_pipeline_and_log_sightings(request)
    assert pipeline_result.get("sightings_logged", 0) >= 1, (
        f"Pipeline did not log any sightings: {pipeline_result}"
    )

    try:
        await asyncio.wait_for(sub_task, timeout=5.0)
    except TimeoutError:
        stop_event.set()
        await sub_task
        pytest.fail(
            f"No sighting broadcast received within 5s; collected {len(received)} messages"
        )

    assert len(received) >= 1, "Subscriber never received a published sighting event"

    event = received[0]
    assert event["event_type"] == "sighting_logged", f"Unexpected event_type: {event}"
    assert event["sighting"]["course_id"] == str(course.id), (
        f"Broadcast course_id does not match seeded course: {event}"
    )
    assert event["sighting"]["student_id"] == str(student.id), (
        f"Broadcast student_id does not match recognized student: {event}"
    )
    assert event["sighting"]["camera_id"] == "cam-integration"
    assert isinstance(event["event_id"], str)
    assert isinstance(event["emitted_at"], str)
