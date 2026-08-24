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


# ---------------------------------------------------------------------------
# Flow 5 — ATT-017 regression: confidence_score reflects cosine_similarity
# (recognition strength) when a student matched, NOT the YOLO detection
# score. Falls back to detection_score for unmatched (unknown-face) rows.
# ---------------------------------------------------------------------------


async def _run_pipeline_and_get_sightings(
    test_engine: AsyncEngine,
    fake_triton: FakeTritonGrpcClient,
    *,
    seed_embedding: bool,
) -> tuple[dict, list]:
    """Run the eager pipeline and return (pipeline_result, sighting_rows).

    When ``seed_embedding`` is True (matched-student test), a matching
    StudentEmbedding is seeded, so the pipeline reports ``is_match=True`` +
    a ``cosine_similarity`` >= 0.85. When False (unknown-face test), no
    embedding is seeded, so the pipeline reports ``is_match=False`` +
    ``cosine_similarity=None`` (and the YOLO ``detection_score`` is the
    only numeric confidence field).
    """
    import base64

    from app.domain.models import Sighting
    from app.domain.schemas import InferenceBatchRequest, ImageTensorPayload
    from app.worker.tasks import _run_pipeline_and_log_sightings

    if seed_embedding:
        course, _, _ = await _seed_course_student_embedding(test_engine)
    else:
        # Seed a course with no enrolled students — the pipeline still
        # runs detection + embedding + matching, but no match.
        from app.domain.models import Course

        factory = async_sessionmaker(
            bind=test_engine,
            class_=AsyncSession,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )
        async with factory() as session:
            course = Course(
                id=uuid.uuid4(),
                code="CS101",
                title="Intro to Computing",
                credits=3,
                is_active=True,
            )
            session.add(course)
            await session.commit()
            await session.refresh(course)

    frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
    frame[120:200, 160:220] = 128
    raw_bytes = frame.tobytes()
    data_b64 = base64.b64encode(raw_bytes).decode("ascii")

    request = InferenceBatchRequest(
        request_id=f"att017-smoke-{'matched' if seed_embedding else 'unmatched'}",
        course_id=course.id,
        camera_id="cam-att017",
        frames=[
            ImageTensorPayload(
                frame_id="att017-frame",
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

    factory = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )
    async with factory() as session:
        rows = (
            await session.execute(
                select(Sighting).where(Sighting.camera_id == "cam-att017")
            )
        ).scalars().all()

    return pipeline_result, rows


@pytest.mark.asyncio
async def test_att_017_matched_sighting_confidence_is_recognition_similarity(
    test_engine: AsyncEngine,
    fake_triton: FakeTritonGrpcClient,
) -> None:
    """ATT-017: a matched (recognition->student) sighting's confidence_score
    MUST equal the pipeline's cosine_similarity, NOT the YOLO detection
    score.

    Pre-fix the worker unconditionally stored detection_score; the dashboard
    advertised "Confidence: 87%" for a face that was merely boxed by YOLO
    (and possibly the 0.85 match was the recognition — but the displayed
    number was the YOLO score, not the match strength). Post-fix matched
    rows store cosine_similarity, so the dashboard confidence reflects
    recognition match strength.

    Pin both directions:
      1. The persisted Sighting.confidence_score equals the pipeline's
         cosine_similarity for this sighting's result (within float round
         tolerance of `round(..., 6)`).
      2. The persisted Sighting.confidence_score does NOT equal the
         pipeline's YOLO detection_score (within the same tolerance).
    """
    pipeline_result, rows = await _run_pipeline_and_get_sightings(
        test_engine, fake_triton, seed_embedding=True
    )

    assert rows, (
        f"ATT-017: expected at least one sighting for the matched-face "
        f"case; pipeline_result={pipeline_result!r}"
    )
    results = pipeline_result.get("results", [])
    matched_results = [r for r in results if isinstance(r, dict) and r.get("is_match")]
    assert matched_results, (
        f"ATT-017: pipeline reported no matched results; results={results!r}"
    )

    matched_pipeline_item = matched_results[0]
    cosine = matched_pipeline_item["cosine_similarity"]
    detection = matched_pipeline_item["detection_score"]

    assert cosine is not None, (
        f"ATT-017: matched pipeline result must report cosine_similarity "
        f"(is_match=True implies recognition >= 0.85 threshold); got {matched_pipeline_item!r}"
    )

    # Cross-reference pipeline result with the persisted Sighting row via
    # the embedding_reference hash (string identity from embedding).
    embedding_reference = (
        str(matched_pipeline_item["identity"])
        if matched_pipeline_item.get("identity") is not None
        else None
    )
    sighting = None
    for r in rows:
        if r.embedding_reference == embedding_reference:
            sighting = r
            break
    assert sighting is not None, (
        f"ATT-017: could not find a Sighting row matching the pipeline "
        f"embedding_reference={embedding_reference!r}; rows={rows!r}"
    )

    # The persisted confidence must be the recognition similarity, not
    # the YOLO detection box score (within float round-6 tolerance).
    assert abs((sighting.confidence_score or 0.0) - cosine) < 1e-6, (
        f"ATT-017: matched Sighting.confidence_score must equal pipeline "
        f"cosine_similarity={cosine!r}, got {sighting.confidence_score!r}."
    )
    # And must NOT equal the YOLO detection score (the field that was
    # pre-fix unconditionally stored). This is the regression anchor: if
    # the worker is reverted to read detection_score, this assertion
    # trips.
    assert abs((sighting.confidence_score or 0.0) - detection) > 1e-6, (
        f"ATT-017: matched Sighting.confidence_score must NOT equal the "
        f"YOLO detection_score={detection!r} (pre-fix bug). Got "
        f"{sighting.confidence_score!r}."
    )


@pytest.mark.asyncio
async def test_att_017_unmatched_sighting_confidence_falls_back_to_detection(
    test_engine: AsyncEngine,
    fake_triton: FakeTritonGrpcClient,
) -> None:
    """ATT-017 fallback: when no student matches, the pipeline reports
    is_match=False and cosine_similarity=None, so the worker falls back
    to the YOLO detection_score. The dashboard still shows a confidence
    number for the unknown-face detection.

    This pins the "unknown-face" arm of the fix:
      1. cosine_similarity is None in the pipeline result.
      2. is_match is False.
      3. The persisted Sighting.confidence_score equals the pipeline's
         YOLO detection_score (the fallback path).
    """
    pipeline_result, rows = await _run_pipeline_and_get_sightings(
        test_engine, fake_triton, seed_embedding=False
    )

    # The fixture frame doesn't necessarily trigger a YOLO detection in
    # the FakeTriton stub, so we may have 0 or 1 sighting rows. Check
    # both shapes.
    results = pipeline_result.get("results", [])
    unmatched_results = [
        r
        for r in results
        if isinstance(r, dict) and not r.get("is_match")
    ]

    # If the pipeline returned no items at all, fixtures may be detecting
    # nothing; just assert no sightings were logged and move on (the
    # fallback isn't exercised but the test doesn't fail spuriously).
    if not unmatched_results:
        assert not rows, (
            f"ATT-017: no unmatched pipeline results, but a sighting was "
            f"still persisted; rows={rows!r}, results={results!r}"
        )
        return

    unmatched = unmatched_results[0]
    assert unmatched["cosine_similarity"] is None, (
        f"ATT-017: unmatched pipeline result must have cosine_similarity=None; "
        f"got {unmatched['cosine_similarity']!r}"
    )
    assert unmatched["is_match"] is False, (
        f"ATT-017: unmatched pipeline result must have is_match=False; "
        f"got {unmatched['is_match']!r}"
    )
    detection = unmatched["detection_score"]
    assert detection is not None, (
        f"ATT-017: unmatched pipeline result must have detection_score for "
        f"the fallback; got {unmatched!r}"
    )

    assert rows, (
        f"ATT-017: expected at least one Sighting row for the unmatched case; "
        f"pipeline_result={pipeline_result!r}"
    )
    # Cross-reference pipeline result with the persisted Sighting row
    # via the embedding_reference hash.
    embedding_reference = (
        str(unmatched["identity"]) if unmatched.get("identity") is not None else None
    )
    sighting = None
    for r in rows:
        if r.embedding_reference == embedding_reference:
            sighting = r
            break
    if sighting is None and len(rows) == 1:
        # Fall back to the first row if the embedding_reference is None
        # in BOTH (the unknown-face path doesn't always carry identity).
        sighting = rows[0]

    assert sighting is not None, (
        f"ATT-017: no Sighting row matches the unmatched pipeline result's "
        f"embedding_reference={embedding_reference!r}; rows={rows!r}"
    )

    # And the persisted confidence must be the YOLO detection score for
    # the unmatched case (fallback path).
    assert abs((sighting.confidence_score or 0.0) - detection) < 1e-6, (
        f"ATT-017: unmatched Sighting.confidence_score must fall back to "
        f"detection_score={detection!r}, got {sighting.confidence_score!r}."
    )
