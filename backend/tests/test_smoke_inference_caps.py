"""Regression anchors for ATT-013 intake caps and ATT-015 task ownership.

Covers the /inference/batch per-frame + aggregate byte caps (the JSON-side
sibling of the /stream upload cap — without it the batch endpoint admits
~128 x 256 MiB of declared tensor data, the same worker-OOM class ATT-013
describes), chunked early-abort on /stream, and the owner/admin/404
authorization matrix on GET /inference/tasks/{task_id}.

NOTE for local runs: these tests are validated by the full suite (the same
way CI runs it). Running single async test files in isolation trips a
pytest-asyncio 0.26 session-loop provisioning quirk on Python 3.12 that is
unrelated to this change — see follow-up issue notes in the PR body.
"""

from __future__ import annotations

import base64
import uuid
from unittest.mock import MagicMock, patch

import pytest
from httpx import AsyncClient


def _frame(frame_id: str, *, width: int, height: int, channels: int = 3, dtype: str = "uint8") -> dict:
    """Minimal valid ImageTensorPayload dict; declared size drives the caps."""
    return {
        "frame_id": frame_id,
        "data_base64": base64.b64encode(b"\x01" * 16).decode("ascii"),
        "width": width,
        "height": height,
        "channels": channels,
        "dtype": dtype,
    }


def _mock_celery_enqueue(task_id: str):
    mock_task = MagicMock()
    mock_task.id = task_id
    return patch("app.api.v1.inference.run_inference_pipeline"), mock_task


# ---------------------------------------------------------------------------
# /stream — oversized uploads get 413 (early abort, not post-buffer)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_rejects_oversized_upload_with_413(
    async_client: AsyncClient,
    admin_user,
    auth_cookie,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A frame larger than the (shrunken) cap is refused mid-stream with 413."""
    import app.api.v1.inference as m

    monkeypatch.setattr(m, "_MAX_FRAME_BYTES", 1024)
    response = await async_client.post(
        "/api/v1/inference/stream",
        data={"frame_id": "f1", "width": 8, "height": 8},
        files={"frame_file": ("frame.bin", b"\x00" * 2048, "application/octet-stream")},
        cookies=auth_cookie(admin_user),
    )
    assert response.status_code == 413, response.text
    assert "ATTENDANCE_MAX_FRAME_BYTES" in response.json()["detail"]


@pytest.mark.asyncio
async def test_stream_accepts_undersized_upload_after_cap_shrink(
    async_client: AsyncClient,
    admin_user,
    auth_cookie,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control: below the cap the stream endpoint still enqueues normally."""
    import app.api.v1.inference as m

    monkeypatch.setattr(m, "_MAX_FRAME_BYTES", 1024)
    patcher, mock_task = _mock_celery_enqueue("cap-ctrl-001")
    with patcher as mock_fn:
        mock_fn.apply_async.return_value = mock_task
        response = await async_client.post(
            "/api/v1/inference/stream",
            data={"frame_id": "f2", "width": 8, "height": 8},
            files={"frame_file": ("frame.bin", b"\x00" * 512, "application/octet-stream")},
            cookies=auth_cookie(admin_user),
        )
    assert response.status_code == 202, response.text
    assert response.json()["state"] == "PENDING"


# ---------------------------------------------------------------------------
# /batch — per-frame and aggregate caps close the JSON-side bypass
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_batch_rejects_single_oversized_declared_frame(
    async_client: AsyncClient,
    admin_user,
    auth_cookie,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One frame declaring more than _MAX_FRAME_BYTES is refused with 413."""
    import app.api.v1.inference as m

    monkeypatch.setattr(m, "_MAX_FRAME_BYTES", 1024)
    payload = {"frames": [_frame("big", width=64, height=64)]}  # 64*64*3 = 12288 > 1024
    response = await async_client.post(
        "/api/v1/inference/batch", json=payload, cookies=auth_cookie(admin_user)
    )
    assert response.status_code == 413, response.text
    assert "'big'" in response.json()["detail"]


@pytest.mark.asyncio
async def test_batch_rejects_aggregate_over_batch_cap(
    async_client: AsyncClient,
    admin_user,
    auth_cookie,
) -> None:
    """Many individually-small frames summing past _MAX_BATCH_BYTES get 413."""
    # Default aggregate cap 64 MiB = 67_108_864. Each frame declares
    # 2048*256*3 = 1_572_864 bytes (< the 4 MiB per-frame cap), so only the
    # aggregate rule can refuse: 43 * 1_572_864 = 67_633_152 > 64 MiB.
    payload = {"frames": [_frame(f"f{i}", width=2048, height=256) for i in range(43)]}
    response = await async_client.post(
        "/api/v1/inference/batch", json=payload, cookies=auth_cookie(admin_user)
    )
    assert response.status_code == 413, response.text
    assert "aggregate maximum" in response.json()["detail"]


@pytest.mark.asyncio
async def test_batch_accepts_payload_just_under_aggregate_cap(
    async_client: AsyncClient,
    admin_user,
    auth_cookie,
) -> None:
    """Boundary control: 42 frames (~65.9 MB total) sit under the 64 MiB cap."""
    patcher, mock_task = _mock_celery_enqueue("batch-boundary-001")
    frames = [_frame(f"f{i}", width=2048, height=256) for i in range(42)]
    with patcher as mock_fn:
        mock_fn.apply_async.return_value = mock_task
        response = await async_client.post(
            "/api/v1/inference/batch",
            json={"frames": frames},
            cookies=auth_cookie(admin_user),
        )
    assert response.status_code == 202, response.text


@pytest.mark.asyncio
async def test_batch_accepts_small_payload_control(
    async_client: AsyncClient,
    admin_user,
    auth_cookie,
) -> None:
    """Control: a normal batch still returns 202 with a task id."""
    patcher, mock_task = _mock_celery_enqueue("batch-control-001")
    with patcher as mock_fn:
        mock_fn.apply_async.return_value = mock_task
        response = await async_client.post(
            "/api/v1/inference/batch",
            json={"frames": [_frame("ok", width=8, height=8)]},
            cookies=auth_cookie(admin_user),
        )
    assert response.status_code == 202, response.text
    assert response.json()["task_id"] == "batch-control-001"


# ---------------------------------------------------------------------------
# GET /tasks/{task_id} — ownership matrix (ATT-015)
# ---------------------------------------------------------------------------

async def _enqueue_one(async_client: AsyncClient, cookie: dict) -> str:
    task_id = f"own-{uuid.uuid4().hex[:8]}"
    patcher, mock_task = _mock_celery_enqueue(task_id)
    with patcher as mock_fn:
        mock_fn.apply_async.return_value = mock_task
        response = await async_client.post(
            "/api/v1/inference/batch",
            json={"frames": [_frame("f", width=8, height=8)]},
            cookies=cookie,
        )
    assert response.status_code == 202, response.text
    return response.json()["task_id"]


@pytest.mark.asyncio
async def test_owner_can_read_own_task_status(
    async_client: AsyncClient,
    instructor_user,
    auth_cookie,
) -> None:
    cookie = auth_cookie(instructor_user)
    task_id = await _enqueue_one(async_client, cookie)
    response = await async_client.get(
        f"/api/v1/inference/tasks/{task_id}", cookies=cookie
    )
    assert response.status_code == 200, response.text
    assert response.json()["task_id"] == task_id


@pytest.mark.asyncio
async def test_non_owner_gets_404_for_foreign_task(
    async_client: AsyncClient,
    admin_user,
    instructor_user,
    auth_cookie,
) -> None:
    """Deny existence: a non-owner sees 404, not 403, for someone else's task."""
    task_id = await _enqueue_one(async_client, auth_cookie(admin_user))
    response = await async_client.get(
        f"/api/v1/inference/tasks/{task_id}", cookies=auth_cookie(instructor_user)
    )
    assert response.status_code == 404, response.text


@pytest.mark.asyncio
async def test_admin_reads_any_task_status(
    async_client: AsyncClient,
    admin_user,
    instructor_user,
    auth_cookie,
) -> None:
    task_id = await _enqueue_one(async_client, auth_cookie(instructor_user))
    response = await async_client.get(
        f"/api/v1/inference/tasks/{task_id}", cookies=auth_cookie(admin_user)
    )
    assert response.status_code == 200, response.text


@pytest.mark.asyncio
async def test_missing_owner_key_denies_non_admin(
    async_client: AsyncClient,
    instructor_user,
    auth_cookie,
) -> None:
    """Fail closed: unknown/never-owned task ids are 404 for non-admins."""
    random_task_id = uuid.uuid4().hex
    response = await async_client.get(
        f"/api/v1/inference/tasks/{random_task_id}", cookies=auth_cookie(instructor_user)
    )
    assert response.status_code == 404, response.text
