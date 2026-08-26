"""Live enrollment preview endpoint (no storage, no embeddings).

Pins the phone-style guided-capture contract: detection/liveness-only
analysis reported in-band, numpy capture diagnostics as reason codes, and
the ingest guard inherited from the inference routes.
"""

from __future__ import annotations

from contextlib import contextmanager
from io import BytesIO

import numpy as np
import pytest
from httpx import AsyncClient
from PIL import Image
from sqlalchemy import select

from tests._fakes import FakeTritonGrpcClient

# Detection rows are xyxy pixel-space on a 640x480 frame: [x1, y1, x2, y2, score, class].
_LARGE_CENTERED = [192.0, 120.0, 448.0, 360.0, 0.95, 0.0]
_LARGE_CENTERED_LEFT = [48.0, 140.0, 272.0, 340.0, 0.90, 0.0]
_SMALL_OFFCENTER = [10.0, 10.0, 50.0, 50.0, 0.88, 0.0]


class _RowsFakeTriton(FakeTritonGrpcClient):
    """FakeTritonGrpcClient with caller-chosen YOLO detection rows."""

    def __init__(self, rows: list[list[float]]) -> None:
        super().__init__()
        self._rows = np.asarray(rows, dtype=np.float32).reshape(len(rows), 6)

    def infer_fp32(self, *, model_name, tensors, output_names=None, **kwargs):
        if model_name == "yolov12":
            if self._rows.shape[0] == 0:
                return {"OUTPUT__0": np.empty((1, 0, 6), dtype=np.float32)}
            return {"OUTPUT__0": self._rows[np.newaxis, :, :].astype(np.float32)}
        return super().infer_fp32(
            model_name=model_name, tensors=tensors, output_names=output_names, **kwargs
        )


@contextmanager
def _override_triton(rows: list[list[float]]):
    from app.infrastructure.triton.client import set_triton_client_override

    fake = _RowsFakeTriton(rows)
    set_triton_client_override(fake)
    try:
        yield fake
    finally:
        set_triton_client_override(None)


def _preview_jpeg() -> bytes:
    """Mid-luma noisy JPEG: passes lighting + blur diagnostics by construction."""
    rng = np.random.default_rng(7)
    frame = rng.integers(96, 160, size=(480, 640, 3), dtype=np.uint8)
    buffer = BytesIO()
    Image.fromarray(frame).save(buffer, format="JPEG")
    return buffer.getvalue()


async def _post_preview(async_client: AsyncClient, cookies: dict | None = None):
    kwargs: dict = {
        "files": {"image_file": ("preview.jpg", _preview_jpeg(), "image/jpeg")}
    }
    if cookies is not None:
        kwargs["cookies"] = cookies
    return await async_client.post("/api/v1/students/enroll/preview", **kwargs)


@pytest.mark.asyncio
async def test_preview_reports_no_face_without_raising(
    async_client: AsyncClient, admin_user, auth_cookie
) -> None:
    with _override_triton([]):
        response = await _post_preview(async_client, auth_cookie(admin_user))

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["detected"] is False
    assert body["num_faces"] == 0
    assert body["ok"] is False
    assert body["reasons"] == ["NO_FACE_DETECTED"]
    assert body["bbox"] is None
    assert body["quality_score"] is None


@pytest.mark.asyncio
async def test_preview_ok_for_single_large_centered_face(
    async_client: AsyncClient, admin_user, auth_cookie
) -> None:
    with _override_triton([_LARGE_CENTERED]):
        response = await _post_preview(async_client, auth_cookie(admin_user))

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["detected"] is True
    assert body["num_faces"] == 1
    assert body["reasons"] == []
    assert body["ok"] is True
    assert body["bbox"] == pytest.approx([0.3, 0.25, 0.4, 0.5])
    assert 0.0 <= body["quality_score"] <= 1.0


@pytest.mark.asyncio
async def test_preview_flags_multiple_faces(
    async_client: AsyncClient, admin_user, auth_cookie
) -> None:
    with _override_triton([_LARGE_CENTERED, _LARGE_CENTERED_LEFT]):
        response = await _post_preview(async_client, auth_cookie(admin_user))

    body = response.json()
    assert body["num_faces"] == 2
    assert body["reasons"] == ["MULTIPLE_FACES"]
    assert body["ok"] is False


@pytest.mark.asyncio
async def test_preview_flags_small_offcenter_face(
    async_client: AsyncClient, admin_user, auth_cookie
) -> None:
    with _override_triton([_SMALL_OFFCENTER]):
        response = await _post_preview(async_client, auth_cookie(admin_user))

    body = response.json()
    assert set(body["reasons"]) == {"FACE_TOO_SMALL", "NOT_CENTERED"}
    assert body["bbox"][2] < 0.1
    assert body["ok"] is False


@pytest.mark.asyncio
async def test_preview_writes_nothing_and_returns_no_embedding(
    async_client: AsyncClient, admin_user, auth_cookie, _session_factory
) -> None:
    from app.domain.models import StudentEmbedding, TemplateAuditLog

    with _override_triton([_LARGE_CENTERED]):
        response = await _post_preview(async_client, auth_cookie(admin_user))

    assert response.status_code == 200
    assert "embedding" not in response.json()
    async with _session_factory() as session:
        assert (await session.scalar(select(StudentEmbedding.id).limit(1))) is None
        assert (await session.scalar(select(TemplateAuditLog.id).limit(1))) is None


@pytest.mark.asyncio
async def test_preview_denies_auditor(
    async_client: AsyncClient, auditor_user, auth_cookie
) -> None:
    response = await _post_preview(async_client, auth_cookie(auditor_user))
    assert response.status_code == 403, response.text
    assert "privileges are required" in response.json()["detail"]


@pytest.mark.asyncio
async def test_preview_denies_operator(
    async_client: AsyncClient, operator_user, auth_cookie
) -> None:
    response = await _post_preview(async_client, auth_cookie(operator_user))
    assert response.status_code == 403, response.text
    assert "privileges are required" in response.json()["detail"]
