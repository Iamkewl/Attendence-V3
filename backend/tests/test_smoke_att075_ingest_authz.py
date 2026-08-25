"""ATT-075 (#80): the ingest endpoints must be instructor-or-admin only.

Before this fix, ``POST /inference/stream`` and ``POST /inference/batch``
accepted frames from ANY authenticated role — AUDITOR (deliberately
read-only, decision D5) and OPERATOR (a worker-system principal) could push
raw camera frames into the biometric pipeline. This file pins the guard:

* every non-privileged role gets a 403 with a clear detail on all three
  ingest endpoints (per role x endpoint matrix below);
* the route handlers actually declare ``CurrentIngestUser`` (source scan, so
  a future refactor back to ``CurrentUser`` fails CI even if the denial
  tests themselves are edited).
"""

from __future__ import annotations

import base64
import copy
import inspect
import textwrap

import pytest
from httpx import AsyncClient

# A schema-valid /inference/batch body (2x2x3 uint8 frame = 12 raw bytes) so a
# 403 can only come from the ingest guard, never from 422 body parsing.
_FRAME_B64 = base64.b64encode(b"\x00" * (2 * 2 * 3)).decode("ascii")
_BATCH_BODY = {
    "frames": [
        {
            "frame_id": "att075-probe",
            "data_base64": _FRAME_B64,
            "width": 2,
            "height": 2,
            "channels": 3,
            "dtype": "uint8",
        }
    ]
}

# (method, path, request kwargs) for each ingest endpoint.
_INGEST_ROUTES = [
    (
        "POST",
        "/api/v1/inference/stream",
        {
            "files": {"frame_file": ("probe.bin", b"\x00" * 4, "application/octet-stream")},
            "data": {"frame_id": "att075-probe", "width": "2", "height": "2"},
        },
    ),
    ("POST", "/api/v1/inference/batch", {"json": _BATCH_BODY}),
    ("POST", "/api/v1/inference/photo", {"files": {"file": ("p.png", b"x", "image/png")}}),
]

_INGEST_ROUTE_IDS = [f"POST:{path.rsplit('/', 1)[-1]}" for _, path, _ in _INGEST_ROUTES]


def _route(index: int) -> tuple[str, str, dict]:
    """Deep-copying accessor: httpx mutates multipart kwargs during send."""
    method, path, kwargs = _INGEST_ROUTES[index]
    return method, path, copy.deepcopy(kwargs)


async def _deny(
    async_client: AsyncClient,
    user,
    auth_cookie,
    *,
    role_label: str,
    route_index: int,
) -> None:
    method, path, kwargs = _route(route_index)
    headers = {"Authorization": f"Bearer {next(iter(auth_cookie(user).values()))}"}
    response = await async_client.request(method, path, headers=headers, **kwargs)
    assert response.status_code == 403, (
        f"{method} {path} as {role_label} expected 403, got "
        f"{response.status_code}: {response.text}"
    )
    assert "privileges are required" in response.json()["detail"], (
        f"{method} {path}: 403 detail must tell the caller which privilege "
        f"is missing; got {response.json()['detail']!r}"
    )


@pytest.mark.parametrize("route_index", range(len(_INGEST_ROUTES)), ids=_INGEST_ROUTE_IDS)
@pytest.mark.asyncio
async def test_ingest_route_denies_auditor(
    async_client: AsyncClient,
    auditor_user,
    auth_cookie,
    route_index: int,
) -> None:
    """AUDITOR is read-only by design (D5) and must never submit frames."""
    await _deny(
        async_client, auditor_user, auth_cookie, role_label="AUDITOR", route_index=route_index
    )


@pytest.mark.parametrize("route_index", range(len(_INGEST_ROUTES)), ids=_INGEST_ROUTE_IDS)
@pytest.mark.asyncio
async def test_ingest_route_denies_operator(
    async_client: AsyncClient,
    operator_user,
    auth_cookie,
    route_index: int,
) -> None:
    """OPERATOR drives worker systems, not cameras; ingest is denied."""
    await _deny(
        async_client, operator_user, auth_cookie, role_label="OPERATOR", route_index=route_index
    )


# ---------------------------------------------------------------------------
# Source scan: the three handlers must depend on CurrentIngestUser.
# ---------------------------------------------------------------------------


def test_att_075_all_three_ingest_handlers_declare_current_ingest_user() -> None:
    """A swap back to ``CurrentUser`` must fail CI even if denials are edited.

    Mirrors the ATT-048 source-scan style: proxy-immune to runtime wiring.
    """
    from app.api.v1 import inference as inference_module

    for handler_name in (
        "enqueue_stream_inference",
        "enqueue_batch_inference",
        "recognize_photo",
    ):
        handler = getattr(inference_module, handler_name)
        src = textwrap.dedent(inspect.getsource(handler))
        assert "current_user: CurrentIngestUser" in src, (
            f"ATT-075 regression: {handler_name} no longer declares "
            f"`current_user: CurrentIngestUser`; the ingest guard was dropped."
        )


def test_att_075_current_ingest_user_is_exported_from_deps() -> None:
    """The dependency alias must stay part of deps.py's public surface."""
    import app.api.deps as deps_module

    assert "CurrentIngestUser" in deps_module.__all__
    assert hasattr(deps_module, "CurrentIngestUser")
    assert hasattr(deps_module, "get_current_ingest_user")
