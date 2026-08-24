"""Tier-1 CI hardening: RBAC denial tests (ATT T1c).

Before this file, the suite contained exactly three status-code assertions
(all 2xx) and zero 4xx assertions anywhere. Every role-gated route was only
ever exercised by a caller who was already allowed through, so a guard that
silently regresses -- e.g. an `allowed_roles` set widened to include
everyone, or a route's `CurrentAdminUser` swapped for `CurrentUser` -- would
not fail a single existing test. This file closes that gap.

Guards under test (backend/app/api/deps.py):
  * CurrentAdminUser      -> ADMIN only
  * CurrentInstructorUser -> ADMIN or INSTRUCTOR

Roles (backend/app/domain/models/_base.py): ADMIN, INSTRUCTOR, AUDITOR,
OPERATOR. There is no STUDENT role. `CurrentWorkerSystem` (ADMIN or
OPERATOR) is defined and exported in deps.py but used by zero routes
(tracked as ATT-076) -- it has nothing to deny-test against and is
intentionally absent below.

Fixture note: the `auditor_user` fixture in conftest.py provisions an
AUDITOR account. It is used below purely as "a caller who is neither
ADMIN nor INSTRUCTOR" -- comments call out the AUDITOR role explicitly
wherever `auditor_user` is referenced, to reflect the actual role
(ATT-031 renamed `student_user` -> `auditor_user` to remove the prior
mislabel where a fixture named after a non-existent STUDENT role was
really an AUDITOR).

Implementation note: role fixtures are always requested directly as normal
test parameters (never via `request.getfixturevalue()`). The latter does
not reliably resolve async fixtures from inside an already-running async
test under pytest-asyncio and raises
`RuntimeError: There is no current event loop in thread 'MainThread'` --
confirmed against this suite's actual CI run before landing this version.
"""

from __future__ import annotations

import uuid

import pytest
from httpx import AsyncClient


# Syntactically valid UUIDs that are never inserted into the database. Every
# route below performs its role check as a FastAPI `Depends` dependency,
# which is resolved -- and, on rejection, raises -- before the path
# operation body (and therefore before any DB lookup keyed on this id) ever
# runs. The row does not need to exist for a denial test.
_UNUSED_USER_ID = "00000000-0000-4000-8000-000000000001"
_UNUSED_STUDENT_ID = "00000000-0000-4000-8000-000000000002"

# Schema-valid bodies/files for every POST/PATCH route below. Kept
# deliberately valid (rather than empty/garbage) so that a 403 can only ever
# come from the RBAC guard -- never from a 422 raised while parsing the
# payload -- regardless of the exact order FastAPI solves dependencies vs.
# request body in a given release.
_VALID_USER_CREATE_BODY = {
    "email": "rbac-denial-target@test.example",
    "full_name": "Rbac Denial Target",
    "password": "TestPass1!",
    "role": "instructor",
    "is_active": True,
}

_VALID_STUDENT_CREATE_BODY = {
    "user_id": str(uuid.uuid4()),
    "student_number": "RBAC0001",
    "program": "Computer Science",
    "enrollment_year": 2024,
}

_FAKE_IMAGE_FILE = ("probe.png", b"not-a-real-image-the-guard-runs-first", "image/png")


# ---------------------------------------------------------------------------
# Route tables: (http_method, path, extra httpx kwargs)
# ---------------------------------------------------------------------------

ADMIN_ONLY_ROUTES = [
    pytest.param("POST", "/api/v1/users", {"json": _VALID_USER_CREATE_BODY}, id="POST:users"),
    pytest.param("GET", "/api/v1/users", {}, id="GET:users"),
    pytest.param("GET", f"/api/v1/users/{_UNUSED_USER_ID}", {}, id="GET:users-id"),
    pytest.param("PATCH", f"/api/v1/users/{_UNUSED_USER_ID}", {"json": {}}, id="PATCH:users-id"),
    pytest.param("DELETE", f"/api/v1/users/{_UNUSED_USER_ID}", {}, id="DELETE:users-id"),
    pytest.param("DELETE", f"/api/v1/students/{_UNUSED_STUDENT_ID}", {}, id="DELETE:students-id"),
]

INSTRUCTOR_PLUS_ROUTES = [
    pytest.param("POST", "/api/v1/students", {"json": _VALID_STUDENT_CREATE_BODY}, id="POST:students"),
    pytest.param("GET", "/api/v1/students", {}, id="GET:students"),
    pytest.param("GET", f"/api/v1/students/{_UNUSED_STUDENT_ID}", {}, id="GET:students-id"),
    pytest.param("PATCH", f"/api/v1/students/{_UNUSED_STUDENT_ID}", {"json": {}}, id="PATCH:students-id"),
    pytest.param(
        "POST",
        f"/api/v1/students/{_UNUSED_STUDENT_ID}/enroll",
        {"files": {"image_file": _FAKE_IMAGE_FILE}},
        id="POST:students-id-enroll",
    ),
    pytest.param(
        "POST",
        "/api/v1/inference/photo",
        {"files": {"file": _FAKE_IMAGE_FILE}},
        id="POST:inference-photo",
    ),
]

# A representative guarded route per tier, used for the unauthenticated
# (no cookie at all) 401 checks.
UNAUTHENTICATED_ROUTES = [
    pytest.param("GET", "/api/v1/users", {}, id="GET:users"),
    pytest.param("GET", "/api/v1/students", {}, id="GET:students"),
    pytest.param(
        "GET",
        f"/api/v1/attendance/sessions?course_id={_UNUSED_USER_ID}",
        {},
        id="GET:attendance-sessions",
    ),
]


async def _assert_status(
    async_client: AsyncClient,
    method: str,
    path: str,
    kwargs: dict,
    expected_status: int,
    *,
    cookies: dict | None = None,
    role_label: str = "unauthenticated",
) -> None:
    # Send the session token as a Bearer header rather than as per-request
    # cookies. httpx deprecated `cookies=` on individual requests, and
    # filterwarnings = ["error"] promotes that DeprecationWarning to a failure.
    # get_current_user accepts either (deps.py:27), and the bearer path also
    # avoids leaking client-level cookie state between parametrized cases.
    headers = dict(kwargs.pop("headers", {}))
    if cookies:
        headers["Authorization"] = f"Bearer {next(iter(cookies.values()))}"

    response = await async_client.request(method, path, headers=headers, **kwargs)
    assert response.status_code == expected_status, (
        f"{method} {path} as {role_label} expected {expected_status}, got "
        f"{response.status_code}: {response.text}"
    )


# ---------------------------------------------------------------------------
# ADMIN-only routes: every non-ADMIN role must be refused with 403.
# Three dedicated test functions (one per non-admin role) rather than a
# fixture-name parametrize, so every role fixture is requested directly and
# resolved by pytest-asyncio in the normal way (see module docstring).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method,path,kwargs", ADMIN_ONLY_ROUTES)
@pytest.mark.asyncio
async def test_admin_only_route_denies_instructor(
    async_client: AsyncClient,
    instructor_user,
    auth_cookie,
    method: str,
    path: str,
    kwargs: dict,
) -> None:
    """INSTRUCTOR is privileged but is not ADMIN -- must still be refused."""
    await _assert_status(
        async_client, method, path, kwargs, 403,
        cookies=auth_cookie(instructor_user), role_label="INSTRUCTOR",
    )


@pytest.mark.parametrize("method,path,kwargs", ADMIN_ONLY_ROUTES)
@pytest.mark.asyncio
async def test_admin_only_route_denies_operator(
    async_client: AsyncClient,
    operator_user,
    auth_cookie,
    method: str,
    path: str,
    kwargs: dict,
) -> None:
    """OPERATOR must be refused on ADMIN-only routes."""
    await _assert_status(
        async_client, method, path, kwargs, 403,
        cookies=auth_cookie(operator_user), role_label="OPERATOR",
    )


@pytest.mark.parametrize("method,path,kwargs", ADMIN_ONLY_ROUTES)
@pytest.mark.asyncio
async def test_admin_only_route_denies_auditor(
    async_client: AsyncClient,
    auditor_user,  # ATT-031: AUDITOR fixture name reflects the actual role.
    auth_cookie,
    method: str,
    path: str,
    kwargs: dict,
) -> None:
    """AUDITOR must be refused on ADMIN-only routes."""
    await _assert_status(
        async_client, method, path, kwargs, 403,
        cookies=auth_cookie(auditor_user), role_label="AUDITOR",
    )


# ---------------------------------------------------------------------------
# INSTRUCTOR-plus routes (ADMIN or INSTRUCTOR): OPERATOR and AUDITOR must be
# refused.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method,path,kwargs", INSTRUCTOR_PLUS_ROUTES)
@pytest.mark.asyncio
async def test_instructor_plus_route_denies_operator(
    async_client: AsyncClient,
    operator_user,
    auth_cookie,
    method: str,
    path: str,
    kwargs: dict,
) -> None:
    """OPERATOR must be refused on CurrentInstructorUser routes."""
    await _assert_status(
        async_client, method, path, kwargs, 403,
        cookies=auth_cookie(operator_user), role_label="OPERATOR",
    )


@pytest.mark.parametrize("method,path,kwargs", INSTRUCTOR_PLUS_ROUTES)
@pytest.mark.asyncio
async def test_instructor_plus_route_denies_auditor(
    async_client: AsyncClient,
    auditor_user,  # ATT-031: AUDITOR fixture name reflects the actual role.
    auth_cookie,
    method: str,
    path: str,
    kwargs: dict,
) -> None:
    """AUDITOR must be refused on CurrentInstructorUser routes."""
    await _assert_status(
        async_client, method, path, kwargs, 403,
        cookies=auth_cookie(auditor_user), role_label="AUDITOR",
    )


# ---------------------------------------------------------------------------
# No credentials at all -> 401, not silently allowed through.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method,path,kwargs", UNAUTHENTICATED_ROUTES)
@pytest.mark.asyncio
async def test_guarded_route_denies_unauthenticated_caller(
    async_client: AsyncClient,
    method: str,
    path: str,
    kwargs: dict,
) -> None:
    await _assert_status(async_client, method, path, kwargs, 401, cookies=None)
