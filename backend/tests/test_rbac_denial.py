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

Fixture note: the `student_user` fixture in conftest.py actually provisions
an AUDITOR account, not a STUDENT (tracked as ATT-031, owned by another
workstream; not fixed here). It is used below purely as "a caller who is
neither ADMIN nor INSTRUCTOR" -- comments call out the AUDITOR role
explicitly wherever `student_user` is referenced, to avoid perpetuating the
mislabel in new code.
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

# Fixture names (from conftest.py) for every role that must be refused by
# CurrentAdminUser -- i.e. everyone except ADMIN.
NON_ADMIN_ROLE_FIXTURES = ["instructor_user", "operator_user", "student_user"]

# Fixture names for every role that must be refused by CurrentInstructorUser
# -- i.e. everyone except ADMIN/INSTRUCTOR.
NON_INSTRUCTOR_ROLE_FIXTURES = ["operator_user", "student_user"]

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


@pytest.mark.parametrize("role_fixture", NON_ADMIN_ROLE_FIXTURES)
@pytest.mark.parametrize("method,path,kwargs", ADMIN_ONLY_ROUTES)
@pytest.mark.asyncio
async def test_admin_only_route_denies_non_admin_role(
    request: pytest.FixtureRequest,
    async_client: AsyncClient,
    auth_cookie,
    method: str,
    path: str,
    kwargs: dict,
    role_fixture: str,
) -> None:
    """A caller who is not ADMIN must be refused with 403 on an ADMIN-only route.

    Covers INSTRUCTOR, OPERATOR, and AUDITOR (via the `student_user` fixture)
    against every CurrentAdminUser route in users.py and students.py.
    """
    caller = request.getfixturevalue(role_fixture)
    response = await async_client.request(method, path, cookies=auth_cookie(caller), **kwargs)
    assert response.status_code == 403, (
        f"{method} {path} as role={caller.role!s} ({role_fixture}) expected 403, "
        f"got {response.status_code}: {response.text}"
    )


@pytest.mark.parametrize("role_fixture", NON_INSTRUCTOR_ROLE_FIXTURES)
@pytest.mark.parametrize("method,path,kwargs", INSTRUCTOR_PLUS_ROUTES)
@pytest.mark.asyncio
async def test_instructor_plus_route_denies_lower_role(
    request: pytest.FixtureRequest,
    async_client: AsyncClient,
    auth_cookie,
    method: str,
    path: str,
    kwargs: dict,
    role_fixture: str,
) -> None:
    """A caller who is neither ADMIN nor INSTRUCTOR must be refused with 403.

    Covers OPERATOR and AUDITOR (via the `student_user` fixture) against
    every CurrentInstructorUser route in students.py and inference.py.
    """
    caller = request.getfixturevalue(role_fixture)
    response = await async_client.request(method, path, cookies=auth_cookie(caller), **kwargs)
    assert response.status_code == 403, (
        f"{method} {path} as role={caller.role!s} ({role_fixture}) expected 403, "
        f"got {response.status_code}: {response.text}"
    )


@pytest.mark.parametrize("method,path,kwargs", UNAUTHENTICATED_ROUTES)
@pytest.mark.asyncio
async def test_guarded_route_denies_unauthenticated_caller(
    async_client: AsyncClient,
    method: str,
    path: str,
    kwargs: dict,
) -> None:
    """No credentials at all must be refused with 401, not silently allowed through."""
    response = await async_client.request(method, path, **kwargs)
    assert response.status_code == 401, (
        f"{method} {path} with no credentials expected 401, got "
        f"{response.status_code}: {response.text}"
    )
