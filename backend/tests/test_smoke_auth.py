"""Flow 1 — POST /auth/login returns 200 and sets both auth cookies."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_login_sets_cookies(async_client: AsyncClient, admin_user) -> None:
    response = await async_client.post(
        "/api/v1/auth/login",
        json={"email": admin_user.email, "password": "TestPass1!"},
    )

    assert response.status_code == 200, response.text

    body = response.json()
    assert str(admin_user.id) == body["user"]["id"]

    set_cookie_headers = response.headers.get_list("set-cookie")
    assert len(set_cookie_headers) >= 2, (
        f"Expected at least 2 Set-Cookie headers, got {set_cookie_headers}"
    )
