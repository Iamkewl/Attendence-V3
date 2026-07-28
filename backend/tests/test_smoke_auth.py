"""Authentication smoke tests — login cookie-set, plus regressions for
ATT-018 (login timing-oracle equalization), ATT-032 (issue_websocket_ticket
reuses a module-level RedisPubSubManager singleton), and ATT-046 (cookie
SameSite is configurable via ATTENDANCE_COOKIE_SAMESITE)."""

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


# ---------------------------------------------------------------------------
# ATT-018 — /auth/login must always run one Argon2 verify per request so the
# response time cannot be used to enumerate registered emails.
#
# Pre-fix behaviour: `if user is None or not verify_password(...)` short-
# circuits when the user lookup misses. Post-fix behaviour: `_verify_login_
# credentials` runs verify_password against a cached dummy Argon2 hash when
# the user is None.
#
# We monkeypatch `app.api.v1.auth.verify_password` (the binding the route
# module reads) with a recording spy, then assert the call count on the miss
# branch is at least 1. Without the fix, the miss branch never calls
# verify_password at all (short-circuit), so the assertion would fail.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_att_018_login_absent_user_still_runs_argon2_verify(
    async_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An absent email must still trigger one Argon2 verify on the server.

    Pre-fix this fails because the `or` short-circuits when user is None.
    """
    import app.api.v1.auth as auth_module

    calls: list[tuple[str, str]] = []

    def _spy(password: str, password_hash: str) -> bool:
        calls.append((password, password_hash))
        from app.core.security import verify_password as real_verify

        return real_verify(password, password_hash)

    monkeypatch.setattr(auth_module, "verify_password", _spy)

    response = await async_client.post(
        "/api/v1/auth/login",
        json={"email": "definitely-absent-user-no-row@example.org", "password": "SomePassword1!"},
    )

    assert response.status_code == 401, response.text
    assert len(calls) >= 1, (
        "verify_password was not called for an absent-user login — the "
        "ATT-018 timing-oracle short-circuit is still present in the code. "
        "An attacker can enumerate registered emails via median response "
        "times because the miss branch skips Argon2 entirely."
    )


@pytest.mark.asyncio
async def test_att_018_login_present_user_wrong_password_runs_argon2_verify(
    async_client: AsyncClient,
    admin_user,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Baseline guard: a wrong password for a present user DOES call verify_password.

    This already passes pre-fix (the short-circuit only fires on the miss
    branch), but the test pins the contract so a future refactor cannot
    accidentally remove the verify call on the hit branch.
    """
    import app.api.v1.auth as auth_module

    calls: list[tuple[str, str]] = []

    def _spy(password: str, password_hash: str) -> bool:
        calls.append((password, password_hash))
        from app.core.security import verify_password as real_verify

        return real_verify(password, password_hash)

    monkeypatch.setattr(auth_module, "verify_password", _spy)

    response = await async_client.post(
        "/api/v1/auth/login",
        json={"email": admin_user.email, "password": "WrongPassword1!"},
    )

    assert response.status_code == 401, response.text
    assert len(calls) >= 1


@pytest.mark.asyncio
async def test_att_018_absent_and_present_login_both_run_verify(
    async_client: AsyncClient,
    admin_user,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Combined contract: login runs verify_password in both miss and hit branches.

    This is the test that the pre-fix code fails (miss branch skips verify)
    and the post-fix code passes (both branches run verify).
    """
    import app.api.v1.auth as auth_module

    real_verify = auth_module.verify_password
    miss_calls = [0]
    hit_calls = [0]

    def _miss_spy(password: str, password_hash: str) -> bool:
        miss_calls[0] += 1
        return real_verify(password, password_hash)

    monkeypatch.setattr(auth_module, "verify_password", _miss_spy)

    await async_client.post(
        "/api/v1/auth/login",
        json={"email": "absent-no-row@example.org", "password": "SomePassword1!"},
    )

    def _hit_spy(password: str, password_hash: str) -> bool:
        hit_calls[0] += 1
        return real_verify(password, password_hash)

    monkeypatch.setattr(auth_module, "verify_password", _hit_spy)

    await async_client.post(
        "/api/v1/auth/login",
        json={"email": admin_user.email, "password": "WrongPassword1!"},
    )

    assert miss_calls[0] >= 1, (
        f"Absent-user login did not run verify_password — ATT-018 still "
        f"vulnerable. counts: miss={miss_calls[0]} hit={hit_calls[0]}"
    )
    assert hit_calls[0] >= 1, (
        f"Present-user wrong-password login did not run verify_password. "
        f"counts: miss={miss_calls[0]} hit={hit_calls[0]}"
    )


def test_att_018_dummy_password_hash_is_lazy_and_stable() -> None:
    """The dummy hash is generated lazily on first miss and cached thereafter.

    Two consecutive calls must return the same hash, otherwise the lru_cache
    wasn't wired (and timing would drift between first miss and subsequent
    misses because Argon2 hashing time itself would be added to the first
    request's wall-clock)."""
    import app.api.v1.auth as auth_module

    auth_module._dummy_password_hash.cache_clear()

    first = auth_module._dummy_password_hash()
    second = auth_module._dummy_password_hash()
    assert first == second, "ATT-018 dummy hash should be lru_cache'd (one-time)"
    assert isinstance(first, str)
    assert first.startswith("$argon2"), (
        f"ATT-018 dummy hash is not an Argon2 hash — got {first!r}"
    )


# ---------------------------------------------------------------------------
# ATT-032 — /auth/ws-ticket should reuse the module-level _auth_pubsub_manager
# instead of constructing RedisPubSubManager() per call.
# ---------------------------------------------------------------------------


def test_att_032_module_level_singleton_exists() -> None:
    """Module must expose `_auth_pubsub_manager` (ATT-032 fix anchor).

    Pre-fix the module had no `_auth_pubsub_manager` and the ws-ticket
    endpoint constructed RedisPubSubManager() inside the function body.
    """
    import app.api.v1.auth as auth_module

    obj = getattr(auth_module, "_auth_pubsub_manager", None)
    assert obj is not None, (
        "ATT-032: _auth_pubsub_manager singleton absent — ws-ticket endpoint "
        "is still constructing RedisPubSubManager per call."
    )


@pytest.mark.asyncio
async def test_att_032_issue_ticket_does_not_construct_manager_per_call(
    async_client: AsyncClient,
    admin_user,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """issue_websocket_ticket must reuse the singleton, not call RedisPubSubManager().

    We patch `app.api.v1.auth.RedisPubSubManager` to AssertionError() so any
    fresh construction inside the endpoint body trips. Pre-fix the endpoint
    body had `pubsub_manager = RedisPubSubManager()` which would call the
    patched ctor. Post-fix the endpoint body just reads the module singleton
    and never constructs.
    """
    import app.api.v1.auth as auth_module

    def _forbid(*args, **kwargs):
        raise AssertionError(
            "ATT-032: issue_websocket_ticket constructed a fresh "
            "RedisPubSubManager() — must reuse the module singleton "
            "_auth_pubsub_manager instead."
        )

    monkeypatch.setattr(auth_module, "RedisPubSubManager", _forbid)

    login_resp = await async_client.post(
        "/api/v1/auth/login",
        json={"email": admin_user.email, "password": "TestPass1!"},
    )
    assert login_resp.status_code == 200, login_resp.text

    response = await async_client.post("/api/v1/auth/ws-ticket")
    assert response.status_code == 200, response.text
    body = response.json()
    assert "ticket" in body


# ---------------------------------------------------------------------------
# ATT-046 — SameSite must be configurable via ATTENDANCE_COOKIE_SAMESITE.
#
# Default behaviour (no env var) keeps `strict` so existing deployments do
# not regress. Setting `ATTENDANCE_COOKIE_SAMESITE=lax` makes the Set-Cookie
# header emit `samesite=lax`. Invalid values raise during request handling
# (fail-closed per §6: the cookie flag must not silently default to a more
# permissive mode).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_att_046_default_samesite_is_strict(
    async_client: AsyncClient,
    admin_user,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ATTENDANCE_COOKIE_SAMESITE, the cookie still says strict (default)."""
    monkeypatch.delenv("ATTENDANCE_COOKIE_SAMESITE", raising=False)
    response = await async_client.post(
        "/api/v1/auth/login",
        json={"email": admin_user.email, "password": "TestPass1!"},
    )
    assert response.status_code == 200, response.text
    set_cookie_headers = response.headers.get_list("set-cookie")
    assert all("samesite=strict" in h.lower() for h in set_cookie_headers), (
        f"ATT-046: default SameSite should be 'strict', got: {set_cookie_headers}"
    )


@pytest.mark.asyncio
async def test_att_046_samesite_lax_env_overrides_default(
    async_client: AsyncClient,
    admin_user,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setting ATTENDANCE_COOKIE_SAMESITE=lax emits samesite=lax.

    This test FAILS on the pre-fix code because the cookie flag was
    hard-coded to `samesite="strict"` and ignored the env var entirely.
    """
    monkeypatch.setenv("ATTENDANCE_COOKIE_SAMESITE", "lax")
    response = await async_client.post(
        "/api/v1/auth/login",
        json={"email": admin_user.email, "password": "TestPass1!"},
    )
    assert response.status_code == 200, response.text
    set_cookie_headers = response.headers.get_list("set-cookie")
    assert any("samesite=lax" in h.lower() for h in set_cookie_headers), (
        f"ATT-046: ATTENDANCE_COOKIE_SAMESITE=lax did not propagate — got: "
        f"{set_cookie_headers}"
    )
    assert all("samesite=lax" in h.lower() for h in set_cookie_headers), (
        f"ATT-046: only some cookies carry the override — got: {set_cookie_headers}"
    )


@pytest.mark.asyncio
async def test_att_046_samesite_none_env_overrides_default(
    async_client: AsyncClient,
    admin_user,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setting ATTENDANCE_COOKIE_SAMESITE=none emits samesite=none (cross-site)."""
    monkeypatch.setenv("ATTENDANCE_COOKIE_SAMESITE", "none")
    response = await async_client.post(
        "/api/v1/auth/login",
        json={"email": admin_user.email, "password": "TestPass1!"},
    )
    assert response.status_code == 200, response.text
    set_cookie_headers = response.headers.get_list("set-cookie")
    assert all("samesite=none" in h.lower() for h in set_cookie_headers), (
        f"ATT-046: ATTENDANCE_COOKIE_SAMESITE=none did not propagate — got: "
        f"{set_cookie_headers}"
    )


@pytest.mark.asyncio
async def test_att_046_invalid_samesite_env_fails_closed(
    async_client: AsyncClient,
    admin_user,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid ATTENDANCE_COOKIE_SAMESITE reject the request rather than emit unsafe cookies.

    The CVE-class risk here is the cookie flag silently defaulting to
    `none` (or to anything else) when the operator's env var has a typo.
    The validator raises RuntimeError so the app fails fast at cookie
    emission time. The endpoint returns a 500-level error rather than
    emitting unsafe cookies.
    """
    monkeypatch.setenv("ATTENDANCE_COOKIE_SAMESITE", "l@x")
    response = await async_client.post(
        "/api/v1/auth/login",
        json={"email": admin_user.email, "password": "TestPass1!"},
    )
    # Per the fail-closed rule for security fixes (§6): an invalid config
    # must NOT silently fall back to a permissive mode.
    assert response.status_code >= 400, response.text
