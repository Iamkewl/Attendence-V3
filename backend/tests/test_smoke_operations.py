"""Operations router smoke tests — covers ATT-021 (no DeprecationWarning
on /ready) and ATT-022 (/ready must not leak exception strings to
unauthenticated callers)."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


# ---------------------------------------------------------------------------
# ATT-021 — _probe_triton must use client.assert_server_ready_async(), not
# asyncio.get_event_loop().run_in_executor(...). The deprecated get_event_loop
# call emits DeprecationWarning under Python 3.12+ (and trips filterwarnings
# = ["error"] in pyproject.toml test config).
# ---------------------------------------------------------------------------


def test_att_021_probe_triton_uses_async_wrapper_source() -> None:
    """Source inspection: _probe_triton must call assert_server_ready_async, not the deprecated pattern."""
    import inspect

    from app.api.operations import _probe_triton

    src = inspect.getsource(_probe_triton)
    assert "assert_server_ready_async" in src, (
        "ATT-021: _probe_triton must call client.assert_server_ready_async() "
        "instead of asyncio.get_event_loop().run_in_executor(...) — the async "
        "wrapper is implemented on the client as `await asyncio.to_thread(...)`."
    )
    assert "get_event_loop" not in src, (
        "ATT-021: _probe_triton still uses asyncio.get_event_loop() — that emits "
        "DeprecationWarning under Python 3.12+ and is the explicit FIX target."
    )


def test_att_021_no_get_event_loop_in_module() -> None:
    """Module scan: no remaining get_event_loop calls anywhere in operations.py.

    This catches DeprecationWarning regression in any helper that may later be
    added. The pre-fix file used get_event_loop in exactly one place — _probe_
    triton — but the regression anchor protects the whole module.
    """
    import inspect

    import app.api.operations as ops_module

    src = inspect.getsource(ops_module)
    assert "get_event_loop" not in src, (
        "ATT-021: operations.py contains `get_event_loop` (deprecated under "
        "Python 3.12+). Use `asyncio.get_running_loop()` or, better, "
        "`await asyncio.to_thread(...)` or the per-client async wrappers."
    )


# ---------------------------------------------------------------------------
# ATT-022 — /ready must not leak exception strings. Internal errors must be
# replaced with opaque "unavailable" and the full error logged server-side.
#
# Probe strategy:
#  1. Inject probe failures by monkeypatching the inner helpers and watching
#     the body's `error` field for the opaque string ("unavailable").
#  2. Pre-fix the body's `error` was str(exc) — a fake exc message like
#     "DSN: postgresql://attendance:attendance@db:5432/attendance" would
#     propaget ut. Post-fix the body says "unavailable" + logger.exception
#     captures the real exc server-side.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_att_022_ready_does_not_leak_database_errors(
    async_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the DB probe fails the body must NOT carry the exception string.

    Pre-fix `/ready` returned `{"error": str(exc)}` which would leak asyncpg
    DSN (`postgresql://attendance:attendance@db:5432/...`). Post-fix the body
    carries the opaque `"unavailable"`.
    """
    import app.api.operations as ops_module

    LEAKED_DSN = "postgresql://attendance:supersecret@db-cluster.internal:5432/attendance_test"

    async def _raising_probe_database():
        raise RuntimeError(f"db probe failed: {LEAKED_DSN}")

    monkeypatch.setattr(ops_module, "_probe_database", _raising_probe_database)
    # Leave the other probes alone; we only care about the DB error string not
    # propagating.

    response = await async_client.get("/ready")

    # The endpoint returns 503 on db failure; we don't care about the HTTP
    # code here, we just want to inspect the body's error field.
    body = response.json()
    db_error = body["checks"]["database"]["error"]
    assert db_error == ops_module.ERROR_GENERIC_DATABASE, (
        f"ATT-022: /ready leaked the database exception string — got: {db_error!r}. "
        f"Expected the opaque {ops_module.ERROR_GENERIC_DATABASE!r}. "
        f"DSN should have been logged server-side, NOT emitted in the body."
    )
    # Belt-and-braces: assert the leaked DSN string does NOT appear anywhere
    # in the JSON body, even if some other field were mis-coded.
    body_str = repr(body)
    assert "attendance:supersecret" not in body_str, (
        f"ATT-022: /ready body contains the leaked DB credentials substring — "
        f"body={body_str}"
    )


@pytest.mark.asyncio
async def test_att_022_ready_does_not_leak_redis_errors(
    async_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same as DB, but the redis probe."""
    import app.api.operations as ops_module

    LEAKED_REDIS_URL = "redis://supersecret:bar@redis-cluster.internal:6379/0"

    async def _raising_probe_redis():
        raise RuntimeError(f"redis probe failed: {LEAKED_REDIS_URL}")

    monkeypatch.setattr(ops_module, "_probe_redis", _raising_probe_redis)

    response = await async_client.get("/ready")

    body = response.json()
    redis_error = body["checks"]["redis"]["error"]
    assert redis_error == ops_module.ERROR_GENERIC_REDIS, (
        f"ATT-022: /ready leaked the redis exception string — got: {redis_error!r}. "
        f"Expected the opaque {ops_module.ERROR_GENERIC_REDIS!r}."
    )
    body_str = repr(body)
    assert "supersecret:bar" not in body_str, (
        f"ATT-022: /ready body contains the leaked Redis credentials substring — "
        f"body={body_str}"
    )


@pytest.mark.asyncio
async def test_att_022_ready_does_not_leak_triton_errors(
    async_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Triton probe errors must also be opaque."""
    import app.api.operations as ops_module

    LEAKED_TRITON = "triton://triton-host:8001/models/yolo_model_v2"

    async def _raising_probe_triton():
        raise RuntimeError(f"triton probe failed: {LEAKED_TRITON}")

    monkeypatch.setattr(ops_module, "_probe_triton", _raising_probe_triton)

    response = await async_client.get("/ready")

    body = response.json()
    triton_error = body["checks"]["triton"]["error"]
    assert triton_error == ops_module.ERROR_GENERIC_TRITON, (
        f"ATT-022: /ready leaked the triton exception string — got: "
        f"{triton_error!r}. Expected the opaque "
        f"{ops_module.ERROR_GENERIC_TRITON!r}."
    )
    body_str = repr(body)
    assert "triton://triton-host:8001" not in body_str, (
        f"ATT-022: /ready body contains the leaked Triton host:port — "
        f"body={body_str}"
    )


@pytest.mark.asyncio
async def test_att_022_ready_200_when_all_probes_ok(
    async_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanity: when all probes succeed, all error fields are None and status is 200.

    Pre-fix the `error` field was None on the success path too — this test
    pins the success contract so a regression that "simplifies" the
    opaque-string refactor doesn't accidentally break the ok branch.
    """
    import app.api.operations as ops_module

    async def _ok_probe_database():
        return {"ok": True, "latency_ms": 0.5, "error": None}

    async def _ok_probe_redis():
        return {"ok": True, "latency_ms": 0.5, "error": None}

    async def _ok_probe_triton():
        return {"ok": True, "mode": "demo", "latency_ms": 0.5, "error": None}

    monkeypatch.setattr(ops_module, "_probe_database", _ok_probe_database)
    monkeypatch.setattr(ops_module, "_probe_redis", _ok_probe_redis)
    monkeypatch.setattr(ops_module, "_probe_triton", _ok_probe_triton)

    response = await async_client.get("/ready")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ok"
    assert body["checks"]["database"]["error"] is None
    assert body["checks"]["redis"]["error"] is None
    assert body["checks"]["triton"]["error"] is None


# ---------------------------------------------------------------------------
# ATT-021 wall-clock — _probe_triton must NOT raise DeprecationWarning.
# Verified by monkeypatching `get_triton_client` to return a stub whose
# assert_server_ready_async returns immediately, and asserting `pytest`'s
# filterwarnings(error) doesn't trip during the probe call.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_att_021_probe_triton_does_not_raise_deprecation_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_probe_triton with a stub client should not raise DeprecationWarning.

    Pre-fix the probe did `asyncio.get_event_loop().run_in_executor(None,
    client.assert_server_ready)` which emits DeprecationWarning under
    Python 3.12+ (there is no current event loop in thread 'MainThread' if
    pytest-asyncio has reset it). With pyproject's filterwarnings=["error"]
    that would surface as a TypeError. Post-fix uses the explicit
    `assert_server_ready_async` wrapper which uses `asyncio.to_thread()`.
    """
    import warnings

    import app.api.operations as ops_module

    class _StubTritonClient:
        async def assert_server_ready_async(self) -> None:
            return None

    def _stub_get_triton_client():
        return _StubTritonClient()

    # Force the production (non-demo) code path through the fixed probe.
    monkeypatch.delenv("ATTENDANCE_TRITON_DEMO_MODE", raising=False)
    monkeypatch.setattr(
        "app.infrastructure.triton.get_triton_client",
        _stub_get_triton_client,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        result = await ops_module._probe_triton()

    assert result["ok"] is True, result


# ---------------------------------------------------------------------------
# ATT-022 opt-in loopback tighten — when ATTENDANCE_OPS_REQUIRE_LOOPBACK=1,
# requests from non-loopback client_addr are refused with 403.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_att_022_loopback_opt_in_default_off(
    async_client: AsyncClient,
) -> None:
    """Default (no env var): /health is open to everyone.

    The default-off guard prevents breaking existing k8s probes / external
    monitoring. A toggle (ATTENDANCE_OPS_REQUIRE_LOOPBACK=1) turns the
    tighten on.
    """
    import app.api.operations as ops_module

    ops_module.os.environ.pop("ATTENDANCE_OPS_REQUIRE_LOOPBACK", None)
    response = await async_client.get("/health")
    assert response.status_code == 200, response.text
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_att_022_loopback_opt_in_when_set_with_no_client_addr(
    async_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ATTENDANCE_OPS_REQUIRE_LOOPBACK=1 and the ASGI client has no
    client_addr attached (typical for httpx.ASGITransport), the loopback check
    is treated as safe (returns True from _client_addr_loopback) so in-process
    test traffic still works.
    """

    monkeypatch.setenv("ATTENDANCE_OPS_REQUIRE_LOOPBACK", "1")
    response = await async_client.get("/health")
    # In-process ASGI transport has client=None (no client_addr), and the
    # helper treats that as loopback-safe so tests on the FastAPI app via
    # TestClient or AsyncClient-ASGITransport keep working.
    assert response.status_code == 200, response.text
    assert response.json() == {"status": "ok"}


def test_att_022_loopback_helper_recognizes_ipv4_mapped_ipv6() -> None:
    """_client_addr_loopback must treat ::ffff:127.0.0.1 as loopback (dual-stack containers)."""
    from app.api.operations import _client_addr_loopback

    class _StubRequest:
        class _Client:
            host = "::ffff:127.0.0.1"
            port = 1

        client = _Client()

    assert _client_addr_loopback(_StubRequest()) is True


def test_att_022_loopback_helper_recognizes_localhost() -> None:
    """_client_addr_loopback must treat 'localhost' as loopback (some proxies)."""
    from app.api.operations import _client_addr_loopback

    class _StubRequest:
        class _Client:
            host = "localhost"
            port = 1

        client = _Client()

    assert _client_addr_loopback(_StubRequest()) is True
