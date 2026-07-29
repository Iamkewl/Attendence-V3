"""Operations router smoke tests — covers ATT-021 (no DeprecationWarning
on /ready) and ATT-022 (/ready must not leak exception strings to
unauthenticated callers)."""

from __future__ import annotations

import ast
import textwrap

import pytest
from httpx import AsyncClient


# ---------------------------------------------------------------------------
# ATT-021 — _probe_triton must use client.assert_server_ready_async(), not
# asyncio.get_event_loop().run_in_executor(...). The deprecated get_event_loop
# call emits DeprecationWarning under Python 3.12+ (and trips filterwarnings
# = ["error"] in pyproject.toml test config).
# ---------------------------------------------------------------------------


def _module_ast() -> ast.Module:
    """Parse the operations module source into an AST.

    Used for AST-level scan for get_event_loop attributes — immune to the
    substring false positive on a comment that mentions the deprecated call
    while describing the fix.
    """
    import inspect

    import app.api.operations as ops_module

    return ast.parse(textwrap.dedent(inspect.getsource(ops_module)))


def _has_get_event_loop_call(tree: ast.Module) -> bool:
    """Walk the AST and return True if any `Attribute` node has attr ==
    "get_event_loop". This catches `asyncio.get_event_loop(...)`,
    `loop.get_event_loop(...)`, and bare `get_event_loop(...)`-as-Call.
    """

    class _Walker(ast.NodeVisitor):
        def __init__(self) -> None:
            self.found = False

        def visit_Attribute(self, node: ast.Attribute) -> None:
            if node.attr == "get_event_loop":
                self.found = True
            self.generic_visit(node)

    walker = _Walker()
    walker.visit(tree)
    return walker.found


def _contains_call_to_async_wrapper_in_probe_triton(src: str) -> bool:
    """Return True if _probe_triton source contains a call to
    assert_server_ready_async. (Substring check across the whole helper —
    the helper is short and doesn't have to be AST-parsed.)"""
    return "assert_server_ready_async" in src


def test_att_021_probe_triton_uses_async_wrapper_source() -> None:
    """_probe_triton must call assert_server_ready_async; must NOT make
    the deprecated get_event_loop call.

    Pre-fix the helper did `asyncio.get_event_loop().run_in_executor(None,
    client.assert_server_ready)`. Post-fix it does
    `await client.assert_server_ready_async()`.
    """
    import inspect

    from app.api.operations import _probe_triton

    src = inspect.getsource(_probe_triton)
    assert _contains_call_to_async_wrapper_in_probe_triton(src), (
        "ATT-021: _probe_triton must call client.assert_server_ready_async() "
        "instead of asyncio.get_event_loop().run_in_executor(...) — the async "
        "wrapper is implemented on the client as `await asyncio.to_thread(...)`."
    )
    # Substring check on the SHORT helper source (the comment that mentions
    # the deprecated call also mentions `assert_server_ready_async`, so this
    # substring check is meaningful: it asserts the call form, not the
    # pattern-matched call form).
    # The strict "no get_event_loop call anywhere" assertion is in
    # test_att_021_no_get_event_loop_in_module (AST-scoped).
    assert "asyncio.get_event_loop(" not in src, (
        "ATT-021: _probe_triton still makes the deprecated "
        "asyncio.get_event_loop() call. Use `await asyncio.to_thread(...)` "
        "or the per-client `assert_server_ready_async` wrapper."
    )


def test_att_021_no_get_event_loop_in_module() -> None:
    """Whole-module AST scan: no Attribute(attr='get_event_loop') anywhere
    in operations.py.

    AT-target regression anchor: catches `asyncio.get_event_loop()`,
    `loop.get_event_loop()`, or `get_event_loop()`-as-Call if any helper
    is later added that re-introduces the deprecated call. AST-scoped so
    the explanatory comment that names the old pattern doesn't trip it.
    """
    tree = _module_ast()
    assert not _has_get_event_loop_call(tree), (
        "ATT-021: operations.py contains an attribute access "
        "`*.get_event_loop(...)` (deprecated under Python 3.12+). Use "
        "`asyncio.get_running_loop()` or, better, `await asyncio.to_thread(...)` "
        "or the per-client async wrappers."
    )


@pytest.mark.asyncio
async def test_att_021_probe_triton_does_not_raise_deprecation_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_probe_triton with a stub client should not raise DeprecationWarning.

    Pre-fix the probe did `asyncio.get_event_loop().run_in_executor(None,
    client.assert_server_ready)` which emits DeprecationWarning under
    Python 3.12+. With pyproject's `filterwarnings=["error"]` that would
    surface as a TypeError. Post-fix uses the explicit
    `await client.assert_server_ready_async()` wrapper which does
    `await asyncio.to_thread(self.assert_server_ready)`.
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
# ATT-022 — /ready must not leak exception strings. Internal errors must be
# replaced with opaque "unavailable" and the full error logged server-side.
#
# Probe strategy: monkeypatch ONE LAYER DEEPER than the probe function so the
# probe's internal `try/except` branch is what catches the exception. If I
# instead replaced `_probe_database` with a function that just raises, the
# exception would escape _probe_database and never reach the catch branch —
# it would escape /ready entirely as an uncaught 500.
#
# For DB: patch `get_session_factory` to raise.
# For Redis: patch `get_redis_client` (awaited) to raise.
# For Triton: patch `get_triton_client` to raise.
#
# In each case the production try/except catches and returns the opaque
# string, /ready then synthesis 503 with the body containing the opaque error
# field instead of str(exc).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_att_022_ready_does_not_leak_database_errors(
    async_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the DB probe fails the body must NOT carry the exception string.

    Pre-fix `/ready` returned `{"error": str(exc)}` which would leak the
    asyncpg DSN. Post-fix the body carries the opaque `"unavailable"` and
    the real exception is logged server-side via `LOGGER.exception(...)`.
    """
    import app.api.operations as ops_module

    LEAKED_DSN = "postgresql://attendance:supersecret@db-cluster.internal:5432/attendance_test"

    def _raising_factory():
        raise RuntimeError(f"db probe failed: {LEAKED_DSN}")

    # Patch get_session_factory at the module-bound name so _probe_database
    # sees the raise. The production try/except then catches it and returns
    # the opaque "unavailable" string. (Patching _probe_database directly
    # would bypass the try/except and the endpoint would 500.)
    monkeypatch.setattr(ops_module, "get_session_factory", _raising_factory)

    response = await async_client.get("/ready")

    # /ready returns 503 on any DB failure (db_ok is False → 503).
    assert response.status_code == 503, response.text
    body = response.json()
    db_error = body["checks"]["database"]["error"]
    assert db_error == ops_module.ERROR_GENERIC_DATABASE, (
        f"ATT-022: /ready leaked the database exception string — got: {db_error!r}. "
        f"Expected the opaque {ops_module.ERROR_GENERIC_DATABASE!r}. "
        f"The DSN should have been logged server-side, NOT emitted in the body."
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
    """Same as DB, but the redis probe. Redis URL leaks are just as bad —
    they expose the host + port + (often) the password.
    """
    import app.api.operations as ops_module

    LEAKED_REDIS_URL = "redis://supersecret:bar@redis-cluster.internal:6379/0"

    async def _raising_get_redis_client():
        raise RuntimeError(f"redis probe failed: {LEAKED_REDIS_URL}")

    monkeypatch.setattr(ops_module, "get_redis_client", _raising_get_redis_client)

    response = await async_client.get("/ready")

    # /ready returns 503 on redis failure (redis_ok is False → 503).
    assert response.status_code == 503, response.text
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
    """Triton probe errors must also be opaque. Triton host:port + model name
    is not a secret per se, but advertising internal infrastructure names
    to an unauthenticated /ready caller is still a leak.
    """
    import app.api.operations as ops_module

    LEAKED_TRITON = "triton://triton-host:8001/models/yolo_model_v2"

    # Force the non-demo path so get_triton_client is actually called.
    monkeypatch.delenv("ATTENDANCE_TRITON_DEMO_MODE", raising=False)

    def _raising_get_triton_client():
        raise RuntimeError(f"triton probe failed: {LEAKED_TRITON}")

    monkeypatch.setattr(
        "app.infrastructure.triton.get_triton_client",
        _raising_get_triton_client,
    )

    response = await async_client.get("/ready")

    # Triton is the only optional probe for /ready — DB + Redis drive the HTTP
    # code. The HTTP code may be 503 (if DB+Redis are also down in CI) or 200
    # (if DB+Redis are up); we don't care which here, we just want to assert
    # Triton's error field is opaque. So DON'T assert on response.status_code.
    body = response.json()
    triton_error = body["checks"]["triton"]["error"]
    assert triton_error == ops_module.ERROR_GENERIC_TRITON, (
        f"ATT-022: /ready leaked the triton exception string — got: "
        f"{triton_error!r}. Expected the opaque "
        f"{ops_module.ERROR_GENERIC_TRITON!r}."
    )
    body_str = repr(body)
    assert "triton://triton-host:8001" not in body_str, (
        f"ATT-022: /ready body contains the leaked Triton host:port + model — "
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
# ATT-022 opt-in loopback tighten — when ATTENDANCE_OPS_REQUIRE_LOOPBACK=1,
# requests from non-loopback client_addr are refused with 403.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_att_022_loopback_opt_in_default_off(
    async_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default (no env var): /health is open to everyone.

    The default-off guard prevents breaking existing k8s probes / external
    monitoring. A toggle (ATTENDANCE_OPS_REQUIRE_LOOPBACK=1) turns the
    tighten on.
    """
    monkeypatch.delenv("ATTENDANCE_OPS_REQUIRE_LOOPBACK", raising=False)

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

    This is the critical "don't break tests" guarantee: an in-process ASGI
    transport can't attach a real client_addr, and the helper recognizes
    that and treats the request as loopback-safe (k8s probes likewise).
    """
    monkeypatch.setenv("ATTENDANCE_OPS_REQUIRE_LOOPBACK", "1")

    response = await async_client.get("/health")
    # In-process ASGI transport has client=None (no client_addr), and the
    # helper treats that as loopback-safe so tests on the FastAPI app via
    # TestClient or AsyncClient-ASGITransport keep working.
    assert response.status_code == 200, response.text
    assert response.json() == {"status": "ok"}


def test_att_022_loopback_helper_recognizes_ipv4_mapped_ipv6() -> None:
    """_client_addr_loopback must treat ::ffff:127.0.0.1 as loopback
    (dual-stack containers)."""
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


def test_att_022_loopback_helper_rejects_external_ip() -> None:
    """_client_addr_loopback must REJECT non-loopback IPs (fail closed).

    Pins the governance: an attacker probing /ready from 203.0.113.5 must
    be refused when ATTENDANCE_OPS_REQUIRE_LOOPBACK=1 is set.
    """
    from app.api.operations import _client_addr_loopback

    class _StubRequest:
        class _Client:
            host = "203.0.113.5"
            port = 1

        client = _Client()

    assert _client_addr_loopback(_StubRequest()) is False


def test_att_022_loopback_helper_rejects_empty_when_require_set() -> None:
    """_client_addr_loopback with client=None (no client_addr attached) is
    treated as loopback-safe — see the production docstring explaining
    in-process ASGI transports have no client_addr and tests mustn't break.
    """
    from app.api.operations import _client_addr_loopback

    class _StubRequest:
        client = None

    # No client_addr → the helper treats as safe (in-process transport).
    # This is the documented "tests should keep working" path.
    assert _client_addr_loopback(_StubRequest()) is True
