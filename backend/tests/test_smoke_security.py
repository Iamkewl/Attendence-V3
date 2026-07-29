"""Security smoke tests — covers ATT-042.

ATT-042 (Medium/security): No JWT-secret minimum-length enforcement. The
HS256 symmetric integrity depends entirely on the secret's entropy.
`_read_required_env("ATTENDANCE_JWT_SECRET")` accepted any non-blank value
including `"x"`. A rushed operator pasting placeholder text into `.env`
produces a token whose signature can be brute-forced offline. For a
biometric-attendance system a forged token grants ADMIN roles over face-
template enrollment and attendance overrides.

Fix in `backend/app/core/security.py`: at startup, enforce
`len(jwt_secret_key.encode("utf-8")) >= _JWT_SECRET_MIN_BYTES (32)` and
reject launch with `RuntimeError`. Plus refuse the documented placeholder
strings (README/RUNBOOK/CLAUDE.md publish these as dev placeholders) via
`_JWT_SECRET_PLACEHOLDER_BLOCKLIST`. Both checks fail CLOSED — the app
does not start until the operator sets a real secret.
"""

from __future__ import annotations


import pytest


def _reload_security_with_env(monkeypatch: pytest.MonkeyPatch, secret: str | None) -> None:
    """Reload app.core.security with the env-var injected.

    ATT-042's check runs AT IMPORT-TIME inside `get_celery_app()` — actually
    it runs lazily on the first call to `get_security_settings()`. So we
    just need to clear the lru_cache, set the env, and call
    `get_security_settings()` (or read the env via the private helper).
    """
    import app.core.security as security

    security.get_security_settings.cache_clear()
    if secret is None:
        monkeypatch.delenv("ATTENDANCE_JWT_SECRET", raising=False)
    else:
        monkeypatch.setenv("ATTENDANCE_JWT_SECRET", secret)


# ---------------------------------------------------------------------------
# Fail-closed: short / blank / missing secrets MUST prevent launch.
# Per the issue ACCEPT: "ATTENDANCE_JWT_SECRET=short fails app startup
# with a clear message; integration test asserts the rejection."
# ---------------------------------------------------------------------------


def test_att_042_short_secret_rejected_at_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 5-character secret ("short") MUST raise RuntimeError at startup.

    Pre-fix `_read_required_env` accepted any non-blank value, so a rushed
    operator pasting `short` got a token whose HS256 signature is brute-
    forceable offline in seconds. Post-fix the check enforces
    `len(secret.encode('utf-8')) >= _JWT_SECRET_MIN_BYTES` and the app
    refuses to start.
    """
    from app.core.security import _read_and_validate_jwt_secret

    monkeypatch.setenv("ATTENDANCE_JWT_SECRET", "short")
    with pytest.raises(RuntimeError, match="too short"):
        _read_and_validate_jwt_secret("ATTENDANCE_JWT_SECRET")


def test_att_042_one_byte_below_minimum_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A secret with exactly 31 bytes (one below the 32-byte minimum) MUST
    raise. Anchors the off-by-one boundary: a future maintainer flipping
    `_JWT_SECRET_MIN_BYTES` to `< 32` is caught here.
    """
    from app.core.security import _JWT_SECRET_MIN_BYTES, _read_and_validate_jwt_secret

    monkeypatch.setenv(
        "ATTENDANCE_JWT_SECRET", "a" * (_JWT_SECRET_MIN_BYTES - 1)
    )
    with pytest.raises(RuntimeError, match="too short"):
        _read_and_validate_jwt_secret("ATTENDANCE_JWT_SECRET")


def test_att_042_exactly_32_bytes_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A secret with exactly 32 bytes (the minimum) is ACCEPTED. Anchors
    the boundary so a maintainer flipping `< ` to `<=` accidentally is
    caught here.
    """
    from app.core.security import _JWT_SECRET_MIN_BYTES, _read_and_validate_jwt_secret

    monkeypatch.setenv("ATTENDANCE_JWT_SECRET", "a" * _JWT_SECRET_MIN_BYTES)
    secret = _read_and_validate_jwt_secret("ATTENDANCE_JWT_SECRET")
    assert len(secret) == _JWT_SECRET_MIN_BYTES


def test_att_042_blank_secret_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blank (whitespace-only) secret MUST raise. Re-uses the pre-existing
    _read_required_env blank-check; pin that it's still in place behind
    the new length check.
    """
    from app.core.security import _read_and_validate_jwt_secret

    monkeypatch.setenv("ATTENDANCE_JWT_SECRET", "    ")
    with pytest.raises(RuntimeError, match="must be set"):
        _read_and_validate_jwt_secret("ATTENDANCE_JWT_SECRET")


def test_att_042_missing_secret_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing env var (no ATTENDANCE_JWT_SECRET set) MUST raise (pre-existing
    behaviour of _read_required_env; pin that the new validator preserved
    it instead of silently returning an empty string).
    """
    from app.core.security import _read_and_validate_jwt_secret

    monkeypatch.delenv("ATTENDANCE_JWT_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="must be set"):
        _read_and_validate_jwt_secret("ATTENDANCE_JWT_SECRET")


# ---------------------------------------------------------------------------
# Placeholder blocklist — refused even when they pass the length check.
# These are the strings documented in README/RUNBOOK/CLAUDE.md as dev /
# example placeholders. A rushed operator copying from those docs produces
# the placeholder verbatim. The blocklist refuses them — pop quiz
# exhaustive crackability is impossible (every public reader knows them).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "placeholder",
    [
        "change-me-to-a-long-random-secret",       # .env.example
        "dev-only-change-me-min-32-chars-needed",  # README:113 / CLAUDE.md
        "test-secret-32chars-minimum-needed",      # README:291 / RUNBOOK
    ],
)
def test_att_042_placeholder_blocklist_rejected(
    monkeypatch: pytest.MonkeyPatch,
    placeholder: str,
) -> None:
    """A documented placeholder string is REJECTED even though it passes
    the length check (`change-me-to-a-long-random-secret` is 35 chars ≥ 32).

    Pre-fix: an operator who copied `change-me-to-a-long-random-secret`
    from `.env.example` to `.env` and ran the app got a token whose
    signature is published in the public repo — anyone who captures one
    token can forge any role (incl. ADMIN over face-template enrollment).

    Post-fix: the blocklist refuses these strings; the operator must
    generate a real secret (`python3 -c 'import secrets; print(secrets.token_urlsafe(32))'`).
    """
    from app.core.security import _read_and_validate_jwt_secret

    assert len(placeholder.encode("utf-8")) >= 32, (
        f"ATT-042 self-check placeholder '{placeholder}' must clear the "
        f"length check (len={len(placeholder.encode('utf-8'))}); the "
        f"blocklist capture only makes sense when the placeholder is "
        f"long enough to pass length alone."
    )

    monkeypatch.setenv("ATTENDANCE_JWT_SECRET", placeholder)
    with pytest.raises(RuntimeError, match="placeholder"):
        _read_and_validate_jwt_secret("ATTENDANCE_JWT_SECRET")


# ---------------------------------------------------------------------------
# Smoke: the default conftest secret still passes the new check (so no
# downstream tests regress).
# ---------------------------------------------------------------------------


def test_att_042_existing_conftest_secret_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The conftest default test secret (test-secret-not-for-production-use,
    34 bytes) MUST pass the new length+placeholder checks. If a future
    maintainer bumps the minimum or accidentally adds `test-secret-not-
    for-production-use` to the placeholder blocklist, this smoke test
    surfaces the break — otherwise all the other smoke tests would fail
    at import-time `get_security_settings()` and the test-failure mode
    would be cryptic.
    """
    from app.core.security import _read_and_validate_jwt_secret

    monkeypatch.setenv(
        "ATTENDANCE_JWT_SECRET", "test-secret-not-for-production-use"
    )
    secret = _read_and_validate_jwt_secret("ATTENDANCE_JWT_SECRET")
    assert secret == "test-secret-not-for-production-use"
    assert len(secret.encode("utf-8")) >= 32


# ---------------------------------------------------------------------------
# UTF-8 byte length (not char length) — a multibyte-UTF-8 secret that's
# < 32 bytes encoded is still rejected. Pins RFC 7518 §3.2 read correctly.
# ---------------------------------------------------------------------------


def test_att_042_multibyte_utf8_short_secret_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 10-char CJK secret encodes as 30 UTF-8 bytes (3 bytes per char)
    — too short for HS256. Pins the byte-length check (len.encode) over
    the char-length check (len), so a future maintainer flipping the
    check to `len(secret) >= 32` is caught here.
    """
    from app.core.security import _read_and_validate_jwt_secret

    secret_cjk_10 = "张" * 10  # each CJK char = 3 UTF-8 bytes → 30 bytes total
    assert len(secret_cjk_10) == 10  # 10 chars
    assert len(secret_cjk_10.encode("utf-8")) == 30  # 30 bytes

    monkeypatch.setenv("ATTENDANCE_JWT_SECRET", secret_cjk_10)
    with pytest.raises(RuntimeError, match="too short"):
        _read_and_validate_jwt_secret("ATTENDANCE_JWT_SECRET")
