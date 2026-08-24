"""Regression coverage for the demo-seeder demo-mode gate (ATT-043).

The seeder is a long-running script targeting a Postgres asyncpg DSN; the
smoke suite normally cannot exercise ``seed()`` end-to-end without a real
database. The load-bearing ATT-043 control, however, is a pure Python
``_check_demo_mode_or_exit()`` preflight that refuses to seed unless the
operator has explicitly opted into demo mode via ``ATTENDANCE_DEMO_MODE=1``.

These tests run that preflight in isolation. They are intentionally
*stateless* — they do not import ``app.*`` (so they cannot accidentally
attach to the pytest-asyncio session engine or Redis) and they don't touch
a database. They would fail against the pre-fix seeder because:

  * pre-fix ``seed_demo_data.py`` had no ``_check_demo_mode_or_exit``
    function (this raises ``AttributeError``);
  * pre-fix ``seed_demo_data.py`` hard-coded ``ADMIN_PASSWORD =
    "DemoAdmin1!"`` so ``_new_admin_password()`` did not exist (raises
    ``AttributeError``) and the assertion that two draws are distinct is
    not even reachable.
"""

from __future__ import annotations

import importlib
import os
import sys

import pytest


_BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


def _load_seeder_module(monkeypatch: pytest.MonkeyPatch):
    """Import the seeder fresh so module-level state isn't shared with other tests."""
    importlib.invalidate_caches()
    if "scripts.seed_demo_data" in sys.modules:
        del sys.modules["scripts.seed_demo_data"]
    if "scripts" in sys.modules and hasattr(sys.modules["scripts"], "seed_demo_data"):
        delattr(sys.modules["scripts"], "seed_demo_data")
    return importlib.import_module("scripts.seed_demo_data")


@pytest.fixture()
def seeder(monkeypatch: pytest.MonkeyPatch):
    """Import the seeder fresh, with the DEMO_MODE env reset to unset."""
    monkeypatch.delenv("ATTENDANCE_DEMO_MODE", raising=False)
    return _load_seeder_module(monkeypatch)


# ---------------------------------------------------------------------------
# ATT-043: refuses to seed unless ATTENDANCE_DEMO_MODE == "1"
# ---------------------------------------------------------------------------

def test_seeder_exposes_demo_mode_gate(seeder) -> None:
    """The preflight function must exist (regression anchor for ATT-043)."""
    assert callable(getattr(seeder, "_check_demo_mode_or_exit", None)), (
        "_check_demo_mode_or_exit must exist — ATT-043 demo-mode gate absent."
    )
    assert callable(getattr(seeder, "_new_admin_password", None)), (
        "_new_admin_password must exist — ATT-043 random-password helper absent."
    )


def test_seeder_refuses_when_demo_mode_unset(seeder, monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuses to seed and exits 1 when ATTENDANCE_DEMO_MODE is unset."""
    monkeypatch.delenv("ATTENDANCE_DEMO_MODE", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        seeder._check_demo_mode_or_exit()
    assert excinfo.value.code == 1


def test_seeder_refuses_when_demo_mode_zero(seeder, monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuses to seed and exits 1 when ATTENDANCE_DEMO_MODE is '0'."""
    monkeypatch.setenv("ATTENDANCE_DEMO_MODE", "0")
    with pytest.raises(SystemExit) as excinfo:
        seeder._check_demo_mode_or_exit()
    assert excinfo.value.code == 1


def test_seeder_refuses_when_demo_mode_arbitrary(seeder, monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuses to seed and exits 1 when ATTENDANCE_DEMO_MODE is a non-'1' value."""
    monkeypatch.setenv("ATTENDANCE_DEMO_MODE", "yes")
    with pytest.raises(SystemExit) as excinfo:
        seeder._check_demo_mode_or_exit()
    assert excinfo.value.code == 1


def test_seeder_passes_gate_when_demo_mode_one(seeder, monkeypatch: pytest.MonkeyPatch) -> None:
    """Passes the gate (no SystemExit) when ATTENDANCE_DEMO_MODE == '1'."""
    monkeypatch.setenv("ATTENDANCE_DEMO_MODE", "1")
    seeder._check_demo_mode_or_exit()  # must not raise


# ---------------------------------------------------------------------------
# ATT-043: admin password is random per-run
# ---------------------------------------------------------------------------

def test_admin_password_generator_returns_distinct_values(seeder) -> None:
    """Two draws of the admin password generator must not be identical.

    Pre-fix the password was a constant string ("DemoAdmin1!") so two draws
    would be equal — a backdoor admin with a publicly-known password.
    """
    a = seeder._new_admin_password()
    b = seeder._new_admin_password()
    assert isinstance(a, str) and isinstance(b, str)
    assert a != b, (
        "Per-run random admin password must differ between runs — "
        "ATT-043 random-password fix is not active."
    )


def test_admin_password_generator_is_not_the_published_constant(seeder) -> None:
    """The generated password cannot be the constant string published in the source tree.

    This catches a regression that re-introduces the original literal.
    """
    a = seeder._new_admin_password()
    assert a != "DemoAdmin1!", (
        "Admin password generator returned the source-published constant — "
        "ATT-043 still vulnerable."
    )


def test_admin_password_generator_meets_strength_floor(seeder) -> None:
    """A reasonable length floor: shorter than 16 chars is too easy to brute-force."""
    a = seeder._new_admin_password()
    assert len(a) >= 16, (
        f"Random admin password too short ({len(a)} chars) — fails a sane "
        "password policy and weakens the ATT-043 fix."
    )


# ---------------------------------------------------------------------------
# ATT-043: the module-level ADMIN_PASSWORD constant no longer exists
# ---------------------------------------------------------------------------

def test_no_hardcoded_admin_password_constant(seeder) -> None:
    """The hard-coded ADMIN_PASSWORD constant must be removed entirely.

    Pre-fix the module had ``ADMIN_PASSWORD = "DemoAdmin1!"``. Its presence
    anywhere in the module namespace is a direct regression of ATT-043.
    """
    assert not hasattr(seeder, "ADMIN_PASSWORD"), (
        "Module exposes an ADMIN_PASSWORD constant — ATT-043 hardcoded-password "
        "finding is still present."
    )


# ---------------------------------------------------------------------------
# ATT-010 (Makefile login hint): the seeder's ADMIN_EMAIL is the canonical one
# ---------------------------------------------------------------------------

def test_seeder_admin_email_matches_makefile_claim(seeder) -> None:
    """The seeder's ADMIN_EMAIL must be admin@attendance.demo so the Makefile
    login hint ("Login: admin@attendance.demo") matches the row it creates.

    ATT-010 fixed the Makefile to say admin@attendance.demo (was
    admin@demo.local, which never existed in the DB and also fails
    email-validator's .local rejection policy per RUNBOOK §4.4). This test
    pins the source-of-truth so a future edit to the seeder cannot silently
    drift away from the Makefile's claim.
    """
    assert seeder.ADMIN_EMAIL == "admin@attendance.demo"
