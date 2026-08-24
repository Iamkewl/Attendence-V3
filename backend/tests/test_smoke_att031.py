"""Regression tests for ATT-031: rename `student_user` -> `auditor_user`.

The conftest fixture `student_user` provisioned a User with role=AUDITOR
but a misleading name (there is no STUDENT role in the UserRole enum).
ATT-031 renames it to `auditor_user` so tests honestly reflect the
provisioned role.

These tests guard against reintroducing the misnomer.
"""

from __future__ import annotations

import pathlib
import re


CONFTEST_PATH = pathlib.Path(__file__).with_name("conftest.py")
RBAC_DENIAL_PATH = pathlib.Path(__file__).with_name("test_smoke_rbac_denial.py")


def test_att_031_no_student_user_fixture_in_conftest() -> None:
    """`student_user` must not exist as a fixture name in conftest.py.

    ATT-031 ACCEPT criterion: 'No `student_user` fixture exists'.
    """
    text = CONFTEST_PATH.read_text()
    # The old fixture decorator+def is gone.
    assert not re.search(
        r"@pytest\.fixture\(\)\s*\nasync\s+def\s+student_user\b", text
    ), (
        "ATT-031 regression: `student_user` fixture was reintroduced in conftest.py"
    )
    # Any lingering textual mention of `student_user` should not be a fixture
    # definition or `student_user` parameter.
    assert "async def student_user" not in text, (
        "ATT-031 regression: `async def student_user` reappeared in conftest.py"
    )


def test_att_031_auditor_user_fixture_exists_in_conftest() -> None:
    """`auditor_user` fixture must exist in conftest.py."""
    text = CONFTEST_PATH.read_text()
    assert re.search(
        r"@pytest\.fixture\(\)\s*\nasync\s+def\s+auditor_user\b", text
    ), "ATT-031: `auditor_user` fixture is missing from conftest.py"


def test_att_031_auditor_user_fixture_provisions_auditor_role() -> None:
    """The `auditor_user` fixture body must provision a UserRole.AUDITOR."""
    text = CONFTEST_PATH.read_text()
    # Locate the fixture body and confirm it sets role=UserRole.AUDITOR.
    fixture_match = re.search(
        r"async\s+def\s+auditor_user\b.*?(?=\nasync\s+def\s|\n@pytest|\Z)",
        text,
        re.DOTALL,
    )
    assert fixture_match is not None, "Could not locate `auditor_user` fixture body"
    body = fixture_match.group(0)
    assert "UserRole.AUDITOR" in body, (
        "ATT-031: `auditor_user` fixture does not provision UserRole.AUDITOR"
    )


def test_att_031_no_student_user_references_in_rbac_denial() -> None:
    """test_smoke_rbac_denial.py must no longer reference `student_user`.

    Previously two test functions parameterised `student_user` as the
    AUDITOR caller; ATT-031 renamed those parameters to `auditor_user`.
    """
    text = RBAC_DENIAL_PATH.read_text()
    # No remaining function-arg named student_user.
    assert not re.search(r"^\s*student_user\s*,?\s*$", text, re.MULTILINE), (
        "ATT-031 regression: test_smoke_rbac_denial.py still references "
        "`student_user` as a fixture parameter"
    )
    # No remaining `cookies=auth_cookie(student_user)` call site.
    assert "auth_cookie(student_user)" not in text, (
        "ATT-031 regression: auth_cookie(student_user) call site remains"
    )


def test_att_031_conftest_imports_as_module_namespace() -> None:
    """The conftest.py module source must define `auditor_user` and not
    `student_user`.

    We deliberately inspect the source via `pathlib` rather than importlib-
    reload the module: this repo's conftest.py runs real bootstrap side
    effects (alembic config import, Triton client, event-loop setup) that
    are unsafe to fire a second time inside an unrelated unit test, and
    that depend on `tests._fakes` which is only importable with
    `rootdir=backend/`. The simpler source-text checks above already
    cover fixture presence + AUDITOR provisioning.
    """
    text = CONFTEST_PATH.read_text()
    # Definitive: the fixture-name identifier appears in a def line.
    assert re.search(r"\basync\s+def\s+auditor_user\b", text), (
        "ATT-031: `auditor_user` def missing from conftest.py"
    )
    assert not re.search(r"\basync\s+def\s+student_user\b", text), (
        "ATT-031 regression: `student_user` def reintroduced in conftest.py"
    )
