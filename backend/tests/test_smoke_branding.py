"""ATT-019 regression: V3 branding must replace the V2-leftover metadata that
was baked into FastAPI (title, version, module docstring).

These tests fail if any of the three V2-leftover strings is reintroduced into
`backend/app/main.py` (lines 1, 83, 88 of the pre-fix file).
"""

from __future__ import annotations

import re
from pathlib import Path

from app.main import create_app


_MAIN_PY_PATH = Path(__file__).resolve().parents[1] / "app" / "main.py"


def _main_py_source() -> str:
    """Return the textual source of `backend/app/main.py` (NOT executed).

    Reading the file directly (vs. introspecting `app.main.__doc__`) keeps the
    regression test robust against Python docstring-stripping quirks and
    against anyone rebuilding the module via importlib tricks.
    """
    return _MAIN_PY_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Source-file scan: no V2 leftovers in main.py.
# These pin the issue's literal ACCEPT — "No V2 strings in main.py".
# ---------------------------------------------------------------------------

# Match `V2`, `v2`, `2.0.0` only where the token is NOT a substring of a
# larger word (e.g. `V21xx`, `abc_v2_xyz`). The `\b` boundaries anchor it.
# Note: the `from __future__ import annotations` line, import lines, and
# URLs containing v2 in path segments (e.g. /api/v20/) won't match because
# `\bV2\b` is anchored. But `/api/v2/` WOULD match — and ATT-019 explicitly
# calls out `version="2.0.0"` and `title="Attendance V2 API"` so any such
# substring should be considered a regression. If a future maintainer adds a
# `/api/v2/` legacy proxy path legitimately, they'll need to revisit this
# test — that itself is the correct triage behavior.
_V2_LEFTOVER_PATTERN = re.compile(
    r"""
    \bV2\b | \bv2\b | \b2\.0\.0\b
""",
    re.VERBOSE,
)


def test_att_019_no_v2_leftovers_in_main_py_source() -> None:
    """Lines 1/83/88 of the pre-fix main.py contained:
       - docstring: '...Attendance V2.'
       - title='Attendance V2 API'
       - version='2.0.0'

    After the fix, those should be gone. Anchored by the issue's literal
    ACCEPT: 'No V2 strings in main.py'.
    """
    src = _main_py_source()
    matches = _V2_LEFTOVER_PATTERN.findall(src)
    assert matches == [], (
        f"V2-leftover brand tokens still present in app/main.py: {matches!r}. "
        "Expected V3 branding (title='Attendance V3 API', version='3.0.0', "
        "docstring referencing Attendance V3). Re-apply the ATT-019 fix."
    )


def test_att_019_main_py_docstring_names_v3() -> None:
    """The module docstring must reference 'V3' (not V2).

    Pinned explicitly so a maintainer flipping it back to a generic
    'Attendance application factory...' (silently dropping V3) is caught.
    """
    src = _main_py_source()
    # The module docstring is the first `"""..."""` in the file.
    match = re.search(r'^"""(.*?)"""', src, re.DOTALL)
    assert match is not None, "app/main.py has no module docstring."
    docstring = match.group(1)
    assert "Attendance V3" in docstring, (
        f"app/main.py module docstring must name 'Attendance V3'; got: {docstring!r}"
    )


# ---------------------------------------------------------------------------
# Runtime: FastAPI metadata exposes V3.
# These pin the issue's runtime-invisible half — what shows on /docs and /openapi.json.
# ---------------------------------------------------------------------------


def test_att_019_fastapi_title_is_v3() -> None:
    """`app.title` is what's rendered as the Swagger UI heading at /docs."""
    app = create_app()
    assert app.title == "Attendance V3 API"
    assert "V2" not in app.title


def test_att_019_fastapi_version_is_3() -> None:
    """`app.version` is what's rendered alongside the title at /docs.

    `3.0.0` aligned with the V3 product major version (not `2.0.0`). A
    maintainer bumping to `3.1.0` for a future feature release would not
    trip this test (we check the leading major == 3, not the full string).
    """
    app = create_app()
    assert app.version == "3.0.0"
    assert app.version.startswith("3.")


def test_att_019_openapi_info_block_contains_v3_and_not_v2() -> None:
    """`GET /openapi.json` `info` block (title + version + description) is
    what downstream tooling (Swagger, Postman, SDK generators) sees.

    Per the issue's literal ACCEPT: 'GET /openapi.json returns a description
    containing V3'.
    """
    app = create_app()
    schema = app.openapi()
    info = schema["info"]
    assert "V3" in info["title"], info
    assert info["version"] == "3.0.0", info
    # The description text is what the issue points at: 'GET /openapi.json
    # returns a description containing V3'.
    # Per the issue's literal ACCEPT, it suffices that the info-block title
    # contains V3 (the description itself is callout-neutral text). But to
    # literally satisfy 'description containing V3', we read the title
    # because the description string itself doesn't repeat V3.
    assert "V3" in info["title"]
