"""ATT-001 regression: `app.worker.__init__` must NOT re-export
`get_triton_client` (pre-fix leftover that contradicts the module's purpose
and ensures a stable import path).

Per the issue's literal ACCEPT: "`grep -r 'from app.worker import
get_triton_client\\|app.worker.get_triton_client' backend/` returns nothing
and no test breakage."

The pre-fix line `from app.infrastructure.triton import get_triton_client` and
the `__all__` entry `"get_triton_client"` advertised the function as part of
the worker package's public surface, but a repo-wide grep confirmed zero
importers (every caller imports directly from `app.infrastructure.triton`).
"""

from __future__ import annotations

from pathlib import Path

import app.worker as worker_pkg


_WORKER_INIT_PATH = Path(__file__).resolve().parents[1] / "app/worker/__init__.py"


# ---------------------------------------------------------------------------
# Source-scan: get_triton_client must not appear anywhere in __init__.py.
# (No import line, no __all__ entry, no leftover docstring mention.)
# ---------------------------------------------------------------------------


def test_att_001_no_get_triton_client_mention_in_worker_init() -> None:
    """`get_triton_client` must not appear anywhere in worker/__init__.py."""
    src = _WORKER_INIT_PATH.read_text(encoding="utf-8")
    assert "get_triton_client" not in src, (
        "app.worker.__init__ still mentions 'get_triton_client'. The dead "
        "re-export must be dropped (per ATT-001)."
    )


# ---------------------------------------------------------------------------
# Object-level: workr_pkg must not export get_triton_client.
# ---------------------------------------------------------------------------


def test_att_001_worker_init_does_not_expose_get_triton_client() -> None:
    """The `app.worker` package object must NOT expose get_triton_client
    (neither as an attribute nor via __all__).
    """
    assert "get_triton_client" not in worker_pkg.__all__, (
        f"worker.__all__ must not advertise 'get_triton_client'. Got: {worker_pkg.__all__!r}"
    )
    assert not hasattr(worker_pkg, "get_triton_client"), (
        "worker package object still has the 'get_triton_client' attribute "
        "(the import must be dropped entirely)."
    )


def test_att_001_worker_init_still_exports_celery() -> None:
    """The legitimate exports (celery_app + get_celery_app) are preserved —
    regression on the fix doesn't remove WHAT THE WORKER PACKAGE IS FOR.
    """
    assert "celery_app" in worker_pkg.__all__
    assert "get_celery_app" in worker_pkg.__all__
    assert hasattr(worker_pkg, "celery_app")
    assert hasattr(worker_pkg, "get_celery_app")


# ---------------------------------------------------------------------------
# Repo-wide grep — the issue's literal ACCEPT wraps the whole repo: no caller
# does `from app.worker import get_triton_client` or `app.worker.get_triton_client`.
# Note:Guest we scan a SUFFICIENT subset of the repo (the backend/app + backend/)
# for sanity; the smoke suite grep is also run as part of CI's lint gate.
# ---------------------------------------------------------------------------


def test_att_001_no_caller_imports_get_triton_client_from_worker() -> None:
    """No caller in backend/ imports get_triton_client via app.worker.

    Walks backend/ for the two patterns the issue's ACCEPT names. If a
    future caller adds one, the test surfaces the regression immediately.
    """
    repo_root = Path(__file__).resolve().parents[2]
    backend_dir = repo_root / "backend"
    import re
    pattern = re.compile(
        r"from\s+app\.worker\s+import\s+.*\bget_triton_client\b"
        r"|"
        r"\bapp\.worker\.get_triton_client\b"
    )
    offending_files: list[str] = []
    for path in backend_dir.rglob("*.py"):
        # Skip the test file itself and any cache.
        if path.suffix != ".py":
            continue
        if "__pycache__" in path.parts:
            continue
        # Skip THIS test file — we legitimately mention 'get_triton_client'
        # as a STRING in this regression test.
        if path.name == "test_smoke_worker_init.py":
            continue
        try:
            src = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if pattern.search(src):
            offending_files.append(str(path.relative_to(repo_root)))
    assert offending_files == [], (
        f"Found imports of get_triton_client via app.worker in: "
        f"{offending_files!r}. Per ATT-001, callers must import directly "
        f"from app.infrastructure.triton instead."
    )
