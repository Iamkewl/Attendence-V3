"""ATT-005 regression: pipeline_service facade must re-export only the public
surface of the pipeline subpackage (per ARCHITECTURE.md §2.4 + §4.2).

Pre-fix the facade re-exported 18 underscore-prefixed private helpers via
``# noqa: F401``, advertising an internal API surface as a public contract
and duplicating the symbol list across two files.

Post-fix the facade imports ONLY the symbols in
``backend/app/services/pipeline/__init__.py``'s ``__all__`` (the subpackage's
own public surface). Downstream callers —
``app.worker.tasks``, ``app.api.v1.inference``, ``app.api.v1.students`` —
only use ``process_inference_batch`` and ``extract_enrollment_embedding``,
both still re-exported here.

These tests pin the issue's literal ACCEPT:
  - ``pipeline_service.__all__`` matches ``pipeline/__init__.py.__all__``.
  - No underscore-prefixed symbol imported by the facade.
  - Downstream callers still import and run unchanged.
"""

from __future__ import annotations

from pathlib import Path

import app.services.pipeline as pipeline_pkg
import app.services.pipeline_service as facade


_PIPELINE_INIT_PATH = (
    Path(__file__).resolve().parents[1]
    / "app/services/pipeline/__init__.py"
)
_FACADE_PATH = (
    Path(__file__).resolve().parents[1] / "app/services/pipeline_service.py"
)


# ---------------------------------------------------------------------------
# ACCEPT #1: pipeline_service.__all__ matches pipeline/__init__.py.__all__.
# ---------------------------------------------------------------------------


def test_att_005_facade_all_matches_subpackage_all() -> None:
    """``pipeline_service.__all__`` MUST match ``pipeline/__init__.__all__``
    per the issue's literal ACCEPT.
    """
    facade_all = list(facade.__all__)
    subpackage_all = list(pipeline_pkg.__all__)
    # Sort both for comparison (list ordering is not part of the contract).
    assert sorted(facade_all) == sorted(subpackage_all), (
        f"pipeline_service.__all__ does NOT match pipeline/__init__.__all__.\n"
        f"  facade:    {sorted(facade_all)!r}\n"
        f"  subpackage: {sorted(subpackage_all)!r}"
    )


# ---------------------------------------------------------------------------
# ACCEPT #2: no underscore-prefixed symbol imported by the facade.
# ---------------------------------------------------------------------------


def test_att_005_no_underscore_symbol_imported_by_facade() -> None:
    """The facade must NOT import any ``_underscore``-prefixed symbol.

    Pre-fix it imported 18 such private helpers (see the issue's LOCATION
    list). Post-fix, the facade's import block is just the public symbols
    from ``pipeline/__init__.py``. This test parses the facade's AST and
    inspects actual ``from`` and ``import`` statements — not the docstring
    or comment text (which legitimately discusses the historical private
    re-exports as a thing of the past).
    """
    import ast
    src = _FACADE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    underscore_tokens_in_imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.name
                if name.startswith("_"):
                    underscore_tokens_in_imports.append(name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name.startswith("_"):
                    underscore_tokens_in_imports.append(name)

    assert underscore_tokens_in_imports == [], (
        f"pipeline_service.py imports underscore-prefixed private symbols: "
        f"{underscore_tokens_in_imports!r}. The facade should re-export ONLY "
        f"the public surface of pipeline/__init__.py."
    )


def test_att_005_no_noqa_f401_leftover() -> None:
    """The pre-fix facade used ``# noqa: F401`` markers on every import line
    because every imported symbol was unused (the imports existed only to
    re-export).

    Post-fix the marker may still be present (importing public symbols that
    are unused locally still triggers F401 in ruff), but the pattern of
    noqa-on-private-imports is gone. We assert the count of
    ``# noqa: F401`` markers in actual IMPORT lines of the AST is at most 1
    — the single noqa on the public import line.
    """
    import ast
    import re
    src = _FACADE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    noqa_count = 0
    # Walk the AST to find all ImportFrom nodes; check each one's source
    # segment for the "noqa F401" marker (rendered here in pieces so ruff
    # doesn't parse the comment as a directive).
    for node in ast.walk(tree):
        if isinstance(node, (ast.ImportFrom, ast.Import)):
            # Extract the source segment for this import node.
            segment = ast.get_source_segment(src, node) or ""
            noqa_count += len(re.findall(r"#\s*noqa:\s*F401", segment))
    assert noqa_count <= 1, (
        f"pipeline_service.py has {noqa_count} `# noqa: F401` markers in import "
        f"statements; expected at most 1 (the public import line). Leftover "
        f"markers from the pre-fix private re-exports were not removed."
    )


# ---------------------------------------------------------------------------
# ACCEPT #3: downstream callers still import and run unchanged.
# We verify imports, not runtime, because runtime requires Triton+Postgres.
# ---------------------------------------------------------------------------


def test_att_005_downstream_tasks_imports_process_inference_batch() -> None:
    """backend/app/worker/tasks.py imports process_inference_batch from
    app.services.pipeline_service. Verify the import statement exists and
    that the symbol is actually findable via the facade.
    """
    from_path = Path(__file__).resolve().parents[1] / "app/worker/tasks.py"
    src = from_path.read_text(encoding="utf-8")
    assert "from app.services.pipeline_service import process_inference_batch" in src, (
        "backend/app/worker/tasks.py must still import process_inference_batch "
        "from app.services.pipeline_service (issue ACCEPT #3)."
    )
    # The symbol is findable on the facade object.
    assert hasattr(facade, "process_inference_batch")


def test_att_005_downstream_inference_imports_process_inference_batch() -> None:
    """backend/app/api/v1/inference.py imports process_inference_batch."""
    from_path = Path(__file__).resolve().parents[1] / "app/api/v1/inference.py"
    src = from_path.read_text(encoding="utf-8")
    assert "from app.services.pipeline_service import process_inference_batch" in src, (
        "backend/app/api/v1/inference.py must still import "
        "process_inference_batch from app.services.pipeline_service."
    )
    assert hasattr(facade, "process_inference_batch")


def test_att_005_downstream_students_imports_extract_enrollment_embedding() -> None:
    """backend/app/api/v1/students.py imports extract_enrollment_embedding."""
    from_path = Path(__file__).resolve().parents[1] / "app/api/v1/students.py"
    src = from_path.read_text(encoding="utf-8")
    assert (
        "from app.services.pipeline_service import extract_enrollment_embedding"
        in src
    ), (
        "backend/app/api/v1/students.py must still import "
        "extract_enrollment_embedding from app.services.pipeline_service."
    )
    assert hasattr(facade, "extract_enrollment_embedding")


# ---------------------------------------------------------------------------
# Bonus: pipeline/__init__.py is the source of truth — facade imports all
# listed symbols correctly.
# ---------------------------------------------------------------------------


def test_att_005_facade_exposes_every_symbol_in_subpackage_all() -> None:
    """Each name in ``pipeline/__init__.py.__all__`` must be findable on
    ``pipeline_service`` (the facade re-exports it).
    """
    for name in pipeline_pkg.__all__:
        assert hasattr(facade, name), (
            f"facade does NOT re-export the public symbol {name!r} "
            f"that pipeline/__init__.py declares in __all__."
        )
