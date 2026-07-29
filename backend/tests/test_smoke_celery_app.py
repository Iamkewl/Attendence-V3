"""Celery app smoke tests — covers ATT-030.

ATT-030 (Low, bug): Beat schedule cadence env vars
(`ATTENDANCE_*_INTERVAL_SECONDS`, `ATTENDANCE_DEMO_MODE`) are read at
module-import time inside the @lru_cache-decorated `get_celery_app()`. A
mid-run change to those env vars does NOT update `beat_schedule` — the beat
process must be restarted.

The fix is documentation: a module docstring note + an inline comment at
the `_read_demo_flags`/`demo_sighting_interval_seconds` env read pointing
to RUNBOOK §6. The regression anchor is therefore documentation-shaped:
assert the module docstring mentions the restart caveat, and assert the
accompanying behavioural invariant (`get_celery_app()`'s `@lru_cache`
ignores env-var changes between calls — that's the bug, and the
documentation note is the user-facing mitigation).

Technical note: we read the celery_app.py file source DIRECTLY via
pathlib (not via `import celery_module`) because Celery 5+ installs a
module-level `__getattr__` proxy that intercepts attribute access on
the module and proxies it to the current Celery app. That proxy returns
the Celery instance for `celery_module.get_celery_app` (instead of the
function), and returns the Celery class docstring for
`celery_module.__doc__` (instead of the module's docstring). Reading
the file source as text is proxy-immune.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _celery_app_py_source() -> str:
    """Read the literal text of backend/app/worker/celery_app.py.

    Proxy-immune: imports trigger Celery 5+'s module `__getattr__`
    proxy which intercepts attribute access. Reading the file as text
    bypasses that.
    """
    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / "backend" / "app" / "worker" / "celery_app.py").read_text()


def _celery_app_module_docstring(src: str) -> str:
    """Extract the module docstring from celery_app.py source.

    Returns the text between the first triple-quoted block at the top
    of the file (with any `from __future__ import annotations` line
    skipped if it appears first).
    """
    # Walk past any leading `from __future__ import ...` lines, blank
    # lines, and comments. The module docstring is the first
    # triple-quoted string that appears at module level.
    triple_quote = '"""'
    start = src.find(triple_quote)
    if start < 0:
        return ""
    end = src.find(triple_quote, start + 3)
    if end < 0:
        return ""
    return src[start + 3 : end]


# ---------------------------------------------------------------------------
# ATT-030 — module docstring must call out that env-var changes during a
# running beat don't take effect until restart.
# ---------------------------------------------------------------------------


def test_att_030_module_docstring_documents_restart_requirement() -> None:
    """Module docstring must mention the restart caveat for env-var re-reads.

    Pre-fix the module docstring was a one-liner. Post-fix it lists the
    snapshotted env vars and points to RUNBOOK §6.
    """
    src = _celery_app_py_source()
    doc = _celery_app_module_docstring(src).strip()
    assert doc, "ATT-030: celery_app.py module docstring must exist"
    # Lowercase the doc for case-insensitive keyword scans so a future
    # tweak of the wording doesn't break the assertion as long as the
    # concepts are present.
    doc_lower = doc.lower()
    assert "lru_cache" in doc_lower, (
        "ATT-030: celery_app.py module docstring must mention the "
        "`@lru_cache` snapshot time on get_celery_app() — that's "
        "what makes mid-run env-var changes invisible."
    )
    assert "restart" in doc_lower, (
        "ATT-030: celery_app.py module docstring must say the beat "
        "process must be RESTARTED for cadence env-var changes to "
        "take effect."
    )
    # Plus the specific env-var names that are snapshotted at app init
    # — proves the doc author did the homework, not just a generic note.
    assert "attendance_demo_sighting_interval_seconds" in doc_lower, (
        "ATT-030: celery_app.py module docstring must list "
        "ATTENDANCE_DEMO_SIGHTING_INTERVAL_SECONDS as one of the "
        "snapshotted-at-init env vars (the cited env in ATT-030)."
    )


def test_att_030_inline_comment_at_demo_interval_read_mentions_restart() -> None:
    """The inline comment at the `demo_sighting_interval_seconds` env read
    must call out the restart caveat.

    The module docstring is the high-level note; this inline comment is
    what a reader standing at the cited line 108-115 will see. Pre-fix
    the env read had no comment; post-fix it explains the snapshot
    semantics + RUNBOOK §6 pointer.
    """
    src = _celery_app_py_source()
    # Look for the ATT-030 anchor near the
    # `demo_sighting_interval_seconds = _read_int_env(` call.
    assert "ATT-030" in src, (
        "ATT-030: get_celery_app() must mention ATT-030 in an inline "
        "comment near the demo_sighting_interval_seconds env read."
    )
    assert "restart" in src.lower(), (
        "ATT-030: inline comment must mention 'restart' for env-var "
        "cadence changes to take effect."
    )
    # Specifically, the env-var name we document must appear so the
    # reader can connect the comment to the variable being read.
    assert "ATTENDANCE_DEMO_SIGHTING_INTERVAL_SECONDS" in src, (
        "ATT-030: the docstring/comment block must reference "
        "ATTENDANCE_DEMO_SIGHTING_INTERVAL_SECONDS by name."
    )


# ---------------------------------------------------------------------------
# ATT-030 behavioural anchor — the lru_cache IS what makes mid-run env
# changes invisible. Pin that property: calling get_celery_app() twice with
# a different env var set between calls returns the SAME app (env changes
# ignored after the first call).
#
# Implementation note: `from app.worker.celery_app import get_celery_app` is
# required because Celery 5+'s module `__getattr__` proxy intercepts
# attribute access through the module namespace and returns the current
# Celery app instance for `celery_module.get_celery_app`. Importing the NAME
# directly via `from ... import` snapshots the function reference at import
# time, before the proxy has been installed.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_att_030_lru_cache_snapshots_beat_cadence_env_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_celery_app()'s @lru_cache is what makes ATT-030 a bug: the first
    call snapshot-reads the env vars, the second call returns the cached
    app (env changes ignored). This test pins that property so a future
    attempt to "fix" this with a separate app instance per env-set
    would surface here (and require a follow-up PR + RUNBOOK update).

    The test does NOT depend on actually starting a Celery worker — it
    inspects the cached app's `beat_schedule`. To run, it needs Redis
    configured (the broker URL); conftest already sets ATTENDANCE_REDIS_URL.
    """
    # Import the NAME directly so Celery 5+'s module __getattr__ proxy
    # can't intercept and return the current Celery app instance
    # instead of the function.
    from app.worker.celery_app import get_celery_app

    # Ensure the lru_cache starts clean so the FIRST call (with the
    # initial env) snapshots the schedule at the value we control.
    get_celery_app.cache_clear()

    monkeypatch.setenv("ATTENDANCE_DEMO_MODE", "true")
    monkeypatch.setenv("ATTENDANCE_DEMO_SIGHTING_INTERVAL_SECONDS", "7")

    app_first = get_celery_app()
    schedule_first = app_first.conf.beat_schedule

    # In a DEMO-mode config, the schedule must include the
    # demo-synthetic-sighting entry, with the cadence we set.
    assert "demo-synthetic-sighting" in schedule_first, (
        "ATT-030 (sanity): demo-synthetic-sighting must be present in "
        "beat_schedule when ATTENDANCE_DEMO_MODE=true"
    )

    # Now flip the cadence env var WITHOUT clearing the lru_cache. The
    # cached app's beat_schedule must NOT change — that's the bug
    # ATT-030 documents (and the documentation fix refers the operator to
    # `restart beat` so they get the new value).
    monkeypatch.setenv("ATTENDANCE_DEMO_SIGHTING_INTERVAL_SECONDS", "999")

    app_second = get_celery_app()
    schedule_second = app_second.conf.beat_schedule

    # Same app instance — lru_cache returned the cached value.
    assert app_first is app_second, (
        "ATT-030: get_celery_app()'s @lru_cache must return the same "
        "app instance on the second call (the cache is the bug)."
    )
    # Same schedule (env change ignored).
    assert schedule_first == schedule_second, (
        "ATT-030: a mid-run env-var change must NOT update beat_schedule "
        "without clearing the lru_cache; this is the documented behaviour "
        "(operator must restart beat). If this assertion fails the env "
        "read moved OUTSIDE the cache, and the RUNBOOK §6 note needs "
        "updating too."
    )

    # Belt-and-braces: after clearing the cache and re-import, the new
    # value DOES take effect. This pins the restart mitigation the
    # RUNBOOK note documents — `cache_clear()` is what a beat restart
    # does in practice.
    get_celery_app.cache_clear()
    app_third = get_celery_app()
    schedule_third = app_third.conf.beat_schedule
    assert schedule_first != schedule_third, (
        "ATT-030: after `cache_clear()` (i.e. a beat process restart), "
        "the new env value MUST take effect on the rebuilt beat_schedule. "
        "If this fails the env read happens before the cache even "
        "though the cache is cleared — that would be a different bug "
        "and RUNBOOK §6 would be wrong."
    )
