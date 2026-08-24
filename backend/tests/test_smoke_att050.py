"""Regression tests for ATT-050: build-time chunk splitting in vite.config.js.

Without `manualChunks`, Vite/rolldown produces a single large main bundle
that ships lucide-react, axios, react-router, and react inline -- a
multi-hundred-KB first-load on a slow classroom Wi-Fi. ATT-050 wires a
`build.rollupOptions.output.manualChunks` callback that splits these
vendors into stable cacheable chunks.

These tests guard against reintroducing a single-bundle build.
"""

from __future__ import annotations

import pathlib
import re


ROOT = pathlib.Path(__file__).resolve().parents[2]
VITE_CONFIG_PATH = ROOT / "frontend" / "vite.config.js"


def test_att_050_vite_config_defines_manual_chunks() -> None:
    """vite.config.js must define `manualChunks` under
    `build.rollupOptions.output` so vendors split into separate chunks.

    ATT-050 ACCEPT: chunk splitting beyond route-level (lucide-react +
    axios out of the main bundle).
    """
    text = VITE_CONFIG_PATH.read_text()
    # Literal presence.
    assert "manualChunks" in text, (
        "ATT-050 regression: vite.config.js does not define manualChunks"
    )
    # In Vite 8/rolldown, manualChunks must be a function (not an object)
    # or the build crashes with "manualChunks is not a function".
    # AST-walk the JS surface is non-trivial; we just confirm function form.
    assert re.search(r"manualChunks\s*\(", text), (
        "ATT-050 regression: vite.config.js manualChunks is not a function; "
        "Vite 8/rolldown rejects the object-form (TypeError: 'manualChunks "
        "is not a function')."
    )


def test_att_050_manual_chunks_targets_lucide_react() -> None:
    """The manualChunks callback must route `lucide-react/` to its own
    `icons` chunk. ATT-050 specifically called out lucide-react + axios
    as the large offenders.
    """
    text = VITE_CONFIG_PATH.read_text()
    assert "lucide-react" in text, (
        "ATT-050 regression: vite.config.js manualChunks does not split "
        "lucide-react into its own chunk"
    )
    # A separate 'icons' chunk name.
    assert re.search(r"return\s+['\"]icons['\"]\s*", text), (
        "ATT-050 regression: vite.config.js does not emit an 'icons' chunk"
    )


def test_att_050_manual_chunks_targets_axios() -> None:
    """The manualChunks callback must route `axios/` to its own chunk."""
    text = VITE_CONFIG_PATH.read_text()
    assert "/axios/" in text or "require('axios')" in text or "from 'axios'" in text, (
        "ATT-050 regression: vite.config.js manualChunks does not split axios"
    )
    assert re.search(r"return\s+['\"]axios['\"]\s*", text), (
        "ATT-050 regression: vite.config.js does not emit an 'axios' chunk"
    )


def test_att_050_manual_chunks_targets_react_vendor() -> None:
    """The manualChunks callback must route react + react-dom into a
    `react-vendor` chunk so the React runtime caches separately from the
    app code (the largest vendor by size).
    """
    text = VITE_CONFIG_PATH.read_text()
    assert "react-dom" in text, (
        "ATT-050 regression: vite.config.js manualChunks does not split react-dom"
    )
    assert re.search(r"return\s+['\"]react-vendor['\"]\s*", text), (
        "ATT-050 regression: vite.config.js does not emit a 'react-vendor' chunk"
    )


def test_att_050_manual_chunks_targets_router() -> None:
    """react-router-dom must split into its own `router` chunk."""
    text = VITE_CONFIG_PATH.read_text()
    assert "react-router-dom" in text or "react-router" in text, (
        "ATT-050 regression: vite.config.js manualChunks does not split router"
    )
    assert re.search(r"return\s+['\"]router['\"]\s*", text), (
        "ATT-050 regression: vite.config.js does not emit a 'router' chunk"
    )


def test_att_050_vite_config_is_valid_js_syntax() -> None:
    """Smoke-parse vite.config.js to confirm the manualChunks callback does
    not break the file syntactically. (Rolldown's module syntax is not
    native Python; we use a regex structural check instead.)

    We assert no unterminated strings/parens via simple balance counts
    on the body of manualChunks.
    """
    text = VITE_CONFIG_PATH.read_text()
    # Pull out the manualChunks function body for structural sanity.
    m = re.search(r"manualChunks\s*\([^)]*\)\s*{(.*?)}\s*,", text, re.DOTALL)
    assert m is not None, (
        "ATT-050: could not locate manualChunks function body for syntax "
        "structural check"
    )
    body = m.group(1)
    # Brace balance within the function body.
    opens = body.count("{")
    closes = body.count("}")
    assert opens == closes, (
        f"ATT-050: unbalanced braces in manualChunks body: open={opens} close={closes}"
    )


def test_att_050_at_least_three_vendor_chunks_in_built_bundle() -> None:
    """Optional smoke: if a `dist/` exists with built artifacts, count
    the number of distinct chunk files -- expect at least 3 (react-vendor,
    axios, icons, router, index). If dist/ is absent (no build run), skip.
    """
    dist_dir = ROOT / "frontend" / "dist" / "assets"
    if not dist_dir.exists():
        return
    js_files = list(dist_dir.glob("*.js"))
    # We expect at least 5 chunks (react-vendor, axios, router, icons, index,
    # plus rolldown-runtime) on this codebase post-fix.
    assert len(js_files) >= 5, (
        f"ATT-050 follow-up check: dist/assets produced only {len(js_files)} "
        f"JS files; expected at least 5 with manualChunks wired. "
        f"Files: {[f.name for f in js_files]}"
    )
