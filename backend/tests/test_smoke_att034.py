"""Regression tests for ATT-034: do not set Content-Type multipart/form-data
manually on FormData axios POSTs.

When sending a `FormData` body via axios, manually setting
`Content-Type: 'multipart/form-data'` causes axios to forward that header
verbatim (without the `boundary=...` parameter). python-multipart (FastAPI /
Starlette) rejects such requests with "no boundary found" / 422. Letting
axios infer the header means it appends `boundary=...` as required.

These tests guard against reintroducing the explicit Content-Type set on
FormData POSTs in frontend/src/pages/Recognize.jsx and across the React
frontend in general.
"""

from __future__ import annotations

import pathlib
import re


ROOT = pathlib.Path(__file__).resolve().parents[3]
RECOGNIZE_PATH = ROOT / "frontend" / "src" / "pages" / "Recognize.jsx"


def _strip_block_comments(text: str) -> str:
    """Strip /* ... */ block comments so we don't false-match docstrings."""
    return re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)


def _strip_line_comments(text: str) -> str:
    """Strip // ... line comments. Naive but works for our small surface."""
    # Remove content after `//` but skip URLs (http://, https://).
    out_lines = []
    for line in text.splitlines():
        # Cheap URL protection: replace '://' with a placeholder before
        # scanning for '//' comment markers, then restore.
        protected = line.replace("://", "\x00URLCOLON\x00")
        idx = protected.find("//")
        if idx >= 0:
            protected = protected[:idx]
        out_lines.append(protected.replace("\x00URLCOLON\x00", "://"))
    return "\n".join(out_lines)


def test_att_034_recognize_jsx_does_not_set_multipart_formdata_header() -> None:
    """Recognize.jsx must not contain `Content-Type: 'multipart/form-data'`
    on the FormData POST. ATT-034 dropped it.
    """
    text = RECOGNIZE_PATH.read_text()
    code = _strip_line_comments(_strip_block_comments(text))
    # The bad header on the axios call should be gone.
    assert "'Content-Type': 'multipart/form-data'" not in code, (
        "ATT-031 regression: Recognize.jsx re-introduced the explicit "
        "Content-Type:'multipart/form-data' header on a FormData POST."
    )
    assert '"Content-Type": "multipart/form-data"' not in code, (
        "ATT-034 regression: Recognize.jsx re-introduced the explicit "
        'Content-Type:"multipart/form-data" header on a FormData POST.'
    )


def test_att_034_recognize_jsx_lets_axios_infer_content_type() -> None:
    """The Recognize.jsx POST to /api/v1/inference/photo must still pass
    FormData and NOT specify a Content-Type header at all (so axios infers
    multipart/form-data; boundary=...).
    """
    text = RECOGNIZE_PATH.read_text()
    code = _strip_line_comments(_strip_block_comments(text))
    # The axios call should still include the FormData and the timeout.
    assert "RECOGNIZE_ENDPOINT" in code, "RECOGNIZE_ENDPOINT constant missing?"
    assert "client.post(RECOGNIZE_ENDPOINT, formData" in code, (
        "Recognize.jsx must still POST formData to the recognize endpoint"
    )
    assert "timeout: 60000" in code, "60s timeout was dropped by mistake"


def test_att_034_no_other_frontend_formdata_post_manually_sets_content_type() -> None:
    """Repo-wide guard: no other .jsx/.js file should manually set
    `Content-Type: multipart/form-data` either, since this is the same
    foot-gun. ATT-034 specifically called out Recognize.jsx as the offender;
    the /students/{id}/enroll flow and run_webcam_test correctly let the
    form be inferred.
    """
    frontend_src = ROOT / "frontend" / "src"
    if not frontend_src.exists():
        # Test runs in CI where the frontend is checked out.
        return
    offending_files: list[str] = []
    for path in frontend_src.rglob("*.jsx"):
        text = path.read_text()
        code = _strip_line_comments(_strip_block_comments(text))
        if (
            "'Content-Type': 'multipart/form-data'" in code
            or '"Content-Type": "multipart/form-data"' in code
        ):
            offending_files.append(str(path.relative_to(ROOT)))
    # ATT-034 fix removed the offending set; no reintroduction anywhere.
    assert not offending_files, (
        "ATT-034 regression: explicit Content-Type multipart/form-data "
        f"found in: {offending_files}"
    )
