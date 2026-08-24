"""Regression tests for ATT-004: scripts/run_webcam_test.py docstring
must not mention the stale "V2" tag.

The project is Attendance V3 and the default endpoint is
`/api/v1/inference/stream` (V3). The previous docstring said
"V2 inference endpoint" -- a stale leftover from a prior generation
which misled readers about which API contract was actually in use.
"""

from __future__ import annotations

import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "run_webcam_test.py"


def test_att_004_run_webcam_test_docstring_does_not_mention_v2() -> None:
    """The docstring of scripts/run_webcam_test.py must not contain
    the stale tag 'V2'.

    ATT-004 ACCEPT: 'Docstring no longer mentions V2.'
    """
    text = SCRIPT_PATH.read_text()
    # We only care about the module docstring (top of file). The script
    # may still reference V3 by name (which is correct); we only forbid
    # the stale 'V2' string.
    # Find the module docstring (first triple-quoted block).
    import re

    m = re.search(r"^\"\"\"(.+?)\"\"\"", text, re.DOTALL | re.MULTILINE)
    assert m is not None, (
        "ATT-004: could not locate module docstring in run_webcam_test.py"
    )
    doc = m.group(1)
    assert "V2" not in doc, (
        f"ATT-004 regression: docstring still mentions 'V2': {doc!r}"
    )


def test_att_004_run_webcam_test_docstring_mentions_attendance_v3_or_stream() -> None:
    """The docstring must positively identify the current endpoint as
    either 'Attendance V3' or '/stream' (the V3 inference contract).
    """
    text = SCRIPT_PATH.read_text()
    import re

    m = re.search(r"^\"\"\"(.+?)\"\"\"", text, re.DOTALL | re.MULTILINE)
    assert m is not None
    doc = m.group(1)
    assert "Attendance V3" in doc or "/stream" in doc or "V3" in doc, (
        f"ATT-004: docstring lacks V3 identification: {doc!r}"
    )
