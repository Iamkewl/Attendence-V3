"""DB-less unit tests for the bulk enrollment importer (slice 1).

Covers manifest parsing, folder scanning, path safety, taxonomy constants,
journal resume semantics, summary/coverage math, pose normalization parity,
the quality-gate relocation identity, and the report-privacy grep — all
without an engine, Triton, or Redis. Integration behavior lives in
``test_enrollment_import_integration.py``.

The module-under-test is ``backend/scripts/import_enrollments.py``, loaded
via the namespace-package import used by ``test_smoke_seed_demo_data.py``
(backend dir on sys.path; fresh import per test to avoid shared state).
"""

from __future__ import annotations

import importlib
import json
import os
import re
import sys

import pytest


_BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


def _load_importer_module():
    """Import the importer fresh so module-level state isn't shared."""
    importlib.invalidate_caches()
    if "scripts.import_enrollments" in sys.modules:
        del sys.modules["scripts.import_enrollments"]
    return importlib.import_module("scripts.import_enrollments")


@pytest.fixture()
def importer():
    return _load_importer_module()


# ---------------------------------------------------------------------------
# Manifest parsing
# ---------------------------------------------------------------------------


def _write_manifest(path, text: str) -> None:
    path.write_bytes(text.encode("utf-8"))


def test_manifest_happy_path(importer, tmp_path) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        "student_number,image_path,pose_label\r\n"
        "2024CS101,a/front.jpg,front\r\n"
        "2024CS102,b/left.JPG,left\r\n",
    )
    rows, issues = importer.parse_manifest(manifest)
    assert issues == []
    assert [(r.student_number, r.image_path, r.pose_label) for r in rows] == [
        ("2024CS101", "a/front.jpg", "front"),
        ("2024CS102", "b/left.JPG", "left"),
    ]
    assert [r.row_index for r in rows] == [2, 3]


def test_manifest_bom_tolerated_and_two_column_header(importer, tmp_path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_bytes(
        b"\xef\xbb\xbfstudent_number,image_path\n2024CS101,front.jpg\n"
    )
    rows, issues = importer.parse_manifest(manifest)
    assert issues == []
    assert rows[0].pose_label == "front"


def test_manifest_blank_lines_skipped(importer, tmp_path) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        "student_number,image_path\n2024CS101,one.jpg\n\n2024CS102,two.jpg\n",
    )
    rows, issues = importer.parse_manifest(manifest)
    assert issues == []
    assert [r.student_number for r in rows] == ["2024CS101", "2024CS102"]


def test_manifest_duplicate_header_is_invalid_row_not_usage_error(importer, tmp_path) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        "student_number,image_path,image_path\n2024CS101,x.jpg,y.jpg\n",
    )
    rows, issues = importer.parse_manifest(manifest)
    assert len(rows) == 1  # DictReader keeps the last duplicated column
    assert len(issues) == 1
    assert issues[0].row_index == 1


def test_manifest_bad_header_is_usage_error(importer, tmp_path) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, "ref,path\n2024CS101,x.jpg\n")
    with pytest.raises(importer.ImportUsageError):
        importer.parse_manifest(manifest)


def test_manifest_empty_file_is_usage_error(importer, tmp_path) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, "")
    with pytest.raises(importer.ImportUsageError):
        importer.parse_manifest(manifest)


@pytest.mark.parametrize(
    ("line", "detail_fragment"),
    [
        (",x.jpg\n", "student_number must not be blank"),
        ("2024CS101,\n", "image_path must not be blank"),
        ("2024CS101\n", "image_path must not be blank"),
        ("2024CS101,x.jpg,this_pose_label_name_is_way_too_long_for_the_column\n",
         "pose_label must be at most 32"),
    ],
)
def test_manifest_malformed_lines_become_invalid_row_issues(
    importer, tmp_path, line, detail_fragment
) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, "student_number,image_path,pose_label\n" + line)
    rows, issues = importer.parse_manifest(manifest)
    assert rows == []
    assert len(issues) == 1
    assert detail_fragment in issues[0].detail


# ---------------------------------------------------------------------------
# Folder scanner
# ---------------------------------------------------------------------------


def test_scan_folder_convention(importer, tmp_path) -> None:
    (tmp_path / "2024CS101.jpg").write_bytes(b"a")
    (tmp_path / "2024CS102__Left.JPG").write_bytes(b"b")  # case-insensitive ext
    # Genuinely unbindable stem: student number longer than String(32).
    # (A bare ".jpeg" is a dotfile in pathlib terms — no suffix — so it lands
    # in the ignored bucket, not the image bucket.)
    (tmp_path / "notes.csv").write_text("ignore me")
    (tmp_path / ".DS_Store").write_bytes(b"x")
    (tmp_path / ".jpeg").write_bytes(b"c")
    long_name = f"{'x' * 33}.png"
    (tmp_path / long_name).write_bytes(b"d")

    rows, unbindable, ignored = importer.scan_folder(tmp_path)

    assert [(r.student_number, r.pose_label, r.image_path) for r in rows] == [
        ("2024CS101", "front", "2024CS101.jpg"),
        ("2024CS102", "left", "2024CS102__Left.JPG"),
    ]
    assert len(unbindable) == 1
    name, ref_hint, detail = unbindable[0]
    assert name == long_name
    assert "at most 32" in detail
    assert ignored == 3  # notes.csv + .DS_Store + dotfile ".jpeg"


# ---------------------------------------------------------------------------
# Path safety (fail closed)
# ---------------------------------------------------------------------------


def test_resolve_path_rejects_traversal_escape(importer, tmp_path) -> None:
    outside = tmp_path.parent / "outside.jpg"
    outside.write_bytes(b"x")
    with pytest.raises(ValueError, match="escapes"):
        importer.resolve_image_path(tmp_path, "../outside.jpg")


def test_resolve_path_rejects_absolute_outside_root(importer, tmp_path) -> None:
    with pytest.raises(ValueError, match="escapes"):
        importer.resolve_image_path(tmp_path, "/etc/passwd")


def test_resolve_path_accepts_relative_and_nested_inside_root(importer, tmp_path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    target = sub / "face.jpg"
    target.write_bytes(b"x")
    resolved = importer.resolve_image_path(tmp_path, "sub/face.jpg")
    assert resolved == target.resolve()


def test_resolve_path_is_lexical_missing_file_left_to_reader(importer, tmp_path) -> None:
    """The resolver checks escape only; missing files surface at read time."""
    resolved = importer.resolve_image_path(tmp_path, "missing.jpg")
    assert not resolved.exists()
    with pytest.raises(FileNotFoundError):
        resolved.read_bytes()


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------


def test_decode_rejects_garbage_bytes(importer) -> None:
    with pytest.raises(ValueError, match="decodable"):
        importer.decode_image_bytes(b"\x00\x01\x02not-an-image")


def test_decode_rejects_tiny_image(importer) -> None:
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (4, 4), color=(255, 255, 255)).save(buf, format="PNG")
    with pytest.raises(ValueError, match="too small"):
        importer.decode_image_bytes(buf.getvalue())


def test_decode_accepts_valid_16x16_png(importer) -> None:
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (16, 16), color=(10, 20, 30)).save(buf, format="PNG")
    tensor = importer.decode_image_bytes(buf.getvalue())
    assert tensor.shape == (16, 16, 3)
    assert tensor.dtype.name == "float32"


# ---------------------------------------------------------------------------
# Taxonomy closed set + normalization parity
# ---------------------------------------------------------------------------


def test_taxonomy_is_exact_closed_set(importer) -> None:
    assert set(importer.REJECT_REASON_CODES) == {
        "UNREADABLE",
        "NO_FACE",
        "LOW_QUALITY",
        "UNKNOWN_REF",
        "INACTIVE_STUDENT",
        "INVALID_ROW",
    }
    # D15: DUPLICATE_IMAGE is a warning marker, never a reject.
    assert importer.REASON_DUPLICATE_IMAGE not in importer.REJECT_REASON_CODES


def test_normalize_parity_with_api_rules(importer) -> None:
    assert importer.normalize_pose_label(None) == "front"
    assert importer.normalize_pose_label("") == "front"
    assert importer.normalize_pose_label("  LEFT ") == "left"
    with pytest.raises(ValueError):
        importer.normalize_pose_label("   ")
    with pytest.raises(ValueError):
        importer.normalize_pose_label("x" * 33)
    with pytest.raises(ValueError):
        importer.normalize_student_number("")
    with pytest.raises(ValueError):
        importer.normalize_student_number("x" * 33)
    assert importer.normalize_student_number(" 2024CS101 ") == "2024CS101"


# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------


def test_journal_round_trip_and_skip_hit(importer, tmp_path) -> None:
    journal = tmp_path / ".import_journal.jsonl"
    records = [
        {"run_id": "r1", "student_number": "S1", "image_sha256": "aa", "pose_label": "front"},
        {"run_id": "r1", "student_number": "S2", "image_sha256": "bb", "pose_label": "left"},
    ]
    with importer.JournalWriter(journal) as writer:
        for record in records:
            writer.append(record)

    keys = importer.load_journal(journal)
    assert ("S1", "aa") in keys
    assert ("S2", "bb") in keys
    assert len(keys) == 2


def test_journal_load_tolerates_corrupt_lines(importer, tmp_path, capsys) -> None:
    journal = tmp_path / ".import_journal.jsonl"
    journal.write_text(
        '{"student_number": "S1", "image_sha256": "aa"}\n'
        "{not json at all}\n"
        '{"no_sha_key": true}\n'
        "\n",
        encoding="utf-8",
    )
    keys = importer.load_journal(journal)
    assert keys == {("S1", "aa")}  # corrupt lines warned + skipped, not fatal


def test_journal_absent_returns_empty(importer, tmp_path) -> None:
    assert importer.load_journal(tmp_path / "does-not-exist.jsonl") == set()


# ---------------------------------------------------------------------------
# Summary math + COVERAGE line format
# ---------------------------------------------------------------------------


def _sample_summary(importer) -> dict:
    return {
        "enrolled_active_templates": 812,
        "roster_active_students": 1000,
        "rejected": {code: 0 for code in importer.REJECT_REASON_CODES},
        "db_errors": 0,
        "extraction_errors": 0,
    }


def test_coverage_line_format_snapshot(importer) -> None:
    summary = _sample_summary(importer)
    summary["rejected"]["LOW_QUALITY"] = 48
    summary["rejected"]["UNREADABLE"] = 5
    assert (
        importer.coverage_line(summary)
        == "COVERAGE: 812 of 1000 enrolled, 48 low-quality, 53 failed"
    )


def test_coverage_line_counts_db_and_extraction_errors_as_failed(importer) -> None:
    summary = _sample_summary(importer)
    summary["db_errors"] = 2
    summary["extraction_errors"] = 1
    assert "3 failed" in importer.coverage_line(summary)


# ---------------------------------------------------------------------------
# Privacy: generated reports carry identifiers/codes only
# ---------------------------------------------------------------------------


_FLOAT_RUN_RE = re.compile(r"(?:-?\d+\.\d+\s*,\s*){100,}-?\d+\.\d+")


def test_reports_never_carry_embeddings_or_long_float_runs(importer, tmp_path) -> None:
    reporter = importer.RejectsReporter(tmp_path, run_id="run-privacy-1")
    reporter.reject(
        student_number="2024CS101",
        image_path="front.jpg",
        pose_label="front",
        reason_code=importer.REASON_LOW_QUALITY,
        detail="quality_score=0.0731 below ATTENDANCE_ENROLLMENT_MIN_QUALITY=0.5000",
        quality_score=0.0731,
    )
    reporter.reject(
        student_number="GHOST",
        image_path="ghost.jpg",
        pose_label="front",
        reason_code=importer.REASON_UNKNOWN_REF,
        detail="no student with this student_number",
    )
    reporter.close()

    reasons_text = (tmp_path / "reasons.csv").read_text(encoding="utf-8")
    assert '"embedding"' not in reasons_text
    assert '"embedding":' not in reasons_text
    assert _FLOAT_RUN_RE.search(reasons_text) is None
    # scalar quality exposure is allowed and present
    assert "0.0731" in reasons_text

    summary = {"ok": True}
    summary_text = json.dumps(summary)
    assert '"embedding":' not in summary_text


def test_sanitize_filename_strips_separators(importer) -> None:
    safe = importer._sanitize_filename("..\\..\\evil:name.jpg")
    assert "\\" not in safe
    assert ":" not in safe
    assert safe.endswith(".jpg")


def test_reject_copy_collision_suffixes(importer, tmp_path) -> None:
    original = tmp_path / "face.jpg"
    original.write_bytes(b"first")
    out_dir = tmp_path / "out"

    reporter = importer.RejectsReporter(out_dir, run_id="r-a")
    try:
        reporter._copy_reject("UNREADABLE", original)
        reporter._copy_reject("UNREADABLE", original)
    finally:
        reporter.close()

    shared = out_dir / "rejects" / "UNREADABLE"
    names = sorted(p.name for p in shared.iterdir())
    assert len(names) == 2
    assert names[1].startswith("face__2")
    assert (shared / names[0]).read_bytes() == b"first"


# ---------------------------------------------------------------------------
# Quality-gate relocation identity (design §8 item 10)
# ---------------------------------------------------------------------------


def test_quality_gate_is_relocated_single_source() -> None:
    from app.api.v1.students import _resolve_enrollment_min_quality as from_api
    from app.services.enrollment_quality import _resolve_enrollment_min_quality as from_services

    assert from_api is from_services


def test_quality_gate_default_and_fail_closed(monkeypatch) -> None:
    from app.services.enrollment_quality import _resolve_enrollment_min_quality

    monkeypatch.delenv("ATTENDANCE_ENROLLMENT_MIN_QUALITY", raising=False)
    assert _resolve_enrollment_min_quality() == 0.5
    monkeypatch.setenv("ATTENDANCE_ENROLLMENT_MIN_QUALITY", "0.75")
    assert _resolve_enrollment_min_quality() == 0.75
    monkeypatch.setenv("ATTENDANCE_ENROLLMENT_MIN_QUALITY", "bogus")
    with pytest.raises(RuntimeError, match="must be a float"):
        _resolve_enrollment_min_quality()


def test_orchestrator_exports_noface_error_through_facade() -> None:
    from app.services import pipeline_service
    from app.services.pipeline.orchestrator import NoFaceDetectedError

    assert pipeline_service.NoFaceDetectedError is NoFaceDetectedError
    assert issubclass(NoFaceDetectedError, ValueError)
