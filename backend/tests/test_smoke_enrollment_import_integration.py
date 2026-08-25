"""Integration tests for the bulk enrollment importer (live DB + FakeTriton).

Mirrors the idioms of ``test_smoke_inference_caps.py`` / conftest fixtures:
session-scoped event loop, ``_session_factory`` for seeding/assertions, and
``FakeTritonGrpcClient`` injected through ``set_triton_client_override``.

Practical quality-gate note (mirrors the ATT-029 test approach): the fake's
liveness output decodes to a LOW quality (~0.07), so ACCEPT paths run with
``ATTENDANCE_ENROLLMENT_MIN_QUALITY=0.0`` via monkeypatch; LOW_QUALITY paths
raise the threshold instead of patching the extractor. The boundary-equality
test patches the service-level extractor (through the facade module — never a
pipeline submodule) to return an exact scalar.

The script is exercised through ``_run_import_async`` (tests already run
inside a loop; ``main()``'s single-``asyncio.run`` contract is covered by the
sync exit-code tests at the bottom).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid

import numpy as np
import pytest
from tests._fakes import FakeTritonGrpcClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_importer():
    import importlib
    import sys

    backend_dir = os.path.join(os.path.dirname(__file__), "..")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    importlib.invalidate_caches()
    return importlib.import_module("scripts.import_enrollments")


@pytest.fixture()
def importer():
    return _load_importer()


async def _seed_student(session_factory, number: str, *, is_active: bool = True):
    """Insert one user+student pair; returns the Student row id."""
    from app.core.security import hash_password
    from app.domain.models import Student, User, UserRole

    async with session_factory() as session:
        user = User(
            id=uuid.uuid4(),
            email=f"{number.lower()}-owner@test.example",
            full_name=f"Owner of {number}",
            password_hash=hash_password("TestPass1!"),
            role=UserRole.INSTRUCTOR,
            is_active=True,
        )
        session.add(user)
        await session.flush()
        student = Student(
            id=uuid.uuid4(),
            user_id=user.id,
            student_number=number,
            program="Import Testing",
            enrollment_year=2024,
            is_active=is_active,
        )
        session.add(student)
        await session.commit()
        return student.id


def _write_jpeg(directory, name: str, *, color=(90, 140, 200), size=None) -> str:
    """Synthetic photo large enough for the fake's fixed YOLO box to parse."""
    from PIL import Image

    path = directory / name
    Image.new("RGB", size or (640, 480), color=color).save(path, format="JPEG")
    return name


def _parse_args(importer, photos_dir, out_dir, *extra: str):
    return importer.build_arg_parser().parse_args(
        ["--photos-dir", str(photos_dir), "--out", str(out_dir), *extra]
    )


async def _count_embeddings(session_factory) -> int:
    from sqlalchemy import select, func

    from app.domain.models import StudentEmbedding

    async with session_factory() as session:
        return int(
            (await session.execute(select(func.count()).select_from(StudentEmbedding))).scalar_one()
        )


async def _active_rows_for(session_factory, student_id):
    from sqlalchemy import select

    from app.domain.models import StudentEmbedding

    async with session_factory() as session:
        return list(
            (
                await session.execute(
                    select(StudentEmbedding)
                    .where(StudentEmbedding.student_id == student_id)
                    .order_by(StudentEmbedding.created_at)
                )
            ).scalars().all()
        )


async def _audit_rows_for(session_factory, student_id):
    from sqlalchemy import select

    from app.domain.models import TemplateAuditLog

    async with session_factory() as session:
        return list(
            (
                await session.execute(
                    select(TemplateAuditLog)
                    .where(TemplateAuditLog.student_id == student_id)
                    .order_by(TemplateAuditLog.created_at)
                )
            ).scalars().all()
        )


def _read_reasons(out_dir) -> list[dict[str, str]]:
    import csv as _csv

    reasons = out_dir / "reasons.csv"
    if not reasons.exists():
        return []
    with open(reasons, encoding="utf-8", newline="") as fh:
        return list(_csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_folder_mode_writes_active_templates(
    importer, tmp_path, monkeypatch, _session_factory,
    fake_triton,
) -> None:
    monkeypatch.setenv("ATTENDANCE_ENROLLMENT_MIN_QUALITY", "0.0")
    sid_a = await _seed_student(_session_factory, "IMP-A-001")
    sid_b = await _seed_student(_session_factory, "IMP-B-002")

    photos = tmp_path / "photos"
    photos.mkdir()
    _write_jpeg(photos, "IMP-A-001.jpg")
    _write_jpeg(photos, "IMP-B-002__left.jpg", color=(10, 20, 30))
    (photos / "README.txt").write_text("not a photo")

    out = tmp_path / "out"
    args = _parse_args(importer, photos, out, "--template-version", "3")
    summary = await importer._run_import_async(args)

    assert summary["imported"] == 2
    assert summary["rotated"] == 0
    assert summary["skipped_resumed"] == 0
    assert summary["rejected"] == {code: 0 for code in importer.REJECT_REASON_CODES}
    assert summary["ignored_files"] == 1  # README.txt
    assert summary["enrolled_active_templates"] >= 2
    assert summary["template_version"] == 3

    rows_a = await _active_rows_for(_session_factory, sid_a)
    assert len(rows_a) == 1 and rows_a[0].is_active
    assert rows_a[0].pose_label == "front"
    assert rows_a[0].template_version == 3
    assert rows_a[0].quality_score > 0.0

    rows_b = await _active_rows_for(_session_factory, sid_b)
    assert rows_b[0].pose_label == "left"

    audits = await _audit_rows_for(_session_factory, sid_b)
    assert [a.action for a in audits] == ["created"]
    assert audits[0].event_metadata["source"] == "enrollment_import_cli"
    assert audits[0].event_metadata["run_id"] == summary["run_id"]

    journal_lines = (out / ".import_journal.jsonl").read_text().strip().splitlines()
    assert len(journal_lines) == 2
    record = json.loads(journal_lines[0])
    assert set(record) == {
        "run_id",
        "student_number",
        "image_sha256",
        "pose_label",
        "embedding_row_id",
        "template_version",
        "committed_at",
    }


@pytest.mark.asyncio
async def test_manifest_mode_with_pose_column_and_crlf(
    importer, tmp_path, monkeypatch, _session_factory,
    fake_triton,
) -> None:
    monkeypatch.setenv("ATTENDANCE_ENROLLMENT_MIN_QUALITY", "0.0")
    sid = await _seed_student(_session_factory, "IMP-C-003")

    photos = tmp_path / "photos"
    nested = photos / "sub"
    nested.mkdir(parents=True)
    _write_jpeg(nested, "shot.jpg")

    manifest = tmp_path / "manifest.csv"
    manifest.write_bytes(
        b"\xef\xbb\xbfstudent_number,image_path,pose_label\r\nIMP-C-003,sub/shot.jpg,up\r\n"
    )

    args = importer.build_arg_parser().parse_args(
        [
            "--photos-dir", str(photos),
            "--manifest", str(manifest),
            "--out", str(tmp_path / "out"),
        ]
    )
    summary = await importer._run_import_async(args)

    assert summary["imported"] == 1
    rows = await _active_rows_for(_session_factory, sid)
    assert rows[0].pose_label == "up"


# ---------------------------------------------------------------------------
# Idempotent resume + overwrite rotation (D12)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rerun_resumes_from_journal_without_new_triton_calls(
    importer, tmp_path, monkeypatch, _session_factory, fake_triton
) -> None:
    monkeypatch.setenv("ATTENDANCE_ENROLLMENT_MIN_QUALITY", "0.0")
    await _seed_student(_session_factory, "IMP-R-001")
    photos = tmp_path / "photos"
    photos.mkdir()
    _write_jpeg(photos, "IMP-R-001.jpg")
    out = tmp_path / "out"

    first = await importer._run_import_async(_parse_args(importer, photos, out))
    assert first["imported"] == 1
    calls_after_first_run = len(fake_triton.calls)

    second = await importer._run_import_async(_parse_args(importer, photos, out))
    assert second["skipped_resumed"] == 1
    assert second["imported"] == 0
    assert len(fake_triton.calls) == calls_after_first_run  # zero new Triton calls
    assert await _count_embeddings(_session_factory) == 1


@pytest.mark.asyncio
async def test_overwrite_rotates_api_style_with_audit_trail(
    importer, tmp_path, monkeypatch, _session_factory,
    fake_triton,
) -> None:
    monkeypatch.setenv("ATTENDANCE_ENROLLMENT_MIN_QUALITY", "0.0")
    sid = await _seed_student(_session_factory, "IMP-W-001")
    photos = tmp_path / "photos"
    photos.mkdir()
    _write_jpeg(photos, "IMP-W-001.jpg")
    out = tmp_path / "out"

    first = await importer._run_import_async(_parse_args(importer, photos, out))
    assert first["imported"] == 1

    # Fresh --out: no journal, so the active template from run 1 is a real
    # conflict for D12. Same bytes + same journal would be skipped_resumed
    # even under --overwrite (identical input already imported).
    second = await importer._run_import_async(
        _parse_args(importer, photos, tmp_path / "out2", "--overwrite")
    )
    assert second["rotated"] == 1
    assert second["imported"] == 1  # rotated rows count as written

    rows = await _active_rows_for(_session_factory, sid)
    assert len(rows) == 2
    assert sorted(r.is_active for r in rows) == [False, True]

    audits = await _audit_rows_for(_session_factory, sid)
    assert [a.action for a in audits] == ["created", "archived", "created"]
    assert audits[1].event_metadata == {"reason": "bulk_import_rotation"}
    assert audits[2].event_metadata["source"] == "enrollment_import_cli"


@pytest.mark.asyncio
async def test_default_skip_when_active_pose_exists(
    importer, tmp_path, monkeypatch, _session_factory,
    fake_triton,
) -> None:
    monkeypatch.setenv("ATTENDANCE_ENROLLMENT_MIN_QUALITY", "0.0")
    sid = await _seed_student(_session_factory, "IMP-S-001")
    # Pre-existing template enrolled earlier via API/kiosk (no journal entry).
    from app.domain.models import StudentEmbedding

    async with _session_factory() as session:
        session.add(
            StudentEmbedding(
                student_id=sid,
                embedding=np.zeros(512, dtype=np.float32).tolist(),
                pose_label="front",
                quality_score=0.99,
                is_active=True,
            )
        )
        await session.commit()

    photos = tmp_path / "photos"
    photos.mkdir()
    _write_jpeg(photos, "IMP-S-001.jpg")
    summary = await importer._run_import_async(_parse_args(importer, photos, tmp_path / "out"))

    assert summary["skipped_active_exists"] == 1
    assert summary["imported"] == 0
    rows = await _active_rows_for(_session_factory, sid)
    assert len(rows) == 1 and rows[0].quality_score == 0.99  # untouched


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dry_run_performs_zero_db_or_journal_writes(
    importer, tmp_path, monkeypatch, _session_factory,
    fake_triton,
) -> None:
    monkeypatch.setenv("ATTENDANCE_ENROLLMENT_MIN_QUALITY", "0.0")
    await _seed_student(_session_factory, "IMP-D-001")
    photos = tmp_path / "photos"
    photos.mkdir()
    _write_jpeg(photos, "IMP-D-001.jpg")
    out = tmp_path / "out"

    before = await _count_embeddings(_session_factory)
    summary = await importer._run_import_async(
        _parse_args(importer, photos, out, "--dry-run")
    )

    assert summary["dry_run"] is True
    assert summary["would_import"] == 1
    assert summary["imported"] == 0
    assert await _count_embeddings(_session_factory) == before
    assert not (out / ".import_journal.jsonl").exists()
    assert (out / "import_summary.json").exists()  # reports ARE written


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_high_min_quality_rejects_low_quality_and_writes_nothing(
    importer, tmp_path, monkeypatch, _session_factory,
    fake_triton,
) -> None:
    monkeypatch.setenv("ATTENDANCE_ENROLLMENT_MIN_QUALITY", "0.99")
    await _seed_student(_session_factory, "IMP-Q-001")
    photos = tmp_path / "photos"
    photos.mkdir()
    photo_name = _write_jpeg(photos, "IMP-Q-001.jpg")
    out = tmp_path / "out"

    summary = await importer._run_import_async(_parse_args(importer, photos, out))

    assert summary["rejected"]["LOW_QUALITY"] == 1
    assert summary["imported"] == 0
    assert await _count_embeddings(_session_factory) == 0

    reason_rows = _read_reasons(out)
    assert reason_rows[0]["reason_code"] == "LOW_QUALITY"
    copied = out / "rejects" / "LOW_QUALITY" / photo_name
    assert copied.read_bytes() == (photos / photo_name).read_bytes()


@pytest.mark.asyncio
async def test_boundary_equality_passes_strict_less_than_gate(
    importer, tmp_path, monkeypatch, _session_factory,
    fake_triton,
) -> None:
    """quality == min_quality must be accepted (gate is strict <)."""
    monkeypatch.setenv("ATTENDANCE_ENROLLMENT_MIN_QUALITY", "0.5")

    async def _fixed_quality(tensor, *, triton_client=None, require_detection=False):
        embedding = np.ones(512, dtype=np.float32)
        embedding /= np.linalg.norm(embedding)
        return embedding, 0.5

    import app.services.pipeline_service as facade

    monkeypatch.setattr(facade, "extract_enrollment_embedding", _fixed_quality)

    sid = await _seed_student(_session_factory, "IMP-EQ-01")
    photos = tmp_path / "photos"
    photos.mkdir()
    _write_jpeg(photos, "IMP-EQ-01.jpg")
    summary = await importer._run_import_async(_parse_args(importer, photos, tmp_path / "out"))

    assert summary["imported"] == 1
    rows = await _active_rows_for(_session_factory, sid)
    assert float(rows[0].quality_score) == 0.5


# ---------------------------------------------------------------------------
# Detection-presence gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_zero_detections_map_to_no_face_reject(
    importer, tmp_path, monkeypatch, _session_factory
) -> None:
    from app.infrastructure.triton.client import set_triton_client_override

    monkeypatch.setenv("ATTENDANCE_ENROLLMENT_MIN_QUALITY", "0.0")
    await _seed_student(_session_factory, "IMP-N-001")
    photos = tmp_path / "photos"
    photos.mkdir()
    photo_name = _write_jpeg(photos, "IMP-N-001.jpg")
    out = tmp_path / "out"

    zero_face_fake = FakeTritonGrpcClient(zero_detections=True)
    set_triton_client_override(zero_face_fake)
    try:
        summary = await importer._run_import_async(_parse_args(importer, photos, out))
    finally:
        set_triton_client_override(None)

    assert summary["rejected"]["NO_FACE"] == 1
    assert await _count_embeddings(_session_factory) == 0
    assert _read_reasons(out)[0]["reason_code"] == "NO_FACE"
    assert (out / "rejects" / "NO_FACE" / photo_name).exists()


# ---------------------------------------------------------------------------
# Identity resolution: UNKNOWN_REF / INACTIVE_STUDENT (D14)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_ref_and_inactive_student_fail_closed(
    importer, tmp_path, monkeypatch, _session_factory,
    fake_triton,
) -> None:
    monkeypatch.setenv("ATTENDANCE_ENROLLMENT_MIN_QUALITY", "0.0")
    await _seed_student(_session_factory, "IMP-GONE-1", is_active=False)
    photos = tmp_path / "photos"
    photos.mkdir()
    _write_jpeg(photos, "NO-SUCH-STUDENT.jpg")
    _write_jpeg(photos, "IMP-GONE-1.jpg")
    out = tmp_path / "out"

    summary = await importer._run_import_async(_parse_args(importer, photos, out))

    assert summary["rejected"]["UNKNOWN_REF"] == 1
    assert summary["rejected"]["INACTIVE_STUDENT"] == 1
    assert summary["imported"] == 0
    codes = {row["student_number"]: row["reason_code"] for row in _read_reasons(out)}
    assert codes["NO-SUCH-STUDENT"] == "UNKNOWN_REF"
    assert codes["IMP-GONE-1"] == "INACTIVE_STUDENT"
    assert await _count_embeddings(_session_factory) == 0


# ---------------------------------------------------------------------------
# Unreadable input
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_garbage_image_bytes_rejected_unreadable(
    importer, tmp_path, monkeypatch, _session_factory,
    fake_triton,
) -> None:
    monkeypatch.setenv("ATTENDANCE_ENROLLMENT_MIN_QUALITY", "0.0")
    sid = await _seed_student(_session_factory, "IMP-U-001")
    photos = tmp_path / "photos"
    photos.mkdir()
    garbage = b"\x00\xff\xfe definitely not jpeg payload"
    (photos / "IMP-U-001.jpg").write_bytes(garbage)
    out = tmp_path / "out"

    summary = await importer._run_import_async(_parse_args(importer, photos, out))

    assert summary["rejected"]["UNREADABLE"] == 1
    assert summary["imported"] == 0
    copied = out / "rejects" / "UNREADABLE" / "IMP-U-001.jpg"
    assert copied.read_bytes() == garbage  # byte-identical copy for triage
    assert await _count_embeddings(_session_factory) == 0
    assert await _audit_rows_for(_session_factory, sid) == []


# ---------------------------------------------------------------------------
# D15 duplicate-image warn counter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_identical_image_two_refs_imports_both_and_warns(
    importer, tmp_path, monkeypatch, _session_factory,
    fake_triton,
) -> None:
    monkeypatch.setenv("ATTENDANCE_ENROLLMENT_MIN_QUALITY", "0.0")
    await _seed_student(_session_factory, "IMP-P-001")
    await _seed_student(_session_factory, "IMP-P-002")
    photos = tmp_path / "photos"
    photos.mkdir()
    photo_name = _write_jpeg(photos, "IMP-P-001.jpg")
    (photos / "IMP-P-002.jpg").write_bytes((photos / photo_name).read_bytes())
    out = tmp_path / "out"

    summary = await importer._run_import_async(_parse_args(importer, photos, out))

    assert summary["imported"] == 2  # D15: later duplicates still import
    assert summary["duplicate_image_warnings"] == 1
    assert summary["rejected"].get(importer.REASON_DUPLICATE_IMAGE, 0) == 0
    assert await _count_embeddings(_session_factory) == 2
    # Not treated as a reject: no rejects/DUPLICATE_IMAGE tree exists.
    assert not (out / "rejects" / "DUPLICATE_IMAGE").exists()


# ---------------------------------------------------------------------------
# Infrastructure abort (exit 3 semantics)
# ---------------------------------------------------------------------------


class _FailsAfterCallsFake(FakeTritonGrpcClient):
    """Fake that raises TritonServerUnavailableError after N infer calls."""

    def __init__(self, fail_after: int) -> None:
        super().__init__()
        self._fail_after = fail_after
        self._infer_count = 0

    async def infer_fp32_async(self, **kwargs):
        self._infer_count += 1
        if self._infer_count > self._fail_after:
            from app.infrastructure.triton import TritonServerUnavailableError

            raise TritonServerUnavailableError("simulated outage")
        return await super().infer_fp32_async(**kwargs)


@pytest.mark.asyncio
async def test_triton_outage_aborts_run_but_keeps_committed_rows(
    importer, tmp_path, monkeypatch, _session_factory
) -> None:
    from app.infrastructure.triton.client import set_triton_client_override

    monkeypatch.setenv("ATTENDANCE_ENROLLMENT_MIN_QUALITY", "0.0")
    # Shrink retry backoff so the abort path doesn't sleep 2s+8s.
    monkeypatch.setattr(importer, "TRITON_RETRY_BACKOFF_SECONDS", (0.01, 0.01))

    await _seed_student(_session_factory, "IMP-X-001")
    await _seed_student(_session_factory, "IMP-X-002")
    photos = tmp_path / "photos"
    photos.mkdir()
    _write_jpeg(photos, "IMP-X-001.jpg")
    _write_jpeg(photos, "IMP-X-002.jpg")
    out = tmp_path / "out"

    flaky = _FailsAfterCallsFake(fail_after=2)  # row 1 succeeds (2 calls), row 2 dies
    set_triton_client_override(flaky)
    try:
        with pytest.raises(importer.ImportInfraError):
            await importer._run_import_async(_parse_args(importer, photos, out))
    finally:
        set_triton_client_override(None)

    assert await _count_embeddings(_session_factory) == 1  # row 1 stays committed
    journal_lines = (out / ".import_journal.jsonl").read_text().strip().splitlines()
    assert len(journal_lines) == 1
    assert json.loads(journal_lines[0])["student_number"] == "IMP-X-001"


# ---------------------------------------------------------------------------
# Privacy scan over a mixed-run report tree
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_report_tree_carries_no_embeddings_anywhere(
    importer, tmp_path, monkeypatch, _session_factory,
    fake_triton,
) -> None:
    import re

    monkeypatch.setenv("ATTENDANCE_ENROLLMENT_MIN_QUALITY", "0.99")  # force rejects too
    await _seed_student(_session_factory, "IMP-V-001")
    await _seed_student(_session_factory, "GHOST-V-2")
    photos = tmp_path / "photos"
    photos.mkdir()
    _write_jpeg(photos, "IMP-V-001.jpg")
    _write_jpeg(photos, "GHOST-V-2.jpg")
    out = tmp_path / "out"

    await importer._run_import_async(_parse_args(importer, photos, out))

    text_keys = [".csv", ".json", ".jsonl"]
    float_run = re.compile(r"(?:-?\d+\.\d+(?:e[-+]\d+)?\s*,\s*){100,}-?\d+\.\d+")
    for path in out.rglob("*"):
        if path.is_file() and path.suffix in text_keys:
            content = path.read_text(encoding="utf-8")
            assert '"embedding":' not in content, f"embedding key leaked into {path}"
            assert float_run.search(content) is None, f"512-float run leaked into {path}"


# ---------------------------------------------------------------------------
# CLI exit codes through main() — subprocess-isolated
# ---------------------------------------------------------------------------


def _run_cli(*cli_args: str, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Run the script as a real subprocess from the backend directory.

    Subprocess isolation is deliberate: main() owns the process's single
    asyncio.run(), and calling it in-process would unset the current event
    loop (asyncio.run does that on exit) and poison every later async test
    in the session. The suite never calls asyncio.run() in-process.
    """
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return subprocess.run(
        [sys.executable, "-m", "scripts.import_enrollments", *cli_args],
        cwd=backend_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_cli_exit_zero_and_coverage_line(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ATTENDANCE_ENROLLMENT_MIN_QUALITY", "0.0")
    photos = tmp_path / "photos"
    photos.mkdir()

    result = _run_cli(
        "--photos-dir", str(photos),
        "--out", str(tmp_path / "out"),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("COVERAGE: ")
    assert "low-quality" in result.stdout
    assert "failed" in result.stdout
    assert "rows: total=0 imported=0" in result.stdout


def test_cli_exit_two_on_missing_photos_dir(tmp_path) -> None:
    result = _run_cli(
        "--photos-dir", str(tmp_path / "does-not-exist"),
        "--out", str(tmp_path / "out"),
    )
    assert result.returncode == 2
    assert "USAGE ERROR" in result.stderr


def test_cli_exit_two_on_bad_template_version(tmp_path) -> None:
    photos = tmp_path / "photos"
    photos.mkdir()
    result = _run_cli(
        "--photos-dir", str(photos),
        "--out", str(tmp_path / "out"),
        "--template-version", "0",
    )
    assert result.returncode == 2
    assert "positive integer" in result.stderr


def test_cli_exit_three_on_unreachable_database(tmp_path, monkeypatch) -> None:
    """DB down ≡ infrastructure abort: exit 3, no partial-report lies."""
    monkeypatch.setenv("ATTENDANCE_ENROLLMENT_MIN_QUALITY", "0.0")
    photos = tmp_path / "photos"
    photos.mkdir()
    result = _run_cli(
        "--photos-dir", str(photos),
        "--out", str(tmp_path / "out"),
        extra_env={
            "ATTENDANCE_DATABASE_URL": "postgresql://attendance:attendance@localhost:15499/nope"
        },
    )
    assert result.returncode == 3, result.stderr
