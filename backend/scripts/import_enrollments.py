"""Bulk enrollment importer — offline batch folder/CSV ingest (ROADMAP §2 slice 1).

Reads photos bound to students by ``student_number`` (decision D11), runs each
through detect→embed→quality-gate, and writes active pose templates honoring the
one-active-template-per-pose invariant (decision D12: skip when an active
template already exists unless ``--overwrite``, which rotates API-style with
TemplateAuditLog rows). Stamps ``template_version`` (D13, migration
20260824_0005) on every row it writes.

CLI (run from ``backend/`` — same layout as ``scripts/seed_demo_data.py``)::

    python -m scripts.import_enrollments --photos-dir ./intake \\
        [--manifest ./intake/manifest.csv] [--out ./import_out] \\
        [--dry-run] [--overwrite] [--template-version 1]

Input modes
    --manifest   CSV, utf-8-sig, header ``student_number,image_path[,pose_label]``;
                 image_path relative to --photos-dir (absolute accepted only when
                 it resolves under --photos-dir; traversal/symlink escape ⇒ reject).
    folder       ``<student_number>.jpg|.jpeg|.png`` (case-insensitive ext) or
                 ``<student_number>__<pose>.jpg`` (double underscore is the pose
                 separator because real student numbers contain single underscores
                 and dashes). Other image-extension files are unbindable ⇒
                 UNKNOWN_REF reject; non-image files are ignored (counted).
    Both modes funnel into the same internal row type; folder mode is just a
    generated manifest. Pose labels follow API normalization rules (strip,
    lowercase, non-blank, ≤32); the default pose is ``front``.

Exit codes
    0  run completed (rejects are data problems reported, not run failures)
    2  usage/config error (bad args, missing dirs, bad manifest header,
       malformed ATTENDANCE_ENROLLMENT_MIN_QUALITY — fail closed)
    3  infrastructure abort (Triton repeatedly failing after in-process
       retries, DB down). Committed rows stay committed; rerun resumes.

Reason-code taxonomy (closed set; snake-free uppercase for dashboard parity)
    UNREADABLE        not decodable as an image, <8px, missing/unreadable file,
                      or path escapes --photos-dir (fail closed)
    NO_FACE           zero YOLO detections with require_detection=True
    LOW_QUALITY       quality_score < ATTENDANCE_ENROLLMENT_MIN_QUALITY (strict,
                      same boundary as the API gate; equality passes)
    DUPLICATE_IMAGE   identical image bytes bound to multiple student_numbers:
                      later ones STILL IMPORT plus a warn counter (D15) — this
                      is a warning, never a reject; it never appears in rejects/
    UNKNOWN_REF       student_number absent from the DB (fail closed)
    INACTIVE_STUDENT  student_number resolves but Student.is_active=False
                      (D14: dedicated code so dashboards distinguish bad-roster
                      from photo problems)
    INVALID_ROW       manifest-line malformation (missing/blank/oversized
                      fields, bad pose label, duplicate header columns)

Idempotency / resume
    ``<out>/.import_journal.jsonl`` — one JSON object per committed row keyed by
    ``(student_number, sha256(image bytes))``. Re-runs skip journaled rows
    entirely (no Triton call). Journal is honored even under ``--overwrite``
    (identical bytes already imported ⇒ nothing to rotate). Rows whose DB write
    failed are NOT journaled so reruns retry them. One commit per row: a crash
    between commit and journal append re-imports that row once more, idempotently.

Quality gate provenance (the "choose one" decision required by slice spec)
    RELOCATED, not replicated: ``_resolve_enrollment_min_quality`` moved verbatim
    into ``app/services/enrollment_quality.py`` (single fail-closed source of
    truth) and is re-exported from ``app.api.v1.students`` for compatibility.
    This script imports it from the services module directly. No local copy.

Event-loop discipline (attendance-v3-context rules)
    ``main()`` calls ``asyncio.run()`` exactly ONCE over
    ``_run_import_async``; the asyncpg engine + session factory are built
    INSIDE that coroutine (loop-bound lifetime confined to this loop) and
    disposed in ``finally`` — the same shape as the Celery tasks'
    ``dispose_engine`` rule in ``app/worker/celery_app.py``. No Redis anywhere;
    no FastAPI-loop singletons touched. App/pipeline imports happen lazily
    inside functions so the module stays cheap to import for pure unit tests
    (same pattern as ``scripts/seed_demo_data.py``).

Privacy hard invariant (biometric data of possibly-minors)
    Embeddings are NEVER logged, printed, serialized, or written by this
    script — including verbose logs, tracebacks, and repr()s of ORM objects
    (only ids are logged). Reports carry identifiers (student_number, UUIDs),
    reason codes, scalar quality_score (same exposure as the API INFO line),
    timestamps, and paths — nothing else. Audit metadata is limited to
    ``{"source": "enrollment_import_cli", "run_id": ...}`` /
    ``{"reason": "bulk_import_rotation"}``. Consent plumbing is deferred to
    the consent sprint (D16) and deliberately absent here.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import shutil
import sys
import traceback
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Constants / taxonomy
# ---------------------------------------------------------------------------

STUDENT_NUMBER_MAX_LEN = 32  # matches Student.student_number String(32)
POSE_LABEL_MAX_LEN = 32  # matches API pose-label rules
DEFAULT_POSE_LABEL = "front"
DEFAULT_TEMPLATE_VERSION = 1
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

REASON_UNREADABLE = "UNREADABLE"
REASON_NO_FACE = "NO_FACE"
REASON_LOW_QUALITY = "LOW_QUALITY"
REASON_UNKNOWN_REF = "UNKNOWN_REF"
REASON_INACTIVE_STUDENT = "INACTIVE_STUDENT"
REASON_INVALID_ROW = "INVALID_ROW"

REJECT_REASON_CODES = (
    REASON_UNREADABLE,
    REASON_NO_FACE,
    REASON_LOW_QUALITY,
    REASON_UNKNOWN_REF,
    REASON_INACTIVE_STUDENT,
    REASON_INVALID_ROW,
)

# Warning-only marker (D15): rows flagged with this still import.
REASON_DUPLICATE_IMAGE = "DUPLICATE_IMAGE"

JOURNAL_FILENAME = ".import_journal.jsonl"
SUMMARY_FILENAME = "import_summary.json"
REJECTS_CSV_FILENAME = "reasons.csv"
REJECTS_DIRNAME = "rejects"

TRITON_MAX_RETRIES = 2  # retries AFTER the initial attempt
TRITON_RETRY_BACKOFF_SECONDS = (2.0, 8.0)


class ImportUsageError(Exception):
    """Bad arguments/configuration — maps to exit code 2."""


class ImportInfraError(Exception):
    """Infrastructure failure mid-run — maps to exit code 3."""


# ---------------------------------------------------------------------------
# Pure helpers (DB-less, unit-testable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImportRow:
    """One bindable input row; identical shape for manifest and folder modes."""

    row_index: int
    student_number: str
    image_path: str  # as written in the manifest / folder listing
    pose_label: str


@dataclass(frozen=True)
class ParseIssue:
    """An input-authoring error found before any I/O on the row itself."""

    row_index: int | None
    student_number: str
    image_path: str
    pose_label: str
    detail: str


def normalize_student_number(raw: str | None) -> str:
    """Trim + validate a student number; ValueError on unusable input."""
    ref = (raw or "").strip()
    if not ref:
        raise ValueError("student_number must not be blank.")
    if len(ref) > STUDENT_NUMBER_MAX_LEN:
        raise ValueError(
            f"student_number must be at most {STUDENT_NUMBER_MAX_LEN} characters long."
        )
    return ref


def normalize_pose_label(raw: str | None) -> str:
    """API-equivalent pose normalization (strip/lower/default front/≤32)."""
    pose_label = (raw or DEFAULT_POSE_LABEL).strip().lower()
    if not pose_label:
        raise ValueError("pose_label must not be blank.")
    if len(pose_label) > POSE_LABEL_MAX_LEN:
        raise ValueError(f"pose_label must be at most {POSE_LABEL_MAX_LEN} characters long.")
    return pose_label


def parse_manifest(
    manifest_path: Path,
) -> tuple[list[ImportRow], list[ParseIssue]]:
    """Parse a manifest CSV into rows + input-authoring issues.

    Header must be exactly ``student_number,image_path[,pose_label]``
    (utf-8-sig tolerates a BOM; CRLF tolerated). A malformed header is a
    usage/config error (exit 2), not a row outcome. Blank lines are skipped;
    duplicate header column names are recorded as an INVALID_ROW issue.
    """
    with open(manifest_path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = [(name or "").strip() for name in (reader.fieldnames or [])]
        if not any(fieldnames):
            raise ImportUsageError(f"Manifest {manifest_path} has no header row.")

        known_columns = {"student_number", "image_path", "pose_label"}
        if (
            not set(fieldnames) <= known_columns
            or "student_number" not in fieldnames
            or "image_path" not in fieldnames
        ):
            raise ImportUsageError(
                f"Manifest header must be 'student_number,image_path[,pose_label]'; "
                f"got {fieldnames!r}."
            )

        issues: list[ParseIssue] = []
        rows: list[ImportRow] = []
        if len(fieldnames) != len(set(fieldnames)):
            issues.append(
                ParseIssue(
                    row_index=1,
                    student_number="",
                    image_path="",
                    pose_label="",
                    detail=f"duplicate header column names: {reader.fieldnames!r}",
                )
            )

        for lineno, raw_row in enumerate(reader, start=2):
            values = {
                (key or "").strip(): (value.strip() if isinstance(value, str) else "")
                for key, value in raw_row.items()
            }
            student_raw = values.get("student_number") or ""
            image_raw = values.get("image_path") or ""
            pose_raw = values.get("pose_label") or ""

            if not student_raw and not image_raw and not pose_raw:
                continue  # blank line (all-empty fields)

            try:
                student_number = normalize_student_number(student_raw)
            except ValueError as exc:
                issues.append(ParseIssue(lineno, student_raw, image_raw, pose_raw, str(exc)))
                continue
            if not image_raw:
                issues.append(ParseIssue(lineno, student_number, "", pose_raw, "image_path must not be blank."))
                continue
            try:
                pose_label = normalize_pose_label(pose_raw)
            except ValueError as exc:
                issues.append(ParseIssue(lineno, student_number, image_raw, pose_raw, str(exc)))
                continue

            rows.append(ImportRow(lineno, student_number, image_raw, pose_label))

    return rows, issues


def scan_folder(
    photos_dir: Path,
) -> tuple[list[ImportRow], list[tuple[str, str, str]], int]:
    """Folder-convention scan (no manifest).

    Returns ``(rows, unbindable, ignored_files)`` where ``unbindable`` holds
    ``(filename, student_number_field, detail)`` tuples for image-extension
    files whose stem does not parse as ``<ref>[__<pose>]`` — these become
    UNKNOWN_REF rejects (with the file copied for re-shoot triage). Files with
    a pose/ref part violating length rules are likewise unbindable here:
    a filename we cannot parse cannot be bound to any roster entry.
    Non-image files are ignored silently and counted.
    """
    rows: list[ImportRow] = []
    unbindable: list[tuple[str, str, str]] = []
    ignored_files = 0

    for entry in sorted(photos_dir.iterdir()):
        if not entry.is_file():
            continue
        extension = entry.suffix.lower()
        if extension not in VALID_IMAGE_EXTENSIONS:
            ignored_files += 1
            continue

        stem = entry.stem
        if "__" in stem:
            ref_raw, _, pose_raw = stem.partition("__")
        else:
            ref_raw, pose_raw = stem, DEFAULT_POSE_LABEL

        try:
            student_number = normalize_student_number(ref_raw)
            pose_label = normalize_pose_label(pose_raw)
        except ValueError as exc:
            unbindable.append((entry.name, ref_raw.strip(), str(exc)))
            continue

        rows.append(ImportRow(len(rows) + 1, student_number, entry.name, pose_label))

    return rows, unbindable, ignored_files


def resolve_image_path(photos_dir: Path, image_path: str) -> Path:
    """Resolve a manifest/folder image path strictly inside ``photos_dir``.

    Raises FileNotFoundError (missing) or ValueError (absolute outside root,
    traversal/symlink escape — fail closed). Callers map both to UNREADABLE.
    """
    candidate = Path(image_path)
    if candidate.is_absolute():
        resolved_root = photos_dir.resolve()
        resolved = candidate.resolve()
        if resolved != resolved_root and resolved_root not in resolved.parents:
            raise ValueError(f"path escapes photos directory: {image_path}")
        return resolved

    resolved_root = photos_dir.resolve()
    resolved = (photos_dir / candidate).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"path escapes photos directory: {image_path}")
    return resolved


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode image bytes to a float32 HWC RGB tensor (≥8×8) for extraction.

    Mirrors the API's ``_decode_enrollment_image`` checks; raises ValueError
    on anything undecodable or too small.
    """
    from io import BytesIO

    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(BytesIO(data)) as image:
            rgb_image = image.convert("RGB")
            tensor = np.asarray(rgb_image, dtype=np.float32)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"not a decodable image: {exc}") from exc

    if tensor.ndim != 3 or tensor.shape[0] < 8 or tensor.shape[1] < 8:
        raise ValueError("image dimensions are too small for enrollment.")

    return np.ascontiguousarray(tensor, dtype=np.float32)


def sha256_bytes(data: bytes) -> str:
    """Hex digest of raw image bytes (journal/duplicate-image key)."""
    return hashlib.sha256(data).hexdigest()


def load_journal(path: Path) -> set[tuple[str, str]]:
    """Load journal keys ``(student_number, image_sha256)``; tolerate corruption."""
    keys: set[tuple[str, str]] = set()
    if not path.exists():
        return keys
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                keys.add((str(record["student_number"]), str(record["image_sha256"])))
            except (json.JSONDecodeError, KeyError, TypeError):
                print(
                    f"WARNING: ignoring corrupt journal line {lineno} in {path}",
                    file=sys.stderr,
                )
    return keys


class JournalWriter:
    """Append-only JSONL journal with flush+fsync after every record."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fh = open(path, "a", encoding="utf-8")

    def append(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, sort_keys=True) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JournalWriter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _sanitize_filename(name: str) -> str:
    """Filename-safe form for reject copies (keeps identifiers readable)."""
    cleaned = "".join(char if char.isalnum() or char in {".", "_", "-"} else "_" for char in name)
    return cleaned[:120] or "unnamed"


class RejectsReporter:
    """Writes reasons.csv and copies rejected originals under rejects/<CODE>/."""

    HEADER = [
        "run_id",
        "student_number",
        "image_path",
        "pose_label",
        "reason_code",
        "detail",
        "quality_score",
        "rejected_at",
    ]

    def __init__(self, out_dir: Path, run_id: str) -> None:
        self._run_id = run_id
        self.rejects_dir = out_dir / REJECTS_DIRNAME
        self.rejects_dir.mkdir(parents=True, exist_ok=True)
        self.reasons_path = out_dir / REJECTS_CSV_FILENAME
        self._fh = open(self.reasons_path, "a", encoding="utf-8", newline="")
        self._writer = csv.writer(self._fh)
        if self.reasons_path.stat().st_size == 0:
            self._writer.writerow(self.HEADER)
            self._fh.flush()

    def reject(
        self,
        *,
        student_number: str,
        image_path: str,
        pose_label: str,
        reason_code: str,
        detail: str,
        quality_score: float | None = None,
        source_path: Path | None = None,
    ) -> None:
        """Record one rejection; copy the original image when we have one."""
        if source_path is not None:
            self._copy_reject(reason_code, source_path)
        self._writer.writerow(
            [
                self._run_id,
                student_number,
                image_path,
                pose_label,
                reason_code,
                detail,
                "" if quality_score is None else f"{float(quality_score):.4f}",
                datetime.now(tz=UTC).isoformat(),
            ]
        )
        self._fh.flush()

    def _copy_reject(self, reason_code: str, source_path: Path) -> None:
        target_dir = self.rejects_dir / reason_code
        target_dir.mkdir(parents=True, exist_ok=True)
        base = _sanitize_filename(source_path.name)
        target = target_dir / base
        suffix = 2
        while target.exists():
            target = target_dir / f"{source_path.stem}__{suffix}{source_path.suffix}"
            suffix += 1
        try:
            shutil.copyfile(source_path, target)
        except OSError as exc:
            print(
                f"WARNING: could not copy reject original {source_path.name}: {exc}",
                file=sys.stderr,
            )

    def close(self) -> None:
        self._fh.close()


def coverage_line(summary: dict[str, Any]) -> str:
    """The end-of-run coverage line: 'N of M enrolled, K low-quality, J failed'."""
    enrolled = int(summary["enrolled_active_templates"])
    roster = int(summary["roster_active_students"])
    low_quality = int(summary["rejected"].get(REASON_LOW_QUALITY, 0))
    failed = sum(int(count) for count in summary["rejected"].values())
    failed += int(summary.get("db_errors", 0)) + int(summary.get("extraction_errors", 0))
    return f"COVERAGE: {enrolled} of {roster} enrolled, {low_quality} low-quality, {failed} failed"


# ---------------------------------------------------------------------------
# Async pipeline (one asyncio.run per process; engine lives on this loop only)
# ---------------------------------------------------------------------------


async def _load_roster(session: Any) -> tuple[dict[str, tuple[Any, bool]], int]:
    """One-query roster snapshot: student_number -> (id, is_active), + active count.

    Includes INACTIVE students deliberately: D14 requires distinguishing
    "photo of a deactivated student" (INACTIVE_STUDENT) from "unknown ref".
    """
    from sqlalchemy import select

    from app.domain.models import Student

    result = await session.execute(
        select(Student.student_number, Student.id, Student.is_active)
    )
    roster: dict[str, tuple[Any, bool]] = {}
    active_count = 0
    for number, student_id, is_active in result.all():
        roster[number] = (student_id, bool(is_active))
        if is_active:
            active_count += 1
    return roster, active_count


async def _extract_with_retry(tensor: np.ndarray) -> tuple[np.ndarray, float]:
    """Extraction with require_detection=True + bounded infra retries.

    Any Triton availability/timeout/model/inference error surviving the retry
    budget aborts the whole run (ImportInfraError → exit 3): infrastructure
    failure is not a photo problem, so no NO_FACE-style reject is written.
    """
    from app.infrastructure.triton import (
        TritonInferenceError,
        TritonModelUnavailableError,
        TritonServerUnavailableError,
        TritonTimeoutError,
    )
    from app.services.pipeline_service import extract_enrollment_embedding

    attempt = 0
    while True:
        try:
            return await extract_enrollment_embedding(tensor, require_detection=True)
        except (
            TritonTimeoutError,
            TritonServerUnavailableError,
            TritonModelUnavailableError,
            TritonInferenceError,
        ) as exc:
            if attempt >= TRITON_MAX_RETRIES:
                raise ImportInfraError(
                    "Triton inference kept failing; aborting run. Committed rows "
                    "stay committed — rerun resumes from the journal."
                ) from exc
            delay = TRITON_RETRY_BACKOFF_SECONDS[min(attempt, len(TRITON_RETRY_BACKOFF_SECONDS) - 1)]
            attempt += 1
            print(
                f"WARNING: Triton inference failed ({exc.__class__.__name__}); "
                f"retry {attempt}/{TRITON_MAX_RETRIES} in {delay:.0f}s",
                file=sys.stderr,
            )
            await asyncio.sleep(delay)


async def _upsert_template(
    session_factory: Any,
    *,
    student_id: Any,
    student_number: str,
    pose_label: str,
    embedding: np.ndarray,
    quality_score: float,
    template_version: int,
    run_id: str,
    overwrite: bool,
) -> tuple[str, Any]:
    """D12 upsert: insert, or rotate when --overwrite; else skip.

    Mirrors the API rotation in ``students.enroll_student_template``: archive
    actives under FOR UPDATE + audit each, insert new active row + audit
    created, single commit. Returns ``(outcome, embedding_row_id)`` where
    outcome is "imported", "rotated", or "skipped_active_exists".
    """
    from sqlalchemy import select
    from sqlalchemy.exc import IntegrityError

    from app.domain.models import StudentEmbedding, TemplateAuditLog

    async with session_factory() as session:
        try:
            actives = list(
                (
                    await session.execute(
                        select(StudentEmbedding)
                        .where(StudentEmbedding.student_id == student_id)
                        .where(StudentEmbedding.pose_label == pose_label)
                        .where(StudentEmbedding.is_active.is_(True))
                        .with_for_update()
                    )
                ).scalars().all()
            )

            if actives and not overwrite:
                return "skipped_active_exists", None

            for active_embedding in actives:
                active_embedding.is_active = False
                session.add(
                    TemplateAuditLog(
                        student_id=student_id,
                        student_embedding_id=active_embedding.id,
                        action="archived",
                        pose_label=active_embedding.pose_label,
                        quality_score=active_embedding.quality_score,
                        event_metadata={"reason": "bulk_import_rotation"},
                    )
                )

            created = StudentEmbedding(
                student_id=student_id,
                # PRIVACY: this is the ONLY place a vector leaves this process —
                # straight into the biometric store. Never logged/serialized elsewhere.
                embedding=[float(value) for value in embedding.tolist()],
                pose_label=pose_label,
                quality_score=float(quality_score),
                is_active=True,
                template_version=int(template_version),
            )
            session.add(created)
            await session.flush()

            session.add(
                TemplateAuditLog(
                    student_id=student_id,
                    student_embedding_id=created.id,
                    action="created",
                    pose_label=pose_label,
                    quality_score=float(quality_score),
                    event_metadata={"source": "enrollment_import_cli", "run_id": run_id},
                )
            )
            await session.commit()
            return ("rotated" if actives else "imported"), created.id
        except IntegrityError:
            await session.rollback()
            raise


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI contract (see module docstring)."""
    parser = argparse.ArgumentParser(
        prog="import_enrollments",
        description="Bulk face-template enrollment importer (folder/CSV ingest).",
    )
    parser.add_argument(
        "--photos-dir",
        required=True,
        type=str,
        help="Root of the batch folder; manifest image_path entries resolve under it.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Optional CSV manifest (student_number,image_path[,pose_label]); "
        "default: folder-convention scan.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="import_out",
        help="Output directory for journal, reasons.csv, rejects/, summary "
        "(must not live inside --photos-dir). Default: ./import_out.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Full pipeline incl. Triton; zero DB writes, zero journal writes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rotate existing active (student, pose) templates API-style; "
        "default: skip those rows.",
    )
    parser.add_argument(
        "--template-version",
        type=int,
        default=DEFAULT_TEMPLATE_VERSION,
        metavar="N",
        help="template_version stamped on written embeddings (default 1).",
    )
    return parser


async def _run_import_async(args: argparse.Namespace) -> dict[str, Any]:
    """Run one import pass; returns the summary dict.

    Raises ImportUsageError (exit 2) or ImportInfraError (exit 3). Engine and
    session factory are created here and disposed in finally — they never
    outlive this coroutine's event loop.
    """
    from app.services.enrollment_quality import _resolve_enrollment_min_quality
    from app.services.pipeline_service import NoFaceDetectedError

    photos_dir = Path(args.photos_dir)
    if not photos_dir.is_dir():
        raise ImportUsageError(f"--photos-dir does not exist or is not a directory: {photos_dir}")

    out_dir = Path(args.out)
    resolved_photos = photos_dir.resolve()
    resolved_out = out_dir.resolve()
    if resolved_photos == resolved_out or resolved_photos in resolved_out.parents:
        raise ImportUsageError("--out must not live inside --photos-dir (it pollutes rescans).")

    if int(args.template_version) < 1:
        raise ImportUsageError("--template-version must be a positive integer.")

    # Fail closed BEFORE touching anything: malformed env is a config error.
    try:
        min_quality = _resolve_enrollment_min_quality()
    except RuntimeError as exc:
        raise ImportUsageError(str(exc)) from exc

    run_id = str(uuid.uuid4())
    started_at = datetime.now(tz=UTC)

    out_dir.mkdir(parents=True, exist_ok=True)
    reporter = RejectsReporter(out_dir, run_id)
    summary: dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "dry_run": bool(args.dry_run),
        "overwrite": bool(args.overwrite),
        "template_version": int(args.template_version),
        "min_quality": min_quality,
        "photos_dir": str(resolved_photos),
        "manifest": str(Path(args.manifest).resolve()) if args.manifest else None,
        "total_rows": 0,
        "imported": 0,
        "rotated": 0,
        "would_import": 0,
        "skipped_resumed": 0,
        "skipped_active_exists": 0,
        "duplicate_image_warnings": 0,
        "db_errors": 0,
        "extraction_errors": 0,
        "ignored_files": 0,
        "rejected": {code: 0 for code in REJECT_REASON_CODES},
    }

    db_url = os.environ.get("ATTENDANCE_DATABASE_URL")
    if not db_url:
        reporter.close()
        raise ImportUsageError("ATTENDANCE_DATABASE_URL is not set.")
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    engine = None
    reporter_closed = False
    journal_writer: JournalWriter | None = None
    try:
        # --- Input parsing (pure) ------------------------------------------
        unbindable: list[tuple[str, str, str]] = []
        if args.manifest:
            manifest_path = Path(args.manifest)
            if not manifest_path.is_file():
                raise ImportUsageError(f"--manifest does not exist: {manifest_path}")
            rows, issues = parse_manifest(manifest_path)
        else:
            rows, unbindable, ignored_files = scan_folder(photos_dir)
            summary["ignored_files"] = ignored_files
            issues = []

        # Input-authoring issues first (no image copies; authoring errors).
        for issue in issues:
            summary["rejected"][REASON_INVALID_ROW] += 1
            summary["total_rows"] += 1
            reporter.reject(
                student_number=issue.student_number,
                image_path=issue.image_path,
                pose_label=issue.pose_label,
                reason_code=REASON_INVALID_ROW,
                detail=issue.detail,
            )
            print(f"INVALID_ROW row={issue.row_index}: {issue.detail}", file=sys.stderr)

        # Folder-mode unbindable image files → UNKNOWN_REF rejects w/ copy.
        for filename, ref_hint, detail in unbindable:
            summary["rejected"][REASON_UNKNOWN_REF] += 1
            summary["total_rows"] += 1
            source = photos_dir / filename
            reporter.reject(
                student_number=ref_hint,
                image_path=filename,
                pose_label="",
                reason_code=REASON_UNKNOWN_REF,
                detail=f"filename does not match <student_number>[__<pose>].jpg: {detail}",
                source_path=source if source.is_file() else None,
            )

        # --- DB bootstrap (loop-bound; disposed in finally) -----------------
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

        engine = create_async_engine(db_url, pool_pre_ping=True, future=True)
        session_factory = async_sessionmaker(
            engine, class_=AsyncSession, autoflush=False, expire_on_commit=False
        )

        async with session_factory() as session:
            roster, roster_active = await _load_roster(session)
        summary["roster_active_students"] = roster_active

        journal_path = out_dir / JOURNAL_FILENAME
        journaled = load_journal(journal_path)
        seen_hashes: dict[str, str] = {}  # sha -> first student_number bound to it

        if not args.dry_run:
            journal_writer = JournalWriter(journal_path)

        # --- Row loop (strictly sequential; cheapest checks first) ----------
        for row in rows:
            summary["total_rows"] += 1

            # 1. Identity resolution (fail closed; D14 dedicated code).
            entry = roster.get(row.student_number)
            if entry is None:
                summary["rejected"][REASON_UNKNOWN_REF] += 1
                reporter.reject(
                    student_number=row.student_number,
                    image_path=row.image_path,
                    pose_label=row.pose_label,
                    reason_code=REASON_UNKNOWN_REF,
                    detail="no student with this student_number",
                )
                continue
            student_id, is_active = entry
            if not is_active:
                summary["rejected"][REASON_INACTIVE_STUDENT] += 1
                reporter.reject(
                    student_number=row.student_number,
                    image_path=row.image_path,
                    pose_label=row.pose_label,
                    reason_code=REASON_INACTIVE_STUDENT,
                    detail="student exists but is deactivated (is_active=False)",
                )
                continue

            # 2. Read bytes + hash (journal key needs only the bytes).
            try:
                source_path = resolve_image_path(photos_dir, row.image_path)
                data = source_path.read_bytes()
            except (ValueError, FileNotFoundError, OSError) as exc:
                summary["rejected"][REASON_UNREADABLE] += 1
                reporter.reject(
                    student_number=row.student_number,
                    image_path=row.image_path,
                    pose_label=row.pose_label,
                    reason_code=REASON_UNREADABLE,
                    detail=str(exc),
                    source_path=None,
                )
                continue
            digest = sha256_bytes(data)

            # 3. Journal resume skip (even under --overwrite: identical bytes
            #    already imported ⇒ nothing to rotate).
            if (row.student_number, digest) in journaled:
                summary["skipped_resumed"] += 1
                continue

            # 4. D15 duplicate-image warning: identical bytes bound to another
            #    student_number still imports, counted as a warning.
            first_owner = seen_hashes.get(digest)
            if first_owner is not None and first_owner != row.student_number:
                summary["duplicate_image_warnings"] += 1
                print(
                    f"WARNING: DUPLICATE_IMAGE — image already bound to "
                    f"{first_owner}; importing anyway for {row.student_number}.",
                    file=sys.stderr,
                )
            seen_hashes.setdefault(digest, row.student_number)

            # 5. Decode.
            try:
                tensor = decode_image_bytes(data)
            except ValueError as exc:
                summary["rejected"][REASON_UNREADABLE] += 1
                reporter.reject(
                    student_number=row.student_number,
                    image_path=row.image_path,
                    pose_label=row.pose_label,
                    reason_code=REASON_UNREADABLE,
                    detail=str(exc),
                    source_path=source_path,
                )
                continue

            # 6. Detection-presence extraction (NO_FACE instead of the
            #    whole-frame-resize fallback).
            try:
                embedding, quality_score = await _extract_with_retry(tensor)
            except ImportInfraError:
                raise
            except NoFaceDetectedError:
                summary["rejected"][REASON_NO_FACE] += 1
                reporter.reject(
                    student_number=row.student_number,
                    image_path=row.image_path,
                    pose_label=row.pose_label,
                    reason_code=REASON_NO_FACE,
                    detail="detector found zero faces in the image",
                    source_path=source_path,
                )
                continue
            except Exception as exc:  # noqa: BLE001 — unexpected ≠ photo problem;
                # count it and leave the row unjournaled so reruns retry it.
                summary["extraction_errors"] += 1
                print(
                    f"ERROR: extraction failed for {row.student_number} "
                    f"({exc.__class__.__name__}: {exc}); row left unjournaled.",
                    file=sys.stderr,
                )
                continue

            # 7. Quality gate (strict <, same boundary as the API).
            if quality_score < min_quality:
                summary["rejected"][REASON_LOW_QUALITY] += 1
                reporter.reject(
                    student_number=row.student_number,
                    image_path=row.image_path,
                    pose_label=row.pose_label,
                    reason_code=REASON_LOW_QUALITY,
                    detail=(
                        f"quality_score={quality_score:.4f} below "
                        f"ATTENDANCE_ENROLLMENT_MIN_QUALITY={min_quality:.4f}"
                    ),
                    quality_score=float(quality_score),
                    source_path=source_path,
                )
                continue

            # 8. Dry-run stop: everything validated, nothing persisted.
            if args.dry_run:
                summary["would_import"] += 1
                continue

            # 9. D12 upsert (skip active pose unless --overwrite; rotation
            #    mirrors enroll_student_template incl. audit rows).
            try:
                outcome, embedding_row_id = await _upsert_template(
                    session_factory,
                    student_id=student_id,
                    student_number=row.student_number,
                    pose_label=row.pose_label,
                    embedding=embedding,
                    quality_score=float(quality_score),
                    template_version=int(args.template_version),
                    run_id=run_id,
                    overwrite=bool(args.overwrite),
                )
            except Exception as exc:  # noqa: BLE001 — per-row DB failure: count + retry on rerun
                summary["db_errors"] += 1
                print(
                    f"ERROR: DB write failed for {row.student_number}/{row.pose_label} "
                    f"({exc.__class__.__name__}: {exc}); row NOT journaled (rerun retries it).",
                    file=sys.stderr,
                )
                continue

            if outcome == "skipped_active_exists":
                summary["skipped_active_exists"] += 1
                continue
            if outcome == "rotated":
                summary["rotated"] += 1
            summary["imported"] += 1

            # 10. Commit-first ordering: journal only AFTER a successful commit.
            assert journal_writer is not None
            journal_writer.append(
                {
                    "run_id": run_id,
                    "student_number": row.student_number,
                    "image_sha256": digest,
                    "pose_label": row.pose_label,
                    "embedding_row_id": str(embedding_row_id),
                    "template_version": int(args.template_version),
                    "committed_at": datetime.now(tz=UTC).isoformat(),
                }
            )

        # --- Coverage --------------------------------------------------------
        from sqlalchemy import text as sql_text

        async with session_factory() as session:
            enrolled_n = int(
                (
                    await session.execute(
                        sql_text(
                            "SELECT COUNT(DISTINCT student_id) FROM student_embeddings "
                            "WHERE is_active"
                        )
                    )
                ).scalar_one()
            )
        summary["enrolled_active_templates"] = enrolled_n

        finished_at = datetime.now(tz=UTC)
        summary["finished_at"] = finished_at.isoformat()
        summary["duration_seconds"] = round((finished_at - started_at).total_seconds(), 3)

        summary_path = out_dir / SUMMARY_FILENAME
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
            fh.write("\n")

        reporter.close()
        reporter_closed = True
        return summary
    finally:
        if not reporter_closed:
            reporter.close()
        if journal_writer is not None:
            journal_writer.close()
        if engine is not None:
            await engine.dispose()  # loop-bound engine dies with its loop


def main(argv: list[str] | None = None) -> int:
    """Sync entrypoint: exactly ONE asyncio.run; maps failures to exit codes."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        summary = asyncio.run(_run_import_async(args))
    except ImportUsageError as exc:
        print(f"USAGE ERROR (exit 2): {exc}", file=sys.stderr)
        return 2
    except ImportInfraError as exc:
        print(f"INFRASTRUCTURE ABORT (exit 3): {exc}", file=sys.stderr)
        return 3
    except Exception:  # noqa: BLE001 — unexpected crash ≡ infra abort class
        print("UNEXPECTED FAILURE (exit 3); committed rows stay committed.", file=sys.stderr)
        traceback.print_exc()
        return 3

    _print_report(summary, args)
    return 0


def _print_report(summary: dict[str, Any], args: argparse.Namespace) -> None:
    """Stdout report: coverage line first, then per-counter breakdown."""
    print(coverage_line(summary))
    rejected = summary["rejected"]
    print(
        "rows: total={} imported={} rotated={} skipped_resumed={} skipped_active_exists={}".format(
            summary["total_rows"],
            summary["imported"],
            summary["rotated"],
            summary["skipped_resumed"],
            summary["skipped_active_exists"],
        )
    )
    print(
        "rejected: "
        + " ".join(f"{code}={rejected[code]}" for code in REJECT_REASON_CODES)
        + f" | duplicate_image_warnings={summary['duplicate_image_warnings']}"
        f" | db_errors={summary['db_errors']}"
        f" | extraction_errors={summary['extraction_errors']}"
    )
    print(f"summary: {Path(args.out).resolve() / SUMMARY_FILENAME}")


if __name__ == "__main__":
    sys.exit(main())
