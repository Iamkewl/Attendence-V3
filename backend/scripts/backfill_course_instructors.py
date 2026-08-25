"""Human-gated backfill of course_instructors from a reviewed CSV mapping.

ATT-016 / decision D9: production rollout is driven by a human-produced CSV
(``course_code,instructor_email,role_in_course``), committed nowhere. The
script ships inert without it and refuses to guess: unknown course codes or
emails are collected and reported, never fuzzy-matched.

Usage:
    python backend/scripts/backfill_course_instructors.py --mapping course_owners.csv            # dry-run (default)
    ATTENDANCE_BACKFILL_CONFIRM=YES python backend/scripts/backfill_course_instructors.py \
        --mapping course_owners.csv --apply

Dry-run is the default and prints planned (course_id, user_id, role) triples
plus a coverage report ("N of M active courses unassigned"). ``--apply``
additionally requires ``ATTENDANCE_BACKFILL_CONFIRM=YES`` in the environment
— two independent human actions before any write. Inserts are idempotent
(ON CONFLICT DO NOTHING).

Exit codes: 0 success, 1 refusal/validation failure.
"""

from __future__ import annotations

import asyncio
import csv
import os
import sys

CONFIRM_ENV = "ATTENDANCE_BACKFILL_CONFIRM"
REQUIRED_HEADER = ["course_code", "instructor_email", "role_in_course"]
VALID_ROLES = {"owner", "ta"}

# TODO(audit-sprint): each applied insert must also write a GovernanceLog row
# (action='course_instructor.backfill', entity_type='course_instructors',
# actor=executing admin) once the governance audit service lands; see
# decisions D1-D7 in /tmp/opencode/design/audit-service-design.md.


def _parse_mapping(path: str) -> list[dict[str, str]]:
    """Read and validate the CSV shape; raise ValueError on structural problems."""
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or [f.strip().lower() for f in reader.fieldnames] != REQUIRED_HEADER:
            raise ValueError(
                f"CSV header must be exactly {','.join(REQUIRED_HEADER)}; "
                f"got {reader.fieldnames!r}."
            )
        rows = []
        for line_no, row in enumerate(reader, start=2):
            code = (row["course_code"] or "").strip()
            email = (row["instructor_email"] or "").strip()
            role = (row["role_in_course"] or "").strip().lower()
            if not code or not email or role not in VALID_ROLES:
                raise ValueError(
                    f"CSV line {line_no}: course_code, instructor_email must be non-blank "
                    f"and role_in_course must be one of {sorted(VALID_ROLES)}; got "
                    f"{code!r}, {email!r}, {(row['role_in_course'] or '').strip()!r}."
                )
            rows.append({"course_code": code, "instructor_email": email.lower(), "role_in_course": role})
    if not rows:
        raise ValueError("CSV contains no data rows.")
    return rows


async def run(mapping_path: str, *, apply: bool) -> int:
    """Resolve the mapping against the DB, report, and optionally apply."""
    from sqlalchemy import func, select
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession, create_async_engine

    from app.domain.models import Course, CourseInstructor, User

    try:
        rows = _parse_mapping(mapping_path)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    db_url = os.environ.get("ATTENDANCE_DATABASE_URL")
    if not db_url:
        print("ERROR: ATTENDANCE_DATABASE_URL is not set.", file=sys.stderr)
        return 1
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    engine = create_async_engine(db_url, pool_pre_ping=True, future=True)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    planned: list[dict[str, object]] = []
    errors: list[str] = []
    try:
        async with session_factory() as session:
            for index, row in enumerate(rows, start=1):
                course_id = (
                    await session.scalar(
                        select(Course.id).where(Course.code == row["course_code"])
                    )
                )
                if course_id is None:
                    errors.append(
                        f"mapping #{index}: unknown course_code {row['course_code']!r}"
                        " — refusing to guess."
                    )
                    continue
                user_id = (
                    await session.scalar(
                        select(User.id).where(func.lower(User.email) == row["instructor_email"])
                    )
                )
                if user_id is None:
                    errors.append(
                        f"mapping #{index}: unknown instructor_email "
                        f"{row['instructor_email']!r} — refusing to guess."
                    )
                    continue
                planned.append({**row, "course_id": course_id, "user_id": user_id})

            total_active = (
                await session.scalar(select(func.count()).select_from(Course).where(Course.is_active.is_(True)))
            ) or 0
            assigned_ids = set(
                await session.scalars(
                    select(CourseInstructor.course_id).where(CourseInstructor.role_in_course == "owner")
                )
            )
            active_ids = set(
                await session.scalars(select(Course.id).where(Course.is_active.is_(True)))
            )
            unassigned = len(active_ids - assigned_ids)

            for entry in planned:
                print(
                    f"PLAN  course={entry['course_id']} user={entry['user_id']} "
                    f"code={entry['course_code']} email={entry['instructor_email']} "
                    f"role={entry['role_in_course']}"
                )
            print(f"Coverage: {unassigned} of {total_active} active courses have no owner.")

            if errors:
                for error in errors:
                    print(f"ERROR: {error}", file=sys.stderr)
                print(
                    f"Aborted: {len(errors)} unresolved mapping entr(y|ies); "
                    "nothing was written.",
                    file=sys.stderr,
                )
                return 1

            if not apply:
                print("Dry-run only — no rows written. Re-run with --apply to persist.")
                return 0

            confirm = os.environ.get(CONFIRM_ENV, "").strip().upper()
            if confirm != "YES":
                print(
                    f"ERROR: --apply requires {CONFIRM_ENV}=YES in the environment.",
                    file=sys.stderr,
                )
                return 1

            inserted = 0
            for entry in planned:
                result = await session.execute(
                    pg_insert(CourseInstructor)
                    .values(
                        course_id=entry["course_id"],
                        user_id=entry["user_id"],
                        role_in_course=entry["role_in_course"],
                    )
                    .on_conflict_do_nothing(
                        index_elements=[CourseInstructor.course_id, CourseInstructor.user_id]
                    )
                )
                inserted += result.rowcount or 0
            await session.commit()
            # TODO(audit-sprint): write GovernanceLog rows for applied inserts here.
            print(f"Applied: {inserted} row(s) inserted ({len(planned) - inserted} already existed).")
            return 0
    finally:
        await engine.dispose()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mapping", required=True, help="Path to the reviewed CSV mapping.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Report only (default).")
    mode.add_argument("--apply", action="store_true", help="Insert rows (requires confirmation env).")
    args = parser.parse_args()
    try:
        sys.exit(asyncio.run(run(args.mapping, apply=args.apply)))
    except Exception as exc:  # noqa: BLE001 — operator-facing script boundary
        print(f"BACKFILL FAILED: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
