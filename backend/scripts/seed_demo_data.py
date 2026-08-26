"""Idempotent demo data seeder for Attendance v3.

Run inside the api container:
    python /app/backend/scripts/seed_demo_data.py

Reads ATTENDANCE_DATABASE_URL and ATTENDANCE_DEMO_MODE from the environment.
Exits 0 on success, 1 on failure.

DEMO-MODE GATE (ATT-043): refuses to seed unless ATTENDANCE_DEMO_MODE=1 is
set in the environment, generates a fresh random admin password each run,
and prints it to stderr once.  This guards against an accidental `make demo`
against a reachable production database leaving a publicly-known admin
backdoor.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import sys
import uuid

# Fixed deterministic UUIDs so re-runs are always idempotent.
DEMO_COURSE_ID = uuid.UUID("00000000-0000-4000-a000-000000000001")
DEMO_ADMIN_ID = uuid.UUID("00000000-0000-4000-a000-000000000002")
# ATT-016: deterministic owner link binding the demo admin to the demo
# course, so `make demo` keeps working once ATTENDANCE_COURSE_SCOPED_AUTHZ
# is flipped on. Deliberately NOT bound to the five INSTRUCTOR-roled demo
# student accounts — those must not inherit instructor-grade read paths.
DEMO_COURSE_INSTRUCTOR_ID = uuid.UUID("00000000-0000-4000-a000-000000000003")
DEMO_STUDENT_IDS = [
    uuid.UUID(f"00000000-0000-4000-a000-0000000000{10 + i:02d}")
    for i in range(5)
]
DEMO_STUDENT_USER_IDS = [
    uuid.UUID(f"00000000-0000-4000-a000-0000000000{20 + i:02d}")
    for i in range(5)
]

ADMIN_EMAIL = "admin@attendance.demo"
ADMIN_FULL_NAME = "Demo Administrator"

# Hard requirement: this script must not silently bootstrap an admin
# account with a fixed password on an arbitrary environment. The demo
# gate and per-run random password below close the
# ATT-043 "backdoor admin" finding.
_DEMO_MODE_ENV = "ATTENDANCE_DEMO_MODE"


def _new_admin_password() -> str:
    """Generate a fresh random admin password for this run.

    `secrets.token_urlsafe(16)` yields ~22 chars of URL-safe base64
    (120 bits of entropy) — well beyond any sane password policy.
    """
    return secrets.token_urlsafe(16)


def _check_demo_mode_or_exit() -> None:
    """Refuse to seed unless the operator has explicitly opted into demo mode.

    Closes ATT-043: prior to this gate `make demo` could bootstrap an admin
    account on whatever database URL happened to be reachable, with a
    constant password published in the source tree. Now `make demo` exits 1
    unless ``ATTENDANCE_DEMO_MODE=1`` is set in the environment.
    """
    mode = os.environ.get(_DEMO_MODE_ENV, "")
    if mode != "1":
        print(
            "ERROR: refusing to seed demo data — "
            f"{_DEMO_MODE_ENV}={mode!r}, expected '1'. "
            "Set ATTENDANCE_DEMO_MODE=1 to opt into demo seeding. "
            "Demo seeding must never run against a production database.",
            file=sys.stderr,
        )
        sys.exit(1)


def _warn_if_target_db_not_demo_marked(db_url: str) -> None:
    """Best-effort warn if ATTENDANCE_DATABASE_URL doesn't look demo-targeted.

    The issue's recommended fix suggests rejecting non-`_demo` URLs, but
    doing so hard would break legitimate test setups (e.g.
    ``attendance_test`` used by the smoke suite). We emit a warning
    instead and let the demo_mode gate (the load-bearing control) refuse
    runs that didn't opt in.
    """
    # Best-effort parse from the URL path (e.g. .../attendance_demo).
    path = db_url.split("://", 1)[-1]
    if "/" in path:
        db_name = path.rsplit("/", 1)[-1]
    else:
        db_name = ""
    if db_name and not (db_name.endswith("_demo") or db_name == "demo"):
        print(
            f"WARNING: ATTENDANCE_DATABASE_URL target '{db_name}' does not "
            "end in _demo. Confirm this is a demo database before continuing.",
            file=sys.stderr,
        )


async def seed() -> None:
    """Run the full idempotent seed sequence."""
    # Import here so the module can be syntax-checked without the app installed.
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

    from app.core.security import hash_password
    from app.domain.models import User, UserRole, Course, CourseInstructor, Student

    _check_demo_mode_or_exit()

    db_url = os.environ.get("ATTENDANCE_DATABASE_URL")
    if not db_url:
        print("ERROR: ATTENDANCE_DATABASE_URL is not set.", file=sys.stderr)
        sys.exit(1)
    _warn_if_target_db_not_demo_marked(db_url)

    # Normalise to asyncpg dialect.
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    engine = create_async_engine(db_url, pool_pre_ping=True, future=True)
    session_factory = async_sessionmaker(engine, class_=AsyncSession,
                                         autoflush=False, expire_on_commit=False)

    created: list[str] = []
    skipped: list[str] = []

    admin_password_printed = ""  # populated below for fresh-seed summary

    async with session_factory() as session:
        # --- Admin user ---
        existing_admin = await session.get(User, DEMO_ADMIN_ID)
        if existing_admin is None:
            # ATT-043: random per-run admin password, printed to stderr only
            # so it lands in the operator's terminal rather than a server log.
            admin_password = _new_admin_password()
            pw_hash = hash_password(admin_password)
            admin = User(
                id=DEMO_ADMIN_ID,
                email=ADMIN_EMAIL,
                full_name=ADMIN_FULL_NAME,
                password_hash=pw_hash,
                role=UserRole.ADMIN,
                is_active=True,
            )
            session.add(admin)
            created.append(f"User  {ADMIN_EMAIL} (ADMIN)")
            admin_password_printed = admin_password
        else:
            skipped.append(f"User  {ADMIN_EMAIL} (ADMIN) — already exists (unchanged)")

        # --- Demo course ---
        existing_course = await session.get(Course, DEMO_COURSE_ID)
        if existing_course is None:
            course = Course(
                id=DEMO_COURSE_ID,
                code="DEMO-101",
                title="Demo Course — Attendance v3",
                description="Seeded for reviewer demo purposes.",
                credits=3,
                is_active=True,
            )
            session.add(course)
            created.append("Course DEMO-101")
        else:
            skipped.append("Course DEMO-101 — already exists")

        # autoflush is disabled on this session; the link INSERT below
        # FK-checks user/course rows that are still only pending in the
        # session — flush them first or the INSERT fails.
        await session.flush()

        # --- Demo course owner link (ATT-016) ---
        existing_link = await session.get(CourseInstructor, DEMO_COURSE_INSTRUCTOR_ID)
        if existing_link is None:
            session.add(
                CourseInstructor(
                    id=DEMO_COURSE_INSTRUCTOR_ID,
                    course_id=DEMO_COURSE_ID,
                    user_id=DEMO_ADMIN_ID,
                    role_in_course="owner",
                )
            )
            created.append("CourseInstructor admin@attendance.demo -> DEMO-101 (owner)")
        else:
            skipped.append("CourseInstructor admin@attendance.demo -> DEMO-101 — already exists")

        # --- Demo students (each needs a User row first) ---
        for i in range(5):
            label = f"Demo Student {i + 1:02d}"
            user_id = DEMO_STUDENT_USER_IDS[i]
            student_id = DEMO_STUDENT_IDS[i]
            email = f"student{i + 1:02d}@attendance.demo"

            existing_user = await session.get(User, user_id)
            if existing_user is None:
                user = User(
                    id=user_id,
                    email=email,
                    full_name=label,
                    password_hash=hash_password(f"DemoStudent{i + 1:02d}!"),
                    role=UserRole.INSTRUCTOR,
                    is_active=True,
                )
                session.add(user)
                created.append(f"User  {email}")
            else:
                skipped.append(f"User  {email} — already exists")

            existing_student = await session.get(Student, student_id)
            if existing_student is None:
                student = Student(
                    id=student_id,
                    user_id=user_id,
                    student_number=f"S{2026000 + i + 1}",
                    program="Demo Program",
                    enrollment_year=2026,
                    is_active=True,
                )
                session.add(student)
                created.append(f"Student {label} (#{student.student_number})")
            else:
                skipped.append(f"Student {label} — already exists")

        await session.commit()

    await engine.dispose()

    print("\n=== Demo Seed Results ===")
    for line in created:
        print(f"  CREATED  {line}")
    for line in skipped:
        print(f"  SKIPPED  {line}")
    print(f"\nDone: {len(created)} created, {len(skipped)} skipped.")
    if admin_password_printed:
        # Print the fresh admin password to stderr so server-side log
        # aggregators that capture stdout don't accidentally persist a
        # credential. The demo operator reads it from their terminal.
        print(
            f"\nFresh admin credentials (ATTN: capture now, not re-printed):\n"
            f"  Login: {ADMIN_EMAIL}\n"
            f"  Password: {admin_password_printed}",
            file=sys.stderr,
        )
    else:
        print(
            "\nAdmin account already existed; its password was NOT reset. "
            "If you do not know it, drop the users table and re-seed.",
            file=sys.stderr,
        )


def main() -> None:
    try:
        asyncio.run(seed())
    except Exception as exc:
        print(f"SEED FAILED: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
