"""Idempotent demo data seeder for Attendance v3.

Run inside the api container:
    python /app/backend/scripts/seed_demo_data.py

Reads ATTENDANCE_DATABASE_URL from the environment.
Exits 0 on success, 1 on failure.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid

# Fixed deterministic UUIDs so re-runs are always idempotent.
DEMO_COURSE_ID = uuid.UUID("00000000-0000-4000-a000-000000000001")
DEMO_ADMIN_ID = uuid.UUID("00000000-0000-4000-a000-000000000002")
DEMO_STUDENT_IDS = [
    uuid.UUID(f"00000000-0000-4000-a000-0000000000{10 + i:02d}")
    for i in range(5)
]
DEMO_STUDENT_USER_IDS = [
    uuid.UUID(f"00000000-0000-4000-a000-0000000000{20 + i:02d}")
    for i in range(5)
]

ADMIN_EMAIL = "admin@attendance.demo"
ADMIN_PASSWORD = "DemoAdmin1!"
ADMIN_FULL_NAME = "Demo Administrator"


async def seed() -> None:
    """Run the full idempotent seed sequence."""
    # Import here so the module can be syntax-checked without the app installed.
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

    from app.core.security import hash_password
    from app.domain.models import User, UserRole, Course, Student

    db_url = os.environ.get("ATTENDANCE_DATABASE_URL")
    if not db_url:
        print("ERROR: ATTENDANCE_DATABASE_URL is not set.", file=sys.stderr)
        sys.exit(1)

    # Normalise to asyncpg dialect.
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    engine = create_async_engine(db_url, pool_pre_ping=True, future=True)
    session_factory = async_sessionmaker(engine, class_=AsyncSession,
                                         autoflush=False, expire_on_commit=False)

    created: list[str] = []
    skipped: list[str] = []

    async with session_factory() as session:
        # --- Admin user ---
        existing_admin = await session.get(User, DEMO_ADMIN_ID)
        if existing_admin is None:
            pw_hash = hash_password(ADMIN_PASSWORD)
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
        else:
            skipped.append(f"User  {ADMIN_EMAIL} (ADMIN) — already exists")

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
    print("\nLogin: admin@attendance.demo / DemoAdmin1!")


def main() -> None:
    try:
        asyncio.run(seed())
    except Exception as exc:
        print(f"SEED FAILED: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
