"""Pytest fixtures for the Attendance v3 smoke-test suite."""

from __future__ import annotations

import os

# Set required env vars BEFORE any app module is imported.  Module-level
# lru_cache calls in celery_app.py, security.py, etc. fire at import time
# and will hard-fail if these vars are missing.
os.environ.setdefault("ATTENDANCE_ALLOWED_ORIGINS", "http://localhost:3000")
os.environ.setdefault("ATTENDANCE_JWT_SECRET", "test-secret-not-for-production-use")
os.environ.setdefault("ATTENDANCE_REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault(
    "ATTENDANCE_DATABASE_URL",
    "postgresql://attendance:attendance@localhost:15432/attendance_test",
)
os.environ.setdefault("ATTENDANCE_TRITON_URL", "fake-host:8001")

import uuid
from typing import AsyncGenerator

import numpy as np
import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine, async_sessionmaker

from tests._fakes import FakeTritonGrpcClient


# ---------------------------------------------------------------------------
# Session-scoped event loop policy for pytest-asyncio 0.26+
# Without this, session-scoped async fixtures emit a DeprecationWarning which
# is promoted to an error by filterwarnings = ["error"] in pyproject.toml.
# ---------------------------------------------------------------------------

def pytest_configure(config: pytest.Config) -> None:
    # Suppress pytest-asyncio session-scope deprecation warnings (promoted to
    # errors by filterwarnings = ["error"] in pyproject.toml).
    config.addinivalue_line(
        "filterwarnings",
        "ignore::DeprecationWarning:pytest_asyncio",
    )
    # Celery 5.x emits a DeprecationWarning for task_always_eager.
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*task_always_eager.*:DeprecationWarning",
    )
    # SQLAlchemy may emit ImplicitlyBegunTransactionWarning in some versions.
    config.addinivalue_line(
        "filterwarnings",
        "ignore::sqlalchemy.exc.SAWarning",
    )


# ---------------------------------------------------------------------------
# Helper: resolve and normalise DB URL
# ---------------------------------------------------------------------------

def _resolve_db_url() -> str | None:
    url = os.environ.get("ATTENDANCE_DATABASE_URL_TEST") or os.environ.get("ATTENDANCE_DATABASE_URL")
    if not url:
        return None
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if url.startswith("postgresql+asyncpg://"):
        return url
    if url.startswith("postgresql+psycopg://"):
        return url.replace("postgresql+psycopg://", "postgresql+asyncpg://", 1)
    if url.startswith("postgresql+psycopg2://"):
        return url.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
    return url


def _resolve_redis_url() -> str | None:
    return os.environ.get("ATTENDANCE_REDIS_URL_TEST") or os.environ.get("ATTENDANCE_REDIS_URL")


# ---------------------------------------------------------------------------
# Session-scoped engine + alembic migration
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_engine() -> AsyncEngine:
    db_url = _resolve_db_url()
    if not db_url:
        pytest.skip("ATTENDANCE_DATABASE_URL not set — skipping DB-dependent tests")

    # Keep the test DB URL in the standard env var so alembic env.py picks it up.
    raw_url = os.environ.get("ATTENDANCE_DATABASE_URL_TEST") or os.environ.get("ATTENDANCE_DATABASE_URL", "")
    os.environ["ATTENDANCE_DATABASE_URL"] = raw_url

    from alembic.config import Config as AlembicConfig
    from alembic import command as alembic_command

    alembic_cfg = AlembicConfig("backend/alembic.ini")
    alembic_cfg.set_main_option("script_location", "backend/alembic")
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)
    alembic_command.upgrade(alembic_cfg, "head")

    engine = create_async_engine(db_url, future=True, pool_pre_ping=True)
    return engine


# ---------------------------------------------------------------------------
# Per-test TRUNCATE
# ---------------------------------------------------------------------------

_DOMAIN_TABLES = [
    "governance_logs",
    "class_session_records",
    "sightings",
    "template_audit_logs",
    "student_embeddings",
    "students",
    "courses",
    "rooms",
    "users",
]


@pytest.fixture(autouse=True)
async def truncate_tables(test_engine: AsyncEngine) -> AsyncGenerator[None, None]:
    yield
    table_list = ", ".join(_DOMAIN_TABLES)
    async with test_engine.begin() as conn:
        await conn.execute(
            text(f"TRUNCATE {table_list} RESTART IDENTITY CASCADE")
        )


# ---------------------------------------------------------------------------
# Session-scoped Redis client
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
async def redis_client():
    redis_url = _resolve_redis_url()
    if not redis_url:
        pytest.skip("ATTENDANCE_REDIS_URL not set — skipping Redis-dependent tests")

    import redis.asyncio as aioredis

    client = aioredis.Redis.from_url(redis_url, decode_responses=True)
    try:
        await client.ping()
    except Exception:
        pytest.skip("Redis not reachable — skipping Redis-dependent tests")
    yield client
    await client.aclose()


@pytest.fixture(autouse=True)
async def flush_redis(redis_client) -> AsyncGenerator[None, None]:
    yield
    await redis_client.flushdb()


# ---------------------------------------------------------------------------
# Fake Triton
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_triton() -> FakeTritonGrpcClient:
    from app.infrastructure.triton.client import set_triton_client_override

    fake = FakeTritonGrpcClient()
    set_triton_client_override(fake)
    yield fake
    set_triton_client_override(None)


# ---------------------------------------------------------------------------
# Synthetic frame fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_frame_bytes() -> bytes:
    frame = (255 * np.ones((480, 640, 3), dtype=np.uint8))
    frame[120:200, 160:220] = 128
    return frame.tobytes()


@pytest.fixture()
def synthetic_frame_form(synthetic_frame_bytes: bytes) -> dict:
    return {
        "frame_id": "smoke-frame-001",
        "width": "640",
        "height": "480",
        "channels": "3",
        "dtype": "uint8",
        "normalize": "true",
    }


# ---------------------------------------------------------------------------
# User fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
async def _session_factory(test_engine: AsyncEngine):
    factory = async_sessionmaker(bind=test_engine, class_=AsyncSession, autoflush=False, autocommit=False, expire_on_commit=False)
    return factory


async def _insert_user(
    session_factory,
    *,
    email: str,
    full_name: str,
    role,
):
    from app.domain.models import User
    from app.core.security import hash_password

    async with session_factory() as session:
        user = User(
            id=uuid.uuid4(),
            email=email,
            full_name=full_name,
            password_hash=hash_password("TestPass1!"),
            role=role,
            is_active=True,
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user


@pytest.fixture()
async def admin_user(_session_factory):
    from app.domain.models import UserRole
    return await _insert_user(_session_factory, email="admin@test.example", full_name="Admin User", role=UserRole.ADMIN)


@pytest.fixture()
async def instructor_user(_session_factory):
    from app.domain.models import UserRole
    return await _insert_user(_session_factory, email="instructor@test.example", full_name="Instructor User", role=UserRole.INSTRUCTOR)


@pytest.fixture()
async def operator_user(_session_factory):
    from app.domain.models import UserRole
    return await _insert_user(_session_factory, email="operator@test.example", full_name="Operator User", role=UserRole.OPERATOR)


@pytest.fixture()
async def auditor_user(_session_factory):
    """Provision an AUDITOR-role user for RBAC denial tests.

    ATT-031: this fixture was historically named `student_user`, but the
    UserRole enum (see backend/app/domain/models/_base.py:54-60) has no
    STUDENT role — students are not self-authenticating in this codebase;
    attendance is read-side for instructors/admins. The previous name
    mislabeled tests as exercising 'student permission' when they were
    actually exercising AUDITOR permission and masking gaps in role design.
    The new name honestly reflects what the fixture provisions.
    """
    from app.domain.models import UserRole
    return await _insert_user(
        _session_factory,
        email="auditor@test.example",
        full_name="Auditor User",
        role=UserRole.AUDITOR,
    )


# ---------------------------------------------------------------------------
# Auth cookie factory
# ---------------------------------------------------------------------------

@pytest.fixture()
def auth_cookie():
    def _make(user) -> dict[str, str]:
        from app.core.security import create_access_token, get_security_settings
        token, _ = create_access_token(subject=user.id, role=user.role)
        settings = get_security_settings()
        return {settings.access_cookie_name: token}
    return _make


# ---------------------------------------------------------------------------
# ASGI async client
# ---------------------------------------------------------------------------

@pytest.fixture()
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    from app.main import create_app
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client

