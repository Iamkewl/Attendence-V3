"""Database engine and session lifecycle management for the attendance service."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
from functools import lru_cache


from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine


LOGGER = logging.getLogger(__name__)


def _read_int_env(name: str, default: int) -> int:
    """Read an integer environment variable and validate its type and bounds."""
    value = os.getenv(name)
    if value is None:
        return default

    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer.") from exc

    if parsed < 0:
        raise ValueError(f"Environment variable {name} must be greater than or equal to zero.")

    return parsed


def _require_database_url() -> str:
    """Return the configured PostgreSQL DSN and fail fast when missing."""
    url = os.getenv("ATTENDANCE_DATABASE_URL")
    if not url:
        raise RuntimeError(
            "Environment variable ATTENDANCE_DATABASE_URL must be set to a PostgreSQL DSN."
        )
    return url


def _normalize_async_url(database_url: str) -> str:
    """Normalize supported PostgreSQL URLs to the SQLAlchemy asyncpg dialect."""
    if database_url.startswith("postgresql+asyncpg://"):
        return database_url
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql+asyncpg://", 1)
    if database_url.startswith("postgresql+psycopg2://"):
        return database_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)

    raise ValueError(
        "Database URL must start with postgresql://, postgresql+psycopg://, "
        "postgresql+psycopg2://, or postgresql+asyncpg://."
    )


@lru_cache(maxsize=1)
def get_async_engine() -> AsyncEngine:
    """Create and cache the asynchronous SQLAlchemy engine used by the application."""
    return create_async_engine(
        _normalize_async_url(_require_database_url()),
        pool_size=_read_int_env("ATTENDANCE_DB_POOL_SIZE", 10),
        max_overflow=_read_int_env("ATTENDANCE_DB_MAX_OVERFLOW", 20),
        pool_timeout=_read_int_env("ATTENDANCE_DB_POOL_TIMEOUT", 30),
        pool_recycle=_read_int_env("ATTENDANCE_DB_POOL_RECYCLE", 1800),
        pool_pre_ping=True,
        echo=False,
        future=True,
    )


@lru_cache(maxsize=1)
def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Create and cache the asynchronous session factory with safe production defaults."""
    return async_sessionmaker(
        bind=get_async_engine(),
        class_=AsyncSession,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield a managed asynchronous session for request-scoped database access."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
        except Exception:
            try:
                await session.rollback()
            except Exception:
                LOGGER.exception("Session rollback failed while handling outer exception.")
            raise


async def dispose_engine() -> None:
    """Dispose pooled database connections and clear engine/session caches.

    Caches are cleared in a ``finally`` block: if ``engine.dispose()`` raises,
    the cached factory/engine must still be dropped, otherwise the next task
    would reuse the dead loop-bound engine this function exists to retire.
    """
    engine = get_async_engine()
    try:
        await engine.dispose()
    finally:
        get_session_factory.cache_clear()
        get_async_engine.cache_clear()


__all__ = ["dispose_engine", "get_async_engine", "get_async_session", "get_session_factory"]
