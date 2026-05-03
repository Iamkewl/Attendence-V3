"""Alembic environment configuration for asynchronous PostgreSQL migrations."""

from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from app.domain.models import Base


config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    """Resolve the database URL from environment or Alembic configuration."""
    env_url = os.getenv("ATTENDANCE_DATABASE_URL")
    if env_url:
        if env_url.startswith("postgresql+asyncpg://"):
            return env_url
        if env_url.startswith("postgresql://"):
            return env_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        if env_url.startswith("postgresql+psycopg://"):
            return env_url.replace("postgresql+psycopg://", "postgresql+asyncpg://", 1)
        if env_url.startswith("postgresql+psycopg2://"):
            return env_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
        raise ValueError(
            "ATTENDANCE_DATABASE_URL must start with postgresql://, postgresql+psycopg://, "
            "postgresql+psycopg2://, or postgresql+asyncpg://."
        )

    configured_url = config.get_main_option("sqlalchemy.url")
    if not configured_url:
        raise RuntimeError("Alembic sqlalchemy.url is not configured.")

    return configured_url


def run_migrations_offline() -> None:
    """Run migrations in offline mode using emitted SQL scripts."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Configure migration context and execute migrations for a live connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations in online mode using an asynchronous SQLAlchemy engine."""
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _database_url()

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
