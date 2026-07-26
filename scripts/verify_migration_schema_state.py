#!/usr/bin/env python
"""Assert PostgreSQL schema state for the Alembic migration round-trip CI check.

`alembic upgrade` / `alembic downgrade` exiting 0 only proves the commands ran
without raising -- it does not prove they left the schema in the state they
claim to. This script inspects `information_schema` directly so the CI job
(`migration-round-trip` in .github/workflows/backend-tests.yml) fails loudly
if a downgrade "succeeds" without actually dropping what it says it drops, or
if upgrade head is missing an expected table.

Connects using standard libpq environment variables (PGHOST, PGPORT, PGUSER,
PGPASSWORD, PGDATABASE), which the CI job sets to match the ephemeral
pgvector/pgvector:pg16 service container.

Usage:
    python scripts/verify_migration_schema_state.py empty       # after `alembic downgrade base`
    python scripts/verify_migration_schema_state.py populated   # after `alembic upgrade head`
"""

from __future__ import annotations

import sys

import psycopg2

# Every application table created across the four migrations in
# backend/alembic/versions/. Kept in sync manually -- there are only four
# migrations and this list is meant to catch drift, not chase it.
EXPECTED_TABLES_AT_HEAD = {
    "users",
    "students",
    "courses",
    "rooms",
    "class_session_records",
    "governance_logs",
    "sightings",
    "student_embeddings",
    "template_audit_logs",
}


def _application_tables(cur) -> set[str]:
    """Return base tables in the public schema, excluding Alembic's own bookkeeping table."""
    cur.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_type = 'BASE TABLE'
          AND table_name != 'alembic_version'
        """
    )
    return {row[0] for row in cur.fetchall()}


def _check_empty(cur) -> int:
    tables = _application_tables(cur)
    if tables:
        print(
            "FAIL: expected an empty public schema after 'alembic downgrade base', "
            f"but found leftover tables: {sorted(tables)}",
            file=sys.stderr,
        )
        return 1
    print("OK: public schema has no application tables after 'alembic downgrade base'.")
    return 0


def _check_populated(cur) -> int:
    tables = _application_tables(cur)
    missing = EXPECTED_TABLES_AT_HEAD - tables
    if missing:
        print(
            "FAIL: expected all domain tables to exist after 'alembic upgrade head', "
            f"but these are missing: {sorted(missing)}",
            file=sys.stderr,
        )
        return 1
    print(f"OK: all {len(EXPECTED_TABLES_AT_HEAD)} domain tables present after 'alembic upgrade head'.")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in {"empty", "populated"}:
        print(f"usage: {argv[0]} {{empty|populated}}", file=sys.stderr)
        return 2

    mode = argv[1]
    conn = psycopg2.connect()
    try:
        with conn, conn.cursor() as cur:
            return _check_empty(cur) if mode == "empty" else _check_populated(cur)
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
