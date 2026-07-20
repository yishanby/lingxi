"""Transactional SQLite schema migrations used during application startup."""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine


_SESSIONS_MIGRATION_TABLE = "sessions__md_v2_migration"
_CREATE_SESSIONS_SQL = f"""
CREATE TABLE {_SESSIONS_MIGRATION_TABLE} (
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    character_id INTEGER NOT NULL,
    worldbook_ids TEXT NOT NULL,
    feishu_chat_id VARCHAR(256),
    user_id VARCHAR(256),
    persona_id INTEGER,
    user_name VARCHAR(256) NOT NULL,
    user_persona TEXT NOT NULL,
    messages TEXT,
    summary TEXT NOT NULL,
    summary_up_to INTEGER NOT NULL,
    backend_id INTEGER,
    status VARCHAR(32) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
    FOREIGN KEY(character_id) REFERENCES characters (id),
    FOREIGN KEY(persona_id) REFERENCES personas (id),
    FOREIGN KEY(backend_id) REFERENCES backends (id)
)
"""


async def _table_info(conn: AsyncConnection) -> dict[str, tuple]:
    rows = (await conn.execute(text("PRAGMA table_info(sessions)"))).fetchall()
    return {str(row[1]): tuple(row) for row in rows}


async def _sessions_sql(conn: AsyncConnection) -> str:
    value = (
        await conn.execute(
            text(
                "SELECT sql FROM sqlite_master "
                "WHERE type='table' AND name='sessions'"
            )
        )
    ).scalar_one_or_none()
    return str(value or "")


async def _migrate_sessions(conn: AsyncConnection) -> None:
    columns = await _table_info(conn)
    if not columns:
        return

    if "summary" not in columns:
        await conn.execute(
            text(
                "ALTER TABLE sessions "
                "ADD COLUMN summary TEXT NOT NULL DEFAULT ''"
            )
        )
    if "summary_up_to" not in columns:
        await conn.execute(
            text(
                "ALTER TABLE sessions "
                "ADD COLUMN summary_up_to INTEGER NOT NULL DEFAULT 0"
            )
        )

    columns = await _table_info(conn)
    messages_nullable = columns["messages"][3] == 0
    has_autoincrement = "AUTOINCREMENT" in (await _sessions_sql(conn)).upper()
    if not messages_nullable or not has_autoincrement:
        await conn.execute(text(f"DROP TABLE IF EXISTS {_SESSIONS_MIGRATION_TABLE}"))
        await conn.execute(text(_CREATE_SESSIONS_SQL))
        await conn.execute(
            text(
                f"""
                INSERT INTO {_SESSIONS_MIGRATION_TABLE} (
                    id, character_id, worldbook_ids, feishu_chat_id, user_id,
                    persona_id, user_name, user_persona, messages, summary,
                    summary_up_to, backend_id, status, created_at
                )
                SELECT
                    id, character_id, worldbook_ids, feishu_chat_id, user_id,
                    persona_id, user_name, user_persona, messages,
                    COALESCE(summary, ''), COALESCE(summary_up_to, 0),
                    backend_id, status, created_at
                FROM sessions
                """
            )
        )
        await conn.execute(text("DROP TABLE sessions"))
        await conn.execute(
            text(f"ALTER TABLE {_SESSIONS_MIGRATION_TABLE} RENAME TO sessions")
        )

    await conn.execute(
        text(
            "CREATE INDEX IF NOT EXISTS ix_sessions_feishu_chat_id "
            "ON sessions (feishu_chat_id)"
        )
    )
    violations = (await conn.execute(text("PRAGMA foreign_key_check"))).fetchall()
    if violations:
        raise RuntimeError("database foreign key check failed")


async def migrate_database(engine: AsyncEngine) -> None:
    """Idempotently migrate SQLite tables without exposing a partial rebuild."""
    async with engine.connect() as conn:
        foreign_keys = int(
            (await conn.execute(text("PRAGMA foreign_keys"))).scalar_one()
        )
        await conn.commit()
        if foreign_keys:
            await conn.execute(text("PRAGMA foreign_keys=OFF"))
            await conn.commit()
        try:
            await conn.exec_driver_sql("BEGIN IMMEDIATE")
            try:
                await _migrate_sessions(conn)
            except BaseException:
                await conn.rollback()
                raise
            else:
                await conn.commit()
        finally:
            if conn.in_transaction():
                await conn.rollback()
            await conn.execute(
                text(f"PRAGMA foreign_keys={'ON' if foreign_keys else 'OFF'}")
            )
            await conn.commit()
