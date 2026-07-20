from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)

from app import main
from app.models.tables import Base, Session
from app.services.md_store import ChatRecord, MarkdownMemoryStore
from app.services.session_creation import (
    SessionInitializationError,
    create_session_with_markdown,
)


async def _legacy_engine(path: Path, *, session_id: int = 5) -> AsyncEngine:
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    async with engine.connect() as conn:
        await conn.execute(text("PRAGMA foreign_keys=ON"))
        await conn.commit()
    async with engine.begin() as conn:
        await conn.execute(text("CREATE TABLE characters (id INTEGER PRIMARY KEY)"))
        await conn.execute(text("CREATE TABLE personas (id INTEGER PRIMARY KEY)"))
        await conn.execute(text("CREATE TABLE backends (id INTEGER PRIMARY KEY)"))
        await conn.execute(text("INSERT INTO characters (id) VALUES (1)"))
        await conn.execute(
            text(
                """
                CREATE TABLE sessions (
                    id INTEGER NOT NULL,
                    character_id INTEGER NOT NULL,
                    worldbook_ids TEXT NOT NULL,
                    feishu_chat_id VARCHAR(256),
                    user_id VARCHAR(256),
                    persona_id INTEGER,
                    user_name VARCHAR(256) NOT NULL,
                    user_persona TEXT NOT NULL,
                    messages TEXT NOT NULL,
                    backend_id INTEGER,
                    status VARCHAR(32) NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
                    PRIMARY KEY (id),
                    FOREIGN KEY(character_id) REFERENCES characters (id),
                    FOREIGN KEY(persona_id) REFERENCES personas (id),
                    FOREIGN KEY(backend_id) REFERENCES backends (id)
                )
                """
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX ix_sessions_feishu_chat_id "
                "ON sessions (feishu_chat_id)"
            )
        )
        await conn.execute(
            text(
                "CREATE TABLE session_children ("
                "id INTEGER PRIMARY KEY, session_id INTEGER, "
                "FOREIGN KEY(session_id) REFERENCES sessions (id))"
            )
        )
        await conn.execute(
            text(
                """
                INSERT INTO sessions (
                    id, character_id, worldbook_ids, feishu_chat_id, user_id,
                    persona_id, user_name, user_persona, messages, backend_id,
                    status
                ) VALUES (
                    :session_id, 1, '[]', 'chat-old', 'user-old', NULL, '旧用户', '',
                    :messages, NULL, 'active'
                )
                """
            ),
            {
                "session_id": session_id,
                "messages": '[{"role":"assistant","content":"LEGACY"}]',
            },
        )
        await conn.execute(
            text(
                "INSERT INTO session_children (id, session_id) "
                "VALUES (1, :session_id)"
            ),
            {"session_id": session_id},
        )
    return engine


async def _assert_sessions_schema(engine: AsyncEngine) -> None:
    async with engine.connect() as conn:
        columns = {
            row[1]: row
            for row in (
                await conn.execute(text("PRAGMA table_info(sessions)"))
            ).fetchall()
        }
        assert columns["messages"][3] == 0
        assert "summary" in columns
        assert "summary_up_to" in columns

        table_sql = (
            await conn.execute(
                text(
                    "SELECT sql FROM sqlite_master "
                    "WHERE type='table' AND name='sessions'"
                )
            )
        ).scalar_one()
        assert "AUTOINCREMENT" in table_sql.upper()

        indexes = {
            row[1]
            for row in (
                await conn.execute(text("PRAGMA index_list(sessions)"))
            ).fetchall()
        }
        assert "ix_sessions_feishu_chat_id" in indexes
        assert (
            await conn.execute(text("PRAGMA foreign_key_check"))
        ).fetchall() == []


@pytest.mark.asyncio
async def test_startup_migration_rebuilds_legacy_sessions_without_losing_v1_history(
    tmp_path: Path,
) -> None:
    engine = await _legacy_engine(tmp_path / "legacy.db")
    try:
        await main._migrate_database(engine)
        await main._migrate_database(engine)
        await _assert_sessions_schema(engine)

        async with engine.begin() as conn:
            migrated = (
                await conn.execute(
                    text(
                        "SELECT messages, summary, summary_up_to "
                        "FROM sessions WHERE id=5"
                    )
                )
            ).one()
            assert migrated == (
                '[{"role":"assistant","content":"LEGACY"}]',
                "",
                0,
            )

            first = await conn.execute(
                text(
                    """
                    INSERT INTO sessions (
                        character_id, worldbook_ids, user_name, user_persona,
                        messages, summary, summary_up_to, status
                    ) VALUES (1, '[]', '新用户', '', NULL, '', 0, 'active')
                    """
                )
            )
            first_id = first.lastrowid
            assert first_id is not None
            await conn.execute(
                text("DELETE FROM sessions WHERE id=:id"),
                {"id": first_id},
            )
            second = await conn.execute(
                text(
                    """
                    INSERT INTO sessions (
                        character_id, worldbook_ids, user_name, user_persona,
                        messages, summary, summary_up_to, status
                    ) VALUES (1, '[]', '再建用户', '', NULL, '', 0, 'active')
                    """
                )
            )
            assert second.lastrowid is not None
            assert second.lastrowid > first_id
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_startup_migration_is_idempotent_for_fresh_schema(
    tmp_path: Path,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'fresh.db'}")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        await main._migrate_database(engine)
        await main._migrate_database(engine)
        await _assert_sessions_schema(engine)

        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO characters "
                    "(name, description, personality, scenario, first_message, "
                    "example_dialogues, system_prompt, creator_notes, tags, "
                    "linked_worldbook_ids, source) VALUES "
                    "('角色', '', '', '', '', '', '', '', '[]', '[]', 'manual')"
                )
            )
            await conn.execute(
                text(
                    """
                    INSERT INTO sessions (
                        character_id, worldbook_ids, user_name, user_persona,
                        messages, summary, summary_up_to, status
                    ) VALUES (1, '[]', '用户', '', NULL, '', 0, 'active')
                    """
                )
            )
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_migrated_autoincrement_keeps_new_session_away_from_orphan_chat(
    tmp_path: Path,
) -> None:
    engine = await _legacy_engine(tmp_path / "orphan.db")
    store = MarkdownMemoryStore(tmp_path / "memory")
    try:
        await main._migrate_database(engine)
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM session_children WHERE session_id=5"))
            await conn.execute(text("DELETE FROM sessions WHERE id=5"))
        await store.append_pair(
            5,
            "孤儿问题",
            "孤儿回答",
            char_name="旧角色",
            user_name="旧用户",
        )

        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as db:
            session = Session(
                character_id=1,
                worldbook_ids="[]",
                user_name="新用户",
                user_persona="",
                messages=None,
                summary="",
                summary_up_to=0,
                status="active",
            )
            await create_session_with_markdown(
                db,
                store,
                session,
                [
                    ChatRecord(
                        0,
                        "assistant",
                        "新会话",
                        "2026-07-21 00:00",
                        "新角色",
                    )
                ],
            )

        assert session.id > 5
        assert [record.content for record in await store.load_chat(5)] == [
            "孤儿问题",
            "孤儿回答",
        ]
        assert [
            record.content for record in await store.load_chat(session.id)
        ] == ["新会话"]
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_failed_sessions_rebuild_rolls_back_original_schema_and_data(
    tmp_path: Path,
) -> None:
    engine = await _legacy_engine(tmp_path / "rollback.db")
    try:
        async with engine.connect() as conn:
            await conn.execute(text("PRAGMA foreign_keys=OFF"))
            await conn.commit()
            await conn.execute(
                text(
                    "INSERT INTO session_children (id, session_id) "
                    "VALUES (2, 999)"
                )
            )
            await conn.commit()

        with pytest.raises(RuntimeError, match="foreign key check"):
            await main._migrate_database(engine)

        async with engine.connect() as conn:
            columns = {
                row[1]: row
                for row in (
                    await conn.execute(text("PRAGMA table_info(sessions)"))
                ).fetchall()
            }
            assert columns["messages"][3] == 1
            assert "summary" not in columns
            assert "summary_up_to" not in columns
            assert (
                await conn.execute(
                    text("SELECT messages FROM sessions WHERE id=5")
                )
            ).scalar_one() == '[{"role":"assistant","content":"LEGACY"}]'
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_preupgrade_orphan_above_db_max_does_not_permanently_block_creation(
    tmp_path: Path,
) -> None:
    engine = await _legacy_engine(tmp_path / "preupgrade-orphan.db", session_id=4)
    store = MarkdownMemoryStore(tmp_path / "memory-preupgrade")
    await store.append_pair(
        5,
        "升级前孤儿问题",
        "升级前孤儿回答",
        char_name="旧角色",
        user_name="旧用户",
    )
    for ignored in ("05", "0", "-1", "not-a-session"):
        (store.base / ignored).mkdir()
    (store.base / "999").write_text("not a directory", encoding="utf-8")

    try:
        await main._migrate_database(engine)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        created: Session | None = None
        for attempt in range(2):
            async with factory() as db:
                candidate = Session(
                    character_id=1,
                    worldbook_ids="[]",
                    user_name=f"新用户{attempt}",
                    user_persona="",
                    messages=None,
                    summary="",
                    summary_up_to=0,
                    status="active",
                )
                try:
                    await create_session_with_markdown(
                        db,
                        store,
                        candidate,
                        [
                            ChatRecord(
                                0,
                                "assistant",
                                "新会话",
                                "2026-07-21 00:00",
                                "新角色",
                            )
                        ],
                    )
                except SessionInitializationError:
                    continue
                created = candidate
                break

        assert created is not None
        assert created.id == 6
        assert [record.content for record in await store.load_chat(5)] == [
            "升级前孤儿问题",
            "升级前孤儿回答",
        ]
        assert [record.content for record in await store.load_chat(6)] == [
            "新会话"
        ]
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_markdown_sequence_floor_never_lowers_higher_database_sequence(
    tmp_path: Path,
) -> None:
    engine = await _legacy_engine(tmp_path / "high-sequence.db", session_id=4)
    store = MarkdownMemoryStore(tmp_path / "memory-high-sequence")
    await store.append_pair(
        5,
        "低位孤儿",
        "低位回答",
        char_name="旧角色",
        user_name="旧用户",
    )
    try:
        await main._migrate_database(engine)
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    """
                    INSERT INTO sessions (
                        id, character_id, worldbook_ids, user_name, user_persona,
                        messages, summary, summary_up_to, status
                    ) VALUES (30, 1, '[]', '高位用户', '', NULL, '', 0, 'active')
                    """
                )
            )
            await conn.execute(text("DELETE FROM sessions WHERE id=30"))

        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as db:
            session = Session(
                character_id=1,
                worldbook_ids="[]",
                user_name="新用户",
                user_persona="",
                messages=None,
                summary="",
                summary_up_to=0,
                status="active",
            )
            await create_session_with_markdown(db, store, session, [])

        assert session.id == 31
        async with engine.connect() as conn:
            sequence = (
                await conn.execute(
                    text("SELECT seq FROM sqlite_sequence WHERE name='sessions'")
                )
            ).scalar_one()
            assert sequence == 31
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_post_scan_collision_fails_once_then_next_request_advances(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = await _legacy_engine(tmp_path / "late-orphan.db", session_id=4)
    store = MarkdownMemoryStore(tmp_path / "memory-late-orphan")
    original_create_chat = store.create_chat
    injected = False

    async def collide_once(session_id, records):
        nonlocal injected
        if not injected:
            injected = True
            await original_create_chat(
                session_id,
                [
                    ChatRecord(
                        0,
                        "assistant",
                        "扫描后孤儿",
                        "2026-07-21 00:00",
                        "旧角色",
                    )
                ],
            )
        await original_create_chat(session_id, records)

    monkeypatch.setattr(store, "create_chat", collide_once)
    try:
        await main._migrate_database(engine)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        first = Session(
            character_id=1,
            worldbook_ids="[]",
            user_name="首次",
            user_persona="",
            messages=None,
            summary="",
            summary_up_to=0,
            status="active",
        )
        async with factory() as db:
            with pytest.raises(SessionInitializationError):
                await create_session_with_markdown(db, store, first, [])

        second = Session(
            character_id=1,
            worldbook_ids="[]",
            user_name="再次",
            user_persona="",
            messages=None,
            summary="",
            summary_up_to=0,
            status="active",
        )
        async with factory() as db:
            await create_session_with_markdown(db, store, second, [])

        assert first.id == 5
        assert second.id == 6
        assert [record.content for record in await store.load_chat(5)] == [
            "扫描后孤儿"
        ]
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_concurrent_shared_creates_get_distinct_ids_above_orphan_floor(
    tmp_path: Path,
) -> None:
    engine = await _legacy_engine(tmp_path / "concurrent.db", session_id=4)
    store = MarkdownMemoryStore(tmp_path / "memory-concurrent")
    await store.create_chat(5, [])
    try:
        await main._migrate_database(engine)
        factory = async_sessionmaker(engine, expire_on_commit=False)

        async def create_one(name: str) -> int:
            async with factory() as db:
                session = Session(
                    character_id=1,
                    worldbook_ids="[]",
                    user_name=name,
                    user_persona="",
                    messages=None,
                    summary="",
                    summary_up_to=0,
                    status="active",
                )
                await create_session_with_markdown(db, store, session, [])
                return session.id

        ids = await asyncio.gather(create_one("并发一"), create_one("并发二"))

        assert sorted(ids) == [6, 7]
        assert store.file_path(5, "chat.md").exists()
    finally:
        await engine.dispose()
