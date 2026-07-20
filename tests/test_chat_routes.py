from __future__ import annotations

import asyncio
import datetime as dt
import json
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

from app import main
from app.models.tables import Character, Session
from app.routers import feishu, sessions
from app.schemas.api import SessionCreate, SessionMessageIn
from app.services import memory
from app.services.chat_service import ChatService, DomainContext, TurnRequest
from app.services.md_store import (
    ChatRecord,
    InvalidationIntentError,
    MarkdownMemoryStore,
    MemoryState,
)
from app.services.prompt_policy import REQUIRED_OPENING


class Manager:
    def __init__(self) -> None:
        self.submitted: list[int] = []

    async def submit(self, session_id: int) -> None:
        self.submitted.append(session_id)


class ScalarResult:
    def __init__(self, value: Any) -> None:
        self.value = value

    def scalar_one_or_none(self):
        return self.value

    def scalars(self):
        values = self.value if isinstance(self.value, list) else [self.value]
        return SimpleNamespace(all=lambda: values)


class DB:
    def __init__(self, session: Session, character: Character) -> None:
        self.session = session
        self.character = character
        self.commits = 0

    async def get(self, model, object_id):
        if model is Session:
            return self.session if object_id == self.session.id else None
        if model is Character:
            return self.character if object_id == self.character.id else None
        return None

    async def execute(self, statement):
        return ScalarResult(self.session)

    async def commit(self):
        self.commits += 1

    async def refresh(self, value):
        return None


class AmbiguousCreateDB:
    def __init__(
        self,
        character: Character,
        *,
        session_id: int,
        failure_stage: str,
        failure: BaseException,
    ) -> None:
        self.character = character
        self.session_id = session_id
        self.failure_stage = failure_stage
        self.failure = failure
        self.added: Session | None = None
        self.rows: dict[int, Session] = {}
        self.deleted: Session | None = None
        self.commits = 0
        self.rollback_calls = 0

    async def get(self, model, object_id):
        if model is Character and object_id == self.character.id:
            return self.character
        if model is Session:
            return self.rows.get(object_id)
        return None

    async def execute(self, statement):
        return ScalarResult([])

    def add(self, session):
        self.added = session

    async def flush(self):
        assert self.added is not None
        self.added.id = self.session_id
        self.added.created_at = dt.datetime(2026, 7, 20)

    async def commit(self):
        self.commits += 1
        assert self.added is not None
        if self.deleted is not None:
            self.rows.pop(self.deleted.id, None)
        else:
            self.rows[self.added.id] = self.added
        if self.commits == 1 and self.failure_stage == "commit":
            raise self.failure

    async def refresh(self, value):
        if self.failure_stage == "refresh":
            raise self.failure

    async def rollback(self):
        self.rollback_calls += 1

    async def delete(self, value):
        self.deleted = value


def entities() -> tuple[Session, Character]:
    session = Session(
        id=1,
        character_id=2,
        worldbook_ids="[]",
        user_name="逸山",
        user_persona="",
        messages="[]",
        summary="",
        status="active",
        created_at=dt.datetime(2026, 7, 20),
    )
    character = Character(
        id=2,
        name="灵汐",
        description="",
        personality="",
        scenario="",
        first_message="",
        example_dialogues="EXAMPLES",
        system_prompt="",
    )
    return session, character


@pytest.mark.asyncio
async def test_create_session_with_first_message_seeds_complete_markdown_history(
    route_env,
) -> None:
    _, character = entities()
    character.first_message = "你好，{{user}}。"

    class CreateDB:
        def __init__(self) -> None:
            self.session: Session | None = None
            self.committed = False

        async def get(self, model, object_id):
            return character if model is Character else None

        async def execute(self, statement):
            return ScalarResult([])

        def add(self, session):
            self.session = session

        async def flush(self):
            assert self.session is not None
            session = self.session
            session.id = 9
            session.created_at = dt.datetime(2026, 7, 20)

        async def commit(self):
            assert self.session is not None
            if self.session.id is None:
                self.session.id = 9
                self.session.created_at = dt.datetime(2026, 7, 20)
            self.committed = True

        async def refresh(self, value):
            return None

    db = CreateDB()
    response = await sessions.create_session(
        SessionCreate(character_id=character.id, user_name="逸山"), db
    )

    assert db.committed
    assert db.session is not None
    assert db.session.messages is None
    assert [(message.role, message.content) for message in response.messages] == [
        ("assistant", "你好，逸山。"),
        ("user", "(OOC: 确认开始)"),
        (
            "assistant",
            "背景已经收到并通过校验，我将按照故事背景展开叙事。请开始你的行动。",
        ),
    ]
    assert [
        (record.role, record.content)
        for record in await route_env.store.load_chat(9)
    ] == [
        ("assistant", "你好，逸山。"),
        ("user", "(OOC: 确认开始)"),
        (
            "assistant",
            "背景已经收到并通过校验，我将按照故事背景展开叙事。请开始你的行动。",
        ),
    ]


@pytest.mark.asyncio
async def test_create_session_rolls_back_when_markdown_seed_fails(
    route_env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, character = entities()
    character.first_message = "你好，{{user}}。"

    class CreateDB:
        def __init__(self) -> None:
            self.added: Session | None = None
            self.persisted: list[Session] = []
            self.rolled_back = False

        async def get(self, model, object_id):
            return character if model is Character else None

        async def execute(self, statement):
            return ScalarResult([])

        def add(self, session):
            self.added = session

        async def flush(self):
            assert self.added is not None
            self.added.id = 10
            self.added.created_at = dt.datetime(2026, 7, 20)

        async def commit(self):
            assert self.added is not None
            if self.added.id is None:
                self.added.id = 10
            self.persisted.append(self.added)

        async def rollback(self):
            self.rolled_back = True

        async def refresh(self, value):
            return None

    async def fail_seed(*args, **kwargs):
        raise RuntimeError("secret markdown failure")

    monkeypatch.setattr(route_env.store, "create_chat", fail_seed, raising=False)
    db = CreateDB()

    with pytest.raises(HTTPException) as captured:
        await sessions.create_session(
            SessionCreate(character_id=character.id, user_name="逸山"), db
        )

    assert captured.value.status_code == 500
    assert captured.value.detail == "Failed to initialize session"
    assert "secret" not in captured.value.detail
    assert db.rolled_back
    assert db.persisted == []
    assert db.added is not None
    assert db.added.messages is None
    assert not route_env.store.file_path(10, "chat.md").exists()


@pytest.mark.asyncio
async def test_create_only_markdown_seed_never_reuses_or_deletes_existing_chat(
    route_env,
) -> None:
    await route_env.store.append_pair(
        14,
        "已有问题",
        "已有回答",
        char_name="已有角色",
        user_name="已有用户",
    )

    with pytest.raises(FileExistsError):
        await route_env.store.create_chat(
            14,
            [
                ChatRecord(
                    number=0,
                    role="assistant",
                    content="不得采用的新历史",
                    timestamp="2026-07-20 00:00",
                    name="新角色",
                )
            ],
        )

    assert [record.content for record in await route_env.store.load_chat(14)] == [
        "已有问题",
        "已有回答",
    ]


@pytest.mark.asyncio
async def test_create_session_removes_seeded_markdown_when_commit_fails(
    route_env,
) -> None:
    _, character = entities()
    character.first_message = "你好，{{user}}。"

    class CreateDB:
        def __init__(self) -> None:
            self.added: Session | None = None
            self.rolled_back = False

        async def get(self, model, object_id):
            return character if model is Character else None

        async def execute(self, statement):
            return ScalarResult([])

        def add(self, session):
            self.added = session

        async def flush(self):
            assert self.added is not None
            self.added.id = 11

        async def commit(self):
            raise RuntimeError("secret commit failure")

        async def rollback(self):
            self.rolled_back = True

    db = CreateDB()

    with pytest.raises(HTTPException) as captured:
        await sessions.create_session(
            SessionCreate(character_id=character.id, user_name="逸山"), db
        )

    assert captured.value.status_code == 500
    assert captured.value.detail == "Failed to initialize session"
    assert db.rolled_back
    assert not route_env.store.file_path(11, "chat.md").exists()


@pytest.mark.asyncio
async def test_create_session_removes_committed_row_and_seed_when_refresh_fails(
    route_env,
) -> None:
    _, character = entities()
    character.first_message = "你好，{{user}}。"

    class CreateDB:
        def __init__(self) -> None:
            self.added: Session | None = None
            self.persisted: list[Session] = []
            self.deleted = False

        async def get(self, model, object_id):
            if model is Character:
                return character
            if model is Session and self.persisted:
                return self.persisted[0]
            return None

        async def execute(self, statement):
            return ScalarResult([])

        def add(self, session):
            self.added = session

        async def flush(self):
            assert self.added is not None
            self.added.id = 12

        async def commit(self):
            if self.deleted:
                self.persisted.clear()
            else:
                assert self.added is not None
                self.persisted[:] = [self.added]

        async def refresh(self, value):
            raise RuntimeError("secret refresh failure")

        async def delete(self, value):
            assert value is self.added
            self.deleted = True

        async def rollback(self):
            return None

    db = CreateDB()

    with pytest.raises(HTTPException) as captured:
        await sessions.create_session(
            SessionCreate(character_id=character.id, user_name="逸山"), db
        )

    assert captured.value.status_code == 500
    assert captured.value.detail == "Failed to initialize session"
    assert db.deleted
    assert db.persisted == []
    assert not route_env.store.file_path(12, "chat.md").exists()


@pytest.mark.asyncio
async def test_create_session_rejects_preexisting_markdown_for_reused_id(
    route_env,
) -> None:
    _, character = entities()
    character.first_message = "你好，{{user}}。"
    await route_env.store.append_pair(
        13,
        "孤儿旧问题",
        "孤儿旧回答",
        char_name="旧角色",
        user_name="旧用户",
    )

    class CreateDB:
        def __init__(self) -> None:
            self.added: Session | None = None
            self.committed = False
            self.rolled_back = False

        async def get(self, model, object_id):
            return character if model is Character else None

        async def execute(self, statement):
            return ScalarResult([])

        def add(self, session):
            self.added = session

        async def flush(self):
            assert self.added is not None
            self.added.id = 13
            self.added.created_at = dt.datetime(2026, 7, 20)

        async def commit(self):
            self.committed = True

        async def rollback(self):
            self.rolled_back = True

        async def refresh(self, value):
            return None

    db = CreateDB()

    with pytest.raises(HTTPException) as captured:
        await sessions.create_session(
            SessionCreate(character_id=character.id, user_name="逸山"), db
        )

    assert captured.value.status_code == 500
    assert captured.value.detail == "Failed to initialize session"
    assert not db.committed
    assert db.rolled_back
    assert [record.content for record in await route_env.store.load_chat(13)] == [
        "孤儿旧问题",
        "孤儿旧回答",
    ]


@pytest.mark.asyncio
async def test_failed_db_compensation_never_deletes_preexisting_chat_or_logs_secrets(
    route_env,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _, character = entities()
    character.first_message = "你好，{{user}}。"
    await route_env.store.append_pair(
        14,
        "预存问题",
        "预存回答",
        char_name="旧角色",
        user_name="旧用户",
    )

    class CleanupFailureDB:
        def __init__(self) -> None:
            self.session: Session | None = None

        async def get(self, model, object_id):
            if model is Character:
                return character
            if model is Session:
                raise RuntimeError("secret lookup failure")
            return None

        async def execute(self, statement):
            return ScalarResult([])

        def add(self, session):
            self.session = session

        async def flush(self):
            assert self.session is not None
            self.session.id = 14

        async def rollback(self):
            raise RuntimeError("secret rollback failure")

    with pytest.raises(HTTPException) as captured:
        await sessions.create_session(
            SessionCreate(character_id=character.id, user_name="逸山"),
            CleanupFailureDB(),
        )

    assert captured.value.detail == "Failed to initialize session"
    assert [record.content for record in await route_env.store.load_chat(14)] == [
        "预存问题",
        "预存回答",
    ]
    assert "secret lookup failure" not in caplog.text
    assert "secret rollback failure" not in caplog.text


@pytest.mark.asyncio
async def test_create_session_compensates_when_commit_persists_then_raises(
    route_env,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _, character = entities()
    character.first_message = "你好，{{user}}。"
    db = AmbiguousCreateDB(
        character,
        session_id=15,
        failure_stage="commit",
        failure=RuntimeError("secret commit acknowledgement failure"),
    )

    with pytest.raises(HTTPException) as captured:
        await sessions.create_session(
            SessionCreate(character_id=character.id, user_name="逸山"), db
        )

    assert captured.value.status_code == 500
    assert captured.value.detail == "Failed to initialize session"
    assert db.rows == {}
    assert db.rollback_calls >= 1
    assert not route_env.store.file_path(15, "chat.md").exists()
    assert "secret commit acknowledgement failure" not in caplog.text


@pytest.mark.parametrize("failure_stage", ["commit", "refresh"])
@pytest.mark.asyncio
async def test_create_session_cancellation_completes_compensation_before_propagating(
    route_env,
    failure_stage: str,
) -> None:
    _, character = entities()
    character.first_message = "你好，{{user}}。"
    session_id = 16 if failure_stage == "commit" else 17
    db = AmbiguousCreateDB(
        character,
        session_id=session_id,
        failure_stage=failure_stage,
        failure=asyncio.CancelledError(),
    )

    with pytest.raises(asyncio.CancelledError):
        await sessions.create_session(
            SessionCreate(character_id=character.id, user_name="逸山"), db
        )

    assert db.rows == {}
    assert db.rollback_calls >= 1
    assert not route_env.store.file_path(session_id, "chat.md").exists()


@pytest.fixture
def route_env(tmp_path, monkeypatch):
    store = MarkdownMemoryStore(tmp_path)
    manager = Manager()
    monkeypatch.setattr(memory, "md_store", store)
    monkeypatch.setattr(main, "memory_task_manager", manager)
    backend = {
        "provider": "openai",
        "api_key": "test",
        "model": "model",
        "base_url": "https://invalid.local",
        "params": {},
    }

    async def resolve_backend(backend_id, db):
        return backend

    async def empty(*args, **kwargs):
        return ""

    monkeypatch.setattr(sessions, "_resolve_backend", resolve_backend)
    monkeypatch.setattr(sessions, "load_memory_relevant", empty)
    monkeypatch.setattr(sessions, "load_assets_summary", empty)
    monkeypatch.setattr(sessions, "load_assets", empty)
    monkeypatch.setattr(sessions, "load_mentioned_character_profiles", empty)
    monkeypatch.setattr(sessions, "load_summary", empty)
    monkeypatch.setattr(sessions, "is_asset_relevant", lambda content: False)

    usage_calls: list[dict[str, Any]] = []

    async def usage(**kwargs):
        usage_calls.append(kwargs)

    monkeypatch.setattr(sessions, "record_usage", usage)
    monkeypatch.setattr(feishu, "record_usage", usage)
    return SimpleNamespace(
        store=store,
        manager=manager,
        backend=backend,
        usage_calls=usage_calls,
    )


@pytest.mark.asyncio
async def test_sessions_rest_delegates_guarded_turn_and_records_combined_usage(
    route_env, monkeypatch
) -> None:
    session, character = entities()
    db = DB(session, character)
    calls = 0

    async def complete(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {
                "content": "I cannot continue this story.",
                "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            }
        return {
            "content": "正文",
            "usage": {"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18},
        }

    monkeypatch.setattr(sessions, "chat_completion", complete)

    response = await sessions.send_message(1, SessionMessageIn(content="继续。"), db)

    assert calls == 2
    assert response.messages[-1].content == f"{REQUIRED_OPENING}\n\n正文"
    assert route_env.manager.submitted == [1]
    assert route_env.usage_calls[0]["usage"] == {
        "prompt_tokens": 9,
        "completion_tokens": 14,
        "total_tokens": 23,
    }


@pytest.mark.asyncio
async def test_sessions_retry_delegates_atomic_replacement_and_records_usage(
    route_env, monkeypatch
) -> None:
    session, character = entities()
    db = DB(session, character)
    await route_env.store.append_pair(
        1, "retry user", "old answer", char_name="灵汐", user_name="逸山"
    )
    await route_env.store.write_text(1, "story_state.md", "stale retry state")

    async def complete(**kwargs):
        return {"content": "new answer", "usage": {"total_tokens": 9}}

    monkeypatch.setattr(sessions, "chat_completion", complete)

    response = await sessions.send_message(
        1, SessionMessageIn(content="/retry"), db
    )

    assert [(message.role, message.content) for message in response.messages] == [
        ("user", "retry user"),
        ("assistant", f"{REQUIRED_OPENING}\n\nnew answer"),
    ]
    assert route_env.manager.submitted == [1]
    assert route_env.usage_calls == [
        {
            "session_id": 1,
            "user_id": session.user_id,
            "character_name": "灵汐",
            "model": "model",
            "usage": {"total_tokens": 9},
        }
    ]
    assert await route_env.store.read_text(1, "story_state.md") == ""


@pytest.mark.asyncio
async def test_sessions_undo_delegates_invalidation_and_pipeline_submit(
    route_env,
) -> None:
    session, character = entities()
    db = DB(session, character)
    await route_env.store.append_pair(
        1, "undo user", "undo answer", char_name="灵汐", user_name="逸山"
    )
    await route_env.store.write_text(1, "summary.md", "stale undo summary")

    response = await sessions.send_message(1, SessionMessageIn(content="/undo"), db)

    assert response.messages == []
    assert route_env.manager.submitted == [1]
    assert await route_env.store.read_text(1, "summary.md") == ""


@pytest.mark.asyncio
async def test_sessions_undo_imports_db_only_v1_history_before_locking_turn(
    route_env,
) -> None:
    session, character = entities()
    legacy = [
        {"role": "user", "content": "V1 undo user"},
        {"role": "assistant", "content": "V1 undo answer"},
    ]
    session.messages = json.dumps(legacy)
    original_legacy = session.messages

    response = await sessions.send_message(
        session.id,
        SessionMessageIn(content="/undo"),
        DB(session, character),
    )

    assert response.messages == []
    assert session.messages == original_legacy
    assert route_env.store.file_path(session.id, "chat.md").exists()
    assert await route_env.store.load_chat(session.id) == []


@pytest.mark.asyncio
async def test_sessions_retry_imports_db_only_v1_history_before_locking_turn(
    route_env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, character = entities()
    legacy = [
        {"role": "user", "content": "V1 retry user"},
        {"role": "assistant", "content": "V1 retry answer"},
    ]
    session.messages = json.dumps(legacy)
    original_legacy = session.messages

    async def complete(**kwargs):
        return {"content": "V1 retried", "usage": {}}

    monkeypatch.setattr(sessions, "chat_completion", complete)

    response = await sessions.send_message(
        session.id,
        SessionMessageIn(content="/retry"),
        DB(session, character),
    )

    assert [(message.role, message.content) for message in response.messages] == [
        ("user", "V1 retry user"),
        ("assistant", f"{REQUIRED_OPENING}\n\nV1 retried"),
    ]
    assert session.messages == original_legacy


@pytest.mark.parametrize("command", ["/undo", "/retry"])
@pytest.mark.asyncio
async def test_v1_mutating_commands_ignore_malformed_fallback_when_chat_exists(
    route_env,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    session, character = entities()
    session.messages = "{malformed legacy json"
    await route_env.store.append_pair(
        session.id,
        "MD user",
        "MD answer",
        char_name=character.name,
        user_name=session.user_name,
    )

    async def complete(**kwargs):
        return {"content": "MD retried", "usage": {}}

    monkeypatch.setattr(sessions, "chat_completion", complete)

    response = await sessions.send_message(
        session.id,
        SessionMessageIn(content=command),
        DB(session, character),
    )

    if command == "/undo":
        assert response.messages == []
    else:
        assert response.messages[-2].content == "MD user"
        assert response.messages[-1].content.endswith("MD retried")
    assert session.messages == "{malformed legacy json"


@pytest.mark.parametrize(
    ("command", "history"),
    [
        ("/undo", "empty"),
        ("/undo", "incomplete"),
        ("/retry", "empty"),
        ("/retry", "incomplete"),
    ],
)
@pytest.mark.asyncio
async def test_sessions_mutating_command_maps_only_missing_pair_to_400(
    route_env,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    history: str,
) -> None:
    session, character = entities()
    db = DB(session, character)
    if history == "incomplete":
        await route_env.store.append_record(1, "user", "incomplete", name="User")

    async def fail_completion(**kwargs):
        raise AssertionError("completion must not run without a complete pair")

    monkeypatch.setattr(sessions, "chat_completion", fail_completion)

    with pytest.raises(HTTPException) as captured:
        await sessions.send_message(1, SessionMessageIn(content=command), db)

    assert captured.value.status_code == 400
    assert "No complete message pair" in captured.value.detail


@pytest.mark.parametrize("command", ["/undo", "/retry"])
@pytest.mark.asyncio
async def test_sessions_malformed_journal_is_not_mislabeled_as_missing_pair(
    route_env,
    command: str,
) -> None:
    session, character = entities()
    db = DB(session, character)
    await route_env.store.append_pair(
        1,
        "old user",
        "old assistant",
        char_name="Character",
        user_name="User",
    )
    malformed = "# Invalidation Intent\n\n<!-- boundary-message: 0 -->\n"
    await route_env.store.write_text(1, "invalidation_intent.md", malformed)

    if command == "/undo":
        with pytest.raises(HTTPException) as captured:
            await sessions.send_message(1, SessionMessageIn(content=command), db)
        assert captured.value.status_code == 500
        assert captured.value.detail == "Undo failed"
    else:
        with pytest.raises(InvalidationIntentError) as captured:
            await sessions.send_message(1, SessionMessageIn(content=command), db)
        assert str(captured.value) == "invalid invalidation intent"


@pytest.mark.asyncio
async def test_sessions_retry_resolves_original_user_under_the_turn_lock(
    route_env,
    monkeypatch,
) -> None:
    session, character = entities()
    db = DB(session, character)
    await route_env.store.append_pair(
        1, "first user", "first answer", char_name="灵汐", user_name="逸山"
    )
    backend_started = asyncio.Event()
    release_backend = asyncio.Event()
    relevance_queries: list[str] = []

    async def blocking_backend(backend_id, db):
        backend_started.set()
        await release_backend.wait()
        return route_env.backend

    async def relevant(session_id, content, recent):
        relevance_queries.append(content)
        return ""

    async def route_complete(**kwargs):
        return {"content": "retried answer", "usage": {}}

    async def concurrent_complete(messages):
        return {"content": "second answer", "usage": {}}

    monkeypatch.setattr(sessions, "_resolve_backend", blocking_backend)
    monkeypatch.setattr(sessions, "load_memory_relevant", relevant)
    monkeypatch.setattr(sessions, "chat_completion", route_complete)

    retry_task = asyncio.create_task(
        sessions.send_message(1, SessionMessageIn(content="/retry"), db)
    )
    await backend_started.wait()
    await ChatService(
        route_env.store,
        route_env.manager,
        completion=concurrent_complete,
    ).send(
        TurnRequest(
            session_id=1,
            content="second user",
            character={"name": "灵汐"},
            user_name="逸山",
        )
    )
    release_backend.set()
    await retry_task

    assert relevance_queries == ["second user"]
    assert [record.content for record in await route_env.store.load_chat(1)] == [
        "first user",
        "first answer",
        "second user",
        f"{REQUIRED_OPENING}\n\nretried answer",
    ]


@pytest.mark.asyncio
async def test_established_session_without_examples_does_not_load_history_seed(
    route_env, monkeypatch
) -> None:
    session, character = entities()
    character.example_dialogues = ""
    db = DB(session, character)
    await route_env.store.append_pair(
        1, "旧问题", "旧回答", char_name="灵汐", user_name="逸山"
    )
    seed_calls = 0

    async def seed(*args, **kwargs):
        nonlocal seed_calls
        seed_calls += 1
        return [{"role": "assistant", "content": "SHOULD_NOT_LOAD"}]

    async def complete(**kwargs):
        return {"content": "正文", "usage": {}}

    monkeypatch.setattr(sessions, "get_history_seed", seed)
    monkeypatch.setattr(sessions, "chat_completion", complete)

    await sessions.send_message(1, SessionMessageIn(content="继续。"), db)

    assert seed_calls == 0


@pytest.mark.asyncio
async def test_rest_lazily_imports_v1_database_history_before_new_turn(
    route_env,
    monkeypatch,
) -> None:
    session, character = entities()
    legacy = [
        {
            "role": "user",
            "content": "V1_USER_MARKER",
            "timestamp": "2026-07-01T10:11:12",
        },
        {
            "role": "assistant",
            "content": "V1_ASSISTANT_MARKER",
            "timestamp": "2026-07-01T10:12:13",
        },
    ]
    session.messages = json.dumps(legacy)
    original_database_history = session.messages
    captured: list[list[dict[str, Any]]] = []

    async def complete(**kwargs):
        captured.append(kwargs["messages"])
        return {"content": "新回答", "usage": {}}

    monkeypatch.setattr(sessions, "chat_completion", complete)

    await sessions.send_message(
        1,
        SessionMessageIn(content="继续旧故事。"),
        DB(session, character),
    )

    provider_text = "\n".join(
        str(message.get("content", "")) for message in captured[0]
    )
    assert "V1_USER_MARKER" in provider_text
    assert "V1_ASSISTANT_MARKER" in provider_text
    assert [record.content for record in await route_env.store.load_chat(1)] == [
        "V1_USER_MARKER",
        "V1_ASSISTANT_MARKER",
        "继续旧故事。",
        f"{REQUIRED_OPENING}\n\n新回答",
    ]
    assert session.messages == original_database_history


@pytest.mark.asyncio
async def test_existing_markdown_ignores_malformed_v1_database_fallback(
    route_env,
    monkeypatch,
) -> None:
    session, character = entities()
    session.messages = "{malformed legacy json"
    await route_env.store.append_pair(
        1,
        "MD_USER_MARKER",
        "MD_ASSISTANT_MARKER",
        char_name="灵汐",
        user_name="逸山",
    )
    captured: list[list[dict[str, Any]]] = []

    async def complete(**kwargs):
        captured.append(kwargs["messages"])
        return {"content": "新回答", "usage": {}}

    monkeypatch.setattr(sessions, "chat_completion", complete)

    await sessions.send_message(
        1,
        SessionMessageIn(content="继续。"),
        DB(session, character),
    )

    provider_text = "\n".join(
        str(message.get("content", "")) for message in captured[0]
    )
    assert "MD_USER_MARKER" in provider_text
    assert "MD_ASSISTANT_MARKER" in provider_text


@pytest.mark.parametrize("lite", [False, True])
@pytest.mark.asyncio
async def test_session_list_isolates_malformed_unrelated_v1_history(
    route_env,
    lite: bool,
) -> None:
    malformed, character = entities()
    malformed.id = 2
    malformed.messages = "{malformed legacy json"
    valid, _ = entities()
    valid.messages = json.dumps(
        [
            {"role": "user", "content": "VALID_V1_USER"},
            {"role": "assistant", "content": "VALID_V1_ASSISTANT"},
        ]
    )

    class ListDB:
        async def get(self, model, object_id):
            if model is Character and object_id == character.id:
                return character
            return None

        async def execute(self, statement):
            return ScalarResult([malformed, valid])

    response = await sessions.list_sessions(lite=lite, db=ListDB())

    if lite:
        payload = json.loads(response.body)
        by_id = {item["id"]: item for item in payload}
        assert by_id[2]["msg_count"] == 0
        assert by_id[1]["msg_count"] == 2
    else:
        by_id = {item.id: item for item in response}
        assert by_id[2].messages == []
        assert [message.content for message in by_id[1].messages] == [
            "VALID_V1_USER",
            "VALID_V1_ASSISTANT",
        ]


@pytest.mark.asyncio
async def test_new_session_without_examples_loads_history_seed_once(
    route_env, monkeypatch
) -> None:
    session, character = entities()
    character.example_dialogues = ""
    db = DB(session, character)
    seed_calls = 0
    received: list[list[dict[str, str]]] = []

    async def seed(*args, **kwargs):
        nonlocal seed_calls
        seed_calls += 1
        return [{"role": "assistant", "content": "SEED_MARKER"}]

    async def complete(**kwargs):
        received.append(kwargs["messages"])
        return {"content": "正文", "usage": {}}

    monkeypatch.setattr(sessions, "get_history_seed", seed)
    monkeypatch.setattr(sessions, "chat_completion", complete)

    await sessions.send_message(1, SessionMessageIn(content="继续。"), db)

    assert seed_calls == 1
    assert "SEED_MARKER" in "\n".join(
        message["content"] for message in received[0]
    )


@pytest.mark.asyncio
async def test_sessions_context_preparation_is_serialized_and_sees_prior_commit(
    route_env, monkeypatch
) -> None:
    session, character = entities()
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    histories: list[list[str]] = []
    active = 0
    max_active = 0

    async def blocking_memory(session_id, content, recent):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        histories.append([message["content"] for message in recent])
        try:
            if len(histories) == 1:
                first_started.set()
                await release_first.wait()
            return ""
        finally:
            active -= 1

    async def complete(**kwargs):
        return {"content": "正文", "usage": {}}

    monkeypatch.setattr(sessions, "load_memory_relevant", blocking_memory)
    monkeypatch.setattr(sessions, "chat_completion", complete)

    first = asyncio.create_task(
        sessions.send_message(
            1, SessionMessageIn(content="第一问"), DB(session, character)
        )
    )
    await first_started.wait()
    second = asyncio.create_task(
        sessions.send_message(
            1, SessionMessageIn(content="第二问"), DB(session, character)
        )
    )
    await asyncio.sleep(0)

    observed_before_release = max_active
    release_first.set()
    await asyncio.gather(first, second)

    assert observed_before_release == 1
    assert max_active == 1
    assert histories[1][-2:] == [
        "第一问",
        f"{REQUIRED_OPENING}\n\n正文",
    ]


@pytest.mark.asyncio
async def test_command_receipt_waits_for_turn_and_appends_one_complete_pair(
    route_env,
    monkeypatch,
) -> None:
    session, character = entities()
    loader_started = asyncio.Event()
    release_loader = asyncio.Event()

    async def blocking_memory(session_id, content, recent):
        loader_started.set()
        await release_loader.wait()
        return ""

    async def complete(**kwargs):
        return {"content": "普通回答", "usage": {}}

    monkeypatch.setattr(sessions, "load_memory_relevant", blocking_memory)
    monkeypatch.setattr(sessions, "chat_completion", complete)

    ordinary = asyncio.create_task(
        sessions.send_message(
            1,
            SessionMessageIn(content="普通问题"),
            DB(session, character),
        )
    )
    await loader_started.wait()
    command = asyncio.create_task(
        sessions.send_message(
            1,
            SessionMessageIn(content="/assets"),
            DB(session, character),
        )
    )
    await asyncio.sleep(0)

    assert not command.done()
    release_loader.set()
    await asyncio.gather(ordinary, command)

    records = await route_env.store.load_chat(1)
    assert [(record.role, record.content) for record in records] == [
        ("user", "普通问题"),
        ("assistant", f"{REQUIRED_OPENING}\n\n普通回答"),
        ("user", "/assets"),
        ("assistant", "还没有资产记录。资产会在对话中自动跟踪更新。"),
    ]


@pytest.mark.asyncio
async def test_remember_side_effect_and_receipt_wait_for_same_session_turn_lock(
    route_env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, character = entities()
    original_turn_lock_for = route_env.store.turn_lock_for
    turn_lock = await original_turn_lock_for(session.id)
    lock_requested = asyncio.Event()

    async def tracked_turn_lock_for(session_id: int):
        lock_requested.set()
        return await original_turn_lock_for(session_id)

    monkeypatch.setattr(route_env.store, "turn_lock_for", tracked_turn_lock_for)

    async with turn_lock:
        command = asyncio.create_task(
            sessions.send_message(
                session.id,
                SessionMessageIn(content="/remember 锁内记忆"),
                DB(session, character),
            )
        )
        await asyncio.wait_for(lock_requested.wait(), timeout=1)
        await asyncio.sleep(0)

        assert not command.done()
        assert await route_env.store.read_text(session.id, "memory.md") == ""
        assert await route_env.store.load_chat(session.id) == []

    response = await command

    assert "锁内记忆" in await route_env.store.read_text(session.id, "memory.md")
    assert [(record.role, record.content) for record in await route_env.store.load_chat(1)] == [
        ("user", "/remember 锁内记忆"),
        ("assistant", "✓ Remembered: 锁内记忆"),
    ]
    assert response.messages[-1].content == "✓ Remembered: 锁内记忆"


@pytest.mark.asyncio
async def test_command_append_failure_returns_safe_error_without_rolling_back_side_effect(
    route_env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, character = entities()

    async def fail_append(*args, **kwargs):
        raise RuntimeError("secret append failure")

    monkeypatch.setattr(route_env.store, "append_pair", fail_append)

    with pytest.raises(HTTPException) as captured:
        await sessions.send_message(
            session.id,
            SessionMessageIn(content="/remember 已发生的副作用"),
            DB(session, character),
        )

    assert captured.value.status_code == 500
    assert captured.value.detail == "Failed to record command"
    assert "secret" not in captured.value.detail
    assert "已发生的副作用" in await route_env.store.read_text(
        session.id, "memory.md"
    )
    assert await route_env.store.load_chat(session.id) == []


@pytest.mark.asyncio
async def test_summary_clear_clears_markdown_and_appends_one_complete_pair(
    route_env,
) -> None:
    session, character = entities()
    await route_env.store.write_text(session.id, "summary.md", "旧摘要")

    response = await sessions.send_message(
        session.id,
        SessionMessageIn(content="/summary clear"),
        DB(session, character),
    )

    assert await route_env.store.read_text(session.id, "summary.md") == ""
    assert [(record.role, record.content) for record in await route_env.store.load_chat(1)] == [
        ("user", "/summary clear"),
        ("assistant", "🗑️ 摘要已清空。"),
    ]
    assert response.messages[-2].content == "/summary clear"
    assert response.messages[-1].content == "🗑️ 摘要已清空。"


@pytest.mark.asyncio
async def test_command_turn_lock_does_not_block_a_different_session(
    route_env,
) -> None:
    first, character = entities()
    second, _ = entities()
    second.id = 2
    first_lock = await route_env.store.turn_lock_for(first.id)

    async with first_lock:
        response = await asyncio.wait_for(
            sessions.send_message(
                second.id,
                SessionMessageIn(content="/remember 第二会话"),
                DB(second, character),
            ),
            timeout=1,
        )

    assert response.messages[-1].content == "✓ Remembered: 第二会话"
    assert "第二会话" in await route_env.store.read_text(second.id, "memory.md")
    assert await route_env.store.load_chat(first.id) == []


@pytest.mark.asyncio
async def test_summary_update_uses_v2_semantics_and_preserves_valid_summary(
    route_env,
    monkeypatch,
) -> None:
    session, character = entities()
    await route_env.store.append_pair(
        1,
        "旧问题",
        "旧回答",
        char_name="灵汐",
        user_name="逸山",
    )
    await route_env.store.write_text(1, "summary.md", "应保留的整体剧情。\n")
    prompts: list[list[dict[str, Any]]] = []

    async def invalid_summary(**kwargs):
        prompts.append(kwargs["messages"])
        return {"content": "<script>剧情已完成</script>", "usage": {}}

    monkeypatch.setattr(memory, "chat_completion", invalid_summary)
    monkeypatch.setattr(sessions, "load_summary", memory.load_summary)

    response = await sessions.send_message(
        1,
        SessionMessageIn(content="/summary update"),
        DB(session, character),
    )

    assert "不判断剧情线或伏笔是否完成" in prompts[0][0]["content"]
    assert await route_env.store.read_text(1, "summary.md") == "应保留的整体剧情。\n"
    assert response.messages[-2].content == "/summary update"
    assert response.messages[-1].content.startswith("❌ 更新失败:")


@pytest.mark.asyncio
async def test_rag_provider_failure_log_is_sanitized(
    route_env, monkeypatch, caplog
) -> None:
    import app.services.rag as rag

    session, character = entities()
    db = DB(session, character)

    async def fail_index(session_id):
        raise RuntimeError("provider-secret response-body")

    async def complete(**kwargs):
        return {"content": "正文", "usage": {}}

    monkeypatch.setattr(rag, "load_index", fail_index)
    monkeypatch.setattr(sessions, "chat_completion", complete)

    await sessions.send_message(1, SessionMessageIn(content="继续。"), db)

    assert "provider-secret" not in caplog.text
    assert "response-body" not in caplog.text
    assert "RuntimeError" in caplog.text


@pytest.mark.asyncio
async def test_sessions_rest_maps_context_budget_failure_to_safe_413(
    route_env, monkeypatch
) -> None:
    session, character = entities()
    db = DB(session, character)
    monkeypatch.setattr("app.services.chat_service.settings.total_token_budget", 10)

    async def complete(**kwargs):
        raise AssertionError("provider must not run")

    monkeypatch.setattr(sessions, "chat_completion", complete)

    with pytest.raises(sessions.HTTPException) as exc_info:
        await sessions.send_message(
            1,
            SessionMessageIn(content="超长输入" * 100),
            db,
        )

    assert exc_info.value.status_code == 413
    assert "mandatory" not in str(exc_info.value.detail)
    assert await route_env.store.load_chat(1) == []
    assert route_env.manager.submitted == []


@pytest.mark.asyncio
async def test_sessions_stream_preserves_sse_schema_and_persists_before_done(
    route_env, monkeypatch
) -> None:
    session, character = entities()
    db = DB(session, character)
    monkeypatch.setattr("app.services.chat_service.settings.stream_guard_chars", 1)

    async def stream(**kwargs):
        yield "正文"
        yield {"usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}}

    monkeypatch.setattr(sessions, "chat_completion_stream", stream)

    response = await sessions.send_message_stream(
        1, SessionMessageIn(content="继续。"), db
    )
    chunks = [chunk async for chunk in response.body_iterator]
    body = "".join(
        chunk.decode() if isinstance(chunk, bytes) else chunk for chunk in chunks
    )

    assert '"delta"' in body
    assert REQUIRED_OPENING in body
    assert body.endswith("data: [DONE]\n\n")
    assert len(await route_env.store.load_chat(1)) == 2
    assert route_env.manager.submitted == [1]


@pytest.mark.asyncio
async def test_sessions_stream_disconnect_closes_provider_without_persisting(
    route_env, monkeypatch
) -> None:
    session, character = entities()
    db = DB(session, character)
    monkeypatch.setattr("app.services.chat_service.settings.stream_guard_chars", 1)
    provider_closed = asyncio.Event()

    async def stream(**kwargs):
        try:
            yield "正文"
            await asyncio.Event().wait()
        finally:
            provider_closed.set()

    monkeypatch.setattr(sessions, "chat_completion_stream", stream)
    response = await sessions.send_message_stream(
        1, SessionMessageIn(content="继续。"), db
    )

    first = await anext(response.body_iterator)
    assert REQUIRED_OPENING in first
    await response.body_iterator.aclose()

    assert provider_closed.is_set()
    assert await route_env.store.load_chat(1) == []
    assert route_env.manager.submitted == []


def feishu_event(text: str) -> dict[str, Any]:
    return {
        "message": {
            "chat_id": "chat-1",
            "message_id": "message-1",
            "message_type": "text",
            "content": json.dumps({"text": text}),
        },
        "sender": {"sender_id": {"open_id": "sender-1"}},
    }


@pytest.mark.asyncio
async def test_feishu_ordinary_chat_uses_markdown_and_preserves_token_footer(
    route_env, monkeypatch
) -> None:
    session, character = entities()
    session.feishu_chat_id = "chat-1"
    session.messages = '[{"role":"assistant","content":"STALE_DB"}]'
    db = DB(session, character)

    async def resolve_backend(backend_id, db):
        return route_env.backend

    async def complete(**kwargs):
        return {"content": "正文", "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}}

    cards: list[dict[str, Any]] = []

    async def send_card(chat_id, card):
        cards.append(card)

    monkeypatch.setattr(sessions, "_resolve_backend", resolve_backend)
    monkeypatch.setattr(feishu, "chat_completion", complete)
    monkeypatch.setattr(feishu.feishu_client, "send_interactive_card", send_card)

    await feishu._handle_message(feishu_event("继续。"), db)

    records = await route_env.store.load_chat(1)
    assert records[-1].content == f"{REQUIRED_OPENING}\n\n正文"
    assert "STALE_DB" in session.messages
    assert route_env.manager.submitted == [1]
    assert cards
    assert "Token" in json.dumps(cards, ensure_ascii=False)


@pytest.mark.asyncio
async def test_http_feishu_lazily_imports_v1_database_history(
    route_env,
    monkeypatch,
) -> None:
    session, character = entities()
    session.feishu_chat_id = "chat-1"
    session.messages = json.dumps(
        [
            {"role": "user", "content": "V1_FEISHU_USER"},
            {"role": "assistant", "content": "V1_FEISHU_ASSISTANT"},
        ]
    )
    original_database_history = session.messages
    captured: list[list[dict[str, Any]]] = []

    async def complete(**kwargs):
        captured.append(kwargs["messages"])
        return {"content": "新回答", "usage": {}}

    async def send_card(chat_id, card):
        return None

    monkeypatch.setattr(feishu, "chat_completion", complete)
    monkeypatch.setattr(feishu.feishu_client, "send_interactive_card", send_card)

    await feishu._handle_message(feishu_event("继续。"), DB(session, character))

    provider_text = "\n".join(
        str(message.get("content", "")) for message in captured[0]
    )
    assert "V1_FEISHU_USER" in provider_text
    assert "V1_FEISHU_ASSISTANT" in provider_text
    assert [record.content for record in await route_env.store.load_chat(1)] == [
        "V1_FEISHU_USER",
        "V1_FEISHU_ASSISTANT",
        "继续。",
        f"{REQUIRED_OPENING}\n\n新回答",
    ]
    assert session.messages == original_database_history


@pytest.mark.asyncio
async def test_feishu_session_creation_rejects_orphan_before_sending_success_card(
    route_env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, character = entities()
    character.first_message = "你好，{{user}}。"
    await route_env.store.append_pair(
        20,
        "孤儿问题",
        "孤儿回答",
        char_name="旧角色",
        user_name="旧用户",
    )

    class CardDB:
        def __init__(self) -> None:
            self.session: Session | None = None
            self.rolled_back = False

        async def get(self, model, object_id):
            if model is Character and object_id == character.id:
                return character
            return None

        async def execute(self, statement, params=None):
            return ScalarResult([])

        def add(self, session):
            self.session = session

        async def flush(self):
            assert self.session is not None
            self.session.id = 20

        async def commit(self):
            assert self.session is not None
            if self.session.id is None:
                self.session.id = 20

        async def rollback(self):
            self.rolled_back = True

    sent_cards: list[dict[str, Any]] = []

    async def send_card(chat_id, card):
        sent_cards.append(card)

    monkeypatch.setattr(feishu.feishu_client, "send_interactive_card", send_card)
    db = CardDB()

    with pytest.raises(HTTPException) as captured:
        await feishu._handle_card_action(
            {
                "action": {"option": str(character.id), "value": {}},
                "open_chat_id": "chat-orphan",
            },
            db,
        )

    assert captured.value.status_code == 500
    assert captured.value.detail == "Failed to initialize session"
    assert db.rolled_back
    assert sent_cards == []
    assert [record.content for record in await route_env.store.load_chat(20)] == [
        "孤儿问题",
        "孤儿回答",
    ]
    assert feishu.create_session_with_markdown is sessions.create_session_with_markdown


@pytest.mark.parametrize(
    "pending_case",
    ["memory-assets", "cleanup", "missing-boundary"],
)
@pytest.mark.asyncio
async def test_feishu_ordinary_chat_never_sends_pending_stale_context_to_provider(
    route_env,
    monkeypatch: pytest.MonkeyPatch,
    pending_case: str,
) -> None:
    from app.services import rag

    session, character = entities()
    session.feishu_chat_id = "chat-1"
    db = DB(session, character)
    await route_env.store.append_pair(
        1,
        "AUTHORITATIVE_RECENT_USER",
        "AUTHORITATIVE_RECENT_ASSISTANT",
        char_name="Character",
        user_name="User",
    )
    if pending_case == "memory-assets":
        await route_env.store.save_state(
            1,
            MemoryState(
                rebuild_memory_required=True,
                rebuild_assets_required=True,
                rebuild_from_message=2,
            ),
        )
    await route_env.store.write_text(1, "story_state.md", "STALE_STORY")
    await route_env.store.write_text(1, "summary.md", "STALE_SUMMARY")
    await route_env.store.write_text(
        1,
        "episodes/episode-000001.md",
        "STALE_EPISODE",
    )

    async def memory_context(*args, **kwargs):
        if pending_case == "cleanup":
            await route_env.store.save_state(
                1,
                MemoryState(cleanup_required=True, rebuild_from_message=2),
            )
        elif pending_case == "missing-boundary":
            await route_env.store.save_state(
                1,
                MemoryState(rebuild_story_required=True),
            )
        return "STALE_MEMORY"

    async def stale_profile(*args, **kwargs):
        return "STALE_PROFILE"

    async def stale_assets(*args, **kwargs):
        return "STALE_ASSETS"

    async def stale_index(*args, **kwargs):
        return {"chunks": [{"text": "STALE_RAG"}]}

    async def stale_search(*args, **kwargs):
        return [{"text": "STALE_RAG", "score": 1.0}]

    captured: list[list[dict[str, Any]]] = []

    async def complete(**kwargs):
        captured.append(kwargs["messages"])
        return {"content": "正文", "usage": {}}

    async def send_card(chat_id, card):
        return None

    monkeypatch.setattr(sessions, "load_memory_relevant", memory_context)
    monkeypatch.setattr(
        sessions,
        "load_mentioned_character_profiles",
        stale_profile,
    )
    monkeypatch.setattr(sessions, "load_assets_summary", stale_assets)
    monkeypatch.setattr(sessions, "load_assets", stale_assets)
    monkeypatch.setattr(sessions, "is_asset_relevant", lambda content: True)
    monkeypatch.setattr(rag, "load_index", stale_index)
    monkeypatch.setattr(rag, "search", stale_search)
    monkeypatch.setattr(feishu, "chat_completion", complete)
    monkeypatch.setattr(
        feishu.feishu_client,
        "send_interactive_card",
        send_card,
    )

    await feishu._handle_message(feishu_event("AUTHORITATIVE_INPUT"), db)

    provider_text = "\n".join(
        str(message.get("content", "")) for message in captured[0]
    )
    assert "AUTHORITATIVE_RECENT_USER" in provider_text
    assert "AUTHORITATIVE_RECENT_ASSISTANT" in provider_text
    assert "AUTHORITATIVE_INPUT" in provider_text
    assert "STALE_MEMORY" not in provider_text
    assert "STALE_PROFILE" not in provider_text
    assert "STALE_ASSETS" not in provider_text
    if pending_case != "memory-assets":
        for marker in (
            "STALE_STORY",
            "STALE_EPISODE",
            "STALE_RAG",
            "STALE_SUMMARY",
        ):
            assert marker not in provider_text


@pytest.mark.asyncio
async def test_feishu_reset_clears_markdown_chat(route_env, monkeypatch) -> None:
    monkeypatch.setattr("app.config.settings.episode_size_messages", 2)
    session, character = entities()
    session.feishu_chat_id = "chat-1"
    db = DB(session, character)
    await route_env.store.append_pair(
        1, "问", "答", char_name="灵汐", user_name="逸山"
    )
    await route_env.store.write_text(1, "memory.md", "preserved long-term memory")
    await route_env.store.write_text(1, "story_state.md", "stale story")
    await route_env.store.write_text(1, "summary.md", "stale summary")
    await route_env.store.write_text(
        1,
        "episodes/episode-000001.md",
        (
            "# Episode 000001\n\n<!-- messages: 1-2 -->\n\n"
            "## 剧情摘要\n- 旧剧情\n\n"
            "## 状态变化\n- 旧状态\n\n"
            "## 承诺与伏笔\n- 旧伏笔\n"
        ),
    )
    await route_env.store.write_text(
        1,
        "rag/index.json",
        '{"chunks": [{"text": "stale rag", "start_message": 1, '
        '"end_message": 2}], "embeddings": [[1]], "indexed_messages": 2}',
    )
    sent: list[str] = []

    async def send_text(chat_id, text):
        sent.append(text)

    monkeypatch.setattr(feishu.feishu_client, "send_text_message", send_text)

    await feishu._handle_command("/reset", "chat-1", "sender", db)

    assert await route_env.store.load_chat(1) == []
    assert not (route_env.store.session_dir(1) / "memory.md").exists()
    assert await route_env.store.read_text(1, "story_state.md") == ""
    assert await route_env.store.read_text(1, "summary.md") == ""
    assert not (
        route_env.store.session_dir(1) / "episodes" / "episode-000001.md"
    ).exists()
    assert "stale rag" not in await route_env.store.read_text(1, "rag/index.json")
    assert not (await route_env.store.load_state(1)).rebuild_required
    assert sent


@pytest.mark.asyncio
async def test_feishu_reset_waits_for_locked_context_preparation_and_turn(
    route_env, monkeypatch
) -> None:
    session, character = entities()
    session.feishu_chat_id = "chat-1"
    db = DB(session, character)
    await route_env.store.append_pair(
        1, "旧问", "旧答", char_name="灵汐", user_name="逸山"
    )
    loader_started = asyncio.Event()
    release_loader = asyncio.Event()

    async def loader(records):
        loader_started.set()
        await release_loader.wait()
        return DomainContext()

    async def complete(messages):
        return {"content": "正文", "usage": {}}

    async def send_text(chat_id, text):
        return None

    monkeypatch.setattr(feishu.feishu_client, "send_text_message", send_text)
    turn = asyncio.create_task(
        ChatService(
            route_env.store,
            route_env.manager,
            completion=complete,
            domain_context_loader=loader,
        ).send(
            TurnRequest(
                session_id=1,
                content="新问",
                character={"name": "灵汐"},
                user_name="逸山",
            )
        )
    )
    await loader_started.wait()
    reset = asyncio.create_task(
        feishu._handle_command("/reset", "chat-1", "sender", db)
    )
    await asyncio.sleep(0)

    assert not reset.done()
    assert len(await route_env.store.load_chat(1)) == 2
    release_loader.set()
    await asyncio.gather(turn, reset)

    assert await route_env.store.load_chat(1) == []


@pytest.mark.asyncio
async def test_feishu_context_budget_error_is_safe_and_does_not_persist(
    route_env, monkeypatch
) -> None:
    session, character = entities()
    session.feishu_chat_id = "chat-1"
    db = DB(session, character)
    monkeypatch.setattr("app.services.chat_service.settings.total_token_budget", 10)
    sent: list[str] = []

    async def send_text(chat_id, text):
        sent.append(text)

    monkeypatch.setattr(feishu.feishu_client, "send_text_message", send_text)

    await feishu._handle_message(feishu_event("超长输入" * 100), db)

    assert sent
    assert "mandatory" not in sent[-1]
    assert "secret" not in sent[-1]
    assert await route_env.store.load_chat(1) == []


@pytest.mark.asyncio
async def test_feishu_backend_resolution_failure_sends_safe_error_without_side_effects(
    route_env, monkeypatch
) -> None:
    session, character = entities()
    session.feishu_chat_id = "chat-1"
    db = DB(session, character)
    sent: list[str] = []

    async def fail_backend(backend_id, db):
        raise RuntimeError("context-secret provider-key")

    async def send_text(chat_id, text):
        sent.append(text)

    monkeypatch.setattr(sessions, "_resolve_backend", fail_backend)
    monkeypatch.setattr(feishu.feishu_client, "send_text_message", send_text)

    await feishu._handle_message(feishu_event("继续。"), db)

    assert sent == ["生成失败，请稍后重试。"]
    assert "context-secret" not in sent[0]
    assert "provider-key" not in sent[0]
    assert await route_env.store.load_chat(1) == []
    assert route_env.manager.submitted == []


@pytest.mark.asyncio
async def test_feishu_context_preparation_failure_sends_safe_error_without_side_effects(
    route_env, monkeypatch
) -> None:
    session, character = entities()
    session.feishu_chat_id = "chat-1"
    db = DB(session, character)
    sent: list[str] = []

    async def fail_context(**kwargs):
        raise RuntimeError("context-secret provider-key")

    async def send_text(chat_id, text):
        sent.append(text)

    monkeypatch.setattr(sessions, "_prepare_domain_context", fail_context)
    monkeypatch.setattr(feishu.feishu_client, "send_text_message", send_text)

    await feishu._handle_message(feishu_event("继续。"), db)

    assert sent == ["生成失败，请稍后重试。"]
    assert "context-secret" not in sent[0]
    assert "provider-key" not in sent[0]
    assert await route_env.store.load_chat(1) == []
    assert route_env.manager.submitted == []
