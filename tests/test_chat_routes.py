from __future__ import annotations

import asyncio
import datetime as dt
import json
from types import SimpleNamespace
from typing import Any

import pytest

from app import main
from app.models.tables import Character, Session
from app.routers import feishu, sessions
from app.schemas.api import SessionCreate, SessionMessageIn
from app.services import memory
from app.services.chat_service import ChatService, DomainContext, TurnRequest
from app.services.md_store import MarkdownMemoryStore
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


def entities() -> tuple[Session, Character]:
    session = Session(
        id=1,
        character_id=2,
        worldbook_ids="[]",
        user_name="逸山",
        user_persona="",
        messages='[{"role":"assistant","content":"STALE_DB"}]',
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
async def test_create_session_with_first_message_keeps_legacy_command_path_working(
    route_env,
) -> None:
    _, character = entities()
    character.first_message = "你好，{{user}}。"

    class CreateDB:
        async def get(self, model, object_id):
            return character if model is Character else None

        async def execute(self, statement):
            return ScalarResult([])

        def add(self, session):
            session.id = 9
            session.created_at = dt.datetime(2026, 7, 20)

        async def commit(self):
            return None

        async def refresh(self, value):
            return None

    response = await sessions.create_session(
        SessionCreate(character_id=character.id, user_name="逸山"), CreateDB()
    )

    assert response.messages[0].content == "你好，逸山。"


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
async def test_feishu_reset_clears_markdown_chat(route_env, monkeypatch) -> None:
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
    assert await route_env.store.read_text(1, "memory.md") == "preserved long-term memory"
    assert await route_env.store.read_text(1, "story_state.md") == ""
    assert await route_env.store.read_text(1, "summary.md") == ""
    assert not (
        route_env.store.session_dir(1) / "episodes" / "episode-000001.md"
    ).exists()
    assert "stale rag" not in await route_env.store.read_text(1, "rag/index.json")
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
