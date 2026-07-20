from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

from app.config import settings
from app.services.chat_service import (
    ChatService,
    ContextBudgetExceeded,
    DomainContext,
    NoCompletePairError,
    StreamEvent,
    TurnRequest,
)
from app.services.md_store import InvalidationIntentError, MarkdownMemoryStore
from app.services.md_store import MemoryState
from app.services.output_guard import OutputGuardError
from app.services.prompt_policy import REQUIRED_OPENING
from app.schemas.api import WorldBookEntry


USAGE = {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}


class MemoryManager:
    def __init__(self, *, fail: bool = False) -> None:
        self.submitted: list[int] = []
        self.fail = fail

    async def submit(self, session_id: int) -> None:
        self.submitted.append(session_id)
        if self.fail:
            raise RuntimeError("memory queue unavailable")


class SyncMemoryManager:
    def __init__(self) -> None:
        self.submitted: list[int] = []

    def submit(self, session_id: int) -> None:
        self.submitted.append(session_id)


class CapturingBuilder:
    def __init__(self, messages: list[dict[str, str]] | None = None) -> None:
        self.sources: list[Any] = []
        self.messages = messages or [{"role": "user", "content": "built"}]

    def build(self, sources: Any) -> Any:
        self.sources.append(sources)
        return SimpleNamespace(messages=self.messages)


class CloseFailingStream:
    def __init__(
        self,
        items: list[str | dict[str, Any]],
        *,
        iteration_error: Exception | None = None,
        block_after_items: bool = False,
    ) -> None:
        self.items = list(items)
        self.iteration_error = iteration_error
        self.block_after_items = block_after_items
        self.close_calls = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.items:
            return self.items.pop(0)
        if self.iteration_error is not None:
            error = self.iteration_error
            self.iteration_error = None
            raise error
        if self.block_after_items:
            await asyncio.Event().wait()
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.close_calls += 1
        raise RuntimeError("close-secret-body")


def request(session_id: int = 1, content: str = "继续。", **overrides: Any) -> TurnRequest:
    values: dict[str, Any] = {
        "session_id": session_id,
        "content": content,
        "character": {"name": "灵汐", "description": "CHARACTER_MARKER"},
        "user_name": "逸山",
    }
    values.update(overrides)
    return TurnRequest(**values)


async def collect(stream: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    return [event async for event in stream]


@pytest.mark.asyncio
async def test_send_persists_one_complete_pair_and_submits_memory_once(tmp_path) -> None:
    store = MarkdownMemoryStore(tmp_path)
    manager = MemoryManager()

    async def complete(messages: list[dict[str, Any]]) -> dict[str, Any]:
        return {"content": "正文", "usage": USAGE}

    result = await ChatService(store, manager, completion=complete).send(request())

    records = await store.load_chat(1)
    assert [(record.role, record.content) for record in records] == [
        ("user", "继续。"),
        ("assistant", f"{REQUIRED_OPENING}\n\n正文"),
    ]
    assert result.content.startswith(REQUIRED_OPENING)
    assert result.usage == USAGE
    assert manager.submitted == [1]


@pytest.mark.asyncio
async def test_send_provider_failure_leaves_markdown_unchanged(tmp_path) -> None:
    store = MarkdownMemoryStore(tmp_path)
    manager = MemoryManager()

    async def fail(messages: list[dict[str, Any]]) -> dict[str, Any]:
        raise RuntimeError("provider secret")

    with pytest.raises(RuntimeError, match="provider secret"):
        await ChatService(store, manager, completion=fail).send(request())

    assert await store.load_chat(1) == []
    assert manager.submitted == []


@pytest.mark.asyncio
async def test_send_two_non_substantive_candidates_leave_markdown_unchanged(
    tmp_path,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    manager = MemoryManager()
    candidates = ["", REQUIRED_OPENING]
    calls = 0

    async def complete(messages: list[dict[str, Any]]) -> dict[str, Any]:
        nonlocal calls
        content = candidates[calls]
        calls += 1
        return {"content": content, "usage": USAGE}

    with pytest.raises(OutputGuardError):
        await ChatService(store, manager, completion=complete).send(request())

    assert calls == 2
    assert await store.load_chat(1) == []
    assert manager.submitted == []


@pytest.mark.asyncio
async def test_send_recovers_unresolved_intent_before_context_and_releases_pipeline_lock(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.services.md_store as md_store_module

    store = MarkdownMemoryStore(tmp_path)
    for pair in range(1, 3):
        await store.append_pair(
            1,
            f"user-{pair}",
            f"assistant-{pair}",
            char_name="Character",
            user_name="User",
        )
    records = await store.load_chat(1)
    await store.write_text(1, "story_state.md", "stale story")
    await store.save_state(1, MemoryState(last_story_state_message=4))
    real_replace = md_store_module.os.replace

    def fail_state(source, target) -> None:
        if target.name == "memory_state.md":
            raise OSError("state marker failed")
        real_replace(source, target)

    monkeypatch.setattr(md_store_module.os, "replace", fail_state)
    updated = await store.replace_final_pair(
        1,
        expected_user=records[-2],
        expected_assistant=records[-1],
        assistant_content="replacement assistant",
    )
    assert updated[-1].content == "replacement assistant"
    assert (tmp_path / "1" / "invalidation_intent.md").exists()
    monkeypatch.setattr(md_store_module.os, "replace", real_replace)

    async def complete(messages: list[dict[str, Any]]) -> dict[str, Any]:
        assert await store.read_text(1, "story_state.md") == ""
        pipeline_lock = await store.pipeline_lock_for(1)
        assert not pipeline_lock.locked()
        return {"content": "正文", "usage": USAGE}

    await ChatService(store, None, completion=complete).send(request(content="next"))

    assert len(await store.load_chat(1)) == 6
    assert not (tmp_path / "1" / "invalidation_intent.md").exists()


@pytest.mark.parametrize("operation", ["undo", "retry"])
@pytest.mark.asyncio
async def test_mutating_commands_fail_before_touching_malformed_intent(
    tmp_path,
    operation: str,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "old user",
        "old assistant",
        char_name="Character",
        user_name="User",
    )
    malformed = "# Invalidation Intent\n\n<!-- boundary-message: 0 -->\n"
    await store.write_text(1, "invalidation_intent.md", malformed)
    before_chat = await store.read_text(1, "chat.md")
    completion_calls = 0

    async def complete(messages: list[dict[str, Any]]) -> dict[str, Any]:
        nonlocal completion_calls
        completion_calls += 1
        return {"content": "replacement", "usage": USAGE}

    service = ChatService(store, None, completion=complete)

    with pytest.raises(ValueError, match="invalid invalidation intent"):
        if operation == "undo":
            await service.undo(1)
        else:
            await service.retry(request())

    assert await store.read_text(1, "chat.md") == before_chat
    assert await store.read_text(1, "invalidation_intent.md") == malformed
    assert completion_calls == 0


@pytest.mark.parametrize("operation", ["send", "retry"])
@pytest.mark.parametrize(
    "intent_bytes",
    [b"", b" \t\r\n"],
    ids=["zero-byte", "whitespace"],
)
@pytest.mark.asyncio
async def test_turn_recovery_refuses_present_empty_intent_without_any_mutation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    intent_bytes: bytes,
) -> None:
    from app.config import settings

    monkeypatch.setattr(settings, "episode_size_messages", 2)
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "old user",
        "old assistant",
        char_name="Character",
        user_name="User",
    )
    await store.write_text(1, "story_state.md", "unchanged story")
    await store.write_text(1, "summary.md", "unchanged summary")
    await store.write_text(
        1,
        "episodes/episode-000001.md",
        "# Episode 000001\n\n"
        "<!-- messages: 1-2 -->\n\n"
        "## 剧情摘要\n- unchanged events\n\n"
        "## 状态变化\n- unchanged state\n\n"
        "## 承诺与伏笔\n- unchanged facts\n",
    )
    await store.write_text(
        1,
        "rag/index.json",
        '{"chunks": [], "embeddings": [], "indexed_messages": 2}',
    )
    await store.save_state(
        1,
        MemoryState(
            last_memory_message=2,
            last_story_state_message=2,
            last_summary_message=2,
            last_episode_message=2,
            last_rag_message=2,
            last_character_message=2,
            last_assets_message=2,
        ),
    )
    session = tmp_path / "1"
    (session / "invalidation_intent.md").write_bytes(intent_bytes)
    relatives = (
        "chat.md",
        "memory_state.md",
        "story_state.md",
        "summary.md",
        "episodes/episode-000001.md",
        "rag/index.json",
        "invalidation_intent.md",
    )
    before = {relative: (session / relative).read_bytes() for relative in relatives}
    completion_calls = 0

    async def complete(messages: list[dict[str, Any]]) -> dict[str, Any]:
        nonlocal completion_calls
        completion_calls += 1
        return {"content": "replacement", "usage": USAGE}

    service = ChatService(store, None, completion=complete)

    with pytest.raises(InvalidationIntentError) as captured:
        if operation == "send":
            await service.send(request(content="next"))
        else:
            await service.retry(request())

    assert str(captured.value) == "invalid invalidation intent"
    assert completion_calls == 0
    assert {
        relative: (session / relative).read_bytes() for relative in relatives
    } == before


@pytest.mark.asyncio
async def test_send_refuses_boundaryless_legacy_retry_state_before_completion(
    tmp_path,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    for pair in range(1, 3):
        await store.append_pair(
            1,
            f"user-{pair}",
            f"assistant-{pair}",
            char_name="Character",
            user_name="User",
        )
    await store.write_text(1, "story_state.md", "must remain unchanged")
    await store.save_state(
        1,
        MemoryState(
            last_story_state_message=4,
            cleanup_required=True,
            rebuild_story_required=True,
        ),
    )
    before_chat = await store.read_text(1, "chat.md")
    completion_calls = 0

    async def complete(messages: list[dict[str, Any]]) -> dict[str, Any]:
        nonlocal completion_calls
        completion_calls += 1
        return {"content": "正文", "usage": USAGE}

    with pytest.raises(ValueError, match="no persisted rebuild boundary"):
        await ChatService(store, None, completion=complete).send(request())

    assert completion_calls == 0
    assert await store.read_text(1, "chat.md") == before_chat
    assert await store.read_text(1, "story_state.md") == "must remain unchanged"


@pytest.mark.asyncio
async def test_send_builds_context_from_authoritative_markdown_and_domain_inputs(
    tmp_path,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "旧问题",
        "旧回答",
        char_name="灵汐",
        user_name="逸山",
    )
    builder = CapturingBuilder()
    received: list[list[dict[str, Any]]] = []

    async def complete(messages: list[dict[str, Any]]) -> dict[str, Any]:
        received.append(messages)
        return {"content": "正文", "usage": USAGE}

    turn = request(
        worldbook_entries=[
            WorldBookEntry(keyword="继续", content="WORLDBOOK", position="before_char")
        ],
        user_persona="PERSONA",
        story_state="STORY_STATE",
        memory="MEMORY",
        character_profiles=["PROFILE"],
        assets="ASSETS",
        episodes=["EPISODE"],
        rag=["RAG"],
        overall_summary="SUMMARY",
        history_seed=[{"role": "assistant", "content": "SEED"}],
        is_new_conversation=True,
    )
    await ChatService(
        store,
        MemoryManager(),
        completion=complete,
        context_builder=builder,
    ).send(turn)

    sources = builder.sources[0]
    assert sources.recent == [
        {"role": "user", "content": "旧问题"},
        {"role": "assistant", "content": "旧回答"},
    ]
    assert sources.character["_history_seed"] == turn.history_seed
    assert "WORLDBOOK" in "\n".join(sources.ordered_character_context)
    assert sources.user_persona == "PERSONA"
    assert sources.story_state == "STORY_STATE"
    assert sources.memory == "MEMORY"
    assert sources.character_profiles == ["PROFILE"]
    assert sources.assets == "ASSETS"
    assert sources.episodes == ["EPISODE"]
    assert sources.rag == ["RAG"]
    assert sources.summary == "SUMMARY"
    assert sources.user_message == "继续。"
    assert received == [builder.messages]


@pytest.mark.parametrize(
    ("flag", "suppressed_fields"),
    [
        ("rebuild_story_required", {"story_state"}),
        ("rebuild_memory_required", {"memory", "character_profiles"}),
        ("rebuild_episode_required", {"episodes"}),
        ("rebuild_summary_required", {"summary"}),
        ("rebuild_rag_required", {"rag"}),
        ("rebuild_assets_required", {"assets"}),
    ],
)
@pytest.mark.asyncio
async def test_send_suppresses_each_pending_derived_context_layer(
    tmp_path,
    flag: str,
    suppressed_fields: set[str],
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "authoritative recent user",
        "authoritative recent assistant",
        char_name="Character",
        user_name="User",
    )
    await store.save_state(
        1,
        MemoryState(**{flag: True}, rebuild_from_message=2),
    )
    builder = CapturingBuilder()

    async def complete(messages: list[dict[str, Any]]) -> dict[str, Any]:
        return {"content": "正文", "usage": USAGE}

    await ChatService(
        store,
        None,
        completion=complete,
        context_builder=builder,
    ).send(
        request(
            story_state="STALE_STORY",
            memory="STALE_MEMORY",
            character_profiles=["STALE_PROFILE"],
            assets="STALE_ASSETS",
            episodes=["STALE_EPISODE"],
            rag=["STALE_RAG"],
            overall_summary="STALE_SUMMARY",
        )
    )

    sources = builder.sources[0]
    values = {
        "story_state": sources.story_state,
        "memory": sources.memory,
        "character_profiles": sources.character_profiles,
        "assets": sources.assets,
        "episodes": sources.episodes,
        "rag": sources.rag,
        "summary": sources.summary,
    }
    stale = {
        "story_state": "STALE_STORY",
        "memory": "STALE_MEMORY",
        "character_profiles": ["STALE_PROFILE"],
        "assets": "STALE_ASSETS",
        "episodes": ["STALE_EPISODE"],
        "rag": ["STALE_RAG"],
        "summary": "STALE_SUMMARY",
    }
    expected = dict(stale)
    for field in suppressed_fields:
        expected[field] = [] if isinstance(stale[field], list) else ""
    assert values == expected


@pytest.mark.parametrize(
    "state",
    [
        MemoryState(cleanup_required=True, rebuild_from_message=0),
        MemoryState(rebuild_story_required=True),
    ],
    ids=["cleanup", "missing-boundary"],
)
@pytest.mark.asyncio
async def test_context_snapshot_anomalies_suppress_all_derived_layers(
    tmp_path,
    state: MemoryState,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    await store.save_state(1, state)
    builder = CapturingBuilder()
    service = ChatService(store, None, context_builder=builder)
    turn_lock = await store.turn_lock_for(1)

    async with turn_lock:
        await service._build_messages(
            request(
                story_state="STALE_STORY",
                memory="STALE_MEMORY",
                character_profiles=["STALE_PROFILE"],
                assets="STALE_ASSETS",
                episodes=["STALE_EPISODE"],
                rag=["STALE_RAG"],
                overall_summary="STALE_SUMMARY",
            )
        )

    sources = builder.sources[0]
    assert sources.story_state == ""
    assert sources.memory == ""
    assert sources.character_profiles == []
    assert sources.assets == ""
    assert sources.episodes == []
    assert sources.rag == []
    assert sources.summary == ""


@pytest.mark.asyncio
async def test_send_never_injects_persona_when_position_is_none(tmp_path) -> None:
    received: list[list[dict[str, Any]]] = []

    async def complete(messages: list[dict[str, Any]]) -> dict[str, Any]:
        received.append(messages)
        return {"content": "正文", "usage": USAGE}

    await ChatService(
        MarkdownMemoryStore(tmp_path),
        MemoryManager(),
        completion=complete,
    ).send(
        request(
            character={"name": "灵汐"},
            user_persona="PRIVATE_PERSONA",
            persona_position="none",
            is_new_conversation=False,
        )
    )

    assert "PRIVATE_PERSONA" not in "\n".join(
        message["content"] for message in received[0]
    )


@pytest.mark.asyncio
async def test_same_session_turns_are_serialized_across_service_instances(tmp_path) -> None:
    store = MarkdownMemoryStore(tmp_path)
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    second_started = asyncio.Event()

    async def first_call(messages: list[dict[str, Any]]) -> dict[str, Any]:
        first_started.set()
        await release_first.wait()
        return {"content": "第一答", "usage": USAGE}

    async def second_call(messages: list[dict[str, Any]]) -> dict[str, Any]:
        second_started.set()
        return {"content": "第二答", "usage": USAGE}

    first = asyncio.create_task(
        ChatService(store, MemoryManager(), completion=first_call).send(
            request(content="第一问")
        )
    )
    await first_started.wait()
    second = asyncio.create_task(
        ChatService(store, MemoryManager(), completion=second_call).send(
            request(content="第二问")
        )
    )

    await asyncio.sleep(0)
    assert not second_started.is_set()
    release_first.set()
    await asyncio.gather(first, second)
    assert second_started.is_set()
    records = await store.load_chat(1)
    assert [record.content for record in records if record.role == "user"] == [
        "第一问",
        "第二问",
    ]


@pytest.mark.asyncio
async def test_different_sessions_are_not_globally_serialized(tmp_path) -> None:
    store = MarkdownMemoryStore(tmp_path)
    both_started = asyncio.Event()
    release = asyncio.Event()
    started: set[int] = set()

    def completion_for(session_id: int):
        async def complete(messages: list[dict[str, Any]]) -> dict[str, Any]:
            started.add(session_id)
            if len(started) == 2:
                both_started.set()
            await release.wait()
            return {"content": str(session_id), "usage": USAGE}

        return complete

    tasks = [
        asyncio.create_task(
            ChatService(
                store,
                MemoryManager(),
                completion=completion_for(session_id),
            ).send(request(session_id=session_id))
        )
        for session_id in (1, 2)
    ]

    await asyncio.wait_for(both_started.wait(), timeout=1)
    release.set()
    await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_same_session_domain_context_loaders_are_serialized_and_see_commit(
    tmp_path,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    first_loader_started = asyncio.Event()
    release_first_loader = asyncio.Event()
    histories: list[list[str]] = []
    active = 0
    max_active = 0

    async def loader(records):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        histories.append([record.content for record in records])
        try:
            if len(histories) == 1:
                first_loader_started.set()
                await release_first_loader.wait()
            return DomainContext()
        finally:
            active -= 1

    async def complete(messages: list[dict[str, Any]]) -> dict[str, Any]:
        return {"content": "正文", "usage": USAGE}

    first = asyncio.create_task(
        ChatService(
            store,
            MemoryManager(),
            completion=complete,
            domain_context_loader=loader,
        ).send(request(content="第一问"))
    )
    await first_loader_started.wait()
    second = asyncio.create_task(
        ChatService(
            store,
            MemoryManager(),
            completion=complete,
            domain_context_loader=loader,
        ).send(request(content="第二问"))
    )
    await asyncio.sleep(0)

    assert max_active == 1
    assert len(histories) == 1
    release_first_loader.set()
    await asyncio.gather(first, second)

    assert max_active == 1
    assert histories[1][-2:] == [
        "第一问",
        f"{REQUIRED_OPENING}\n\n正文",
    ]


@pytest.mark.asyncio
async def test_different_session_domain_context_loaders_run_independently(
    tmp_path,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    both_active = asyncio.Event()
    release = asyncio.Event()
    active = 0
    max_active = 0

    async def loader(records):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        if active == 2:
            both_active.set()
        try:
            await release.wait()
            return DomainContext()
        finally:
            active -= 1

    async def complete(messages: list[dict[str, Any]]) -> dict[str, Any]:
        return {"content": "正文", "usage": USAGE}

    tasks = [
        asyncio.create_task(
            ChatService(
                store,
                MemoryManager(),
                completion=complete,
                domain_context_loader=loader,
            ).send(request(session_id=session_id))
        )
        for session_id in (1, 2)
    ]

    await asyncio.wait_for(both_active.wait(), timeout=1)
    assert max_active == 2
    release.set()
    await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_memory_submit_failure_does_not_rollback_or_fail_committed_turn(
    tmp_path, caplog
) -> None:
    store = MarkdownMemoryStore(tmp_path)

    async def complete(messages: list[dict[str, Any]]) -> dict[str, Any]:
        return {"content": "正文", "usage": USAGE}

    result = await ChatService(
        store,
        MemoryManager(fail=True),
        completion=complete,
    ).send(request())

    assert result.content.endswith("正文")
    assert len(await store.load_chat(1)) == 2
    assert "memory queue unavailable" not in caplog.text
    assert "RuntimeError" in caplog.text


@pytest.mark.asyncio
async def test_send_supports_synchronous_memory_submitter_without_false_failure(
    tmp_path, caplog
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    manager = SyncMemoryManager()

    async def complete(messages: list[dict[str, Any]]) -> dict[str, Any]:
        return {"content": "正文", "usage": USAGE}

    result = await ChatService(store, manager, completion=complete).send(request())

    assert result.content.endswith("正文")
    assert manager.submitted == [1]
    assert len(await store.load_chat(1)) == 2
    assert "failed to submit managed memory" not in caplog.text.casefold()


@pytest.mark.asyncio
async def test_context_budget_failure_is_typed_and_has_no_side_effects(tmp_path) -> None:
    class FailingBuilder:
        def build(self, sources: Any) -> Any:
            raise ValueError("mandatory prompt context exceeds total token budget")

    store = MarkdownMemoryStore(tmp_path)
    manager = MemoryManager()

    async def unused(messages: list[dict[str, Any]]) -> dict[str, Any]:
        raise AssertionError("completion must not run")

    with pytest.raises(ContextBudgetExceeded):
        await ChatService(
            store,
            manager,
            completion=unused,
            context_builder=FailingBuilder(),
        ).send(request())

    assert await store.load_chat(1) == []
    assert manager.submitted == []


@pytest.mark.asyncio
async def test_stream_success_normalizes_opening_persists_then_emits_done(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("app.services.chat_service.settings.stream_guard_chars", 4)
    store = MarkdownMemoryStore(tmp_path)
    manager = MemoryManager()

    async def stream(messages: list[dict[str, Any]]):
        yield "正文"
        yield "继续"
        yield {"usage": USAGE}

    events = await collect(
        ChatService(store, manager, stream_completion=stream).stream(request())
    )

    deltas = [event.delta for event in events if event.delta]
    assert deltas[0].startswith(REQUIRED_OPENING)
    assert "".join(deltas) == f"{REQUIRED_OPENING}\n\n正文继续"
    assert events[-1] == StreamEvent(done=True, usage=USAGE)
    records = await store.load_chat(1)
    assert [record.content for record in records] == [
        "继续。",
        f"{REQUIRED_OPENING}\n\n正文继续",
    ]
    assert manager.submitted == [1]


@pytest.mark.asyncio
async def test_stream_recovers_unresolved_intent_before_context_and_provider(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.services.md_store as md_store_module

    monkeypatch.setattr("app.services.chat_service.settings.stream_guard_chars", 1)
    store = MarkdownMemoryStore(tmp_path)
    for pair in range(1, 3):
        await store.append_pair(
            1,
            f"user-{pair}",
            f"assistant-{pair}",
            char_name="Character",
            user_name="User",
        )
    records = await store.load_chat(1)
    await store.write_text(1, "story_state.md", "stale story")
    await store.save_state(1, MemoryState(last_story_state_message=4))
    real_replace = md_store_module.os.replace

    def fail_state(source, target) -> None:
        if target.name == "memory_state.md":
            raise OSError("state marker failed")
        real_replace(source, target)

    monkeypatch.setattr(md_store_module.os, "replace", fail_state)
    updated = await store.replace_final_pair(
        1,
        expected_user=records[-2],
        expected_assistant=records[-1],
        assistant_content="replacement assistant",
    )
    assert updated[-1].content == "replacement assistant"
    assert (tmp_path / "1" / "invalidation_intent.md").exists()
    monkeypatch.setattr(md_store_module.os, "replace", real_replace)

    async def stream(messages: list[dict[str, Any]]):
        assert await store.read_text(1, "story_state.md") == ""
        pipeline_lock = await store.pipeline_lock_for(1)
        assert not pipeline_lock.locked()
        yield "正文"
        yield {"usage": USAGE}

    events = await collect(
        ChatService(store, None, stream_completion=stream).stream(
            request(content="next")
        )
    )

    assert events[-1].done
    assert len(await store.load_chat(1)) == 6
    assert not (tmp_path / "1" / "invalidation_intent.md").exists()


@pytest.mark.asyncio
async def test_stream_supports_synchronous_memory_submitter_without_false_failure(
    tmp_path, monkeypatch, caplog
) -> None:
    monkeypatch.setattr("app.services.chat_service.settings.stream_guard_chars", 1)
    store = MarkdownMemoryStore(tmp_path)
    manager = SyncMemoryManager()

    async def stream(messages: list[dict[str, Any]]):
        yield "正文"
        yield {"usage": USAGE}

    events = await collect(
        ChatService(store, manager, stream_completion=stream).stream(request())
    )

    assert events[-1].done
    assert manager.submitted == [1]
    assert len(await store.load_chat(1)) == 2
    assert "failed to submit managed memory" not in caplog.text.casefold()


@pytest.mark.asyncio
async def test_stream_refusal_retries_before_first_delta_and_combines_usage(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("app.services.chat_service.settings.stream_guard_chars", 100)
    store = MarkdownMemoryStore(tmp_path)
    calls: list[list[dict[str, Any]]] = []
    usages = [
        {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        {"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18},
    ]

    async def stream(messages: list[dict[str, Any]]):
        call_index = len(calls)
        calls.append(messages)
        if call_index == 0:
            yield "I cannot continue this story."
        else:
            yield "正文"
        yield {"usage": usages[call_index]}

    events = await collect(
        ChatService(store, MemoryManager(), stream_completion=stream).stream(request())
    )

    assert len(calls) == 2
    assert calls[1][:-1] == calls[0]
    assert "继续虚构角色扮演" in calls[1][-1]["content"]
    assert [event for event in events if event.delta][0].delta.startswith(REQUIRED_OPENING)
    assert not any("cannot continue" in event.delta for event in events)
    assert events[-1].usage == {
        "prompt_tokens": 9,
        "completion_tokens": 14,
        "total_tokens": 23,
    }


@pytest.mark.asyncio
async def test_stream_does_not_treat_opening_only_boundary_as_guarded_content(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("app.services.chat_service.settings.stream_guard_chars", 1000)
    calls = 0

    async def stream(messages: list[dict[str, Any]]):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield f"{REQUIRED_OPENING}\n\n"
            yield "I cannot continue this story."
        else:
            yield "正文"
        yield {"usage": USAGE}

    events = await collect(
        ChatService(
            MarkdownMemoryStore(tmp_path),
            MemoryManager(),
            stream_completion=stream,
        ).stream(request())
    )

    assert calls == 2
    assert not any("cannot continue" in event.delta for event in events)
    assert "".join(event.delta for event in events) == f"{REQUIRED_OPENING}\n\n正文"


@pytest.mark.asyncio
async def test_stream_waits_when_refusal_prefix_is_split_at_guard_threshold(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("app.services.chat_service.settings.stream_guard_chars", 6)
    calls = 0

    async def stream(messages: list[dict[str, Any]]):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield "I cann"
            yield "ot continue this story."
        else:
            yield "正文"
        yield {"usage": USAGE}

    events = await collect(
        ChatService(
            MarkdownMemoryStore(tmp_path),
            MemoryManager(),
            stream_completion=stream,
        ).stream(request())
    )

    assert calls == 2
    assert not any("cannot continue" in event.delta for event in events)
    assert events[-1].done


@pytest.mark.asyncio
async def test_stream_refusal_prefix_stays_buffered_across_crlf_chunk_boundary(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("app.services.chat_service.settings.stream_guard_chars", 1)
    calls = 0

    async def stream(messages: list[dict[str, Any]]):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield f"{REQUIRED_OPENING}\r"
            yield "\n \tI cann"
            yield "ot continue this story."
        else:
            yield "正文"
        yield {"usage": USAGE}

    events = await collect(
        ChatService(
            MarkdownMemoryStore(tmp_path),
            MemoryManager(),
            stream_completion=stream,
        ).stream(request())
    )

    assert calls == 2
    assert not any("cannot continue" in event.delta for event in events)
    assert events[-1].done


@pytest.mark.parametrize(
    "refusal_chunks",
    [
        ["Sorry, b", "ut I cannot continue this story."],
        ["I need to be direct b", "ut I cannot assist."],
        ["As an AI m", "odel, I cannot continue this story."],
        ["As an AI language m", "odel, I cannot continue this story."],
        ["As an AI ass", "istant, I cannot continue this story."],
        ["Sorry;\r", "\n\tbu", "t I cannot continue this story."],
    ],
)
@pytest.mark.asyncio
async def test_stream_buffers_partial_refusal_preamble_without_leaking(
    tmp_path, monkeypatch, refusal_chunks: list[str]
) -> None:
    monkeypatch.setattr(
        "app.services.chat_service.settings.stream_guard_chars",
        len(refusal_chunks[0]),
    )
    calls = 0

    async def stream(messages: list[dict[str, Any]]):
        nonlocal calls
        calls += 1
        if calls == 1:
            for chunk in refusal_chunks:
                yield chunk
        else:
            yield "正文"
        yield {"usage": USAGE}

    store = MarkdownMemoryStore(tmp_path)
    events = await collect(
        ChatService(
            store,
            MemoryManager(),
            stream_completion=stream,
        ).stream(request())
    )

    assert calls == 2
    assert not any("cannot" in event.delta.casefold() for event in events)
    assert events[-1].done
    assert (await store.load_chat(1))[-1].content.endswith("正文")


@pytest.mark.parametrize(
    "split_at",
    range(1, len("I cannot continue this story.")),
)
@pytest.mark.asyncio
async def test_stream_waits_at_every_refusal_prefix_split_after_opening_whitespace(
    tmp_path, monkeypatch, split_at: int
) -> None:
    refusal = "I cannot continue this story."
    opening = f"{REQUIRED_OPENING}\r\n \t"
    first_chunk = opening + refusal[:split_at]
    monkeypatch.setattr(
        "app.services.chat_service.settings.stream_guard_chars",
        len(first_chunk),
    )
    calls = 0

    async def stream(messages: list[dict[str, Any]]):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield first_chunk
            yield refusal[split_at:]
        else:
            yield "正文"
        yield {"usage": USAGE}

    events = await collect(
        ChatService(
            MarkdownMemoryStore(tmp_path),
            MemoryManager(),
            stream_completion=stream,
        ).stream(request())
    )

    assert calls == 2
    assert not any("cannot continue" in event.delta for event in events)
    assert events[-1].done


@pytest.mark.asyncio
async def test_stream_nonpositive_guard_setting_still_waits_for_classification(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("app.services.chat_service.settings.stream_guard_chars", 0)

    async def stream(messages: list[dict[str, Any]]):
        yield "正文"
        yield {"usage": USAGE}

    events = await collect(
        ChatService(
            MarkdownMemoryStore(tmp_path),
            MemoryManager(),
            stream_completion=stream,
        ).stream(request())
    )

    assert "".join(event.delta for event in events if event.delta).endswith("正文")
    assert events[-1].done


@pytest.mark.asyncio
async def test_stream_does_not_reclassify_later_narrative_as_opening_refusal(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("app.services.chat_service.settings.stream_guard_chars", 5)
    calls = 0

    async def stream(messages: list[dict[str, Any]]):
        nonlocal calls
        calls += 1
        yield "骑士穿过城门。"
        yield " 他低声说：I cannot continue walking in this rain."
        yield {"usage": USAGE}

    events = await collect(
        ChatService(
            MarkdownMemoryStore(tmp_path),
            MemoryManager(),
            stream_completion=stream,
        ).stream(request())
    )

    assert calls == 1
    assert events[-1].done
    assert "I cannot continue walking" in "".join(
        event.delta for event in events
    )


@pytest.mark.parametrize("first_candidate", ["", REQUIRED_OPENING])
@pytest.mark.asyncio
async def test_stream_retries_empty_or_opening_only_candidate_before_emission(
    tmp_path, monkeypatch, first_candidate: str
) -> None:
    monkeypatch.setattr("app.services.chat_service.settings.stream_guard_chars", 1)
    calls = 0

    async def stream(messages: list[dict[str, Any]]):
        nonlocal calls
        calls += 1
        if calls == 1 and first_candidate:
            yield first_candidate
        elif calls == 2:
            yield "正文"
        yield {"usage": USAGE}

    store = MarkdownMemoryStore(tmp_path)
    manager = MemoryManager()
    events = await collect(
        ChatService(store, manager, stream_completion=stream).stream(request())
    )

    assert calls == 2
    assert "".join(event.delta for event in events if event.delta) == (
        f"{REQUIRED_OPENING}\n\n正文"
    )
    assert events[-1].done
    assert events[-1].usage == {
        "prompt_tokens": 6,
        "completion_tokens": 10,
        "total_tokens": 16,
    }
    assert len(await store.load_chat(1)) == 2
    assert manager.submitted == [1]


@pytest.mark.parametrize("candidate", ["", REQUIRED_OPENING, f" {REQUIRED_OPENING}\r\n\t"])
@pytest.mark.asyncio
async def test_stream_two_non_substantive_candidates_emit_error_without_side_effects(
    tmp_path, monkeypatch, candidate: str
) -> None:
    monkeypatch.setattr("app.services.chat_service.settings.stream_guard_chars", 1)

    async def stream(messages: list[dict[str, Any]]):
        if candidate:
            yield candidate
        yield {"usage": USAGE}

    store = MarkdownMemoryStore(tmp_path)
    manager = MemoryManager()
    events = await collect(
        ChatService(store, manager, stream_completion=stream).stream(request())
    )

    assert events[-1].error
    assert not any(event.done for event in events)
    assert not any(event.delta for event in events)
    assert await store.load_chat(1) == []
    assert manager.submitted == []


@pytest.mark.parametrize("fail_after_text", [False, True])
@pytest.mark.asyncio
async def test_stream_provider_failure_never_persists_partial_pair(
    tmp_path, monkeypatch, caplog, fail_after_text: bool
) -> None:
    monkeypatch.setattr("app.services.chat_service.settings.stream_guard_chars", 1)
    store = MarkdownMemoryStore(tmp_path)
    manager = MemoryManager()

    async def stream(messages: list[dict[str, Any]]):
        if fail_after_text:
            yield "正文"
        raise RuntimeError("api-key=secret")
        yield  # pragma: no cover

    events = await collect(
        ChatService(store, manager, stream_completion=stream).stream(request())
    )

    assert events[-1].error
    assert "secret" not in events[-1].error
    assert "api-key=secret" not in caplog.text
    assert "RuntimeError" in caplog.text
    assert not events[-1].done
    assert await store.load_chat(1) == []
    assert manager.submitted == []


@pytest.mark.asyncio
async def test_stream_factory_failure_is_sanitized_without_side_effects(
    tmp_path, caplog
) -> None:
    def stream(messages):
        raise RuntimeError("factory-secret-body")

    store = MarkdownMemoryStore(tmp_path)
    manager = MemoryManager()
    events = await collect(
        ChatService(store, manager, stream_completion=stream).stream(request())
    )

    assert events[-1].error
    assert not any(event.done for event in events)
    assert "factory-secret-body" not in caplog.text
    assert "RuntimeError" in caplog.text
    assert await store.load_chat(1) == []
    assert manager.submitted == []


@pytest.mark.asyncio
async def test_stream_refusal_close_failure_aborts_without_retry_or_side_effects(
    tmp_path, monkeypatch, caplog
) -> None:
    monkeypatch.setattr("app.services.chat_service.settings.stream_guard_chars", 1)
    calls = 0
    candidate = CloseFailingStream(["I cannot continue this story."])

    def stream(messages):
        nonlocal calls
        calls += 1
        return candidate

    store = MarkdownMemoryStore(tmp_path)
    manager = MemoryManager()
    events = await collect(
        ChatService(store, manager, stream_completion=stream).stream(request())
    )

    assert calls == 1
    assert candidate.close_calls == 1
    assert events[-1].error
    assert not any(event.done or event.delta for event in events)
    assert "close-secret-body" not in caplog.text
    assert "session 1" in caplog.text
    assert "RuntimeError" in caplog.text
    assert await store.load_chat(1) == []
    assert manager.submitted == []


@pytest.mark.asyncio
async def test_stream_normal_completion_close_failure_has_no_side_effects(
    tmp_path, monkeypatch, caplog
) -> None:
    monkeypatch.setattr("app.services.chat_service.settings.stream_guard_chars", 1)
    candidate = CloseFailingStream(["正文", {"usage": USAGE}])

    def stream(messages):
        return candidate

    store = MarkdownMemoryStore(tmp_path)
    manager = MemoryManager()
    events = await collect(
        ChatService(store, manager, stream_completion=stream).stream(request())
    )

    assert candidate.close_calls == 1
    assert any(event.delta for event in events)
    assert events[-1].error
    assert not any(event.done for event in events)
    assert "close-secret-body" not in caplog.text
    assert await store.load_chat(1) == []
    assert manager.submitted == []


@pytest.mark.asyncio
async def test_stream_iteration_and_close_failures_preserve_safe_original_outcome(
    tmp_path, caplog
) -> None:
    candidate = CloseFailingStream(
        [], iteration_error=RuntimeError("iteration-secret-body")
    )

    def stream(messages):
        return candidate

    store = MarkdownMemoryStore(tmp_path)
    manager = MemoryManager()
    events = await collect(
        ChatService(store, manager, stream_completion=stream).stream(request())
    )

    assert candidate.close_calls == 1
    assert events[-1].error
    assert not any(event.done for event in events)
    assert "iteration-secret-body" not in caplog.text
    assert "close-secret-body" not in caplog.text
    assert await store.load_chat(1) == []
    assert manager.submitted == []


@pytest.mark.asyncio
async def test_stream_provider_error_closes_before_emitting_terminal_error(
    tmp_path,
) -> None:
    candidate = CloseFailingStream(
        [], iteration_error=RuntimeError("iteration-secret-body")
    )

    def stream(messages):
        return candidate

    response = ChatService(
        MarkdownMemoryStore(tmp_path),
        MemoryManager(),
        stream_completion=stream,
    ).stream(request())
    try:
        assert (await anext(response)).error
        assert candidate.close_calls == 1
    finally:
        await response.aclose()

    assert candidate.close_calls == 1


@pytest.mark.asyncio
async def test_stream_consumer_close_ignores_close_error_and_releases_turn_lock(
    tmp_path, monkeypatch, caplog
) -> None:
    monkeypatch.setattr("app.services.chat_service.settings.stream_guard_chars", 1)
    store = MarkdownMemoryStore(tmp_path)
    manager = MemoryManager()
    candidate = CloseFailingStream(["正文"], block_after_items=True)

    def stream(messages):
        return candidate

    response = ChatService(
        store, manager, stream_completion=stream
    ).stream(request())
    assert (await anext(response)).delta

    await response.aclose()

    assert candidate.close_calls == 1
    assert "close-secret-body" not in caplog.text
    assert await store.load_chat(1) == []
    assert manager.submitted == []

    async def complete(messages):
        return {"content": "解锁", "usage": {}}

    await asyncio.wait_for(
        ChatService(store, manager, completion=complete).send(
            request(content="下一问")
        ),
        timeout=1,
    )


@pytest.mark.asyncio
async def test_stream_cancellation_preserves_cancelled_error_when_close_fails(
    tmp_path, monkeypatch, caplog
) -> None:
    monkeypatch.setattr("app.services.chat_service.settings.stream_guard_chars", 1)
    store = MarkdownMemoryStore(tmp_path)
    manager = MemoryManager()
    candidate = CloseFailingStream(["正文"], block_after_items=True)

    def stream(messages):
        return candidate

    response = ChatService(
        store, manager, stream_completion=stream
    ).stream(request())
    assert (await anext(response)).delta
    pending = asyncio.create_task(anext(response))
    await asyncio.sleep(0)
    pending.cancel()

    with pytest.raises(asyncio.CancelledError):
        await pending

    assert candidate.close_calls == 1
    assert "close-secret-body" not in caplog.text
    assert await store.load_chat(1) == []
    assert manager.submitted == []


@pytest.mark.asyncio
async def test_stream_consumer_close_does_not_persist_or_submit(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("app.services.chat_service.settings.stream_guard_chars", 1)
    store = MarkdownMemoryStore(tmp_path)
    manager = MemoryManager()
    upstream_closed = asyncio.Event()

    async def stream(messages: list[dict[str, Any]]):
        try:
            yield "第一段"
            await asyncio.Event().wait()
        finally:
            upstream_closed.set()

    response_stream = ChatService(
        store,
        manager,
        stream_completion=stream,
    ).stream(request())
    first = await anext(response_stream)
    assert first.delta.startswith(REQUIRED_OPENING)

    await response_stream.aclose()

    assert upstream_closed.is_set()
    assert await store.load_chat(1) == []
    assert manager.submitted == []


@pytest.mark.asyncio
async def test_stream_holds_same_session_turn_lock_until_normal_completion(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("app.services.chat_service.settings.stream_guard_chars", 1)
    store = MarkdownMemoryStore(tmp_path)
    release = asyncio.Event()
    send_started = asyncio.Event()

    async def stream(messages: list[dict[str, Any]]):
        yield "第一段"
        await release.wait()
        yield {"usage": USAGE}

    async def complete(messages: list[dict[str, Any]]) -> dict[str, Any]:
        send_started.set()
        return {"content": "第二答", "usage": USAGE}

    response_stream = ChatService(
        store,
        MemoryManager(),
        stream_completion=stream,
    ).stream(request(content="第一问"))
    assert (await anext(response_stream)).delta
    send_task = asyncio.create_task(
        ChatService(store, MemoryManager(), completion=complete).send(
            request(content="第二问")
        )
    )
    await asyncio.sleep(0)
    assert not send_started.is_set()

    release.set()
    remaining = await collect(response_stream)
    assert remaining[-1].done
    await send_task
    assert send_started.is_set()


@pytest.mark.parametrize(
    "pending_case",
    ["memory-assets", "cleanup", "missing-boundary"],
)
@pytest.mark.asyncio
async def test_stream_provider_never_receives_pending_stale_derived_context(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    pending_case: str,
) -> None:
    monkeypatch.setattr("app.services.chat_service.settings.stream_guard_chars", 1)
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "AUTHORITATIVE_RECENT_USER",
        "AUTHORITATIVE_RECENT_ASSISTANT",
        char_name="Character",
        user_name="User",
    )
    if pending_case == "memory-assets":
        await store.save_state(
            1,
            MemoryState(
                rebuild_memory_required=True,
                rebuild_assets_required=True,
                rebuild_from_message=2,
            ),
        )

    stale = DomainContext(
        story_state="STALE_STORY",
        memory="STALE_MEMORY",
        character_profiles=["STALE_PROFILE"],
        assets="STALE_ASSETS",
        episodes=["STALE_EPISODE"],
        rag=["STALE_RAG"],
        overall_summary="STALE_SUMMARY",
    )

    async def load_domain(records):
        if pending_case == "cleanup":
            await store.save_state(
                1,
                MemoryState(cleanup_required=True, rebuild_from_message=2),
            )
        elif pending_case == "missing-boundary":
            await store.save_state(1, MemoryState(rebuild_story_required=True))
        return stale

    captured: list[list[dict[str, Any]]] = []

    async def stream(messages: list[dict[str, Any]]):
        captured.append(messages)
        yield "正文"

    events = await collect(
        ChatService(
            store,
            None,
            stream_completion=stream,
            domain_context_loader=load_domain,
        ).stream(request(content="AUTHORITATIVE_INPUT"))
    )

    assert events[-1].done
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
async def test_stream_context_budget_failure_emits_safe_error(tmp_path) -> None:
    class FailingBuilder:
        def build(self, sources: Any) -> Any:
            raise ValueError("mandatory prompt context includes secret")

    async def unused(messages: list[dict[str, Any]]):
        raise AssertionError("stream must not run")
        yield  # pragma: no cover

    store = MarkdownMemoryStore(tmp_path)
    events = await collect(
        ChatService(
            store,
            MemoryManager(),
            stream_completion=unused,
            context_builder=FailingBuilder(),
        ).stream(request())
    )

    assert len(events) == 1
    assert events[0].error
    assert "secret" not in events[0].error
    assert await store.load_chat(1) == []


async def _seed_retry_pair(store: MarkdownMemoryStore) -> tuple[Any, Any]:
    await store.append_pair(
        1,
        "prefix user",
        "prefix assistant",
        char_name="灵汐",
        user_name="逸山",
    )
    await store.append_pair(
        1,
        "retry this",
        "original answer",
        char_name="灵汐",
        user_name="Original User",
        msg_type="ooc",
    )
    records = await store.load_chat(1)
    await store.save_state(
        1,
        MemoryState(
            last_memory_message=4,
            last_story_state_message=4,
            last_summary_message=0,
            last_episode_message=0,
            last_rag_message=4,
            last_character_message=4,
            last_assets_message=4,
        ),
    )
    await store.write_text(1, "story_state.md", "stale final-pair story")
    await store.write_text(1, "summary.md", "stale final-pair summary")
    return records[-2], records[-1]


@pytest.mark.asyncio
async def test_retry_replaces_exactly_one_final_pair_from_logically_retained_history(
    tmp_path,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    original_user, original_assistant = await _seed_retry_pair(store)
    manager = MemoryManager()
    builder = CapturingBuilder()
    loader_records: list[tuple[Any, ...]] = []

    async def loader(records):
        loader_records.append(records)
        return DomainContext(
            story_state="stale final-pair story",
            memory="preserved memory",
            character_profiles=["stale profile"],
            assets="stale assets",
            episodes=["stale episode"],
            rag=["stale rag"],
            overall_summary="stale final-pair summary",
        )

    async def complete(messages):
        return {"content": "replacement answer", "usage": USAGE}

    result = await ChatService(
        store,
        manager,
        completion=complete,
        context_builder=builder,
        domain_context_loader=loader,
    ).retry(request(content="/retry"))

    records = await store.load_chat(1)
    assert len(records) == 4
    assert records[-2] == original_user
    assert records[-1].number == original_assistant.number
    assert records[-1].name == original_assistant.name
    assert records[-1].timestamp == original_assistant.timestamp
    assert records[-1].content == f"{REQUIRED_OPENING}\n\nreplacement answer"
    assert len(loader_records) == 1
    assert [record.content for record in loader_records[0]] == [
        "prefix user",
        "prefix assistant",
    ]
    sources = builder.sources[0]
    assert sources.recent == [
        {"role": "user", "content": "prefix user"},
        {"role": "assistant", "content": "prefix assistant"},
    ]
    assert sources.user_message == "retry this"
    assert sources.memory == ""
    assert sources.story_state == ""
    assert sources.summary == ""
    assert sources.episodes == []
    assert sources.rag == []
    assert sources.assets == ""
    assert sources.character_profiles == []
    assert (await store.load_state(1)).rebuild_required
    assert await store.read_text(1, "story_state.md") == ""
    assert await store.read_text(1, "summary.md") == ""
    assert manager.submitted == [1]
    assert result.usage == USAGE


@pytest.mark.asyncio
async def test_retry_provider_failure_preserves_original_pair_and_state(tmp_path) -> None:
    store = MarkdownMemoryStore(tmp_path)
    await _seed_retry_pair(store)
    original_chat = await store.read_text(1, "chat.md")
    original_state = await store.load_state(1)
    manager = MemoryManager()

    async def fail(messages):
        raise RuntimeError("provider unavailable")

    with pytest.raises(RuntimeError, match="provider unavailable"):
        await ChatService(store, manager, completion=fail).retry(
            request(content="/retry")
        )

    assert await store.read_text(1, "chat.md") == original_chat
    assert await store.load_state(1) == original_state
    assert await store.read_text(1, "story_state.md") == "stale final-pair story"
    assert manager.submitted == []


@pytest.mark.asyncio
async def test_retry_context_budget_failure_preserves_original_pair_and_state(
    tmp_path,
) -> None:
    class FailingBuilder:
        def build(self, sources):
            raise ValueError("mandatory input too large")

    store = MarkdownMemoryStore(tmp_path)
    await _seed_retry_pair(store)
    original_chat = await store.read_text(1, "chat.md")
    original_state = await store.load_state(1)
    manager = MemoryManager()

    async def unused(messages):
        raise AssertionError("provider must not run")

    with pytest.raises(ContextBudgetExceeded):
        await ChatService(
            store,
            manager,
            completion=unused,
            context_builder=FailingBuilder(),
        ).retry(request(content="/retry"))

    assert await store.read_text(1, "chat.md") == original_chat
    assert await store.load_state(1) == original_state
    assert manager.submitted == []


@pytest.mark.asyncio
async def test_retry_refusal_exhaustion_preserves_original_pair_and_state(
    tmp_path,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    await _seed_retry_pair(store)
    original_chat = await store.read_text(1, "chat.md")
    original_state = await store.load_state(1)
    manager = MemoryManager()
    calls = 0

    async def refuse(messages):
        nonlocal calls
        calls += 1
        return {"content": "I cannot continue this story.", "usage": USAGE}

    with pytest.raises(OutputGuardError):
        await ChatService(store, manager, completion=refuse).retry(
            request(content="/retry")
        )

    assert calls == 2
    assert await store.read_text(1, "chat.md") == original_chat
    assert await store.load_state(1) == original_state
    assert manager.submitted == []


@pytest.mark.asyncio
async def test_retry_waits_for_pipeline_before_atomic_replacement(tmp_path) -> None:
    store = MarkdownMemoryStore(tmp_path)
    await _seed_retry_pair(store)
    original_chat = await store.read_text(1, "chat.md")
    provider_finished = asyncio.Event()

    async def complete(messages):
        provider_finished.set()
        return {"content": "replacement", "usage": {}}

    pipeline_lock = await store.pipeline_lock_for(1)
    await pipeline_lock.acquire()
    try:
        retry_task = asyncio.create_task(
            ChatService(store, MemoryManager(), completion=complete).retry(
                request(content="/retry")
            )
        )
        await provider_finished.wait()
        await asyncio.sleep(0)
        assert not retry_task.done()
        assert await store.read_text(1, "chat.md") == original_chat
    finally:
        pipeline_lock.release()

    await asyncio.wait_for(retry_task, timeout=1)
    assert (await store.load_chat(1))[-1].content.endswith("replacement")


@pytest.mark.asyncio
async def test_retry_submit_failure_keeps_single_committed_pair(tmp_path) -> None:
    store = MarkdownMemoryStore(tmp_path)
    await _seed_retry_pair(store)
    manager = MemoryManager(fail=True)

    async def complete(messages):
        return {"content": "replacement", "usage": {}}

    await ChatService(store, manager, completion=complete).retry(
        request(content="/retry")
    )

    records = await store.load_chat(1)
    assert len(records) == 4
    assert [record.content for record in records].count("retry this") == 1
    assert [record.content for record in records].count(
        f"{REQUIRED_OPENING}\n\nreplacement"
    ) == 1
    assert manager.submitted == [1]


@pytest.mark.asyncio
async def test_retry_post_commit_cleanup_failure_returns_replacement_and_submits_once(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.services.md_store as md_store_module

    store = MarkdownMemoryStore(tmp_path)
    await _seed_retry_pair(store)
    manager = MemoryManager()
    real_replace = md_store_module.os.replace
    cleanup_failures = 0

    def fail_story_once(source, target) -> None:
        nonlocal cleanup_failures
        if target.name == "story_state.md" and cleanup_failures == 0:
            cleanup_failures += 1
            raise OSError("post-commit cleanup failed")
        real_replace(source, target)

    async def complete(messages):
        return {"content": "replacement", "usage": USAGE}

    monkeypatch.setattr(md_store_module.os, "replace", fail_story_once)

    result = await ChatService(store, manager, completion=complete).retry(
        request(content="/retry")
    )

    committed = await store.load_chat(1)
    assert result.content == f"{REQUIRED_OPENING}\n\nreplacement"
    assert result.usage == USAGE
    assert [record.content for record in committed] == [
        "prefix user",
        "prefix assistant",
        "retry this",
        f"{REQUIRED_OPENING}\n\nreplacement",
    ]
    assert manager.submitted == [1]
    assert cleanup_failures == 1

    monkeypatch.setattr(md_store_module.os, "replace", real_replace)
    await store.recover_invalidation(1)

    assert await store.load_chat(1) == committed
    assert not (tmp_path / "1" / "invalidation_intent.md").exists()


@pytest.mark.asyncio
async def test_retry_of_only_pair_arms_full_rebuild_from_replacement_pair(
    tmp_path,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "only user",
        "only assistant",
        char_name="Character",
        user_name="User",
    )

    async def complete(messages):
        return {"content": "replacement", "usage": USAGE}

    await ChatService(store, MemoryManager(), completion=complete).retry(request())

    state = await store.load_state(1)
    assert state.rebuild_from_message == 0
    assert state.rebuild_story_required
    assert state.rebuild_memory_required
    assert state.rebuild_episode_required
    assert state.rebuild_summary_required
    assert state.rebuild_rag_required
    assert state.rebuild_assets_required


@pytest.mark.asyncio
async def test_undo_only_pair_reconciles_all_active_derivations_to_empty(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "episode_size_messages", 2)
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "only user",
        "only assistant",
        char_name="Character",
        user_name="User",
    )
    await store.write_text(1, "memory.md", "STALE_MEMORY")
    await store.write_text(1, "characters/Old.md", "# Old\n\nSTALE_PROFILE\n")
    await store.write_text(1, "assets.md", "STALE_ASSETS")
    await store.write_text(1, "assets_summary.txt", "STALE_ASSET_SUMMARY")
    await store.write_text(1, "story_state.md", "STALE_STORY")
    await store.write_text(1, "summary.md", "STALE_SUMMARY")
    await store.write_text(
        1,
        "episodes/episode-000001.md",
        "# Episode 000001\n\n"
        "<!-- messages: 1-2 -->\n\n"
        "## 剧情摘要\n- stale\n\n"
        "## 状态变化\n- stale\n\n"
        "## 承诺与伏笔\n- stale\n",
    )
    await store.write_text(
        1,
        "rag/index.json",
        '{"chunks": [{"text": "stale", "start_message": 1, '
        '"end_message": 2}], "embeddings": [[1.0]], "indexed_messages": 2}',
    )
    await store.save_state(
        1,
        MemoryState(
            last_memory_message=2,
            last_story_state_message=2,
            last_summary_message=2,
            last_episode_message=2,
            last_rag_message=2,
            last_character_message=2,
            last_assets_message=2,
        ),
    )

    await ChatService(store, MemoryManager()).undo(1)

    session = tmp_path / "1"
    assert await store.load_chat(1) == []
    assert await store.load_state(1) == MemoryState()
    assert not (session / "memory.md").exists()
    assert not list((session / "characters").glob("*.md"))
    assert not (session / "assets.md").exists()
    assert not (session / "assets_summary.txt").exists()
    assert await store.read_text(1, "story_state.md") == ""
    assert await store.read_text(1, "summary.md") == ""
    assert not list((session / "episodes").glob("episode-*.md"))
    assert json.loads(await store.read_text(1, "rag/index.json")) == {
        "chunks": [],
        "embeddings": [],
        "indexed_messages": 0,
        "source_count": 0,
        "source_sha256": (
            "e3b0c44298fc1c149afbf4c8996fb924"
            "27ae41e4649b934ca495991b7852b855"
        ),
    }


@pytest.mark.asyncio
async def test_undo_removes_one_complete_pair_invalidates_and_submits(tmp_path) -> None:
    store = MarkdownMemoryStore(tmp_path)
    await _seed_retry_pair(store)
    manager = MemoryManager()

    retained = await ChatService(store, manager).undo(1)

    assert [record.content for record in retained] == [
        "prefix user",
        "prefix assistant",
    ]
    assert retained == await store.load_chat(1)
    assert (await store.load_state(1)).rebuild_required
    assert manager.submitted == [1]


@pytest.mark.asyncio
async def test_undo_post_commit_cleanup_failure_returns_retained_and_submits_once(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.services.md_store as md_store_module

    store = MarkdownMemoryStore(tmp_path)
    await _seed_retry_pair(store)
    manager = MemoryManager()
    real_replace = md_store_module.os.replace
    cleanup_failures = 0

    def fail_story_once(source, target) -> None:
        nonlocal cleanup_failures
        if target.name == "story_state.md" and cleanup_failures == 0:
            cleanup_failures += 1
            raise OSError("post-commit cleanup failed")
        real_replace(source, target)

    monkeypatch.setattr(md_store_module.os, "replace", fail_story_once)

    retained = await ChatService(store, manager).undo(1)

    assert [record.content for record in retained] == [
        "prefix user",
        "prefix assistant",
    ]
    assert await store.load_chat(1) == retained
    assert manager.submitted == [1]
    assert cleanup_failures == 1

    monkeypatch.setattr(md_store_module.os, "replace", real_replace)
    await store.recover_invalidation(1)

    assert await store.load_chat(1) == retained
    assert not (tmp_path / "1" / "invalidation_intent.md").exists()


@pytest.mark.asyncio
async def test_undo_rejects_incomplete_final_pair_without_side_effects(tmp_path) -> None:
    store = MarkdownMemoryStore(tmp_path)
    await store.append_record(1, "user", "incomplete", name="User")
    original = await store.read_text(1, "chat.md")
    manager = MemoryManager()

    with pytest.raises(NoCompletePairError, match="complete user/assistant pair"):
        await ChatService(store, manager).undo(1)

    assert await store.read_text(1, "chat.md") == original
    assert manager.submitted == []


@pytest.mark.parametrize("history", ["empty", "incomplete"])
@pytest.mark.parametrize("operation", ["undo", "retry"])
@pytest.mark.asyncio
async def test_mutating_commands_raise_narrow_error_for_no_complete_pair(
    tmp_path,
    history: str,
    operation: str,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    if history == "incomplete":
        await store.append_record(1, "user", "incomplete", name="User")
    original = await store.read_text(1, "chat.md")
    completion_calls = 0

    async def complete(messages):
        nonlocal completion_calls
        completion_calls += 1
        return {"content": "must not run", "usage": {}}

    service = ChatService(store, MemoryManager(), completion=complete)

    with pytest.raises(NoCompletePairError):
        if operation == "undo":
            await service.undo(1)
        else:
            await service.retry(request())

    assert await store.read_text(1, "chat.md") == original
    assert completion_calls == 0


@pytest.mark.asyncio
async def test_undo_waits_for_inflight_send_turn(tmp_path) -> None:
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "old user",
        "old assistant",
        char_name="灵汐",
        user_name="逸山",
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def complete(messages):
        started.set()
        await release.wait()
        return {"content": "new assistant", "usage": {}}

    send_task = asyncio.create_task(
        ChatService(store, MemoryManager(), completion=complete).send(
            request(content="new user")
        )
    )
    await started.wait()
    undo_task = asyncio.create_task(ChatService(store, MemoryManager()).undo(1))
    await asyncio.sleep(0)
    assert not undo_task.done()

    release.set()
    await asyncio.gather(send_task, undo_task)

    assert [record.content for record in await store.load_chat(1)] == [
        "old user",
        "old assistant",
    ]


@pytest.mark.asyncio
async def test_undo_waits_for_inflight_stream_turn(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("app.services.chat_service.settings.stream_guard_chars", 1)
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "old user",
        "old assistant",
        char_name="灵汐",
        user_name="逸山",
    )
    release = asyncio.Event()

    async def stream(messages):
        yield "new assistant"
        await release.wait()

    response = ChatService(
        store, MemoryManager(), stream_completion=stream
    ).stream(request(content="new user"))
    assert (await anext(response)).delta
    pending_stream = asyncio.create_task(anext(response))
    undo_task = asyncio.create_task(ChatService(store, MemoryManager()).undo(1))
    await asyncio.sleep(0)
    assert not undo_task.done()

    release.set()
    assert (await pending_stream).done
    await response.aclose()
    await undo_task

    assert [record.content for record in await store.load_chat(1)] == [
        "old user",
        "old assistant",
    ]
