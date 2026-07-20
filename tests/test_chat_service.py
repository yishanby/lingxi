from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

from app.services.chat_service import (
    ChatService,
    ContextBudgetExceeded,
    StreamEvent,
    TurnRequest,
)
from app.services.md_store import MarkdownMemoryStore
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


class CapturingBuilder:
    def __init__(self, messages: list[dict[str, str]] | None = None) -> None:
        self.sources: list[Any] = []
        self.messages = messages or [{"role": "user", "content": "built"}]

    def build(self, sources: Any) -> Any:
        self.sources.append(sources)
        return SimpleNamespace(messages=self.messages)


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
    assert "memory queue unavailable" in caplog.text


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


@pytest.mark.parametrize("fail_after_text", [False, True])
@pytest.mark.asyncio
async def test_stream_provider_failure_never_persists_partial_pair(
    tmp_path, monkeypatch, fail_after_text: bool
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
    assert not events[-1].done
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
