from __future__ import annotations

import asyncio
import ast
import inspect
import json
import logging
import os
from pathlib import Path
import subprocess

import pytest

from app.config import settings
from app.services import llm, memory, rag, story_memory
from app.services.chat_service import ChatService
from app.services.md_store import (
    ChatRecord,
    InvalidationIntentError,
    MarkdownMemoryStore,
    MemoryState,
    render_chat_records,
)
from app.services.memory_pipeline import MemoryPipeline
from app.services.memory_tasks import MemoryTaskManager
from app.services.stage_receipts import (
    ArtifactIdentity,
    ChatSourceIdentity,
    RECEIPT_MAX_BYTES,
    StageReceipt,
    StageUpdateResult,
    chat_source_identity,
    parse_receipt,
    render_receipt,
    text_artifact,
)


BACKEND = {
    "provider": "openai",
    "api_key": "test-key",
    "model": "test-model",
    "base_url": "https://example.invalid/v1",
    "params": {},
}


@pytest.mark.parametrize(
    "invalid_path",
    [1, "", "../memory.md", "./memory.md", "/memory.md", "characters\\Alice.md"],
)
def test_receipt_parser_rejects_noncanonical_artifact_paths(
    invalid_path: object,
) -> None:
    document = {
        "version": 1,
        "stage": "memory",
        "source_count": 2,
        "source_sha256": "0" * 64,
        "checkpoint": 2,
        "artifacts": [
            {
                "path": invalid_path,
                "sha256": "0" * 64,
                "byte_size": 1,
            }
        ],
        "inputs": [],
    }

    with pytest.raises(ValueError, match="invalid stage receipt"):
        parse_receipt(json.dumps(document))


def test_receipt_renderer_rejects_oversized_document() -> None:
    inputs = tuple(
        ArtifactIdentity(
            path=f"episodes/episode-{number:06d}.md",
            sha256="0" * 64,
            byte_size=0,
        )
        for number in range(1_000)
    )
    receipt = StageReceipt(
        version=1,
        stage="summary",
        source=ChatSourceIdentity(count=2, sha256="0" * 64),
        checkpoint=2,
        artifacts=(),
        inputs=inputs,
    )

    with pytest.raises(ValueError, match="invalid stage receipt"):
        render_receipt(receipt)


def test_receipt_parser_normalizes_deep_json_recursion() -> None:
    deeply_nested = "[" * 30_000 + "]" * 30_000
    assert len(deeply_nested.encode("utf-8")) < RECEIPT_MAX_BYTES

    with pytest.raises(ValueError, match="invalid stage receipt"):
        parse_receipt(deeply_nested)


def test_receipt_parser_rejects_duplicate_json_keys() -> None:
    receipt = StageReceipt(
        version=1,
        stage="memory",
        source=ChatSourceIdentity(count=2, sha256="0" * 64),
        checkpoint=2,
        artifacts=(),
        inputs=(),
    )
    duplicated = render_receipt(receipt).replace(
        '"version":1,',
        '"version":1,"version":1,',
        1,
    )

    with pytest.raises(ValueError, match="invalid stage receipt"):
        parse_receipt(duplicated)


@pytest.mark.parametrize("version", [True, 1.0])
def test_receipt_parser_rejects_non_integer_version(version: object) -> None:
    document = {
        "version": version,
        "stage": "memory",
        "source_count": 2,
        "source_sha256": "0" * 64,
        "checkpoint": 2,
        "artifacts": [],
        "inputs": [],
    }

    with pytest.raises(ValueError, match="invalid stage receipt"):
        parse_receipt(json.dumps(document))


def _episode_document(number: int, start: int, end: int) -> str:
    return (
        f"# Episode {number:06d}\n\n"
        f"<!-- messages: {start}-{end} -->\n\n"
        "## 剧情摘要\n- 已发生的事件\n\n"
        "## 状态变化\n- 状态已变化\n\n"
        "## 承诺与伏笔\n- 保留的事实\n"
    )


async def _completed_story_result(
    store: MarkdownMemoryStore,
    session_id: int,
    source: ChatSourceIdentity,
    *,
    text: str = "story state\n",
) -> StageUpdateResult:
    await store.write_text(session_id, "story_state.md", text)
    return StageUpdateResult(
        stage="story",
        completed=True,
        source=source,
        checkpoint=source.count,
        artifacts=(text_artifact("story_state.md", text),),
    )


async def _completed_memory_result(
    store: MarkdownMemoryStore,
    session_id: int,
    source: ChatSourceIdentity,
    *,
    text: str = "memory\n",
) -> StageUpdateResult:
    await store.write_text(session_id, "memory.md", text)
    return StageUpdateResult(
        stage="memory",
        completed=True,
        source=source,
        checkpoint=source.count,
        artifacts=(text_artifact("memory.md", text),),
    )


async def _completed_rag_result(
    store: MarkdownMemoryStore,
    session_id: int,
    source: ChatSourceIdentity,
) -> StageUpdateResult:
    index = {
        "chunks": [],
        "embeddings": [],
        "indexed_messages": source.count,
        "source_count": source.count,
        "source_sha256": source.sha256,
    }
    text = json.dumps(index, ensure_ascii=False)
    await store.write_text(session_id, "rag/index.json", text)
    return StageUpdateResult(
        stage="rag",
        completed=True,
        source=source,
        checkpoint=source.count,
        artifacts=(text_artifact("rag/index.json", text),),
    )


async def _completed_episode_result(
    store: MarkdownMemoryStore,
    session_id: int,
    source: ChatSourceIdentity,
    checkpoint: int,
) -> StageUpdateResult:
    episode_directory = store.session_dir(session_id) / "episodes"
    artifacts = []
    if episode_directory.exists():
        for path in sorted(episode_directory.glob("episode-*.md")):
            relative = f"episodes/{path.name}"
            text = await store.read_text(session_id, relative)
            artifacts.append(text_artifact(relative, text))
    return StageUpdateResult(
        stage="episode",
        completed=True,
        source=source,
        checkpoint=checkpoint,
        artifacts=tuple(artifacts),
    )


async def _completed_summary_result(
    store: MarkdownMemoryStore,
    session_id: int,
    source: ChatSourceIdentity,
    checkpoint: int,
    *,
    text: str = "summary\n",
) -> StageUpdateResult:
    inputs = []
    episode_directory = store.session_dir(session_id) / "episodes"
    if episode_directory.exists():
        for path in sorted(episode_directory.glob("episode-*.md")):
            relative = f"episodes/{path.name}"
            episode = await store.read_text(session_id, relative)
            inputs.append(text_artifact(relative, episode))
    if checkpoint == 0 and not inputs:
        store.file_path(session_id, "summary.md").unlink(missing_ok=True)
        artifacts = ()
    else:
        await store.write_text(session_id, "summary.md", text)
        artifacts = (text_artifact("summary.md", text),)
    return StageUpdateResult(
        stage="summary",
        completed=True,
        source=source,
        checkpoint=checkpoint,
        artifacts=artifacts,
        inputs=tuple(inputs),
    )


async def _completed_assets_result(
    store: MarkdownMemoryStore,
    session_id: int,
    source: ChatSourceIdentity,
    *,
    assets: str = "assets\n",
    summary: str = "assets summary\n",
) -> StageUpdateResult:
    await store.write_text(session_id, "assets.md", assets)
    await store.write_text(session_id, "assets_summary.txt", summary)
    return StageUpdateResult(
        stage="assets",
        completed=True,
        source=source,
        checkpoint=source.count,
        artifacts=(
            text_artifact("assets.md", assets),
            text_artifact("assets_summary.txt", summary),
        ),
    )


async def _seed_stale_retry_artifacts(
    store: MarkdownMemoryStore,
) -> list[ChatRecord]:
    for pair in range(1, 3):
        await store.append_pair(
            1,
            f"user-{pair}",
            f"assistant-{pair}",
            char_name="Character",
            user_name="User",
        )
    records = await store.load_chat(1)
    await store.write_text(1, "memory.md", "stale memory through four")
    await store.write_text(1, "story_state.md", "stale story through four")
    await store.write_text(1, "summary.md", "stale summary through four")
    await store.write_text(1, "episodes/episode-000001.md", _episode_document(1, 1, 2))
    await store.write_text(1, "episodes/episode-000002.md", _episode_document(2, 3, 4))
    await store.write_text(
        1,
        "rag/index.json",
        '{"chunks": ['
        '{"text": "one", "start_message": 1, "end_message": 2},'
        '{"text": "stale", "start_message": 3, "end_message": 4}'
        '], "embeddings": [[1], [2]], "indexed_messages": 4}',
    )
    await store.save_state(
        1,
        MemoryState(
            last_memory_message=4,
            last_story_state_message=4,
            last_summary_message=4,
            last_episode_message=4,
            last_rag_message=4,
            last_character_message=4,
            last_assets_message=4,
        ),
    )
    return records


def isolate_story_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "story_state_interval_messages", 2)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 100)
    monkeypatch.setattr(settings, "episode_size_messages", 100)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 100)
    monkeypatch.setattr(settings, "assets_interval_messages", 100)


@pytest.mark.asyncio
async def test_failed_story_stage_keeps_checkpoint_and_persists_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isolate_story_stage(monkeypatch)
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "hello",
        "hi",
        char_name="Character",
        user_name="User",
    )

    async def fail_story(*args, **kwargs):
        raise RuntimeError("story broke")

    monkeypatch.setattr(story_memory, "update_story_state", fail_story)

    with pytest.raises(RuntimeError, match="story broke"):
        await MemoryPipeline(store).run(1, BACKEND)

    state = await store.load_state(1)
    assert state.last_story_state_message == 0
    assert state.last_error == "RuntimeError: story broke"


@pytest.mark.asyncio
async def test_retry_processes_the_same_story_range_and_advances_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isolate_story_stage(monkeypatch)
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "hello",
        "hi",
        char_name="Character",
        user_name="User",
    )
    attempted_ranges: list[list[int]] = []

    async def fail_once_then_succeed(
        store_arg,
        session_id,
        records,
        complete,
        *,
        source,
    ):
        attempted_ranges.append([record.number for record in records])
        if len(attempted_ranges) == 1:
            raise RuntimeError("try again")
        return await _completed_story_result(store_arg, session_id, source)

    monkeypatch.setattr(
        story_memory,
        "update_story_state",
        fail_once_then_succeed,
    )
    pipeline = MemoryPipeline(store)

    with pytest.raises(RuntimeError, match="try again"):
        await pipeline.run(1, BACKEND)
    await pipeline.run(1, BACKEND)

    state = await store.load_state(1)
    assert attempted_ranges == [[1, 2], [1, 2]]
    assert state.last_story_state_message == 2
    assert state.last_error == ""


@pytest.mark.asyncio
async def test_duplicate_run_does_not_repeat_finished_story_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isolate_story_stage(monkeypatch)
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "hello",
        "hi",
        char_name="Character",
        user_name="User",
    )
    calls = 0

    async def update_once(
        store_arg,
        session_id,
        records,
        complete,
        *,
        source,
    ):
        nonlocal calls
        calls += 1
        return await _completed_story_result(store_arg, session_id, source)

    monkeypatch.setattr(story_memory, "update_story_state", update_once)
    pipeline = MemoryPipeline(store)

    await pipeline.run(1, BACKEND)
    await pipeline.run(1, BACKEND)

    assert calls == 1


@pytest.mark.asyncio
async def test_pipeline_runs_all_stages_in_order_with_exact_ranges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "story_state_interval_messages", 2)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 2)
    monkeypatch.setattr(settings, "episode_size_messages", 2)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 2)
    monkeypatch.setattr(settings, "assets_interval_messages", 2)
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "hello",
        "hi",
        char_name="Character",
        user_name="User",
    )
    calls: list[str] = []

    async def update_story(
        store_arg,
        session_id,
        records,
        complete,
        *,
        source,
    ):
        assert store_arg is store
        assert [record.number for record in records] == [1, 2]
        calls.append("story")
        return await _completed_story_result(store_arg, session_id, source)

    async def extract(session_id, messages, backend, *, store, source):
        assert messages == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        calls.append("memory")
        return await _completed_memory_result(store, session_id, source)

    async def create_episodes(
        store_arg,
        session_id,
        records,
        last_episode_message,
        complete,
        *,
        episode_size,
        source,
    ):
        assert [record.number for record in records] == [1, 2]
        assert last_episode_message == 0
        assert episode_size == 2
        calls.append("episode")
        document = _episode_document(1, 1, 2)
        await store_arg.write_text(
            session_id,
            "episodes/episode-000001.md",
            document,
        )
        return StageUpdateResult(
            stage="episode",
            completed=True,
            source=source,
            checkpoint=2,
            artifacts=(
                text_artifact("episodes/episode-000001.md", document),
            ),
        )

    async def update_summary(
        store_arg,
        session_id,
        last_summary_message,
        complete,
        *,
        max_tokens,
        source,
    ):
        assert last_summary_message == 0
        assert max_tokens == settings.summary_max_tokens
        calls.append("summary")
        summary = "summary\n"
        episode = await store_arg.read_text(
            session_id,
            "episodes/episode-000001.md",
        )
        await store_arg.write_text(session_id, "summary.md", summary)
        return StageUpdateResult(
            stage="summary",
            completed=True,
            source=source,
            checkpoint=2,
            artifacts=(text_artifact("summary.md", summary),),
            inputs=(
                text_artifact("episodes/episode-000001.md", episode),
            ),
        )

    async def build_rag(session_id, *, store, source, **kwargs):
        calls.append("rag")
        return await _completed_rag_result(store, session_id, source)

    async def update_assets(session_id, messages, backend, *, store, source):
        assert messages == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        calls.append("assets")
        return await _completed_assets_result(store, session_id, source)

    monkeypatch.setattr(story_memory, "update_story_state", update_story)
    monkeypatch.setattr(memory, "extract_memory_and_characters", extract)
    monkeypatch.setattr(story_memory, "create_due_episodes", create_episodes)
    monkeypatch.setattr(
        story_memory,
        "update_summary_from_episodes",
        update_summary,
    )
    monkeypatch.setattr(rag, "build_index", build_rag)
    monkeypatch.setattr(memory, "update_assets", update_assets)

    await MemoryPipeline(store).run(1, BACKEND)

    state = await store.load_state(1)
    assert calls == ["story", "memory", "episode", "summary", "rag", "assets"]
    assert state.last_story_state_message == 2
    assert state.last_memory_message == 2
    assert state.last_character_message == 2
    assert state.last_episode_message == 2
    assert state.last_summary_message == 2
    assert state.last_rag_message == 2
    assert state.last_assets_message == 2


@pytest.mark.asyncio
async def test_complete_text_returns_backend_bound_callback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    async def complete(**kwargs):
        captured.update(kwargs)
        return {"content": "result"}

    monkeypatch.setattr("app.services.llm.chat_completion", complete)
    callback = MemoryPipeline(MarkdownMemoryStore(tmp_path))._complete_text(BACKEND)
    messages = [{"role": "user", "content": "hello"}]

    result = await callback(messages)

    assert result == "result"
    assert captured == {
        "provider": BACKEND["provider"],
        "api_key": BACKEND["api_key"],
        "model": BACKEND["model"],
        "base_url": BACKEND["base_url"],
        "messages": messages,
        "params": {},
    }


@pytest.mark.asyncio
async def test_memory_extraction_preserves_the_exact_pending_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    captured_prompt = ""
    pending = [
        {"role": "user", "content": f"pending-{number}"}
        for number in range(1, 22)
    ]
    for message in pending:
        await store.append_record(
            1,
            message["role"],
            message["content"],
            name="User",
        )
    source = chat_source_identity(await store.load_chat(1))

    async def capture_completion(**kwargs):
        nonlocal captured_prompt
        captured_prompt = kwargs["messages"][1]["content"]
        return {"content": "NO_CHANGE"}

    monkeypatch.setattr(memory, "chat_completion", capture_completion)
    monkeypatch.setattr(settings, "rag_auto_index", False)

    result = await memory.extract_memory_and_characters(
        1,
        pending,
        BACKEND,
        store=store,
        source=source,
    )

    assert result.stage == "memory"
    assert result.completed is True
    assert result.source == source
    assert result.checkpoint == 21
    assert "user: pending-1\n" in captured_prompt
    assert "user: pending-21\n" in captured_prompt


@pytest.mark.asyncio
async def test_asset_update_preserves_the_exact_pending_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    captured_prompt = ""
    pending = [
        {"role": "user", "content": f"asset-pending-{number}"}
        for number in range(1, 22)
    ]
    for message in pending:
        await store.append_record(
            1,
            message["role"],
            message["content"],
            name="User",
        )
    source = chat_source_identity(await store.load_chat(1))

    async def capture_completion(**kwargs):
        nonlocal captured_prompt
        captured_prompt = kwargs["messages"][1]["content"]
        return {"content": "NO_CHANGE"}

    monkeypatch.setattr(memory, "chat_completion", capture_completion)

    result = await memory.update_assets(
        1,
        pending,
        BACKEND,
        store=store,
        source=source,
    )

    assert result == StageUpdateResult(
        stage="assets",
        completed=True,
        source=source,
        checkpoint=21,
    )
    assert "user: asset-pending-1\n" in captured_prompt
    assert "user: asset-pending-21\n" in captured_prompt


@pytest.mark.asyncio
async def test_memory_service_failure_does_not_advance_memory_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isolate_story_stage(monkeypatch)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 2)
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "hello",
        "hi",
        char_name="Character",
        user_name="User",
    )

    async def update_story(
        store_arg,
        session_id,
        records,
        complete,
        *,
        source,
    ):
        return await _completed_story_result(store_arg, session_id, source)

    async def no_memory(session_id):
        return ""

    async def no_characters(session_id):
        return []

    async def fail_completion(**kwargs):
        raise ConnectionError("memory unavailable")

    monkeypatch.setattr(story_memory, "update_story_state", update_story)
    monkeypatch.setattr(memory, "load_memory", no_memory)
    monkeypatch.setattr(memory, "list_character_names", no_characters)
    monkeypatch.setattr(memory, "chat_completion", fail_completion)

    with pytest.raises(ConnectionError, match="memory unavailable"):
        await MemoryPipeline(store).run(1, BACKEND)

    state = await store.load_state(1)
    assert state.last_story_state_message == 2
    assert state.last_memory_message == 0
    assert state.last_character_message == 0
    assert state.last_error == "ConnectionError: memory unavailable"


@pytest.mark.asyncio
async def test_asset_service_failure_does_not_advance_asset_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isolate_story_stage(monkeypatch)
    monkeypatch.setattr(settings, "assets_interval_messages", 2)
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "hello",
        "hi",
        char_name="Character",
        user_name="User",
    )

    async def update_story(
        store_arg,
        session_id,
        records,
        complete,
        *,
        source,
    ):
        return await _completed_story_result(store_arg, session_id, source)

    async def no_assets(session_id):
        return ""

    async def fail_completion(**kwargs):
        raise TimeoutError("assets unavailable")

    monkeypatch.setattr(story_memory, "update_story_state", update_story)
    monkeypatch.setattr(memory, "load_assets", no_assets)
    monkeypatch.setattr(memory, "chat_completion", fail_completion)

    with pytest.raises(TimeoutError, match="assets unavailable"):
        await MemoryPipeline(store).run(1, BACKEND)

    state = await store.load_state(1)
    assert state.last_story_state_message == 2
    assert state.last_assets_message == 0
    assert state.last_error == "TimeoutError: assets unavailable"


@pytest.mark.parametrize(
    ("stage", "flag"),
    [
        ("story", "rebuild_story_required"),
        ("memory", "rebuild_memory_required"),
        ("episode", "rebuild_episode_required"),
        ("summary", "rebuild_summary_required"),
        ("rag", "rebuild_rag_required"),
        ("assets", "rebuild_assets_required"),
    ],
)
@pytest.mark.asyncio
async def test_silent_stage_result_never_advances_forced_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    flag: str,
) -> None:
    monkeypatch.setattr(settings, "story_state_interval_messages", 100)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 100)
    monkeypatch.setattr(settings, "episode_size_messages", 100)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 100)
    monkeypatch.setattr(settings, "assets_interval_messages", 100)
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "user",
        "assistant",
        char_name="Character",
        user_name="User",
    )
    await store.save_state(
        1,
        MemoryState(**{flag: True}, rebuild_from_message=0),
    )

    async def silent(*args, **kwargs):
        return None

    targets = {
        "story": (story_memory, "update_story_state"),
        "memory": (memory, "rebuild_memory_and_profiles"),
        "episode": (story_memory, "create_due_episodes"),
        "summary": (story_memory, "update_summary_from_episodes"),
        "rag": (rag, "build_index"),
        "assets": (memory, "rebuild_assets"),
    }
    module, name = targets[stage]
    monkeypatch.setattr(module, name, silent)

    with pytest.raises(
        RuntimeError,
        match=rf"{stage} stage did not return a completed result",
    ):
        await MemoryPipeline(store).run(1, BACKEND)

    state = await store.load_state(1)
    assert getattr(state, flag)
    assert state.rebuild_from_message == 0


@pytest.mark.asyncio
async def test_asset_summary_failure_does_not_partially_replace_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    await store.append_record(
        1,
        "user",
        "bought a house",
        name="User",
    )
    await store.write_text(1, "assets.md", "old assets")
    source = chat_source_identity(await store.load_chat(1))
    completion_calls = 0

    async def complete(**kwargs):
        nonlocal completion_calls
        completion_calls += 1
        if completion_calls == 1:
            return {"content": "new assets"}
        raise TimeoutError("summary unavailable")

    monkeypatch.setattr(memory, "chat_completion", complete)

    with pytest.raises(TimeoutError, match="summary unavailable"):
        await memory.update_assets(
            1,
            [{"role": "user", "content": "bought a house"}],
            BACKEND,
            store=store,
            source=source,
        )

    assert await store.read_text(1, "assets.md") == "old assets"
    assert not store.file_path(1, "assets_summary.txt").exists()


@pytest.mark.asyncio
async def test_memory_extraction_does_not_build_rag_outside_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    await store.append_record(1, "user", "pending", name="User")
    source = chat_source_identity(await store.load_chat(1))
    rag_calls = 0

    async def complete(**kwargs):
        return {"content": "NO_CHANGE"}

    async def track_rag(*args, **kwargs):
        nonlocal rag_calls
        rag_calls += 1
        return await _completed_rag_result(store, 1, source)

    monkeypatch.setattr(memory, "chat_completion", complete)
    monkeypatch.setattr(rag, "build_index", track_rag)
    monkeypatch.setattr(settings, "rag_auto_index", True)

    result = await memory.extract_memory_and_characters(
        1,
        [{"role": "user", "content": "pending"}],
        BACKEND,
        store=store,
        source=source,
    )

    assert result.stage == "memory"
    assert result.completed is True
    assert result.source == source
    assert result.checkpoint == 1
    assert rag_calls == 0


@pytest.mark.asyncio
async def test_rag_chunks_store_message_ranges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "hello",
        "hi",
        char_name="Character",
        user_name="User",
    )
    embedded: list[str] = []

    async def embed(texts, **kwargs):
        embedded.extend(texts)
        return [[1.0] for _ in texts]

    monkeypatch.setattr(rag, "get_embeddings", embed)

    source = chat_source_identity(await store.load_chat(1))
    result = await rag.build_index(1, store=store, source=source)
    index = await rag.load_index(1, store=store)

    assert result.stage == "rag"
    assert result.completed is True
    assert result.source == source
    assert result.checkpoint == 2
    assert index["chunks"] == [
        {
            "text": "用户: hello\n角色: hi",
            "start_message": 1,
            "end_message": 2,
        }
    ]
    assert embedded == ["用户: hello\n角色: hi"]
    assert index["indexed_messages"] == 2


def test_rag_build_index_requires_explicit_store() -> None:
    store_parameter = inspect.signature(rag.build_index).parameters["store"]

    assert store_parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert store_parameter.default is inspect.Parameter.empty


@pytest.mark.asyncio
async def test_rebuild_character_from_history_builds_with_authoritative_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rag, "MEMORY_BASE", tmp_path)
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "hello",
        "hi",
        char_name="Character",
        user_name="User",
    )
    expected_source = chat_source_identity(await store.load_chat(1))
    build_calls: list[tuple[int, Path, ChatSourceIdentity]] = []

    async def build_index(
        session_id: int,
        *,
        store: MarkdownMemoryStore,
        source: ChatSourceIdentity,
        **kwargs,
    ) -> StageUpdateResult:
        build_calls.append((session_id, store.base, source))
        return await _completed_rag_result(store, session_id, source)

    async def no_results(*args, **kwargs):
        return []

    monkeypatch.setattr(rag, "build_index", build_index)
    monkeypatch.setattr(rag, "search_character", no_results)

    result = await rag.rebuild_character_from_history(1, "Alice")

    assert result == "# Alice\n\n(未找到相关历史记录)"
    assert build_calls == [(1, tmp_path.resolve(), expected_source)]


@pytest.mark.filterwarnings(
    "ignore:datetime.datetime.utcfromtimestamp.*:DeprecationWarning"
)
@pytest.mark.filterwarnings(
    "ignore:There is no current event loop:DeprecationWarning"
)
def test_feishu_chars_index_builds_and_loads_with_authoritative_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.services.feishu_ws_worker as worker

    store = MarkdownMemoryStore(tmp_path)

    async def seed_chat() -> ChatSourceIdentity:
        await store.append_pair(
            1,
            "hello",
            "hi",
            char_name="Character",
            user_name="User",
        )
        return chat_source_identity(await store.load_chat(1))

    expected_source = asyncio.run(seed_chat())
    build_calls: list[tuple[int, Path, ChatSourceIdentity]] = []
    sent: list[str] = []

    async def build_index(
        session_id: int,
        *,
        store: MarkdownMemoryStore,
        source: ChatSourceIdentity,
        **kwargs,
    ) -> StageUpdateResult:
        build_calls.append((session_id, store.base, source))
        index = {
            "chunks": [
                {
                    "text": "用户: hello\n角色: hi",
                    "start_message": 1,
                    "end_message": 2,
                }
            ],
            "embeddings": [[1.0]],
            "indexed_messages": 2,
            "source_count": source.count,
            "source_sha256": source.sha256,
        }
        await rag.save_index(session_id, index, store=store)
        text = json.dumps(index, ensure_ascii=False)
        return StageUpdateResult(
            stage="rag",
            completed=True,
            source=source,
            checkpoint=2,
            artifacts=(text_artifact("rag/index.json", text),),
        )

    monkeypatch.setattr(rag, "MEMORY_BASE", tmp_path)
    monkeypatch.setattr(rag, "build_index", build_index)
    monkeypatch.setattr(
        worker,
        "api_get",
        lambda path: [
            {"id": 1, "feishu_chat_id": "chat-1", "status": "active"}
        ],
    )
    monkeypatch.setattr(worker, "send_text", lambda chat_id, text: sent.append(text))

    worker._handle_command("/chars index", "chat-1", "sender-1")

    assert build_calls == [(1, tmp_path.resolve(), expected_source)]
    assert sent == [
        "🔨 正在建立RAG索引...",
        "✅ 索引完成: 2 条消息, 1 个chunk",
    ]


@pytest.mark.parametrize("chat_mode", ["populated", "empty", "missing"])
@pytest.mark.asyncio
async def test_rag_source_mismatch_preserves_prior_index_without_embedding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    chat_mode: str,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    if chat_mode == "populated":
        await store.append_pair(
            1,
            "hello",
            "hi",
            char_name="Character",
            user_name="User",
        )
    elif chat_mode == "empty":
        await store.write_text(1, "chat.md", "")
    source = chat_source_identity(await store.load_chat(1))
    prior_index = {
        "chunks": [
            {
                "text": "prior valid chunk",
                "start_message": 1,
                "end_message": 2,
            }
        ],
        "embeddings": [[1.0]],
        "indexed_messages": 2,
        "source_count": source.count,
        "source_sha256": source.sha256,
    }
    await rag.save_index(1, prior_index, store=store)
    prior_bytes = store.file_path(1, "rag/index.json").read_bytes()
    embedding_calls = 0

    async def embed(*args, **kwargs):
        nonlocal embedding_calls
        embedding_calls += 1
        return [[2.0]]

    monkeypatch.setattr(rag, "get_embeddings", embed)
    mismatched_source = ChatSourceIdentity(
        count=source.count,
        sha256="0" * 64,
    )

    with pytest.raises(RuntimeError, match="source does not match"):
        await rag.build_index(
            1,
            store=store,
            source=mismatched_source,
            force_rebuild=True,
        )

    assert embedding_calls == 0
    assert store.file_path(1, "rag/index.json").read_bytes() == prior_bytes


@pytest.mark.asyncio
async def test_rag_embedding_failure_preserves_index_and_pipeline_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "story_state_interval_messages", 100)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 100)
    monkeypatch.setattr(settings, "episode_size_messages", 100)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 100)
    monkeypatch.setattr(settings, "assets_interval_messages", 100)
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "hello",
        "hi",
        char_name="Character",
        user_name="User",
    )
    source = chat_source_identity(await store.load_chat(1))
    prior_index = {
        "chunks": [
            {
                "text": "prior valid chunk",
                "start_message": 1,
                "end_message": 2,
            }
        ],
        "embeddings": [[1.0]],
        "indexed_messages": 2,
        "source_count": source.count,
        "source_sha256": source.sha256,
    }
    await rag.save_index(1, prior_index, store=store)
    prior_bytes = store.file_path(1, "rag/index.json").read_bytes()
    await store.save_state(
        1,
        MemoryState(
            last_rag_message=2,
            rebuild_rag_required=True,
            rebuild_from_message=0,
        ),
    )

    async def fail_embeddings(*args, **kwargs):
        raise ConnectionError("embedding unavailable")

    monkeypatch.setattr(rag, "get_embeddings", fail_embeddings)

    with pytest.raises(ConnectionError, match="embedding unavailable"):
        await MemoryPipeline(store).run(1, BACKEND)

    state = await store.load_state(1)
    assert store.file_path(1, "rag/index.json").read_bytes() == prior_bytes
    assert state.last_rag_message == 2
    assert state.rebuild_rag_required
    assert state.rebuild_from_message == 0
    assert state.last_error == "ConnectionError: embedding unavailable"


@pytest.mark.parametrize(
    "embedding_response",
    [
        pytest.param([], id="missing-count"),
        pytest.param([[True]], id="boolean"),
        pytest.param([[float("inf")]], id="nonfinite"),
        pytest.param([["invalid"]], id="nonnumeric"),
        pytest.param([[]], id="empty-vector"),
    ],
)
@pytest.mark.asyncio
async def test_malformed_embedding_response_preserves_index_and_rebuild_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    embedding_response: list,
) -> None:
    monkeypatch.setattr(settings, "story_state_interval_messages", 100)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 100)
    monkeypatch.setattr(settings, "episode_size_messages", 100)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 100)
    monkeypatch.setattr(settings, "assets_interval_messages", 100)
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "hello",
        "hi",
        char_name="Character",
        user_name="User",
    )
    source = chat_source_identity(await store.load_chat(1))
    prior_index = {
        "chunks": [
            {
                "text": "prior valid chunk",
                "start_message": 1,
                "end_message": 2,
            }
        ],
        "embeddings": [[1.0]],
        "indexed_messages": 2,
        "source_count": source.count,
        "source_sha256": source.sha256,
    }
    await rag.save_index(1, prior_index, store=store)
    prior_bytes = store.file_path(1, "rag/index.json").read_bytes()
    await store.save_state(
        1,
        MemoryState(
            last_rag_message=2,
            rebuild_rag_required=True,
            rebuild_from_message=0,
        ),
    )

    async def malformed_embeddings(*args, **kwargs):
        return embedding_response

    monkeypatch.setattr(rag, "get_embeddings", malformed_embeddings)

    with pytest.raises(ValueError, match="invalid RAG index"):
        await MemoryPipeline(store).run(1, BACKEND)

    state = await store.load_state(1)
    assert store.file_path(1, "rag/index.json").read_bytes() == prior_bytes
    assert state.last_rag_message == 2
    assert state.rebuild_rag_required
    assert state.rebuild_from_message == 0
    assert state.last_error == "ValueError: invalid RAG index"


@pytest.mark.asyncio
async def test_embedding_dimension_change_preserves_index_and_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "story_state_interval_messages", 100)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 100)
    monkeypatch.setattr(settings, "episode_size_messages", 100)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 2)
    monkeypatch.setattr(settings, "assets_interval_messages", 100)
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "first user",
        "first assistant",
        char_name="Character",
        user_name="User",
    )
    prefix_source = chat_source_identity(await store.load_chat(1))
    prior_index = {
        "chunks": [
            {
                "text": "用户: first user\n角色: first assistant",
                "start_message": 1,
                "end_message": 2,
            }
        ],
        "embeddings": [[1.0]],
        "indexed_messages": 2,
        "source_count": prefix_source.count,
        "source_sha256": prefix_source.sha256,
    }
    await rag.save_index(1, prior_index, store=store)
    prior_bytes = store.file_path(1, "rag/index.json").read_bytes()
    await store.save_state(1, MemoryState(last_rag_message=2))
    await store.append_pair(
        1,
        "second user",
        "second assistant",
        char_name="Character",
        user_name="User",
    )

    async def changed_dimension(*args, **kwargs):
        return [[1.0, 0.0]]

    monkeypatch.setattr(rag, "get_embeddings", changed_dimension)

    with pytest.raises(ValueError, match="invalid RAG index"):
        await MemoryPipeline(store).run(1, BACKEND)

    state = await store.load_state(1)
    assert store.file_path(1, "rag/index.json").read_bytes() == prior_bytes
    assert state.last_rag_message == 2
    assert not state.rebuild_rag_required
    assert state.last_error == "ValueError: invalid RAG index"


@pytest.mark.asyncio
async def test_same_count_source_change_forces_rag_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "hello",
        "original answer",
        char_name="Character",
        user_name="User",
    )
    embedding_calls: list[list[str]] = []

    async def embed(texts, **kwargs):
        embedding_calls.append(list(texts))
        return [[float(len(embedding_calls))] for _ in texts]

    monkeypatch.setattr(rag, "get_embeddings", embed)
    original_records = await store.load_chat(1)
    original_source = chat_source_identity(original_records)
    await rag.build_index(1, store=store, source=original_source)

    replacement = type(original_records[-1])(
        number=original_records[-1].number,
        role=original_records[-1].role,
        content="replacement answer",
        timestamp=original_records[-1].timestamp,
        name=original_records[-1].name,
        msg_type=original_records[-1].msg_type,
    )
    await store.write_text(
        1,
        "chat.md",
        render_chat_records([*original_records[:-1], replacement]),
    )
    replacement_records = await store.load_chat(1)
    replacement_source = chat_source_identity(replacement_records)

    result = await rag.build_index(
        1,
        store=store,
        source=replacement_source,
    )
    rebuilt = await rag.load_index(1, store=store)

    assert original_source.count == replacement_source.count == 2
    assert original_source.sha256 != replacement_source.sha256
    assert len(embedding_calls) == 2
    assert "original answer" in embedding_calls[0][0]
    assert "replacement answer" in embedding_calls[1][0]
    assert "original answer" not in embedding_calls[1][0]
    assert rebuilt["source_sha256"] == replacement_source.sha256
    assert result.source == replacement_source
    assert result.checkpoint == 2


@pytest.mark.asyncio
async def test_rag_invalidate_after_removes_only_known_later_ranges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    legacy = "legacy chunk"
    through_two = {"text": "one-two", "start_message": 1, "end_message": 2}
    through_four = {"text": "three-four", "start_message": 3, "end_message": 4}
    unknown = "unknown chunk"
    await rag.save_index(
        1,
        {
            "chunks": [legacy, through_two, through_four, unknown],
            "embeddings": [[1.0], [2.0], [3.0], [4.0]],
            "indexed_messages": 4,
        },
        store=store,
    )

    await rag.invalidate_after(1, 2, store=store)

    index = await rag.load_index(1, store=store)
    assert index["chunks"] == [legacy, through_two, unknown]
    assert index["embeddings"] == [[1.0], [2.0], [4.0]]
    assert index["indexed_messages"] == 2


@pytest.mark.parametrize(
    "malformed",
    [
        '{"chunks": null, "embeddings": [], "indexed_messages": 0}',
        '{"chunks": [], "embeddings": null, "indexed_messages": 0}',
        '{"chunks": ["x"], "embeddings": [], "indexed_messages": 1}',
        '{"chunks": [], "embeddings": [], "indexed_messages": true}',
        '{"chunks": [], "embeddings": [], "indexed_messages": -1}',
        '{"chunks": [{}], "embeddings": [[1.0]], "indexed_messages": 1}',
        '{"chunks": ["x"], "embeddings": [[true]], "indexed_messages": 1}',
        '{"chunks": ["x", "y"], "embeddings": [[1.0, 2.0], [1.0]], '
        '"indexed_messages": 2}',
    ],
)
@pytest.mark.asyncio
async def test_malformed_rag_schema_self_heals_to_empty_on_invalidation(
    tmp_path: Path,
    malformed: str,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    await store.write_text(1, "rag/index.json", malformed)
    empty = {
        "chunks": [],
        "embeddings": [],
        "indexed_messages": 0,
        "source_count": 0,
        "source_sha256": chat_source_identity([]).sha256,
    }

    assert await rag.load_index(1, store=store) == empty

    await rag.invalidate_after(1, 0, store=store)

    assert json.loads(await store.read_text(1, "rag/index.json")) == empty


@pytest.mark.parametrize("load_mode", ["store", "transaction", "default"])
@pytest.mark.asyncio
async def test_rag_load_index_normalizes_deep_json_recursion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mode: str,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    deeply_nested = "[" * 30_000 + "]" * 30_000
    await store.write_text(1, "rag/index.json", deeply_nested)

    if load_mode == "store":
        loaded = await rag.load_index(1, store=store)
    elif load_mode == "transaction":
        async with store.transaction(1) as transaction:
            loaded = await rag.load_index(1, transaction=transaction)
    else:
        monkeypatch.setattr(rag, "MEMORY_BASE", tmp_path)
        loaded = await rag.load_index(1)

    assert loaded == rag.empty_index()


@pytest.mark.parametrize(
    "existing_index",
    [
        pytest.param(
            '{"chunks":null,"embeddings":[],"indexed_messages":0}',
            id="malformed",
        ),
        pytest.param(
            '{"chunks":[],"embeddings":[],"indexed_messages":0}',
            id="unstamped",
        ),
    ],
)
@pytest.mark.asyncio
async def test_rag_empty_chat_build_persists_canonical_reset(
    tmp_path: Path,
    existing_index: str,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    await store.write_text(1, "chat.md", "")
    await store.write_text(1, "rag/index.json", existing_index)
    source = chat_source_identity([])

    result = await rag.build_index(1, store=store, source=source)

    assert result.checkpoint == 0
    assert result.source == source
    assert json.loads(await store.read_text(1, "rag/index.json")) == rag.empty_index()


@pytest.mark.parametrize(
    "chunks,indexed_messages,source_count",
    [
        pytest.param(
            [{"text": "truncated", "start_message": 1, "end_message": 3}],
            5,
            5,
            id="truncated-coverage",
        ),
        pytest.param(
            [
                {"text": "one-two", "start_message": 1, "end_message": 2},
                {"text": "four-five", "start_message": 4, "end_message": 5},
            ],
            5,
            5,
            id="gap",
        ),
        pytest.param(
            [
                {"text": "one-three", "start_message": 1, "end_message": 3},
                {"text": "three-five", "start_message": 3, "end_message": 5},
            ],
            5,
            5,
            id="overlap",
        ),
        pytest.param(
            [
                {"text": "three-five", "start_message": 3, "end_message": 5},
                {"text": "one-two", "start_message": 1, "end_message": 2},
            ],
            5,
            5,
            id="out-of-order",
        ),
        pytest.param(
            [{"text": "one-two", "start_message": 1, "end_message": 2}],
            2,
            3,
            id="source-count-mismatch",
        ),
        pytest.param(["legacy"], 1, 1, id="legacy-source-stamped"),
        pytest.param(
            [
                {"text": "one", "start_message": 1, "end_message": 1},
                "legacy",
            ],
            2,
            2,
            id="mixed-source-stamped",
        ),
    ],
)
@pytest.mark.asyncio
async def test_source_stamped_rag_index_requires_complete_ordered_coverage(
    tmp_path: Path,
    chunks: list[object],
    indexed_messages: int,
    source_count: int,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    malformed = {
        "chunks": chunks,
        "embeddings": [[1.0] for _ in chunks],
        "indexed_messages": indexed_messages,
        "source_count": source_count,
        "source_sha256": "0" * 64,
    }
    await store.write_text(1, "rag/index.json", json.dumps(malformed))

    assert await rag.load_index(1, store=store) == rag.empty_index()


@pytest.mark.parametrize(
    "chunks,indexed_messages",
    [
        pytest.param(
            [{"text": "partial", "start_message": 1, "end_message": 2}],
            2,
            id="ranged-partial",
        ),
        pytest.param(["legacy-partial"], 2, id="legacy-partial"),
    ],
)
@pytest.mark.asyncio
async def test_rag_build_fully_rebuilds_unstamped_partial_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    chunks: list[object],
    indexed_messages: int,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "first user",
        "first assistant",
        char_name="Character",
        user_name="User",
    )
    await store.append_pair(
        1,
        "second user",
        "second assistant",
        char_name="Character",
        user_name="User",
    )
    await rag.save_index(
        1,
        {
            "chunks": chunks,
            "embeddings": [[9.0] for _ in chunks],
            "indexed_messages": indexed_messages,
        },
        store=store,
    )
    embedded: list[list[str]] = []

    async def embed(texts, **kwargs):
        embedded.append(list(texts))
        return [[1.0] for _ in texts]

    monkeypatch.setattr(rag, "get_embeddings", embed)
    source = chat_source_identity(await store.load_chat(1))

    await rag.build_index(1, store=store, source=source)

    rebuilt = await rag.load_index(1, store=store)
    assert embedded == [[
        "用户: first user\n角色: first assistant\n"
        "用户: second user\n角色: second assistant"
    ]]
    assert [
        (chunk["start_message"], chunk["end_message"])
        for chunk in rebuilt["chunks"]
    ] == [(1, 4)]
    assert rebuilt["source_count"] == rebuilt["indexed_messages"] == 4


@pytest.mark.asyncio
async def test_rag_search_returns_text_for_new_and_legacy_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def load(session_id):
        return {
            "chunks": [
                {"text": "new chunk", "start_message": 1, "end_message": 2},
                "legacy chunk",
            ],
            "embeddings": [[1.0, 0.0], [0.0, 1.0]],
            "indexed_messages": 2,
        }

    async def query_embedding(text, **kwargs):
        return [1.0, 0.0]

    monkeypatch.setattr(rag, "load_index", load)
    monkeypatch.setattr(rag, "get_embedding", query_embedding)

    results = await rag.search(1, "query", top_k=2)

    assert [result["text"] for result in results] == [
        "new chunk",
        "legacy chunk",
    ]


@pytest.mark.asyncio
async def test_memory_task_manager_has_one_worker_and_deduplicates_submissions() -> None:
    completed = asyncio.Event()
    runs: list[tuple[int, dict[str, str]]] = []

    class Pipeline:
        async def run(self, session_id, backend):
            runs.append((session_id, backend))
            completed.set()

    async def resolve(session_id):
        return {"model": f"session-{session_id}"}

    manager = MemoryTaskManager(Pipeline(), resolve)
    await manager.start()
    first_worker = manager.worker
    await manager.start()
    await manager.submit(7)
    await manager.submit(7)
    await asyncio.wait_for(completed.wait(), timeout=1)
    await asyncio.wait_for(manager.queue.join(), timeout=1)

    assert manager.worker is first_worker
    assert first_worker is not None
    assert first_worker.get_name() == "memory-v2-worker"
    assert runs == [(7, {"model": "session-7"})]
    assert manager.pending == set()

    await manager.stop()
    assert manager.worker is None


@pytest.mark.asyncio
async def test_submit_while_running_queues_one_follow_up_run() -> None:
    first_running = asyncio.Event()
    release_first = asyncio.Event()
    second_completed = asyncio.Event()
    runs = 0

    class Pipeline:
        async def run(self, session_id, backend):
            nonlocal runs
            runs += 1
            if runs == 1:
                first_running.set()
                await release_first.wait()
            else:
                second_completed.set()

    async def resolve(session_id):
        return BACKEND

    manager = MemoryTaskManager(Pipeline(), resolve)
    await manager.start()
    await manager.submit(1)
    await asyncio.wait_for(first_running.wait(), timeout=1)
    await manager.submit(1)
    await manager.submit(1)
    release_first.set()
    await asyncio.wait_for(second_completed.wait(), timeout=1)
    await asyncio.wait_for(manager.queue.join(), timeout=1)

    assert runs == 2

    await manager.stop()


@pytest.mark.asyncio
async def test_startup_scanner_submits_only_sessions_with_pending_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "story_state_interval_messages", 2)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 2)
    monkeypatch.setattr(settings, "episode_size_messages", 2)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 2)
    monkeypatch.setattr(settings, "assets_interval_messages", 2)
    store = MarkdownMemoryStore(tmp_path)
    for session_id in (1, 2):
        await store.append_pair(
            session_id,
            "hello",
            "hi",
            char_name="Character",
            user_name="User",
        )
    await store.save_state(
        2,
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
    (tmp_path / "not-a-session").mkdir()

    async def resolve(session_id):
        return BACKEND

    manager = MemoryTaskManager(MemoryPipeline(store), resolve)

    await manager.scan_pending_sessions()

    assert manager.pending == {1}
    assert manager.queue.qsize() == 1


@pytest.mark.asyncio
async def test_forced_rebuild_runs_below_intervals_and_clears_each_stage_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "story_state_interval_messages", 10)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 10)
    monkeypatch.setattr(settings, "episode_size_messages", 20)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 10)
    monkeypatch.setattr(settings, "assets_interval_messages", 10)
    store = MarkdownMemoryStore(tmp_path)
    for pair in range(1, 6):
        await store.append_pair(
            1,
            f"user-{pair}",
            f"assistant-{pair}",
            char_name="Character",
            user_name="User",
        )
    await store.write_text(1, "memory.md", "stale memory through ten")
    await store.write_text(1, ".last_extract_count", "10")
    await store.write_text(1, "story_state.md", "stale story")
    await store.write_text(1, "summary.md", "stale summary")
    await store.write_text(1, "assets.md", "stale assets")
    await store.save_state(
        1,
        MemoryState(
            last_memory_message=10,
            last_story_state_message=10,
            last_rag_message=10,
            last_character_message=10,
            last_assets_message=10,
        ),
    )

    await store.truncate_chat(1, remove_count=2)

    pending_state = await store.load_state(1)
    assert pending_state.rebuild_required
    assert await store.read_text(1, "memory.md") == "stale memory through ten"
    assert MemoryTaskManager._has_pending(8, pending_state)

    async def resolve(session_id):
        return BACKEND

    startup_manager = MemoryTaskManager(MemoryPipeline(store), resolve)
    await startup_manager.scan_pending_sessions()
    assert startup_manager.pending == {1}

    calls: list[str] = []
    real_create_episodes = story_memory.create_due_episodes
    real_update_summary = story_memory.update_summary_from_episodes

    async def story(
        store_arg,
        session_id,
        records,
        complete,
        *,
        source,
    ):
        calls.append("story")
        assert [record.number for record in records] == list(range(1, 9))
        return await _completed_story_result(
            store_arg,
            session_id,
            source,
            text="rebuilt story",
        )

    async def extract(session_id, messages, backend, *, store, source):
        calls.append("memory")
        assert len(messages) == 8
        return await _completed_memory_result(
            store,
            session_id,
            source,
            text="rebuilt memory",
        )

    async def build_rag(session_id, *, store, source, **kwargs):
        calls.append("rag")
        return await _completed_rag_result(store, session_id, source)

    async def create_episodes(*args, **kwargs):
        calls.append("episode")
        return await real_create_episodes(*args, **kwargs)

    async def update_summary(*args, **kwargs):
        calls.append("summary")
        return await real_update_summary(*args, **kwargs)

    async def assets(session_id, messages, backend, *, store, source):
        calls.append("assets")
        assert len(messages) == 8
        return await _completed_assets_result(
            store,
            session_id,
            source,
            assets="rebuilt assets",
        )

    monkeypatch.setattr(story_memory, "update_story_state", story)
    monkeypatch.setattr(memory, "rebuild_memory_and_profiles", extract)
    monkeypatch.setattr(story_memory, "create_due_episodes", create_episodes)
    monkeypatch.setattr(
        story_memory,
        "update_summary_from_episodes",
        update_summary,
    )
    monkeypatch.setattr(rag, "build_index", build_rag)
    monkeypatch.setattr(memory, "rebuild_assets", assets)

    # Simulate process restart: recovery uses a newly constructed pipeline.
    await MemoryPipeline(store).run(1, BACKEND)

    state = await store.load_state(1)
    assert calls == ["story", "memory", "episode", "summary", "rag", "assets"]
    assert state.last_story_state_message == 8
    assert state.last_memory_message == 8
    assert state.last_character_message == 8
    assert state.last_episode_message == 0
    assert state.last_summary_message == 0
    assert state.last_rag_message == 8
    assert state.last_assets_message == 8
    assert not state.rebuild_required
    assert not MemoryTaskManager._has_pending(8, state)
    assert await store.read_text(1, "memory.md") == "rebuilt memory"
    restarted_manager = MemoryTaskManager(MemoryPipeline(store), resolve)
    await restarted_manager.scan_pending_sessions()
    assert restarted_manager.pending == set()


@pytest.mark.asyncio
async def test_forced_memory_rebuild_excludes_stale_inputs_and_reconciles_profiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "story_state_interval_messages", 100)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 100)
    monkeypatch.setattr(settings, "episode_size_messages", 100)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 100)
    monkeypatch.setattr(settings, "assets_interval_messages", 100)
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "authoritative user",
        "authoritative assistant",
        char_name="Character",
        user_name="User",
    )
    await store.write_text(1, "memory.md", "STALE_MEMORY")
    await store.write_text(1, "characters/Old.md", "# Old\n\nSTALE_PROFILE\n")
    await store.save_state(
        1,
        MemoryState(rebuild_memory_required=True, rebuild_from_message=0),
    )
    captured_prompt = ""

    async def complete(**kwargs):
        nonlocal captured_prompt
        captured_prompt = kwargs["messages"][1]["content"]
        return {
            "content": (
                "# Session Memory\n\n## Key Events\n- rebuilt\n\n"
                "===CHARACTERS===\n"
                "# New\n\n## 基本信息\n- 身份：new\n"
            )
        }

    monkeypatch.setattr(memory, "chat_completion", complete)

    await MemoryPipeline(store).run(1, BACKEND)

    assert "STALE_MEMORY" not in captured_prompt
    assert "STALE_PROFILE" not in captured_prompt
    assert "authoritative user" in captured_prompt
    assert "rebuilt" in await store.read_text(1, "memory.md")
    assert not (tmp_path / "1" / "characters" / "Old.md").exists()
    assert "# New" in await store.read_text(1, "characters/New.md")
    assert not (await store.load_state(1)).rebuild_memory_required


@pytest.mark.skipif(os.name != "nt", reason="Windows junction regression")
@pytest.mark.asyncio
async def test_forced_empty_profiles_rejects_characters_junction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "story_state_interval_messages", 100)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 100)
    monkeypatch.setattr(settings, "episode_size_messages", 100)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 100)
    monkeypatch.setattr(settings, "assets_interval_messages", 100)
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "authoritative user",
        "authoritative assistant",
        char_name="Character",
        user_name="User",
    )
    await store.write_text(1, "memory.md", "OLD_MEMORY")
    await store.save_state(
        1,
        MemoryState(rebuild_memory_required=True, rebuild_from_message=0),
    )
    external = tmp_path / "external-characters"
    external.mkdir()
    victim = external / "Victim.md"
    victim.write_text("EXTERNAL_VICTIM", encoding="utf-8")
    junction = tmp_path / "1" / "characters"
    created = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(junction), str(external)],
        capture_output=True,
        text=True,
        check=False,
    )
    if created.returncode != 0:
        pytest.skip(f"directory junctions are unavailable: {created.stderr}")
    assert junction.is_junction()

    async def complete(**kwargs):
        return {
            "content": (
                "# Session Memory\n\n## Key Events\n- rebuilt\n\n"
                "===CHARACTERS===\nNO_CHANGE"
            )
        }

    monkeypatch.setattr(memory, "chat_completion", complete)

    with pytest.raises(ValueError, match="profile directory escaped"):
        await MemoryPipeline(store).run(1, BACKEND)

    state = await store.load_state(1)
    assert victim.read_text(encoding="utf-8") == "EXTERNAL_VICTIM"
    assert await store.read_text(1, "memory.md") == "OLD_MEMORY"
    assert state.last_memory_message == 0
    assert state.rebuild_memory_required
    assert state.rebuild_from_message == 0
    assert state.last_error.startswith("ValueError:")


@pytest.mark.asyncio
async def test_forced_assets_no_change_clears_stale_asset_views(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "story_state_interval_messages", 100)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 100)
    monkeypatch.setattr(settings, "episode_size_messages", 100)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 100)
    monkeypatch.setattr(settings, "assets_interval_messages", 100)
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "no assets here",
        "none",
        char_name="Character",
        user_name="User",
    )
    await store.write_text(1, "assets.md", "STALE_ASSETS")
    await store.write_text(1, "assets_summary.txt", "STALE_SUMMARY")
    await store.save_state(
        1,
        MemoryState(rebuild_assets_required=True, rebuild_from_message=0),
    )
    prompts: list[str] = []

    async def complete(**kwargs):
        prompts.append(kwargs["messages"][1]["content"])
        return {"content": "NO_CHANGE"}

    monkeypatch.setattr(memory, "chat_completion", complete)

    await MemoryPipeline(store).run(1, BACKEND)

    assert len(prompts) == 1
    assert "STALE_ASSETS" not in prompts[0]
    assert not (tmp_path / "1" / "assets.md").exists()
    assert not (tmp_path / "1" / "assets_summary.txt").exists()
    assert not (await store.load_state(1)).rebuild_assets_required


@pytest.mark.asyncio
async def test_forced_assets_no_change_deletes_alias_without_touching_chat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "story_state_interval_messages", 100)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 100)
    monkeypatch.setattr(settings, "episode_size_messages", 100)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 100)
    monkeypatch.setattr(settings, "assets_interval_messages", 100)
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "authoritative user",
        "authoritative assistant",
        char_name="Character",
        user_name="User",
    )
    await store.save_state(
        1,
        MemoryState(rebuild_assets_required=True, rebuild_from_message=0),
    )
    chat_path = tmp_path / "1" / "chat.md"
    chat_before = chat_path.read_bytes()
    alias = tmp_path / "1" / "assets.md"
    try:
        alias.symlink_to(chat_path)
    except OSError as exc:
        pytest.skip(f"file symlinks are unavailable on this platform: {exc}")

    async def complete(**kwargs):
        return {"content": "NO_CHANGE"}

    monkeypatch.setattr(memory, "chat_completion", complete)

    await MemoryPipeline(store).run(1, BACKEND)

    assert chat_path.read_bytes() == chat_before
    assert not alias.exists()
    assert not alias.is_symlink()
    assert not (await store.load_state(1)).rebuild_assets_required


@pytest.mark.asyncio
async def test_legacy_asset_loader_rejects_alias_to_chat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    monkeypatch.setattr(memory, "MEMORY_BASE", tmp_path)
    monkeypatch.setattr(memory, "md_store", store)
    await store.append_pair(
        1,
        "authoritative user",
        "authoritative assistant",
        char_name="Character",
        user_name="User",
    )
    chat_path = tmp_path / "1" / "chat.md"
    chat_before = chat_path.read_bytes()
    alias = tmp_path / "1" / "assets.md"
    try:
        alias.symlink_to(chat_path)
    except OSError as exc:
        pytest.skip(f"file symlinks are unavailable on this platform: {exc}")

    with pytest.raises(ValueError, match="safe relative path"):
        await memory.load_assets(1)

    assert chat_path.read_bytes() == chat_before
    assert alias.is_symlink()


@pytest.mark.asyncio
async def test_receipt_alias_is_disposed_without_touching_chat(tmp_path: Path) -> None:
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "authoritative user",
        "authoritative assistant",
        char_name="Character",
        user_name="User",
    )
    source = chat_source_identity(await store.load_chat(1))
    chat_path = tmp_path / "1" / "chat.md"
    chat_before = chat_path.read_bytes()
    receipt = tmp_path / "1" / "rebuild_receipts" / "memory.json"
    receipt.parent.mkdir(parents=True)
    try:
        receipt.symlink_to(chat_path)
    except OSError as exc:
        pytest.skip(f"file symlinks are unavailable on this platform: {exc}")

    result = await MemoryPipeline(store)._load_verified_receipt(
        1,
        "memory",
        source,
        source.count,
    )

    assert result is None
    assert chat_path.read_bytes() == chat_before
    assert not receipt.exists()
    assert not receipt.is_symlink()


@pytest.mark.asyncio
async def test_deep_receipt_is_disposed_and_forced_stage_reruns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "story_state_interval_messages", 100)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 100)
    monkeypatch.setattr(settings, "episode_size_messages", 100)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 100)
    monkeypatch.setattr(settings, "assets_interval_messages", 100)
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "authoritative user",
        "authoritative assistant",
        char_name="Character",
        user_name="User",
    )
    await store.save_state(
        1,
        MemoryState(rebuild_memory_required=True, rebuild_from_message=0),
    )
    deeply_nested = "[" * 30_000 + "]" * 30_000
    assert len(deeply_nested.encode("utf-8")) < RECEIPT_MAX_BYTES
    await store.write_text(
        1,
        "rebuild_receipts/memory.json",
        deeply_nested,
    )
    completion_calls = 0

    async def complete(**kwargs):
        nonlocal completion_calls
        completion_calls += 1
        return {
            "content": (
                "# Session Memory\n\n## Key Events\n- rebuilt\n\n"
                "===CHARACTERS===\nNO_CHANGE"
            )
        }

    monkeypatch.setattr(memory, "chat_completion", complete)

    await MemoryPipeline(store).run(1, BACKEND)

    state = await store.load_state(1)
    replacement_receipt = parse_receipt(
        await store.read_text(1, "rebuild_receipts/memory.json")
    )
    assert completion_calls == 1
    assert "rebuilt" in await store.read_text(1, "memory.md")
    assert replacement_receipt.stage == "memory"
    assert replacement_receipt.source.count == 2
    assert state.last_memory_message == 2
    assert not state.rebuild_memory_required


@pytest.mark.asyncio
async def test_memory_rebuild_receipt_avoids_second_llm_after_state_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.services.md_store as md_store_module

    monkeypatch.setattr(settings, "story_state_interval_messages", 100)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 100)
    monkeypatch.setattr(settings, "episode_size_messages", 100)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 100)
    monkeypatch.setattr(settings, "assets_interval_messages", 100)
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "authoritative user",
        "authoritative assistant",
        char_name="Character",
        user_name="User",
    )
    await store.save_state(
        1,
        MemoryState(rebuild_memory_required=True, rebuild_from_message=0),
    )
    completion_calls = 0

    async def complete(**kwargs):
        nonlocal completion_calls
        completion_calls += 1
        return {
            "content": (
                "# Session Memory\n\n## Key Events\n- rebuilt\n\n"
                "===CHARACTERS===\nNO_CHANGE"
            )
        }

    monkeypatch.setattr(memory, "chat_completion", complete)
    real_replace = md_store_module.os.replace
    failed = False

    def fail_state_once(source: Path, target: Path) -> None:
        nonlocal failed
        if target.name == "memory_state.md" and not failed:
            failed = True
            raise OSError("state clear failed")
        real_replace(source, target)

    monkeypatch.setattr(md_store_module.os, "replace", fail_state_once)

    with pytest.raises(OSError, match="state clear failed"):
        await MemoryPipeline(store).run(1, BACKEND)

    assert completion_calls == 1
    assert (await store.load_state(1)).rebuild_memory_required
    receipt_path = tmp_path / "1" / "rebuild_receipts" / "memory.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    artifact_path = tmp_path / "1" / "memory.md"
    assert receipt["artifacts"][0]["byte_size"] == artifact_path.stat().st_size

    monkeypatch.setattr(md_store_module.os, "replace", real_replace)
    await MemoryPipeline(MarkdownMemoryStore(tmp_path)).run(1, BACKEND)

    assert completion_calls == 1
    assert not (await store.load_state(1)).rebuild_memory_required


@pytest.mark.asyncio
async def test_story_stage_result_without_required_output_stays_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "story_state_interval_messages", 100)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 100)
    monkeypatch.setattr(settings, "episode_size_messages", 100)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 100)
    monkeypatch.setattr(settings, "assets_interval_messages", 100)
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "authoritative user",
        "authoritative assistant",
        char_name="Character",
        user_name="User",
    )
    records = await store.load_chat(1)
    source = chat_source_identity(records)
    await store.save_state(
        1,
        MemoryState(rebuild_story_required=True, rebuild_from_message=0),
    )

    async def incomplete_story(*args, **kwargs):
        return StageUpdateResult(
            stage="story",
            completed=True,
            source=source,
            checkpoint=2,
        )

    monkeypatch.setattr(story_memory, "update_story_state", incomplete_story)

    with pytest.raises(RuntimeError, match="reconcile required artifacts"):
        await MemoryPipeline(store).run(1, BACKEND)

    state = await store.load_state(1)
    assert state.rebuild_story_required
    assert state.last_story_state_message == 0
    assert not (tmp_path / "1" / "rebuild_receipts" / "story.json").exists()


@pytest.mark.asyncio
async def test_chat_change_before_state_save_keeps_forced_stage_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "story_state_interval_messages", 100)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 100)
    monkeypatch.setattr(settings, "episode_size_messages", 100)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 100)
    monkeypatch.setattr(settings, "assets_interval_messages", 100)
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "authoritative user",
        "authoritative assistant",
        char_name="Character",
        user_name="User",
    )
    source = chat_source_identity(await store.load_chat(1))
    await store.save_state(
        1,
        MemoryState(rebuild_story_required=True, rebuild_from_message=0),
    )

    async def completed_story(*args, **kwargs):
        story_text = "story state\n"
        await store.write_text(1, "story_state.md", story_text)
        return StageUpdateResult(
            stage="story",
            completed=True,
            source=source,
            checkpoint=2,
            artifacts=(text_artifact("story_state.md", story_text),),
        )

    monkeypatch.setattr(story_memory, "update_story_state", completed_story)
    pipeline = MemoryPipeline(store)
    require_result = pipeline._require_stage_result
    appended = False

    async def require_then_append(*args, **kwargs):
        nonlocal appended
        result = await require_result(*args, **kwargs)
        if not appended:
            appended = True
            await store.append_pair(
                1,
                "racing user",
                "racing assistant",
                char_name="Character",
                user_name="User",
            )
        return result

    monkeypatch.setattr(pipeline, "_require_stage_result", require_then_append)

    with pytest.raises(RuntimeError, match="source changed before state save"):
        await pipeline.run(1, BACKEND)

    state = await store.load_state(1)
    assert state.rebuild_story_required
    assert state.last_story_state_message == 0
    assert state.last_error == "RuntimeError: source changed before state save"


@pytest.mark.asyncio
async def test_memory_stage_result_without_required_output_stays_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "story_state_interval_messages", 100)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 100)
    monkeypatch.setattr(settings, "episode_size_messages", 100)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 100)
    monkeypatch.setattr(settings, "assets_interval_messages", 100)
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "authoritative user",
        "authoritative assistant",
        char_name="Character",
        user_name="User",
    )
    source = chat_source_identity(await store.load_chat(1))
    await store.save_state(
        1,
        MemoryState(rebuild_memory_required=True, rebuild_from_message=0),
    )

    async def incomplete_memory(*args, **kwargs):
        return StageUpdateResult(
            stage="memory",
            completed=True,
            source=source,
            checkpoint=2,
        )

    monkeypatch.setattr(memory, "rebuild_memory_and_profiles", incomplete_memory)

    with pytest.raises(RuntimeError, match="reconcile required artifacts"):
        await MemoryPipeline(store).run(1, BACKEND)

    state = await store.load_state(1)
    assert state.rebuild_memory_required
    assert state.last_memory_message == 0
    assert not (tmp_path / "1" / "rebuild_receipts" / "memory.json").exists()


@pytest.mark.asyncio
async def test_assets_stage_result_with_singleton_output_stays_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "story_state_interval_messages", 100)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 100)
    monkeypatch.setattr(settings, "episode_size_messages", 100)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 100)
    monkeypatch.setattr(settings, "assets_interval_messages", 100)
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "authoritative user",
        "authoritative assistant",
        char_name="Character",
        user_name="User",
    )
    source = chat_source_identity(await store.load_chat(1))
    asset_text = "# 资产总览\n- 现金：¥100"
    await store.write_text(1, "assets.md", asset_text)
    await store.save_state(
        1,
        MemoryState(rebuild_assets_required=True, rebuild_from_message=0),
    )

    async def incomplete_assets(*args, **kwargs):
        return StageUpdateResult(
            stage="assets",
            completed=True,
            source=source,
            checkpoint=2,
            artifacts=(text_artifact("assets.md", asset_text),),
        )

    monkeypatch.setattr(memory, "rebuild_assets", incomplete_assets)

    with pytest.raises(RuntimeError, match="reconcile required artifacts"):
        await MemoryPipeline(store).run(1, BACKEND)

    state = await store.load_state(1)
    assert state.rebuild_assets_required
    assert state.last_assets_message == 0
    assert not (tmp_path / "1" / "rebuild_receipts" / "assets.json").exists()


@pytest.mark.asyncio
async def test_summary_stage_result_without_frontier_output_stays_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "story_state_interval_messages", 100)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 100)
    monkeypatch.setattr(settings, "episode_size_messages", 100)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 100)
    monkeypatch.setattr(settings, "assets_interval_messages", 100)
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "authoritative user",
        "authoritative assistant",
        char_name="Character",
        user_name="User",
    )
    source = chat_source_identity(await store.load_chat(1))
    episode_text = _episode_document(1, 1, 2)
    await store.write_text(1, "episodes/episode-000001.md", episode_text)
    await store.save_state(
        1,
        MemoryState(
            last_episode_message=2,
            rebuild_summary_required=True,
            rebuild_from_message=0,
        ),
    )

    async def incomplete_summary(*args, **kwargs):
        return StageUpdateResult(
            stage="summary",
            completed=True,
            source=source,
            checkpoint=2,
            inputs=(
                text_artifact("episodes/episode-000001.md", episode_text),
            ),
        )

    monkeypatch.setattr(
        story_memory,
        "update_summary_from_episodes",
        incomplete_summary,
    )

    with pytest.raises(RuntimeError, match="reconcile required artifacts"):
        await MemoryPipeline(store).run(1, BACKEND)

    state = await store.load_state(1)
    assert state.rebuild_summary_required
    assert state.last_summary_message == 0
    assert not (tmp_path / "1" / "rebuild_receipts" / "summary.json").exists()


@pytest.mark.asyncio
async def test_forced_summary_stage_result_with_wrong_checkpoint_stays_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "story_state_interval_messages", 100)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 100)
    monkeypatch.setattr(settings, "episode_size_messages", 100)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 100)
    monkeypatch.setattr(settings, "assets_interval_messages", 100)
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "authoritative user",
        "authoritative assistant",
        char_name="Character",
        user_name="User",
    )
    source = chat_source_identity(await store.load_chat(1))
    episode_text = _episode_document(1, 1, 2)
    await store.write_text(1, "episodes/episode-000001.md", episode_text)
    await store.save_state(
        1,
        MemoryState(
            last_episode_message=2,
            rebuild_summary_required=True,
            rebuild_from_message=0,
        ),
    )

    async def wrong_checkpoint(*args, **kwargs):
        summary_text = "summary\n"
        await store.write_text(1, "summary.md", summary_text)
        return StageUpdateResult(
            stage="summary",
            completed=True,
            source=source,
            checkpoint=0,
            artifacts=(text_artifact("summary.md", summary_text),),
            inputs=(
                text_artifact("episodes/episode-000001.md", episode_text),
            ),
        )

    monkeypatch.setattr(
        story_memory,
        "update_summary_from_episodes",
        wrong_checkpoint,
    )

    with pytest.raises(
        RuntimeError,
        match="summary stage did not return a completed result",
    ):
        await MemoryPipeline(store).run(1, BACKEND)

    state = await store.load_state(1)
    assert state.rebuild_summary_required
    assert state.last_summary_message == 0
    assert not (tmp_path / "1" / "rebuild_receipts" / "summary.json").exists()


@pytest.mark.asyncio
async def test_story_rebuild_receipt_avoids_second_llm_after_state_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.services.md_store as md_store_module

    monkeypatch.setattr(settings, "story_state_interval_messages", 100)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 100)
    monkeypatch.setattr(settings, "episode_size_messages", 100)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 100)
    monkeypatch.setattr(settings, "assets_interval_messages", 100)
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "authoritative user",
        "authoritative assistant",
        char_name="Character",
        user_name="User",
    )
    await store.save_state(
        1,
        MemoryState(rebuild_story_required=True, rebuild_from_message=0),
    )
    completion_calls = 0

    async def complete(**kwargs):
        nonlocal completion_calls
        completion_calls += 1
        return {
            "content": (
                "# Story State\n\n"
                "## 时间与地点\n- 当前地点\n\n"
                "## 在场角色\n- 当前角色\n\n"
                "## 当前场景\n- 当前事件\n\n"
                "## 最近变化\n- 新变化"
            )
        }

    monkeypatch.setattr(llm, "chat_completion", complete)
    real_replace = md_store_module.os.replace
    failed = False

    def fail_state_once(source: Path, target: Path) -> None:
        nonlocal failed
        if target.name == "memory_state.md" and not failed:
            failed = True
            raise OSError("state clear failed")
        real_replace(source, target)

    monkeypatch.setattr(md_store_module.os, "replace", fail_state_once)

    with pytest.raises(OSError, match="state clear failed"):
        await MemoryPipeline(store).run(1, BACKEND)

    assert completion_calls == 1
    assert (await store.load_state(1)).rebuild_story_required
    assert (tmp_path / "1" / "rebuild_receipts" / "story.json").exists()

    monkeypatch.setattr(md_store_module.os, "replace", real_replace)
    await MemoryPipeline(MarkdownMemoryStore(tmp_path)).run(1, BACKEND)

    assert completion_calls == 1
    assert not (await store.load_state(1)).rebuild_story_required


@pytest.mark.asyncio
async def test_assets_rebuild_receipt_avoids_second_llm_after_state_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.services.md_store as md_store_module

    monkeypatch.setattr(settings, "story_state_interval_messages", 100)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 100)
    monkeypatch.setattr(settings, "episode_size_messages", 100)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 100)
    monkeypatch.setattr(settings, "assets_interval_messages", 100)
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "authoritative user",
        "authoritative assistant",
        char_name="Character",
        user_name="User",
    )
    await store.save_state(
        1,
        MemoryState(rebuild_assets_required=True, rebuild_from_message=0),
    )
    completion_calls = 0

    async def complete(**kwargs):
        nonlocal completion_calls
        completion_calls += 1
        if completion_calls == 1:
            return {"content": "# 资产总览\n- 现金：¥100"}
        return {"content": "总资产约¥100 | 现金¥100"}

    monkeypatch.setattr(memory, "chat_completion", complete)
    real_replace = md_store_module.os.replace
    failed = False

    def fail_state_once(source: Path, target: Path) -> None:
        nonlocal failed
        if target.name == "memory_state.md" and not failed:
            failed = True
            raise OSError("state clear failed")
        real_replace(source, target)

    monkeypatch.setattr(md_store_module.os, "replace", fail_state_once)

    with pytest.raises(OSError, match="state clear failed"):
        await MemoryPipeline(store).run(1, BACKEND)

    assert completion_calls == 2
    assert (await store.load_state(1)).rebuild_assets_required
    assert (tmp_path / "1" / "rebuild_receipts" / "assets.json").exists()

    monkeypatch.setattr(md_store_module.os, "replace", real_replace)
    await MemoryPipeline(MarkdownMemoryStore(tmp_path)).run(1, BACKEND)

    assert completion_calls == 2
    assert not (await store.load_state(1)).rebuild_assets_required


@pytest.mark.asyncio
async def test_partial_profile_rebuild_keeps_flag_without_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "story_state_interval_messages", 100)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 100)
    monkeypatch.setattr(settings, "episode_size_messages", 100)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 100)
    monkeypatch.setattr(settings, "assets_interval_messages", 100)
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "authoritative user",
        "authoritative assistant",
        char_name="Character",
        user_name="User",
    )
    await store.save_state(
        1,
        MemoryState(rebuild_memory_required=True, rebuild_from_message=0),
    )
    await store.write_text(1, "rebuild_receipts/memory.json", "{")

    async def complete(**kwargs):
        return {
            "content": (
                "# Session Memory\n\n## Key Events\n- rebuilt\n\n"
                "===CHARACTERS===\n"
                "# Alice\n- present\n---\n# Bob\n- present"
            )
        }

    monkeypatch.setattr(memory, "chat_completion", complete)
    real_write_text = store.write_text

    async def fail_second_profile(
        session_id: int,
        relative: str | Path,
        text: str,
    ) -> None:
        if str(relative) == "characters/Bob.md":
            raise OSError("profile write interrupted")
        await real_write_text(session_id, relative, text)

    monkeypatch.setattr(store, "write_text", fail_second_profile)

    with pytest.raises(OSError, match="profile write interrupted"):
        await MemoryPipeline(store).run(1, BACKEND)

    state = await store.load_state(1)
    assert (tmp_path / "1" / "memory.md").exists()
    assert (tmp_path / "1" / "characters" / "Alice.md").exists()
    assert not (tmp_path / "1" / "characters" / "Bob.md").exists()
    assert not (tmp_path / "1" / "rebuild_receipts" / "memory.json").exists()
    assert state.rebuild_memory_required
    assert state.last_memory_message == 0


@pytest.mark.asyncio
async def test_cancelled_partial_assets_rebuild_keeps_flag_without_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "story_state_interval_messages", 100)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 100)
    monkeypatch.setattr(settings, "episode_size_messages", 100)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 100)
    monkeypatch.setattr(settings, "assets_interval_messages", 100)
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "authoritative user",
        "authoritative assistant",
        char_name="Character",
        user_name="User",
    )
    await store.save_state(
        1,
        MemoryState(rebuild_assets_required=True, rebuild_from_message=0),
    )
    await store.write_text(1, "rebuild_receipts/assets.json", "{")
    completion_calls = 0

    async def complete(**kwargs):
        nonlocal completion_calls
        completion_calls += 1
        if completion_calls == 1:
            return {"content": "# 资产总览\n- 现金：¥100"}
        return {"content": "总资产约¥100 | 现金¥100"}

    monkeypatch.setattr(memory, "chat_completion", complete)
    real_write_text = store.write_text

    async def cancel_summary_write(
        session_id: int,
        relative: str | Path,
        text: str,
    ) -> None:
        if str(relative) == "assets_summary.txt":
            raise asyncio.CancelledError
        await real_write_text(session_id, relative, text)

    monkeypatch.setattr(store, "write_text", cancel_summary_write)

    with pytest.raises(asyncio.CancelledError):
        await MemoryPipeline(store).run(1, BACKEND)

    state = await store.load_state(1)
    assert (tmp_path / "1" / "assets.md").exists()
    assert not (tmp_path / "1" / "assets_summary.txt").exists()
    assert not (tmp_path / "1" / "rebuild_receipts" / "assets.json").exists()
    assert state.rebuild_assets_required
    assert state.last_assets_message == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tampering",
    [
        "checkpoint",
        "source",
        "artifact-sha256",
        "artifact-byte-size",
        "artifact-path-set",
        "malformed-json",
    ],
)
async def test_memory_rebuild_disposable_receipt_tampering_reruns_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tampering: str,
) -> None:
    import app.services.md_store as md_store_module

    monkeypatch.setattr(settings, "story_state_interval_messages", 100)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 100)
    monkeypatch.setattr(settings, "episode_size_messages", 100)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 100)
    monkeypatch.setattr(settings, "assets_interval_messages", 100)
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "authoritative user",
        "authoritative assistant",
        char_name="Character",
        user_name="User",
    )
    await store.save_state(
        1,
        MemoryState(rebuild_memory_required=True, rebuild_from_message=0),
    )
    completion_calls = 0

    async def complete(**kwargs):
        nonlocal completion_calls
        completion_calls += 1
        if completion_calls == 2:
            raise RuntimeError("forced stage reran")
        return {
            "content": (
                "# Session Memory\n\n## Key Events\n- rebuilt\n\n"
                "===CHARACTERS===\nNO_CHANGE"
            )
        }

    monkeypatch.setattr(memory, "chat_completion", complete)
    real_replace = md_store_module.os.replace
    failed = False

    def fail_state_once(source: Path, target: Path) -> None:
        nonlocal failed
        if target.name == "memory_state.md" and not failed:
            failed = True
            raise OSError("state clear failed")
        real_replace(source, target)

    monkeypatch.setattr(md_store_module.os, "replace", fail_state_once)

    with pytest.raises(OSError, match="state clear failed"):
        await MemoryPipeline(store).run(1, BACKEND)

    receipt_path = tmp_path / "1" / "rebuild_receipts" / "memory.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if tampering == "checkpoint":
        receipt["checkpoint"] = 999
    elif tampering == "source":
        receipt["source_sha256"] = "0" * 64
    elif tampering == "artifact-sha256":
        receipt["artifacts"][0]["sha256"] = "0" * 64
    elif tampering == "artifact-byte-size":
        receipt["artifacts"][0]["byte_size"] += 1
    elif tampering == "artifact-path-set":
        receipt["artifacts"] = []
    elif tampering == "malformed-json":
        receipt_path.write_text("{", encoding="utf-8")
    else:
        raise AssertionError(f"unknown receipt tampering: {tampering}")
    if tampering != "malformed-json":
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    monkeypatch.setattr(md_store_module.os, "replace", real_replace)
    with pytest.raises(RuntimeError, match="forced stage reran"):
        await MemoryPipeline(MarkdownMemoryStore(tmp_path)).run(1, BACKEND)

    state = await store.load_state(1)
    assert completion_calls == 2
    assert state.last_memory_message == 0
    assert state.rebuild_memory_required
    assert not receipt_path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("input_tampering", "expected_completion_calls"),
    [
        pytest.param("none", 1, id="unchanged-inputs-reused"),
        pytest.param("episode-content", 2, id="changed-content-rerun"),
        pytest.param("missing", 2, id="missing-input-rerun"),
        pytest.param("extra", 2, id="extra-input-rerun"),
        pytest.param("reordered", 2, id="reordered-inputs-rerun"),
        pytest.param("unsafe-path", 2, id="unsafe-input-path-rerun"),
    ],
)
async def test_summary_receipt_reuse_requires_unchanged_episode_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_tampering: str,
    expected_completion_calls: int,
) -> None:
    import app.services.md_store as md_store_module

    monkeypatch.setattr(settings, "story_state_interval_messages", 100)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 100)
    monkeypatch.setattr(settings, "episode_size_messages", 100)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 100)
    monkeypatch.setattr(settings, "assets_interval_messages", 100)
    store = MarkdownMemoryStore(tmp_path)
    for pair in range(2):
        await store.append_pair(
            1,
            f"authoritative user {pair}",
            f"authoritative assistant {pair}",
            char_name="Character",
            user_name="User",
        )
    await store.write_text(
        1,
        "episodes/episode-000001.md",
        _episode_document(1, 1, 2),
    )
    await store.write_text(
        1,
        "episodes/episode-000002.md",
        _episode_document(2, 3, 4),
    )
    await store.save_state(
        1,
        MemoryState(
            last_episode_message=4,
            rebuild_summary_required=True,
            rebuild_from_message=0,
        ),
    )
    completion_calls = 0

    async def complete(**kwargs):
        nonlocal completion_calls
        completion_calls += 1
        return {"content": f"summary version {completion_calls}"}

    monkeypatch.setattr(llm, "chat_completion", complete)
    real_replace = md_store_module.os.replace
    failed = False

    def fail_state_once(source: Path, target: Path) -> None:
        nonlocal failed
        if target.name == "memory_state.md" and not failed:
            failed = True
            raise OSError("state clear failed")
        real_replace(source, target)

    monkeypatch.setattr(md_store_module.os, "replace", fail_state_once)

    with pytest.raises(OSError, match="state clear failed"):
        await MemoryPipeline(store).run(1, BACKEND)

    monkeypatch.setattr(md_store_module.os, "replace", real_replace)
    if input_tampering == "episode-content":
        changed_episode = _episode_document(1, 1, 2).replace(
            "已发生的事件",
            "被重建后改变的事件",
        )
        await store.write_text(1, "episodes/episode-000001.md", changed_episode)
    elif input_tampering != "none":
        receipt_path = tmp_path / "1" / "rebuild_receipts" / "summary.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if input_tampering == "missing":
            receipt["inputs"].pop()
        elif input_tampering == "extra":
            extra = dict(receipt["inputs"][0])
            extra["path"] = "episodes/episode-000003.md"
            receipt["inputs"].append(extra)
        elif input_tampering == "reordered":
            receipt["inputs"].reverse()
        elif input_tampering == "unsafe-path":
            receipt["inputs"][0]["path"] = "../episode.md"
        else:
            raise AssertionError(f"unknown input tampering: {input_tampering}")
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    await MemoryPipeline(MarkdownMemoryStore(tmp_path)).run(1, BACKEND)

    state = await store.load_state(1)
    assert completion_calls == expected_completion_calls
    assert (
        f"summary version {expected_completion_calls}"
        in await store.read_text(1, "summary.md")
    )
    assert not state.rebuild_summary_required


@pytest.mark.asyncio
async def test_zero_frontier_summary_receipt_reuses_empty_ownership_after_state_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.services.md_store as md_store_module

    monkeypatch.setattr(settings, "story_state_interval_messages", 100)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 100)
    monkeypatch.setattr(settings, "episode_size_messages", 100)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 100)
    monkeypatch.setattr(settings, "assets_interval_messages", 100)
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "authoritative user",
        "authoritative assistant",
        char_name="Character",
        user_name="User",
    )
    await store.write_text(1, "summary.md", "stale summary\n")
    await store.save_state(
        1,
        MemoryState(
            rebuild_summary_required=True,
            rebuild_from_message=0,
        ),
    )
    completion_calls = 0

    async def must_not_complete(**kwargs):
        nonlocal completion_calls
        completion_calls += 1
        raise AssertionError("zero-frontier summary must not call the LLM")

    monkeypatch.setattr(llm, "chat_completion", must_not_complete)
    real_replace = md_store_module.os.replace
    failed = False

    def fail_state_once(source: Path, target: Path) -> None:
        nonlocal failed
        if target.name == "memory_state.md" and not failed:
            failed = True
            raise OSError("state clear failed")
        real_replace(source, target)

    monkeypatch.setattr(md_store_module.os, "replace", fail_state_once)
    with pytest.raises(OSError, match="state clear failed"):
        await MemoryPipeline(store).run(1, BACKEND)

    receipt_path = tmp_path / "1" / "rebuild_receipts" / "summary.json"
    assert completion_calls == 0
    assert not (tmp_path / "1" / "summary.md").exists()
    assert receipt_path.exists()
    assert (await store.load_state(1)).rebuild_summary_required

    monkeypatch.setattr(md_store_module.os, "replace", real_replace)
    await MemoryPipeline(MarkdownMemoryStore(tmp_path)).run(1, BACKEND)

    state = await store.load_state(1)
    assert completion_calls == 0
    assert not (tmp_path / "1" / "summary.md").exists()
    assert state.last_summary_message == 0
    assert not state.rebuild_summary_required


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_episode",
    [
        pytest.param(_episode_document(2, 4, 5), id="gapped"),
        pytest.param(_episode_document(2, 2, 3), id="overlapping"),
        pytest.param("# Episode 000002\n\nmalformed", id="malformed"),
    ],
)
async def test_summary_receipt_rejects_matching_invalid_episode_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_episode: str,
) -> None:
    import app.services.md_store as md_store_module

    monkeypatch.setattr(settings, "story_state_interval_messages", 100)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 100)
    monkeypatch.setattr(settings, "episode_size_messages", 100)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 100)
    monkeypatch.setattr(settings, "assets_interval_messages", 100)
    store = MarkdownMemoryStore(tmp_path)
    for pair in range(2):
        await store.append_pair(
            1,
            f"authoritative user {pair}",
            f"authoritative assistant {pair}",
            char_name="Character",
            user_name="User",
        )
    await store.write_text(
        1,
        "episodes/episode-000001.md",
        _episode_document(1, 1, 2),
    )
    await store.write_text(
        1,
        "episodes/episode-000002.md",
        _episode_document(2, 3, 4),
    )
    await store.save_state(
        1,
        MemoryState(
            last_episode_message=4,
            rebuild_summary_required=True,
            rebuild_from_message=0,
        ),
    )

    async def complete(**kwargs):
        return {"content": "valid summary"}

    monkeypatch.setattr(llm, "chat_completion", complete)
    real_replace = md_store_module.os.replace
    failed = False

    def fail_state_once(source: Path, target: Path) -> None:
        nonlocal failed
        if target.name == "memory_state.md" and not failed:
            failed = True
            raise OSError("state clear failed")
        real_replace(source, target)

    monkeypatch.setattr(md_store_module.os, "replace", fail_state_once)
    with pytest.raises(OSError, match="state clear failed"):
        await MemoryPipeline(store).run(1, BACKEND)

    monkeypatch.setattr(md_store_module.os, "replace", real_replace)
    await store.write_text(1, "episodes/episode-000002.md", invalid_episode)
    receipt_path = tmp_path / "1" / "rebuild_receipts" / "summary.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    invalid_identity = text_artifact(
        "episodes/episode-000002.md",
        invalid_episode,
    )
    receipt["inputs"][1] = {
        "path": invalid_identity.path,
        "sha256": invalid_identity.sha256,
        "byte_size": invalid_identity.byte_size,
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError):
        await MemoryPipeline(MarkdownMemoryStore(tmp_path)).run(1, BACKEND)

    state = await store.load_state(1)
    assert state.rebuild_summary_required
    assert state.last_summary_message == 0
    assert not receipt_path.exists()


@pytest.mark.asyncio
async def test_forced_rebuild_retry_skips_stages_with_persisted_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "story_state_interval_messages", 10)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 10)
    monkeypatch.setattr(settings, "episode_size_messages", 20)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 10)
    monkeypatch.setattr(settings, "assets_interval_messages", 10)
    store = MarkdownMemoryStore(tmp_path)
    for pair in range(1, 6):
        await store.append_pair(
            1,
            f"user-{pair}",
            f"assistant-{pair}",
            char_name="Character",
            user_name="User",
        )
    await store.save_state(
        1,
        MemoryState(
            last_memory_message=10,
            last_story_state_message=10,
            last_rag_message=10,
            last_character_message=10,
            last_assets_message=10,
        ),
    )
    await store.truncate_chat(1, remove_count=2)
    calls = {
        "story": 0,
        "memory": 0,
        "episode": 0,
        "summary": 0,
        "rag": 0,
        "assets": 0,
    }

    async def story(store_arg, session_id, *args, source, **kwargs):
        calls["story"] += 1
        return await _completed_story_result(store_arg, session_id, source)

    async def extract(session_id, *args, store, source, **kwargs):
        calls["memory"] += 1
        return await _completed_memory_result(store, session_id, source)

    async def build_rag(session_id, *args, store, source, **kwargs):
        calls["rag"] += 1
        if calls["rag"] == 1:
            raise RuntimeError("rag failed")
        return await _completed_rag_result(store, session_id, source)

    async def episodes(store_arg, session_id, *args, source, **kwargs):
        calls["episode"] += 1
        return await _completed_episode_result(
            store_arg,
            session_id,
            source,
            args[1],
        )

    async def summary(store_arg, session_id, *args, source, **kwargs):
        calls["summary"] += 1
        return await _completed_summary_result(
            store_arg,
            session_id,
            source,
            args[0],
        )

    async def assets(session_id, *args, store, source, **kwargs):
        calls["assets"] += 1
        return await _completed_assets_result(store, session_id, source)

    monkeypatch.setattr(story_memory, "update_story_state", story)
    monkeypatch.setattr(memory, "rebuild_memory_and_profiles", extract)
    monkeypatch.setattr(story_memory, "create_due_episodes", episodes)
    monkeypatch.setattr(story_memory, "update_summary_from_episodes", summary)
    monkeypatch.setattr(rag, "build_index", build_rag)
    monkeypatch.setattr(memory, "rebuild_assets", assets)

    with pytest.raises(RuntimeError, match="rag failed"):
        await MemoryPipeline(store).run(1, BACKEND)

    failed = await store.load_state(1)
    assert not failed.rebuild_story_required
    assert not failed.rebuild_memory_required
    assert not failed.rebuild_episode_required
    assert not failed.rebuild_summary_required
    assert failed.rebuild_rag_required
    assert failed.rebuild_assets_required

    await MemoryPipeline(store).run(1, BACKEND)

    assert calls == {
        "story": 1,
        "memory": 1,
        "episode": 1,
        "summary": 1,
        "rag": 2,
        "assets": 1,
    }
    assert not (await store.load_state(1)).rebuild_required


@pytest.mark.asyncio
async def test_completed_retry_clears_boundary_before_legacy_reset_to_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "story_state_interval_messages", 100)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 100)
    monkeypatch.setattr(settings, "episode_size_messages", 2)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 100)
    monkeypatch.setattr(settings, "assets_interval_messages", 100)
    store = MarkdownMemoryStore(tmp_path)
    records = await _seed_stale_retry_artifacts(store)

    await store.replace_final_pair(
        1,
        expected_user=records[-2],
        expected_assistant=records[-1],
        assistant_content="replacement assistant",
    )

    async def story(store_arg, session_id, *args, source, **kwargs):
        return await _completed_story_result(store_arg, session_id, source)

    async def extract(session_id, *args, store, source, **kwargs):
        return await _completed_memory_result(store, session_id, source)

    async def episodes(store_arg, session_id, *args, source, **kwargs):
        return await _completed_episode_result(store_arg, session_id, source, 2)

    async def summary(store_arg, session_id, *args, source, **kwargs):
        return await _completed_summary_result(
            store_arg,
            session_id,
            source,
            2,
        )

    async def build_rag(session_id, *args, store, source, **kwargs):
        return await _completed_rag_result(store, session_id, source)

    async def assets(session_id, *args, store, source, **kwargs):
        return await _completed_assets_result(store, session_id, source)

    monkeypatch.setattr(story_memory, "update_story_state", story)
    monkeypatch.setattr(memory, "rebuild_memory_and_profiles", extract)
    monkeypatch.setattr(story_memory, "create_due_episodes", episodes)
    monkeypatch.setattr(story_memory, "update_summary_from_episodes", summary)
    monkeypatch.setattr(rag, "build_index", build_rag)
    monkeypatch.setattr(memory, "rebuild_assets", assets)

    await MemoryPipeline(store).run(1, BACKEND)

    completed = await store.load_state(1)
    assert not completed.rebuild_required
    assert completed.rebuild_from_message is None

    # Simulate a pre-journal reset that committed chat before updating V2 state.
    async with store.transaction(1) as transaction:
        await transaction.write_text("chat.md", "")

    await MemoryPipeline(MarkdownMemoryStore(tmp_path)).run(1, {})

    recovered = await store.load_state(1)
    assert not recovered.rebuild_required
    assert recovered.rebuild_from_message is None
    assert not list((tmp_path / "1" / "episodes").glob("episode-*.md"))


@pytest.mark.asyncio
async def test_checkpoint_beyond_truncated_total_triggers_startup_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.services.md_store as md_store_module

    store = MarkdownMemoryStore(tmp_path)
    for pair in range(1, 6):
        await store.append_pair(
            1,
            f"user-{pair}",
            f"assistant-{pair}",
            char_name="Character",
            user_name="User",
        )
    await store.write_text(1, "story_state.md", "stale story")
    await store.save_state(
        1,
        MemoryState(
            last_memory_message=10,
            last_story_state_message=10,
            last_rag_message=10,
            last_character_message=10,
            last_assets_message=10,
        ),
    )
    real_replace = md_store_module.os.replace

    def fail_pending_state(source: Path, target: Path) -> None:
        if target.name == "memory_state.md":
            raise OSError("pending state failed")
        real_replace(source, target)

    monkeypatch.setattr(md_store_module.os, "replace", fail_pending_state)

    retained = await store.truncate_chat(1, remove_count=2)

    assert len(retained) == 8
    assert retained == await store.load_chat(1)
    assert len(await store.load_chat(1)) == 8
    old_state = await store.load_state(1)
    assert old_state.last_memory_message == 10
    assert MemoryTaskManager._has_pending(8, old_state)

    async def resolve(session_id):
        return BACKEND

    startup_manager = MemoryTaskManager(MemoryPipeline(store), resolve)
    await startup_manager.scan_pending_sessions()
    assert startup_manager.pending == {1}

    monkeypatch.setattr(md_store_module.os, "replace", real_replace)
    monkeypatch.setattr(settings, "story_state_interval_messages", 100)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 100)
    monkeypatch.setattr(settings, "episode_size_messages", 100)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 100)
    monkeypatch.setattr(settings, "assets_interval_messages", 100)
    async def story(store_arg, session_id, *args, source, **kwargs):
        return await _completed_story_result(store_arg, session_id, source)

    async def extract(session_id, *args, store, source, **kwargs):
        return await _completed_memory_result(store, session_id, source)

    async def build_rag(session_id, *args, store, source, **kwargs):
        return await _completed_rag_result(store, session_id, source)

    async def no_assets(session_id, *args, store, source, **kwargs):
        return await _completed_assets_result(store, session_id, source)

    monkeypatch.setattr(story_memory, "update_story_state", story)
    monkeypatch.setattr(memory, "rebuild_memory_and_profiles", extract)
    monkeypatch.setattr(rag, "build_index", build_rag)
    monkeypatch.setattr(memory, "rebuild_assets", no_assets)

    await asyncio.wait_for(MemoryPipeline(store).run(1, BACKEND), timeout=1)

    recovered = await store.load_state(1)
    assert not recovered.rebuild_required
    assert not MemoryTaskManager._has_pending(8, recovered)


@pytest.mark.parametrize("failed_target", ["index.json", "story_state.md", "summary.md"])
@pytest.mark.asyncio
async def test_partial_cleanup_failure_is_scannable_and_restart_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_target: str,
) -> None:
    import app.services.md_store as md_store_module

    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "user",
        "assistant",
        char_name="Character",
        user_name="User",
    )
    await store.write_text(1, "story_state.md", "stale story")
    await store.write_text(1, "summary.md", "stale summary")
    await store.write_text(
        1,
        "rag/index.json",
        '{"chunks": [], "embeddings": [], "indexed_messages": 2}',
    )
    await store.save_state(1, MemoryState(last_story_state_message=2))
    real_replace = md_store_module.os.replace

    def fail_target(source: Path, target: Path) -> None:
        if target.name == failed_target:
            raise OSError("cleanup failed")
        real_replace(source, target)

    monkeypatch.setattr(md_store_module.os, "replace", fail_target)

    retained = await store.truncate_chat(1, remove_count=2)

    assert retained == []
    assert await store.load_chat(1) == []
    pending = await store.load_state(1)
    assert pending.cleanup_required
    assert MemoryTaskManager._has_pending(0, pending)

    monkeypatch.setattr(md_store_module.os, "replace", real_replace)
    await MemoryPipeline(store).run(1, {})

    assert not (await store.load_state(1)).rebuild_required
    assert await store.read_text(1, "story_state.md") == ""
    assert await store.read_text(1, "summary.md") == ""


@pytest.mark.asyncio
async def test_cleanup_completion_state_failure_retries_without_llm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.services.md_store as md_store_module

    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "user",
        "assistant",
        char_name="Character",
        user_name="User",
    )
    await store.save_state(1, MemoryState(last_story_state_message=2))
    real_replace = md_store_module.os.replace
    state_writes = 0

    def fail_second_state(source: Path, target: Path) -> None:
        nonlocal state_writes
        if target.name == "memory_state.md":
            state_writes += 1
            if state_writes == 2:
                raise OSError("cleanup completion state failed")
        real_replace(source, target)

    monkeypatch.setattr(md_store_module.os, "replace", fail_second_state)

    retained = await store.truncate_chat(1, remove_count=2)

    assert retained == []
    assert await store.load_chat(1) == []
    assert (await store.load_state(1)).cleanup_required

    monkeypatch.setattr(md_store_module.os, "replace", real_replace)
    await MemoryPipeline(store).run(1, {})

    assert not (await store.load_state(1)).rebuild_required


@pytest.mark.parametrize("operation", ["reset", "undo"])
@pytest.mark.parametrize(
    "fault",
    ["memory", "profile", "assets", "episode", "state"],
)
@pytest.mark.asyncio
async def test_empty_history_reconciliation_fault_is_scannable_and_restart_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    fault: str,
) -> None:
    import app.services.md_store as md_store_module

    monkeypatch.setattr(settings, "story_state_interval_messages", 100)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 100)
    monkeypatch.setattr(settings, "episode_size_messages", 2)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 100)
    monkeypatch.setattr(settings, "assets_interval_messages", 100)
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "only user",
        "only assistant",
        char_name="Character",
        user_name="User",
    )
    await store.write_text(1, "memory.md", "STALE_MEMORY")
    await store.write_text(1, "characters/Old.md", "STALE_PROFILE")
    await store.write_text(1, "assets.md", "STALE_ASSETS")
    await store.write_text(1, "assets_summary.txt", "STALE_ASSET_SUMMARY")
    await store.write_text(1, "story_state.md", "STALE_STORY")
    await store.write_text(1, "summary.md", "STALE_SUMMARY")
    await store.write_text(
        1,
        "episodes/episode-000001.md",
        _episode_document(1, 1, 2),
    )
    await store.write_text(
        1,
        "rag/index.json",
        '{"chunks": [{"text": "STALE_RAG", "start_message": 1, '
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

    real_unlink = Path.unlink
    real_replace = md_store_module.os.replace
    real_finish = md_store_module.MarkdownMemoryTransaction.finish_invalidation_cleanup
    target_names = {
        "memory": "memory.md",
        "profile": "Old.md",
        "assets": "assets.md",
        "episode": "episode-000001.md",
    }
    failed = False
    state_writes = 0

    async def finish_and_restore_episode(self, plan):
        await real_finish(self, plan)
        if fault == "episode":
            await self.write_text(
                "episodes/episode-000001.md",
                _episode_document(1, 1, 2),
            )

    def fail_reconciliation_delete(path: Path, missing_ok: bool = False) -> None:
        nonlocal failed
        if fault != "state" and path.name == target_names[fault] and not failed:
            failed = True
            raise OSError(f"{fault} reconciliation failed")
        real_unlink(path, missing_ok=missing_ok)

    def fail_final_state(source: Path, target: Path) -> None:
        nonlocal state_writes
        if target.name == "memory_state.md":
            state_writes += 1
            if fault == "state" and state_writes == 3:
                raise OSError("state reconciliation failed")
        real_replace(source, target)

    if fault == "episode":
        monkeypatch.setattr(
            md_store_module.MarkdownMemoryTransaction,
            "finish_invalidation_cleanup",
            finish_and_restore_episode,
        )
    monkeypatch.setattr(Path, "unlink", fail_reconciliation_delete)
    monkeypatch.setattr(md_store_module.os, "replace", fail_final_state)

    if operation == "reset":
        retained = await store.truncate_chat(1, remove_count=2)
    else:
        retained = await ChatService(store, None).undo(1)

    assert retained == []
    assert await store.load_chat(1) == []
    pending = await store.load_state(1)
    assert pending.rebuild_required
    assert pending.rebuild_from_message == 0
    assert (tmp_path / "1" / "invalidation_intent.md").exists()
    assert MemoryTaskManager._has_pending(
        0,
        pending,
        has_invalidation_intent=True,
    )

    if fault == "state":
        session = tmp_path / "1"
        assert not (session / "memory.md").exists()
        assert not list((session / "characters").glob("*.md"))
        assert not (session / "assets.md").exists()
        assert not (session / "assets_summary.txt").exists()
        assert not list((session / "episodes").glob("episode-*.md"))
        assert json.loads(await store.read_text(1, "rag/index.json"))["chunks"] == []

    monkeypatch.setattr(Path, "unlink", real_unlink)
    monkeypatch.setattr(md_store_module.os, "replace", real_replace)
    monkeypatch.setattr(
        md_store_module.MarkdownMemoryTransaction,
        "finish_invalidation_cleanup",
        real_finish,
    )

    async def resolve(session_id: int) -> dict[str, object]:
        return BACKEND

    restarted = MarkdownMemoryStore(tmp_path)
    manager = MemoryTaskManager(MemoryPipeline(restarted), resolve)
    await manager.scan_pending_sessions()
    assert manager.pending == {1}

    stage_calls: list[str] = []

    async def fail_stage(*args, **kwargs):
        stage_calls.append("called")
        raise AssertionError("empty-history recovery must not run a derived stage")

    monkeypatch.setattr(story_memory, "update_story_state", fail_stage)
    monkeypatch.setattr(memory, "rebuild_memory_and_profiles", fail_stage)
    monkeypatch.setattr(story_memory, "create_due_episodes", fail_stage)
    monkeypatch.setattr(story_memory, "update_summary_from_episodes", fail_stage)
    monkeypatch.setattr(rag, "build_index", fail_stage)
    monkeypatch.setattr(memory, "rebuild_assets", fail_stage)

    pipeline = MemoryPipeline(restarted)
    await pipeline.run(1, BACKEND)
    await pipeline.run(1, BACKEND)

    session = tmp_path / "1"
    assert stage_calls == []
    assert await restarted.load_state(1) == MemoryState()
    assert not (session / "invalidation_intent.md").exists()
    assert not (session / "memory.md").exists()
    assert not list((session / "characters").glob("*.md"))
    assert not (session / "assets.md").exists()
    assert not (session / "assets_summary.txt").exists()
    assert await restarted.read_text(1, "story_state.md") == ""
    assert await restarted.read_text(1, "summary.md") == ""
    assert not list((session / "episodes").glob("episode-*.md"))
    assert json.loads(await restarted.read_text(1, "rag/index.json")) == rag.empty_index()


@pytest.mark.parametrize(
    ("directory_name", "victim_name"),
    [
        ("characters", "External.md"),
        ("episodes", "episode-external.md"),
    ],
)
@pytest.mark.asyncio
async def test_empty_history_reconciliation_rejects_escaped_derived_directory(
    tmp_path: Path,
    directory_name: str,
    victim_name: str,
) -> None:
    from app.services.md_store import build_invalidation_intent

    store = MarkdownMemoryStore(tmp_path / "memory")
    old_records = [
        ChatRecord(1, "user", "old user", "2026-07-20 00:00", "User"),
        ChatRecord(
            2,
            "assistant",
            "old assistant",
            "2026-07-20 00:01",
            "Character",
        ),
    ]
    await store.save_state(
        1,
        MemoryState(rebuild_memory_required=True, rebuild_from_message=0),
    )
    async with store.transaction(1) as transaction:
        await transaction.save_invalidation_intent(
            build_invalidation_intent(
                "truncate",
                boundary=0,
                old_records=old_records,
                target_records=[],
            )
        )

    external = tmp_path / f"external-{directory_name}"
    external.mkdir()
    victim = external / victim_name
    victim.write_text("EXTERNAL_MUST_SURVIVE", encoding="utf-8")
    link = store.session_dir(1) / directory_name
    try:
        link.symlink_to(external, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlinks are unavailable on this platform: {exc}")

    state_before = await store.read_text(1, "memory_state.md")
    intent_before = await store.read_text(1, "invalidation_intent.md")

    with pytest.raises(ValueError, match="derived artifact directory escaped"):
        async with store.transaction(1) as transaction:
            await transaction.reconcile_empty_history()

    assert victim.read_text(encoding="utf-8") == "EXTERNAL_MUST_SURVIVE"
    assert await store.read_text(1, "memory_state.md") == state_before
    assert await store.read_text(1, "invalidation_intent.md") == intent_before

    async def resolve(session_id: int) -> dict[str, object]:
        return BACKEND

    restarted = MarkdownMemoryStore(tmp_path / "memory")
    manager = MemoryTaskManager(MemoryPipeline(restarted), resolve)
    await manager.scan_pending_sessions()
    assert manager.pending == {1}

    link.unlink()
    await MemoryPipeline(restarted).run(1, BACKEND)

    assert victim.read_text(encoding="utf-8") == "EXTERNAL_MUST_SURVIVE"
    assert await restarted.load_state(1) == MemoryState()
    assert not (restarted.session_dir(1) / "invalidation_intent.md").exists()


@pytest.mark.asyncio
async def test_retry_state_write_failure_recovers_exact_boundary_from_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.services.md_store as md_store_module

    monkeypatch.setattr(settings, "story_state_interval_messages", 100)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 100)
    monkeypatch.setattr(settings, "episode_size_messages", 2)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 100)
    monkeypatch.setattr(settings, "assets_interval_messages", 100)
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
    await store.write_text(1, "memory.md", "stale memory through four")
    await store.write_text(1, "story_state.md", "stale story through four")
    await store.write_text(1, "summary.md", "stale summary through four")
    await store.write_text(1, "episodes/episode-000001.md", _episode_document(1, 1, 2))
    await store.write_text(1, "episodes/episode-000002.md", _episode_document(2, 3, 4))
    await store.write_text(
        1,
        "rag/index.json",
        '{"chunks": ['
        '{"text": "one", "start_message": 1, "end_message": 2},'
        '{"text": "stale", "start_message": 3, "end_message": 4}'
        '], "embeddings": [[1], [2]], "indexed_messages": 4}',
    )
    await store.save_state(
        1,
        MemoryState(
            last_memory_message=4,
            last_story_state_message=4,
            last_summary_message=4,
            last_episode_message=4,
            last_rag_message=4,
            last_character_message=4,
            last_assets_message=4,
        ),
    )
    real_replace = md_store_module.os.replace

    def fail_state(source: Path, target: Path) -> None:
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
    assert (await store.load_chat(1))[-1].content == "replacement assistant"
    assert (tmp_path / "1" / "invalidation_intent.md").exists()
    old_state = await store.load_state(1)
    assert old_state.last_rag_message == 4
    assert MemoryTaskManager._has_pending(
        4,
        old_state,
        has_invalidation_intent=True,
    )

    async def resolve(session_id):
        return BACKEND

    restarted = MarkdownMemoryStore(tmp_path)
    manager = MemoryTaskManager(MemoryPipeline(restarted), resolve)
    await manager.scan_pending_sessions()
    assert manager.pending == {1}

    monkeypatch.setattr(md_store_module.os, "replace", real_replace)
    calls: list[str] = []

    async def story(store_arg, session_id, *args, source, **kwargs):
        calls.append("story")
        return await _completed_story_result(store_arg, session_id, source)

    async def extract(session_id, *args, store, source, **kwargs):
        calls.append("memory")
        assert await restarted.read_text(1, "memory.md") == "stale memory through four"
        assert await restarted.read_text(1, "story_state.md") == "story state\n"
        assert await restarted.read_text(1, "summary.md") == ""
        assert not (tmp_path / "1" / "episodes" / "episode-000002.md").exists()
        index = json.loads(await restarted.read_text(1, "rag/index.json"))
        assert [chunk["end_message"] for chunk in index["chunks"]] == [2]
        return await _completed_memory_result(
            store,
            session_id,
            source,
            text="rebuilt memory",
        )

    async def episodes(store_arg, session_id, *args, source, **kwargs):
        calls.append("episode")
        assert args[1] == 2
        document = _episode_document(2, 3, 4)
        await store_arg.write_text(
            session_id,
            "episodes/episode-000002.md",
            document,
        )
        return await _completed_episode_result(
            store_arg,
            session_id,
            source,
            4,
        )

    async def summary(store_arg, session_id, *args, source, **kwargs):
        calls.append("summary")
        return await _completed_summary_result(
            store_arg,
            session_id,
            source,
            4,
        )

    async def build_rag(session_id, *args, store, source, **kwargs):
        calls.append("rag")
        return await _completed_rag_result(store, session_id, source)

    async def assets(session_id, *args, store, source, **kwargs):
        calls.append("assets")
        return await _completed_assets_result(store, session_id, source)

    monkeypatch.setattr(story_memory, "update_story_state", story)
    monkeypatch.setattr(memory, "rebuild_memory_and_profiles", extract)
    monkeypatch.setattr(story_memory, "create_due_episodes", episodes)
    monkeypatch.setattr(story_memory, "update_summary_from_episodes", summary)
    monkeypatch.setattr(rag, "build_index", build_rag)
    monkeypatch.setattr(memory, "rebuild_assets", assets)

    await MemoryPipeline(restarted).run(1, BACKEND)

    assert calls == ["story", "memory", "episode", "summary", "rag", "assets"]
    assert await restarted.read_text(1, "memory.md") == "rebuilt memory"
    assert not (await restarted.load_state(1)).rebuild_required
    assert not (tmp_path / "1" / "invalidation_intent.md").exists()


@pytest.mark.asyncio
async def test_retry_rag_cleanup_failure_restarts_at_persisted_boundary_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.services.md_store as md_store_module

    monkeypatch.setattr(settings, "story_state_interval_messages", 100)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 100)
    monkeypatch.setattr(settings, "episode_size_messages", 2)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 100)
    monkeypatch.setattr(settings, "assets_interval_messages", 100)
    store = MarkdownMemoryStore(tmp_path)
    records = await _seed_stale_retry_artifacts(store)
    real_replace = md_store_module.os.replace

    def fail_rag(source: Path, target: Path) -> None:
        if target.name == "index.json":
            raise OSError("rag cleanup failed")
        real_replace(source, target)

    monkeypatch.setattr(md_store_module.os, "replace", fail_rag)

    updated = await store.replace_final_pair(
        1,
        expected_user=records[-2],
        expected_assistant=records[-1],
        assistant_content="replacement assistant",
    )

    assert updated[-1].content == "replacement assistant"
    pending = await store.load_state(1)
    assert pending.cleanup_required
    assert pending.rebuild_from_message == 2
    assert (tmp_path / "1" / "invalidation_intent.md").exists()

    monkeypatch.setattr(md_store_module.os, "replace", real_replace)

    async def story(store_arg, session_id, *args, source, **kwargs):
        return await _completed_story_result(store_arg, session_id, source)

    async def extract(session_id, *args, store, source, **kwargs):
        return await _completed_memory_result(store, session_id, source)

    async def episodes(store_arg, session_id, *args, source, **kwargs):
        return await _completed_episode_result(store_arg, session_id, source, 2)

    async def summary(store_arg, session_id, *args, source, **kwargs):
        return await _completed_summary_result(
            store_arg,
            session_id,
            source,
            2,
        )

    async def build_rag(session_id, *args, store, source, **kwargs):
        index = json.loads(await store.read_text(1, "rag/index.json"))
        assert [chunk["end_message"] for chunk in index["chunks"]] == [2]
        assert all(chunk["text"] != "stale" for chunk in index["chunks"])
        return await _completed_rag_result(store, session_id, source)

    async def assets(session_id, *args, store, source, **kwargs):
        return await _completed_assets_result(store, session_id, source)

    monkeypatch.setattr(story_memory, "update_story_state", story)
    monkeypatch.setattr(memory, "rebuild_memory_and_profiles", extract)
    monkeypatch.setattr(story_memory, "create_due_episodes", episodes)
    monkeypatch.setattr(story_memory, "update_summary_from_episodes", summary)
    monkeypatch.setattr(rag, "build_index", build_rag)
    monkeypatch.setattr(memory, "rebuild_assets", assets)

    await MemoryPipeline(MarkdownMemoryStore(tmp_path)).run(1, BACKEND)

    assert not (await store.load_state(1)).rebuild_required
    assert not (tmp_path / "1" / "invalidation_intent.md").exists()


@pytest.mark.asyncio
async def test_retry_episode_delete_failure_reuses_prefix_and_regenerates_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "story_state_interval_messages", 100)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 100)
    monkeypatch.setattr(settings, "episode_size_messages", 2)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 100)
    monkeypatch.setattr(settings, "assets_interval_messages", 100)
    store = MarkdownMemoryStore(tmp_path)
    records = await _seed_stale_retry_artifacts(store)
    first_episode = await store.read_text(1, "episodes/episode-000001.md")
    real_unlink = Path.unlink

    def fail_episode_two(path: Path, missing_ok: bool = False) -> None:
        if path.name == "episode-000002.md":
            raise OSError("episode two deletion failed")
        real_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_episode_two)

    updated = await store.replace_final_pair(
        1,
        expected_user=records[-2],
        expected_assistant=records[-1],
        assistant_content="replacement assistant",
    )

    assert updated[-1].content == "replacement assistant"
    pending = await store.load_state(1)
    assert pending.cleanup_required
    assert pending.rebuild_from_message == 2
    assert (tmp_path / "1" / "episodes" / "episode-000002.md").exists()

    monkeypatch.setattr(Path, "unlink", real_unlink)
    episode_completions = 0

    async def complete(messages):
        nonlocal episode_completions
        episode_completions += 1
        return (
            "## 剧情摘要\n- replacement events\n\n"
            "## 状态变化\n- replacement state\n\n"
            "## 承诺与伏笔\n- replacement facts"
        )

    monkeypatch.setattr(MemoryPipeline, "_complete_text", lambda self, backend: complete)

    async def story(store_arg, session_id, *args, source, **kwargs):
        return await _completed_story_result(store_arg, session_id, source)

    async def extract(session_id, *args, store, source, **kwargs):
        return await _completed_memory_result(store, session_id, source)

    async def summary(store_arg, session_id, *args, source, **kwargs):
        return await _completed_summary_result(
            store_arg,
            session_id,
            source,
            4,
        )

    async def build_rag(session_id, *args, store, source, **kwargs):
        return await _completed_rag_result(store, session_id, source)

    async def assets(session_id, *args, store, source, **kwargs):
        return await _completed_assets_result(store, session_id, source)

    monkeypatch.setattr(story_memory, "update_story_state", story)
    monkeypatch.setattr(memory, "rebuild_memory_and_profiles", extract)
    monkeypatch.setattr(story_memory, "update_summary_from_episodes", summary)
    monkeypatch.setattr(rag, "build_index", build_rag)
    monkeypatch.setattr(memory, "rebuild_assets", assets)

    await MemoryPipeline(MarkdownMemoryStore(tmp_path)).run(1, BACKEND)

    assert await store.read_text(1, "episodes/episode-000001.md") == first_episode
    regenerated = await store.read_text(1, "episodes/episode-000002.md")
    assert "<!-- messages: 3-4 -->" in regenerated
    assert "replacement events" in regenerated
    assert episode_completions == 1
    assert not (await store.load_state(1)).rebuild_required
    assert not (tmp_path / "1" / "invalidation_intent.md").exists()


@pytest.mark.asyncio
async def test_historical_episode_size_delete_failure_restarts_newest_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "story_state_interval_messages", 100)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 100)
    monkeypatch.setattr(settings, "episode_size_messages", 4)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 100)
    monkeypatch.setattr(settings, "assets_interval_messages", 100)
    store = MarkdownMemoryStore(tmp_path)
    for pair in range(1, 5):
        await store.append_pair(
            1,
            f"user-{pair}",
            f"assistant-{pair}",
            char_name="Character",
            user_name="User",
        )
    records = await store.load_chat(1)
    for number, start in enumerate((1, 3, 5, 7), start=1):
        await store.write_text(
            1,
            f"episodes/episode-{number:06d}.md",
            _episode_document(number, start, start + 1),
        )
    await store.save_state(1, MemoryState(last_episode_message=8))

    real_unlink = Path.unlink
    deletion_attempts: list[str] = []
    fail_enabled = True

    def fail_historical_episode(path: Path, missing_ok: bool = False) -> None:
        if path.parent.name == "episodes" and path.name.startswith("episode-"):
            deletion_attempts.append(path.name)
            if fail_enabled and path.name == "episode-000002.md":
                raise OSError("historical episode deletion failed")
        real_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_historical_episode)

    retained = await store.truncate_chat(1, remove_count=2)

    assert len(retained) == 6
    assert retained == await store.load_chat(1)
    assert len(await store.load_chat(1)) == 6
    assert deletion_attempts == [
        "episode-000004.md",
        "episode-000003.md",
        "episode-000002.md",
    ]
    pending = await store.load_state(1)
    assert pending.cleanup_required
    assert pending.rebuild_episode_required
    assert pending.last_episode_message == 0
    assert pending.rebuild_from_message == 6
    assert (tmp_path / "1" / "invalidation_intent.md").exists()
    assert sorted(
        path.name for path in (tmp_path / "1" / "episodes").glob("episode-*.md")
    ) == ["episode-000001.md", "episode-000002.md"]

    async def resolve(session_id: int) -> dict[str, object]:
        return BACKEND

    restarted = MarkdownMemoryStore(tmp_path)
    manager = MemoryTaskManager(MemoryPipeline(restarted), resolve)
    await manager.scan_pending_sessions()
    assert manager.pending == {1}

    fail_enabled = False
    stage_calls: list[str] = []
    episode_completions = 0

    async def story(store_arg, session_id, *args, source, **kwargs):
        stage_calls.append("story")
        return await _completed_story_result(store_arg, session_id, source)

    async def extract(session_id, *args, store, source, **kwargs):
        stage_calls.append("memory")
        return await _completed_memory_result(store, session_id, source)

    async def complete(messages):
        nonlocal episode_completions
        episode_completions += 1
        return (
            "## 剧情摘要\n- rebuilt current-size events\n\n"
            "## 状态变化\n- rebuilt current-size state\n\n"
            "## 承诺与伏笔\n- rebuilt current-size facts"
        )

    async def summary(store_arg, session_id, *args, source, **kwargs):
        stage_calls.append("summary")
        return await _completed_summary_result(
            store_arg,
            session_id,
            source,
            4,
        )

    async def build_rag(session_id, *args, store, source, **kwargs):
        stage_calls.append("rag")
        return await _completed_rag_result(store, session_id, source)

    async def assets(session_id, *args, store, source, **kwargs):
        stage_calls.append("assets")
        return await _completed_assets_result(store, session_id, source)

    monkeypatch.setattr(MemoryPipeline, "_complete_text", lambda self, backend: complete)
    monkeypatch.setattr(story_memory, "update_story_state", story)
    monkeypatch.setattr(memory, "rebuild_memory_and_profiles", extract)
    monkeypatch.setattr(story_memory, "update_summary_from_episodes", summary)
    monkeypatch.setattr(rag, "build_index", build_rag)
    monkeypatch.setattr(memory, "rebuild_assets", assets)

    pipeline = MemoryPipeline(restarted)
    await pipeline.run(1, BACKEND)

    assert deletion_attempts == [
        "episode-000004.md",
        "episode-000003.md",
        "episode-000002.md",
        "episode-000002.md",
        "episode-000001.md",
    ]
    rebuilt_episode = await restarted.read_text(
        1,
        "episodes/episode-000001.md",
    )
    assert "<!-- messages: 1-4 -->" in rebuilt_episode
    assert episode_completions == 1
    state = await restarted.load_state(1)
    assert state.last_episode_message == 4
    assert not state.rebuild_required
    assert not (tmp_path / "1" / "invalidation_intent.md").exists()

    calls_after_recovery = list(stage_calls)
    attempts_after_recovery = list(deletion_attempts)
    await pipeline.run(1, BACKEND)

    assert stage_calls == calls_after_recovery
    assert deletion_attempts == attempts_after_recovery
    assert episode_completions == 1
    assert await restarted.load_state(1) == state


@pytest.mark.parametrize("intent_kind", ["malformed", "neither"])
@pytest.mark.asyncio
async def test_untrusted_intent_fails_without_any_derived_write(
    tmp_path: Path,
    intent_kind: str,
) -> None:
    from app.services.md_store import build_invalidation_intent

    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "authoritative user",
        "authoritative assistant",
        char_name="Character",
        user_name="User",
    )
    await store.write_text(1, "story_state.md", "unchanged story")
    await store.write_text(1, "summary.md", "unchanged summary")
    await store.save_state(1, MemoryState(last_story_state_message=2))
    if intent_kind == "malformed":
        await store.write_text(
            1,
            "invalidation_intent.md",
            "# Invalidation Intent\n\n<!-- boundary-message: 0 -->\n",
        )
        expected_error = "invalid invalidation intent"
    else:
        other = MarkdownMemoryStore(tmp_path / "other")
        for pair in range(1, 3):
            await other.append_pair(
                1,
                f"other-user-{pair}",
                f"other-assistant-{pair}",
                char_name="Character",
                user_name="User",
            )
        old = await other.load_chat(1)
        target = [
            *old[:-1],
            type(old[-1])(
                4,
                "assistant",
                "other replacement",
                old[-1].timestamp,
                old[-1].name,
            ),
        ]
        intent = build_invalidation_intent(
            "replace-final-pair",
            boundary=2,
            old_records=old,
            target_records=target,
        )
        async with store.transaction(1) as transaction:
            await transaction.save_invalidation_intent(intent)
        expected_error = "does not match authoritative chat"
    session = tmp_path / "1"
    relatives = (
        "chat.md",
        "memory_state.md",
        "story_state.md",
        "summary.md",
        "invalidation_intent.md",
    )
    before = {relative: (session / relative).read_bytes() for relative in relatives}

    with pytest.raises(ValueError, match=expected_error):
        await MemoryPipeline(MarkdownMemoryStore(tmp_path)).run(1, BACKEND)

    assert {
        relative: (session / relative).read_bytes() for relative in relatives
    } == before


@pytest.mark.parametrize(
    "intent_bytes",
    [b"", b" \t\r\n"],
    ids=["zero-byte", "whitespace"],
)
@pytest.mark.asyncio
async def test_scanner_and_pipeline_reject_present_empty_intent_without_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    intent_bytes: bytes,
) -> None:
    monkeypatch.setattr(settings, "story_state_interval_messages", 100)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 100)
    monkeypatch.setattr(settings, "episode_size_messages", 100)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 100)
    monkeypatch.setattr(settings, "assets_interval_messages", 100)
    store = MarkdownMemoryStore(tmp_path)
    await _seed_stale_retry_artifacts(store)
    session = tmp_path / "1"
    (session / "invalidation_intent.md").write_bytes(intent_bytes)
    relatives = (
        "chat.md",
        "memory_state.md",
        "memory.md",
        "story_state.md",
        "summary.md",
        "episodes/episode-000001.md",
        "episodes/episode-000002.md",
        "rag/index.json",
        "invalidation_intent.md",
    )
    before = {relative: (session / relative).read_bytes() for relative in relatives}

    async def resolve(session_id: int) -> dict[str, object]:
        return BACKEND

    manager = MemoryTaskManager(MemoryPipeline(store), resolve)
    with caplog.at_level(logging.ERROR, logger="app.services.memory_tasks"):
        await manager.scan_pending_sessions()

    scan_errors = [
        record
        for record in caplog.records
        if record.getMessage() == "Failed to scan memory state for session 1"
    ]
    assert len(scan_errors) == 1
    assert scan_errors[0].exc_info is not None
    scan_error = scan_errors[0].exc_info[1]
    assert isinstance(scan_error, InvalidationIntentError)
    assert str(scan_error) == "invalid invalidation intent"
    assert manager.pending == set()
    assert {
        relative: (session / relative).read_bytes() for relative in relatives
    } == before

    with pytest.raises(InvalidationIntentError) as captured:
        await MemoryPipeline(MarkdownMemoryStore(tmp_path)).run(1, BACKEND)

    assert str(captured.value) == "invalid invalidation intent"
    assert {
        relative: (session / relative).read_bytes() for relative in relatives
    } == before


@pytest.mark.asyncio
async def test_retry_flags_without_boundary_fail_instead_of_guessing_current_total(
    tmp_path: Path,
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
    before_state = await store.read_text(1, "memory_state.md")

    with pytest.raises(ValueError, match="no persisted rebuild boundary"):
        await MemoryPipeline(store).run(1, BACKEND)

    assert await store.read_text(1, "story_state.md") == "must remain unchanged"
    assert await store.read_text(1, "memory_state.md") == before_state


@pytest.mark.asyncio
async def test_startup_scanner_migrates_legacy_extract_checkpoint_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "story_state_interval_messages", 100)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 100)
    monkeypatch.setattr(settings, "episode_size_messages", 100)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 100)
    monkeypatch.setattr(settings, "assets_interval_messages", 100)
    store = MarkdownMemoryStore(tmp_path)
    for number in range(6):
        await store.append_pair(
            1,
            f"user-{number}",
            f"assistant-{number}",
            char_name="Character",
            user_name="User",
        )
    await store.write_text(1, ".last_extract_count", "10")

    async def resolve(session_id):
        return BACKEND

    manager = MemoryTaskManager(MemoryPipeline(store), resolve)

    await manager.scan_pending_sessions()

    state = await store.load_state(1)
    assert state.last_memory_message == 10
    assert state.last_character_message == 10
    assert manager.pending == set()


@pytest.mark.asyncio
async def test_concurrent_runs_for_one_session_do_not_repeat_a_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isolate_story_stage(monkeypatch)
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "hello",
        "hi",
        char_name="Character",
        user_name="User",
    )
    entered = asyncio.Event()
    entered_twice = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def update_story(
        store_arg,
        session_id,
        records,
        complete,
        *,
        source,
    ):
        nonlocal calls
        calls += 1
        entered.set()
        if calls == 2:
            entered_twice.set()
        await release.wait()
        return await _completed_story_result(store_arg, session_id, source)

    monkeypatch.setattr(story_memory, "update_story_state", update_story)
    pipeline = MemoryPipeline(store)
    first = asyncio.create_task(pipeline.run(1, BACKEND))
    await asyncio.wait_for(entered.wait(), timeout=1)
    second = asyncio.create_task(pipeline.run(1, BACKEND))
    try:
        await asyncio.wait_for(entered_twice.wait(), timeout=0.25)
    except TimeoutError:
        pass
    release.set()
    await asyncio.gather(first, second)

    assert calls == 1


@pytest.mark.asyncio
async def test_concurrent_runs_for_different_sessions_can_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isolate_story_stage(monkeypatch)
    store = MarkdownMemoryStore(tmp_path)
    for session_id in (1, 2):
        await store.append_pair(
            session_id,
            "hello",
            "hi",
            char_name="Character",
            user_name="User",
        )
    both_entered = asyncio.Event()
    release = asyncio.Event()
    entered_sessions: set[int] = set()

    async def update_story(
        store_arg,
        session_id,
        records,
        complete,
        *,
        source,
    ):
        entered_sessions.add(session_id)
        if entered_sessions == {1, 2}:
            both_entered.set()
        await release.wait()
        return await _completed_story_result(store_arg, session_id, source)

    monkeypatch.setattr(story_memory, "update_story_state", update_story)
    pipelines = {
        session_id: MemoryPipeline(store)
        for session_id in (1, 2)
    }
    tasks = [
        asyncio.create_task(pipelines[session_id].run(session_id, BACKEND))
        for session_id in (1, 2)
    ]
    await asyncio.wait_for(both_entered.wait(), timeout=1)
    release.set()
    await asyncio.gather(*tasks)

    assert entered_sessions == {1, 2}


@pytest.mark.asyncio
async def test_app_lifespan_manages_memory_worker_and_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.database as database
    import app.main as main
    import app.services.feishu_ws as feishu_ws

    events: list[str] = []

    class Connection:
        async def run_sync(self, callback):
            events.append("database")

    class BeginContext:
        async def __aenter__(self):
            return Connection()

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    class Engine:
        def begin(self):
            return BeginContext()

        async def dispose(self):
            events.append("dispose")

    class Pipeline:
        def __init__(self, store):
            self.store = store
            events.append("pipeline")

    class Manager:
        def __init__(self, pipeline, backend_resolver):
            events.append("manager")

        async def start(self):
            events.append("start")

        async def scan_pending_sessions(self):
            events.append("scan")

        async def stop(self):
            events.append("stop")

    async def no_missing_columns(conn, table, column):
        return False

    monkeypatch.setattr(settings, "memory_v2_enabled", True)
    monkeypatch.setattr(main, "engine", Engine())
    monkeypatch.setattr(main, "_column_missing", no_missing_columns)
    monkeypatch.setattr(main, "MemoryPipeline", Pipeline, raising=False)
    monkeypatch.setattr(main, "MemoryTaskManager", Manager, raising=False)
    monkeypatch.setattr(main, "memory_pipeline", None)
    monkeypatch.setattr(main, "memory_task_manager", None)
    monkeypatch.setattr(database, "async_session", object())
    monkeypatch.setattr(feishu_ws, "set_session_factory", lambda factory: None)
    monkeypatch.setattr(
        feishu_ws,
        "start_ws_client",
        lambda: events.append("ws-start"),
    )
    monkeypatch.setattr(
        feishu_ws,
        "stop_ws_client",
        lambda: events.append("ws-stop"),
    )

    async with main.lifespan(main.app):
        events.append("serving")

    assert events == [
        "database",
        "pipeline",
        "manager",
        "start",
        "scan",
        "ws-start",
        "serving",
        "stop",
        "ws-stop",
        "dispose",
    ]


@pytest.mark.asyncio
async def test_app_lifespan_cleans_up_when_startup_scan_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.main as main

    events: list[str] = []

    class Connection:
        async def run_sync(self, callback):
            events.append("database")

    class BeginContext:
        async def __aenter__(self):
            return Connection()

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    class Engine:
        def begin(self):
            return BeginContext()

        async def dispose(self):
            events.append("dispose")

    class Pipeline:
        def __init__(self, store):
            events.append("pipeline")

    class Manager:
        def __init__(self, pipeline, backend_resolver):
            events.append("manager")

        async def start(self):
            events.append("start")

        async def scan_pending_sessions(self):
            events.append("scan")
            raise RuntimeError("scan failed")

        async def stop(self):
            events.append("stop")

    async def no_missing_columns(conn, table, column):
        return False

    monkeypatch.setattr(settings, "memory_v2_enabled", True)
    monkeypatch.setattr(main, "engine", Engine())
    monkeypatch.setattr(main, "_column_missing", no_missing_columns)
    monkeypatch.setattr(main, "MemoryPipeline", Pipeline)
    monkeypatch.setattr(main, "MemoryTaskManager", Manager)
    monkeypatch.setattr(main, "memory_pipeline", None)
    monkeypatch.setattr(main, "memory_task_manager", None)

    with pytest.raises(RuntimeError, match="scan failed"):
        async with main.lifespan(main.app):
            pytest.fail("lifespan should not yield")

    assert events == [
        "database",
        "pipeline",
        "manager",
        "start",
        "scan",
        "stop",
        "dispose",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("artifact", "relative"),
    [
        ("memory", "memory.md"),
        ("assets", "assets.md"),
        ("assets_summary", "assets_summary.txt"),
        ("character", "characters/Alice.md"),
    ],
)
async def test_derived_artifact_overwrite_failure_preserves_existing_file(
    artifact: str,
    relative: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    monkeypatch.setattr(memory, "MEMORY_BASE", tmp_path)
    monkeypatch.setattr(memory, "md_store", store)
    await store.write_text(1, relative, "old content")

    def fail_replace(source: Path, target: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("app.services.md_store.os.replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        if artifact == "memory":
            await memory.save_memory(1, "new content")
        elif artifact == "assets":
            await memory.save_assets(1, "new content")
        elif artifact == "assets_summary":
            await memory.save_assets_summary(1, "new content")
        else:
            await memory.save_character_profile(1, "Alice", "new content")

    assert await store.read_text(1, relative) == "old content"


@pytest.mark.asyncio
async def test_rag_index_overwrite_failure_preserves_existing_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    monkeypatch.setattr(rag, "MEMORY_BASE", tmp_path)
    old_index = {
        "chunks": ["old chunk"],
        "embeddings": [[1.0]],
        "indexed_messages": 1,
    }
    await store.write_text(
        1,
        "rag/index.json",
        json.dumps(old_index, ensure_ascii=False),
    )

    def fail_replace(source: Path, target: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("app.services.md_store.os.replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        await rag.save_index(
            1,
            {"chunks": ["new chunk"], "embeddings": [[2.0]], "indexed_messages": 2},
        )

    assert await rag.load_index(1) == old_index


@pytest.mark.parametrize("module", [memory, rag])
def test_memory_services_have_no_direct_artifact_writer_bypasses(module) -> None:
    tree = ast.parse(inspect.getsource(module))
    offenders: list[int] = []
    for node in ast.walk(tree):
        is_aiofiles_open = (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "open"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "aiofiles"
        )
        if is_aiofiles_open:
            mode = None
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                mode = node.args[1].value
            for keyword in node.keywords:
                if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
                    mode = keyword.value.value
            if isinstance(mode, str) and any(flag in mode for flag in "wax+"):
                offenders.append(node.lineno)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"copy", "copy2", "copyfile"}
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "shutil"
        ):
            offenders.append(node.lineno)

    assert offenders == []


@pytest.mark.asyncio
async def test_refused_memory_output_preserves_artifact_and_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "story_state_interval_messages", 100)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 2)
    monkeypatch.setattr(settings, "episode_size_messages", 100)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 100)
    monkeypatch.setattr(settings, "assets_interval_messages", 100)
    store = MarkdownMemoryStore(tmp_path)
    monkeypatch.setattr(memory, "MEMORY_BASE", tmp_path)
    monkeypatch.setattr(memory, "md_store", store)
    await store.append_pair(
        1,
        "hello",
        "hi",
        char_name="Character",
        user_name="User",
    )
    await store.write_text(1, "memory.md", "old memory")

    async def refuse(**kwargs):
        return {"content": "## Relationships\n- I won't create memory systems"}

    monkeypatch.setattr(memory, "chat_completion", refuse)

    with pytest.raises(ValueError, match="refusal"):
        await MemoryPipeline(store).run(1, BACKEND)

    state = await store.load_state(1)
    assert await store.read_text(1, "memory.md") == "old memory"
    assert state.last_memory_message == 0
    assert state.last_character_message == 0
    assert "refusal" in state.last_error


@pytest.mark.asyncio
async def test_refused_character_output_preserves_artifacts_and_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "story_state_interval_messages", 100)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 2)
    monkeypatch.setattr(settings, "episode_size_messages", 100)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 100)
    monkeypatch.setattr(settings, "assets_interval_messages", 100)
    store = MarkdownMemoryStore(tmp_path)
    monkeypatch.setattr(memory, "MEMORY_BASE", tmp_path)
    monkeypatch.setattr(memory, "md_store", store)
    await store.append_pair(
        1,
        "hello",
        "hi",
        char_name="Alice",
        user_name="User",
    )
    await store.write_text(1, "memory.md", "old memory")
    await memory.save_character_profile(1, "Alice", "# Alice\nold profile")

    async def refuse(**kwargs):
        return {
            "content": (
                "## Relationships\n- new memory\n"
                "===CHARACTERS===\n# Alice\nI won't produce"
            )
        }

    monkeypatch.setattr(memory, "chat_completion", refuse)

    with pytest.raises(memory.MemoryRefusalError, match="refusal"):
        await MemoryPipeline(store).run(1, BACKEND)

    state = await store.load_state(1)
    assert await store.read_text(1, "memory.md") == "old memory"
    assert await memory.load_character_profile(1, "Alice") == "# Alice\nold profile"
    assert state.last_memory_message == 0
    assert state.last_character_message == 0
    assert "refusal" in state.last_error


@pytest.mark.asyncio
async def test_asset_partial_write_retry_repairs_summary_before_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.services.md_store as md_store_module

    monkeypatch.setattr(settings, "story_state_interval_messages", 100)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 100)
    monkeypatch.setattr(settings, "episode_size_messages", 100)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 100)
    monkeypatch.setattr(settings, "assets_interval_messages", 2)
    store = MarkdownMemoryStore(tmp_path)
    monkeypatch.setattr(memory, "MEMORY_BASE", tmp_path)
    monkeypatch.setattr(memory, "md_store", store)
    await store.append_pair(
        1,
        "bought a house",
        "purchase complete",
        char_name="Character",
        user_name="User",
    )
    await store.write_text(1, "assets.md", "old assets")
    await store.write_text(1, "assets_summary.txt", "old summary")

    responses = iter(
        [
            "new assets",
            "new summary",
            "NO_CHANGE",
            "repaired summary",
        ]
    )

    async def complete(**kwargs):
        return {"content": next(responses)}

    real_replace = md_store_module.os.replace
    failed_summary = False

    def fail_summary_once(source: Path, target: Path) -> None:
        nonlocal failed_summary
        if target.name == "assets_summary.txt" and not failed_summary:
            failed_summary = True
            raise OSError("summary replace failed")
        real_replace(source, target)

    monkeypatch.setattr(memory, "chat_completion", complete)
    monkeypatch.setattr(md_store_module.os, "replace", fail_summary_once)
    pipeline = MemoryPipeline(store)

    with pytest.raises(OSError, match="summary replace failed"):
        await pipeline.run(1, BACKEND)

    failed_state = await store.load_state(1)
    assert await store.read_text(1, "assets.md") == "new assets"
    assert await store.read_text(1, "assets_summary.txt") == "old summary"
    assert failed_state.last_assets_message == 0

    await pipeline.run(1, BACKEND)

    state = await store.load_state(1)
    assert await store.read_text(1, "assets.md") == "new assets"
    assert await store.read_text(1, "assets_summary.txt") == "repaired summary"
    assert state.last_assets_message == 2
    assert state.last_error == ""


@pytest.mark.asyncio
async def test_rag_invalidate_crossing_chunk_rebuilds_retained_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    monkeypatch.setattr(rag, "MEMORY_BASE", tmp_path)
    for number in range(1, 6):
        await store.append_pair(
            1,
            f"u{number}",
            f"a{number}",
            char_name="Character",
            user_name="User",
        )

    async def embed(texts, **kwargs):
        return [[1.0, 0.0] for _ in texts]

    async def embed_query(text, **kwargs):
        return [1.0, 0.0]

    monkeypatch.setattr(rag, "get_embeddings", embed)
    monkeypatch.setattr(rag, "get_embedding", embed_query)
    records = await store.load_chat(1)
    original_source = chat_source_identity(records)
    initial_result = await rag.build_index(
        1,
        store=store,
        source=original_source,
    )
    await store.write_text(1, "chat.md", render_chat_records(records[:8]))

    await rag.invalidate_after(1, 8, store=store)
    invalidated = await rag.load_index(1, store=store)
    retained_records = await store.load_chat(1)
    retained_source = chat_source_identity(retained_records)
    rebuilt_result = await rag.build_index(
        1,
        store=store,
        source=retained_source,
    )
    rebuilt = await rag.load_index(1, store=store)
    results = await rag.search(1, "u4", top_k=10)

    assert initial_result.source == original_source
    assert initial_result.checkpoint == 10
    assert rebuilt_result.source == retained_source
    assert rebuilt_result.checkpoint == 8
    assert invalidated["indexed_messages"] == 5
    assert [
        (chunk["start_message"], chunk["end_message"])
        for chunk in rebuilt["chunks"]
        if isinstance(chunk, dict)
    ] == [(1, 5), (6, 8)]
    assert len(rebuilt["chunks"]) == len(rebuilt["embeddings"])
    searchable = "\n".join(result["text"] for result in results)
    assert "a3" in searchable
    assert "u4" in searchable
    assert "a4" in searchable
    assert "u5" not in searchable
    assert "a5" not in searchable


@pytest.mark.asyncio
async def test_app_lifespan_disposes_engine_when_database_begin_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.main as main

    events: list[str] = []

    class Engine:
        def begin(self):
            events.append("database-begin")
            raise RuntimeError("database unavailable")

        async def dispose(self):
            events.append("dispose")

    monkeypatch.setattr(main, "engine", Engine())
    monkeypatch.setattr(main, "memory_pipeline", None)
    monkeypatch.setattr(main, "memory_task_manager", None)

    with pytest.raises(RuntimeError, match="database unavailable"):
        async with main.lifespan(main.app):
            pytest.fail("lifespan should not yield")

    assert events == ["database-begin", "dispose"]


@pytest.mark.asyncio
async def test_two_pipelines_sharing_store_serialize_same_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isolate_story_stage(monkeypatch)
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "hello",
        "hi",
        char_name="Character",
        user_name="User",
    )
    entered = asyncio.Event()
    entered_twice = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def update_story(
        store_arg,
        session_id,
        records,
        complete,
        *,
        source,
    ):
        nonlocal calls
        calls += 1
        entered.set()
        if calls == 2:
            entered_twice.set()
        await release.wait()
        return await _completed_story_result(store_arg, session_id, source)

    monkeypatch.setattr(story_memory, "update_story_state", update_story)
    first_pipeline = MemoryPipeline(store)
    second_pipeline = MemoryPipeline(store)
    first = asyncio.create_task(first_pipeline.run(1, BACKEND))
    await asyncio.wait_for(entered.wait(), timeout=1)
    second = asyncio.create_task(second_pipeline.run(1, BACKEND))
    try:
        await asyncio.wait_for(entered_twice.wait(), timeout=0.25)
    except TimeoutError:
        pass
    release.set()
    await asyncio.gather(first, second)

    assert calls == 1
