from __future__ import annotations

import asyncio
import ast
import inspect
import json
from pathlib import Path

import pytest

from app.config import settings
from app.services import memory, rag, story_memory
from app.services.md_store import (
    ChatRecord,
    MarkdownMemoryStore,
    MemoryState,
    render_chat_records,
)
from app.services.memory_pipeline import MemoryPipeline
from app.services.memory_tasks import MemoryTaskManager


BACKEND = {
    "provider": "openai",
    "api_key": "test-key",
    "model": "test-model",
    "base_url": "https://example.invalid/v1",
    "params": {},
}


def _episode_document(number: int, start: int, end: int) -> str:
    return (
        f"# Episode {number:06d}\n\n"
        f"<!-- messages: {start}-{end} -->\n\n"
        "## 剧情摘要\n- 已发生的事件\n\n"
        "## 状态变化\n- 状态已变化\n\n"
        "## 承诺与伏笔\n- 保留的事实\n"
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

    async def fail_once_then_succeed(store, session_id, records, complete):
        attempted_ranges.append([record.number for record in records])
        if len(attempted_ranges) == 1:
            raise RuntimeError("try again")
        return "updated"

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

    async def update_once(store, session_id, records, complete):
        nonlocal calls
        calls += 1
        return "updated"

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

    async def update_story(store_arg, session_id, records, complete):
        assert store_arg is store
        assert [record.number for record in records] == [1, 2]
        calls.append("story")
        return "updated"

    async def extract(session_id, messages, backend):
        assert messages == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        calls.append("memory")

    async def create_episodes(
        store_arg,
        session_id,
        records,
        last_episode_message,
        complete,
        *,
        episode_size,
    ):
        assert [record.number for record in records] == [1, 2]
        assert last_episode_message == 0
        assert episode_size == 2
        calls.append("episode")
        return 2

    async def update_summary(
        store_arg,
        session_id,
        last_summary_message,
        complete,
        *,
        max_tokens,
    ):
        assert last_summary_message == 0
        assert max_tokens == settings.summary_max_tokens
        calls.append("summary")
        return 2

    async def build_rag(session_id, **kwargs):
        calls.append("rag")
        return {}

    async def update_assets(session_id, messages, backend):
        assert messages == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        calls.append("assets")
        return False

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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_prompt = ""
    pending = [
        {"role": "user", "content": f"pending-{number}"}
        for number in range(1, 22)
    ]

    async def no_memory(session_id):
        return ""

    async def no_characters(session_id):
        return []

    async def capture_completion(**kwargs):
        nonlocal captured_prompt
        captured_prompt = kwargs["messages"][1]["content"]
        return {"content": "NO_CHANGE"}

    async def ignore_save(session_id, text):
        return None

    monkeypatch.setattr(memory, "load_memory", no_memory)
    monkeypatch.setattr(memory, "list_character_names", no_characters)
    monkeypatch.setattr(memory, "chat_completion", capture_completion)
    monkeypatch.setattr(memory, "save_memory", ignore_save)
    monkeypatch.setattr(settings, "rag_auto_index", False)

    await memory.extract_memory_and_characters(1, pending, BACKEND)

    assert "user: pending-1\n" in captured_prompt
    assert "user: pending-21\n" in captured_prompt


@pytest.mark.asyncio
async def test_asset_update_preserves_the_exact_pending_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_prompt = ""
    pending = [
        {"role": "user", "content": f"asset-pending-{number}"}
        for number in range(1, 22)
    ]

    async def no_assets(session_id):
        return ""

    async def capture_completion(**kwargs):
        nonlocal captured_prompt
        captured_prompt = kwargs["messages"][1]["content"]
        return {"content": "NO_CHANGE"}

    monkeypatch.setattr(memory, "load_assets", no_assets)
    monkeypatch.setattr(memory, "chat_completion", capture_completion)

    changed = await memory.update_assets(1, pending, BACKEND)

    assert changed is False
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

    async def update_story(*args, **kwargs):
        return "updated"

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

    async def update_story(*args, **kwargs):
        return "updated"

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


@pytest.mark.asyncio
async def test_asset_summary_failure_does_not_partially_replace_assets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completion_calls = 0
    saved_assets: list[str] = []
    saved_summaries: list[str] = []

    async def load_existing(session_id):
        return "old assets"

    async def complete(**kwargs):
        nonlocal completion_calls
        completion_calls += 1
        if completion_calls == 1:
            return {"content": "new assets"}
        raise TimeoutError("summary unavailable")

    async def save_assets(session_id, content):
        saved_assets.append(content)

    async def save_summary(session_id, content):
        saved_summaries.append(content)

    monkeypatch.setattr(memory, "load_assets", load_existing)
    monkeypatch.setattr(memory, "chat_completion", complete)
    monkeypatch.setattr(memory, "save_assets", save_assets)
    monkeypatch.setattr(memory, "save_assets_summary", save_summary)

    with pytest.raises(TimeoutError, match="summary unavailable"):
        await memory.update_assets(
            1,
            [{"role": "user", "content": "bought a house"}],
            BACKEND,
        )

    assert saved_assets == []
    assert saved_summaries == []


@pytest.mark.asyncio
async def test_memory_extraction_does_not_build_rag_outside_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rag_calls = 0

    async def no_memory(session_id):
        return ""

    async def no_characters(session_id):
        return []

    async def complete(**kwargs):
        return {"content": "NO_CHANGE"}

    async def ignore_save(session_id, text):
        return None

    async def track_rag(*args, **kwargs):
        nonlocal rag_calls
        rag_calls += 1
        return {}

    monkeypatch.setattr(memory, "load_memory", no_memory)
    monkeypatch.setattr(memory, "list_character_names", no_characters)
    monkeypatch.setattr(memory, "chat_completion", complete)
    monkeypatch.setattr(memory, "save_memory", ignore_save)
    monkeypatch.setattr(rag, "build_index", track_rag)
    monkeypatch.setattr(settings, "rag_auto_index", True)

    await memory.extract_memory_and_characters(
        1,
        [{"role": "user", "content": "pending"}],
        BACKEND,
    )

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

    monkeypatch.setattr(rag, "MEMORY_BASE", tmp_path)
    monkeypatch.setattr(rag, "get_embeddings", embed)

    index = await rag.build_index(1)

    assert index["chunks"] == [
        {
            "text": "用户: hello\n角色: hi",
            "start_message": 1,
            "end_message": 2,
        }
    ]
    assert embedded == ["用户: hello\n角色: hi"]
    assert index["indexed_messages"] == 2


@pytest.mark.asyncio
async def test_rag_invalidate_after_removes_only_known_later_ranges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rag, "MEMORY_BASE", tmp_path)
    legacy = "legacy chunk"
    through_two = {"text": "one-two", "start_message": 1, "end_message": 2}
    through_four = {"text": "three-four", "start_message": 3, "end_message": 4}
    unknown = {"text": "unknown", "start_message": 0, "end_message": 0}
    await rag.save_index(
        1,
        {
            "chunks": [legacy, through_two, through_four, unknown],
            "embeddings": [[1.0], [2.0], [3.0], [4.0]],
            "indexed_messages": 4,
        },
    )

    await rag.invalidate_after(1, 2)

    index = await rag.load_index(1)
    assert index["chunks"] == [legacy, through_two, unknown]
    assert index["embeddings"] == [[1.0], [2.0], [4.0]]
    assert index["indexed_messages"] == 2


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

    async def story(store_arg, session_id, records, complete):
        calls.append("story")
        assert [record.number for record in records] == list(range(1, 9))
        await store_arg.write_text(session_id, "story_state.md", "rebuilt story")
        return "rebuilt story"

    async def extract(session_id, messages, backend):
        calls.append("memory")
        assert len(messages) == 8
        await store.write_text(session_id, "memory.md", "rebuilt memory")

    async def build_rag(session_id, **kwargs):
        calls.append("rag")
        await store.write_text(
            session_id,
            "rag/index.json",
            '{"chunks": [], "embeddings": [], "indexed_messages": 8}',
        )
        return {}

    async def create_episodes(*args, **kwargs):
        calls.append("episode")
        return await real_create_episodes(*args, **kwargs)

    async def update_summary(*args, **kwargs):
        calls.append("summary")
        return await real_update_summary(*args, **kwargs)

    async def assets(session_id, messages, backend):
        calls.append("assets")
        assert len(messages) == 8
        await store.write_text(session_id, "assets.md", "rebuilt assets")
        return True

    monkeypatch.setattr(story_memory, "update_story_state", story)
    monkeypatch.setattr(memory, "extract_memory_and_characters", extract)
    monkeypatch.setattr(story_memory, "create_due_episodes", create_episodes)
    monkeypatch.setattr(
        story_memory,
        "update_summary_from_episodes",
        update_summary,
    )
    monkeypatch.setattr(rag, "build_index", build_rag)
    monkeypatch.setattr(memory, "update_assets", assets)

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

    async def story(*args, **kwargs):
        calls["story"] += 1
        return "story"

    async def extract(*args, **kwargs):
        calls["memory"] += 1

    async def build_rag(*args, **kwargs):
        calls["rag"] += 1
        if calls["rag"] == 1:
            raise RuntimeError("rag failed")
        return {}

    async def episodes(*args, **kwargs):
        calls["episode"] += 1
        return args[3]

    async def summary(*args, **kwargs):
        calls["summary"] += 1
        return args[2]

    async def assets(*args, **kwargs):
        calls["assets"] += 1
        return False

    monkeypatch.setattr(story_memory, "update_story_state", story)
    monkeypatch.setattr(memory, "extract_memory_and_characters", extract)
    monkeypatch.setattr(story_memory, "create_due_episodes", episodes)
    monkeypatch.setattr(story_memory, "update_summary_from_episodes", summary)
    monkeypatch.setattr(rag, "build_index", build_rag)
    monkeypatch.setattr(memory, "update_assets", assets)

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

    with pytest.raises(OSError, match="pending state failed"):
        await store.truncate_chat(1, remove_count=2)

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
    async def story(*args, **kwargs):
        return "story"

    async def noop(*args, **kwargs):
        return None

    async def no_assets(*args, **kwargs):
        return False

    monkeypatch.setattr(story_memory, "update_story_state", story)
    monkeypatch.setattr(memory, "extract_memory_and_characters", noop)
    monkeypatch.setattr(rag, "build_index", noop)
    monkeypatch.setattr(memory, "update_assets", no_assets)

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

    with pytest.raises(OSError, match="cleanup failed"):
        await store.truncate_chat(1, remove_count=2)

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

    with pytest.raises(OSError, match="cleanup completion state failed"):
        await store.truncate_chat(1, remove_count=2)

    assert await store.load_chat(1) == []
    assert (await store.load_state(1)).cleanup_required

    monkeypatch.setattr(md_store_module.os, "replace", real_replace)
    await MemoryPipeline(store).run(1, {})

    assert not (await store.load_state(1)).rebuild_required


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

    with pytest.raises(OSError, match="state marker failed"):
        await store.replace_final_pair(
            1,
            expected_user=records[-2],
            expected_assistant=records[-1],
            assistant_content="replacement assistant",
        )

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

    async def story(*args, **kwargs):
        calls.append("story")
        return "rebuilt story"

    async def extract(*args, **kwargs):
        calls.append("memory")
        assert await restarted.read_text(1, "memory.md") == "stale memory through four"
        assert await restarted.read_text(1, "story_state.md") == ""
        assert await restarted.read_text(1, "summary.md") == ""
        assert not (tmp_path / "1" / "episodes" / "episode-000002.md").exists()
        index = json.loads(await restarted.read_text(1, "rag/index.json"))
        assert [chunk["end_message"] for chunk in index["chunks"]] == [2]
        await restarted.write_text(1, "memory.md", "rebuilt memory")

    async def episodes(*args, **kwargs):
        calls.append("episode")
        assert args[3] == 2
        return 2

    async def summary(*args, **kwargs):
        calls.append("summary")
        return 2

    async def build_rag(*args, **kwargs):
        calls.append("rag")
        return {}

    async def assets(*args, **kwargs):
        calls.append("assets")
        return False

    monkeypatch.setattr(story_memory, "update_story_state", story)
    monkeypatch.setattr(memory, "extract_memory_and_characters", extract)
    monkeypatch.setattr(story_memory, "create_due_episodes", episodes)
    monkeypatch.setattr(story_memory, "update_summary_from_episodes", summary)
    monkeypatch.setattr(rag, "build_index", build_rag)
    monkeypatch.setattr(memory, "update_assets", assets)

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

    with pytest.raises(OSError, match="rag cleanup failed"):
        await store.replace_final_pair(
            1,
            expected_user=records[-2],
            expected_assistant=records[-1],
            assistant_content="replacement assistant",
        )

    pending = await store.load_state(1)
    assert pending.cleanup_required
    assert pending.rebuild_from_message == 2
    assert (tmp_path / "1" / "invalidation_intent.md").exists()

    monkeypatch.setattr(md_store_module.os, "replace", real_replace)

    async def story(*args, **kwargs):
        return "story"

    async def noop(*args, **kwargs):
        return None

    async def episodes(*args, **kwargs):
        return 2

    async def summary(*args, **kwargs):
        return 2

    async def build_rag(*args, **kwargs):
        index = json.loads(await store.read_text(1, "rag/index.json"))
        assert [chunk["end_message"] for chunk in index["chunks"]] == [2]
        assert all(chunk["text"] != "stale" for chunk in index["chunks"])
        return {}

    async def assets(*args, **kwargs):
        return False

    monkeypatch.setattr(story_memory, "update_story_state", story)
    monkeypatch.setattr(memory, "extract_memory_and_characters", noop)
    monkeypatch.setattr(story_memory, "create_due_episodes", episodes)
    monkeypatch.setattr(story_memory, "update_summary_from_episodes", summary)
    monkeypatch.setattr(rag, "build_index", build_rag)
    monkeypatch.setattr(memory, "update_assets", assets)

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

    with pytest.raises(OSError, match="episode two deletion failed"):
        await store.replace_final_pair(
            1,
            expected_user=records[-2],
            expected_assistant=records[-1],
            assistant_content="replacement assistant",
        )

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

    async def story(*args, **kwargs):
        return "story"

    async def noop(*args, **kwargs):
        return None

    async def summary(*args, **kwargs):
        return 4

    async def assets(*args, **kwargs):
        return False

    monkeypatch.setattr(story_memory, "update_story_state", story)
    monkeypatch.setattr(memory, "extract_memory_and_characters", noop)
    monkeypatch.setattr(story_memory, "update_summary_from_episodes", summary)
    monkeypatch.setattr(rag, "build_index", noop)
    monkeypatch.setattr(memory, "update_assets", assets)

    await MemoryPipeline(MarkdownMemoryStore(tmp_path)).run(1, BACKEND)

    assert await store.read_text(1, "episodes/episode-000001.md") == first_episode
    regenerated = await store.read_text(1, "episodes/episode-000002.md")
    assert "<!-- messages: 3-4 -->" in regenerated
    assert "replacement events" in regenerated
    assert episode_completions == 1
    assert not (await store.load_state(1)).rebuild_required
    assert not (tmp_path / "1" / "invalidation_intent.md").exists()


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

    async def update_story(*args, **kwargs):
        nonlocal calls
        calls += 1
        entered.set()
        if calls == 2:
            entered_twice.set()
        await release.wait()
        return "updated"

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

    async def update_story(store_arg, session_id, records, complete):
        entered_sessions.add(session_id)
        if entered_sessions == {1, 2}:
            both_entered.set()
        await release.wait()
        return "updated"

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
    await rag.build_index(1)
    records = await store.load_chat(1)
    await store.write_text(1, "chat.md", render_chat_records(records[:8]))

    await rag.invalidate_after(1, 8)
    invalidated = await rag.load_index(1)
    rebuilt = await rag.build_index(1)
    results = await rag.search(1, "u4", top_k=10)

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

    async def update_story(*args, **kwargs):
        nonlocal calls
        calls += 1
        entered.set()
        if calls == 2:
            entered_twice.set()
        await release.wait()
        return "updated"

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
