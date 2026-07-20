from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.config import settings
from app.services import memory, rag, story_memory
from app.services.md_store import MarkdownMemoryStore, MemoryState
from app.services.memory_pipeline import MemoryPipeline
from app.services.memory_tasks import MemoryTaskManager


BACKEND = {
    "provider": "openai",
    "api_key": "test-key",
    "model": "test-model",
    "base_url": "https://example.invalid/v1",
    "params": {},
}


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
    pipeline = MemoryPipeline(store)
    tasks = [
        asyncio.create_task(pipeline.run(session_id, BACKEND))
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
