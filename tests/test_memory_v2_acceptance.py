from __future__ import annotations

import copy
import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from app import main
from app.config import settings
from app.database import get_db
from app.models.tables import Character, Session
from app.routers import feishu, sessions
from app.services import llm, memory, rag, story_memory
from app.services.chat_service import ChatService, DomainContext, TurnRequest
from app.services.context_builder import ContextBuilder, ContextSources
from app.services.md_store import MarkdownMemoryStore
from app.services.memory_pipeline import MemoryPipeline
from app.services.prompt_policy import REQUIRED_OPENING
from app.services.stage_receipts import (
    StageUpdateResult,
    text_artifact,
)
from app.services.token_utils import estimate_messages_tokens


pytestmark = pytest.mark.filterwarnings(
    "ignore:datetime.datetime.utcfromtimestamp.*:DeprecationWarning"
)


BACKEND = {
    "provider": "openai",
    "api_key": "test-key",
    "model": "test-model",
    "base_url": "https://example.invalid/v1",
    "params": {},
}


@pytest.mark.asyncio
async def test_two_hundred_messages_preserve_story_continuity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "story_state_interval_messages", 20)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 20)
    monkeypatch.setattr(settings, "episode_size_messages", 20)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 20)
    monkeypatch.setattr(settings, "assets_interval_messages", 1_000)
    store = MarkdownMemoryStore(tmp_path)

    completion_calls: list[dict[str, Any]] = []
    embedding_calls: list[dict[str, Any]] = []

    def generated_text(messages: list[dict[str, str]]) -> tuple[str, str]:
        system = messages[0]["content"]
        prompt = "\n".join(message["content"] for message in messages)
        has_key = "银钥匙" in prompt
        has_cave = "山洞" in prompt

        if "当前场景快照维护器" in system:
            if has_cave:
                return "story", (
                    "# Story State\n\n"
                    "## 时间与地点\n"
                    "- 夜晚，山洞地下河畔\n\n"
                    "## 在场角色\n"
                    "- 用户与角色共同在场\n\n"
                    "## 当前场景\n"
                    "- 彼此信任的两人正在山洞地下河畔追踪线索\n\n"
                    "## 最近变化\n"
                    "- 两人刚越过石门，并发现石门后的新鲜足迹"
                )
            return "story", (
                "# Story State\n\n"
                "## 时间与地点\n"
                "- 白天，前往下一处遗迹的路上\n\n"
                "## 在场角色\n"
                "- 用户与角色共同在场\n\n"
                "## 当前场景\n"
                "- 两人正在沿途调查遗迹线索\n\n"
                "## 最近变化\n"
                "- 双方新增了一批调查记录"
            )

        if "章节压缩器" in system:
            if has_cave:
                return "episode", (
                    "## 剧情摘要\n"
                    "- 两人抵达山洞地下河畔，越过石门并发现新鲜足迹。\n\n"
                    "## 状态变化\n"
                    "- 调查地点从沿途遗迹转移到山洞深处。\n\n"
                    "## 承诺与伏笔\n"
                    "- 银钥匙与洞内石壁纹路产生呼应。"
                )
            if has_key:
                return "episode", (
                    "## 剧情摘要\n"
                    "- 最初，用户与角色立下共同寻找银钥匙的约定。\n\n"
                    "## 状态变化\n"
                    "- 两人因这项约定成为共同调查的同伴。\n\n"
                    "## 承诺与伏笔\n"
                    "- 双方明确约定共同寻找银钥匙。"
                )
            return "episode", (
                "## 剧情摘要\n"
                "- 两人继续调查沿途遗迹并记录线索。\n\n"
                "## 状态变化\n"
                "- 调查记录增加，双方保持协同行动。\n\n"
                "## 承诺与伏笔\n"
                "- 双方决定继续按既定方向调查。"
            )

        if "整体剧情回顾" in system:
            if has_cave:
                return "summary", (
                    "故事从用户与角色立下共同寻找银钥匙的约定开始，"
                    "这项早期约定让两人成为彼此信任的同伴。"
                    "经过多处遗迹的连续调查，他们抵达山洞地下河畔，"
                    "刚越过石门并发现新鲜足迹；银钥匙仍是串联这些事件的关键。"
                )
            return "summary", (
                "故事从用户与角色立下共同寻找银钥匙的约定开始，"
                "两人由此成为彼此信任的同伴，并持续调查沿途遗迹线索。"
            )

        if "memory extraction system" in system:
            key_event = (
                "- 早期事件是用户与角色立下共同寻找银钥匙的约定。"
                if has_key
                else "- 两人持续调查沿途遗迹线索。"
            )
            cave_event = (
                "\n- 最近抵达山洞地下河畔并发现石门后的新鲜足迹。"
                if has_cave
                else ""
            )
            return "memory", (
                "## Relationships\n"
                "- 用户与角色因银钥匙约定成为彼此信任的同伴。\n\n"
                "## Key Events\n"
                f"{key_event}{cave_event}\n\n"
                "## Decisions\n"
                "- 双方决定共同寻找银钥匙并持续协同行动。\n\n"
                "## User Preferences\n"
                "- 用户希望故事保持前后连续。\n"
                "===CHARACTERS===\n"
                "NO_CHANGE"
            )

        raise AssertionError("unexpected deterministic completion prompt")

    async def fake_completion(**kwargs):
        assert kwargs["provider"] == BACKEND["provider"]
        assert kwargs["base_url"] == BACKEND["base_url"]
        stage, content = generated_text(kwargs["messages"])
        completion_calls.append(
            {
                "stage": stage,
                "provider": kwargs["provider"],
                "base_url": kwargs["base_url"],
            }
        )
        return {
            "content": content,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

    async def fake_embeddings(texts: list[str], **kwargs):
        embedding_calls.append(
            {"texts": tuple(texts), "base_url": kwargs.get("base_url")}
        )
        return [
            [float((sum(map(ord, text)) % 97) + 1), float(len(text) + 1), 1.0]
            for text in texts
        ]

    def reject_live_http(*args, **kwargs):
        raise AssertionError("live HTTP client must not be constructed")

    monkeypatch.setattr(llm, "chat_completion", fake_completion)
    monkeypatch.setattr(memory, "chat_completion", fake_completion)
    monkeypatch.setattr(rag, "get_embeddings", fake_embeddings)
    monkeypatch.setattr(httpx, "AsyncClient", reject_live_http)

    final_pairs = {
        95: (
            "两人抵达山洞入口，准备沿银钥匙线索深入",
            "彼此信任的两人进入山洞",
        ),
        96: (
            "我们在地下河边短暂休整",
            "角色确认当前场景是山洞地下河畔",
        ),
        97: (
            "石壁纹路似乎能与银钥匙呼应",
            "两人用银钥匙辨认出石门方向",
        ),
        98: (
            "我们越过石门继续追踪",
            "两人发现石门后的新鲜足迹",
        ),
        99: (
            "继续沿足迹前进并记住刚才的发现",
            "最近事件是越过石门并发现新鲜足迹",
        ),
    }
    pipeline = MemoryPipeline(store)

    for index in range(100):
        if index == 0:
            user = "第一章里，我们约定共同寻找银钥匙"
            assistant = "角色接受约定，两人因而成为共同调查的同伴"
        elif index in final_pairs:
            user, assistant = final_pairs[index]
        else:
            user = f"两人调查第 {index} 处沿途遗迹"
            assistant = f"角色记录线索 {index}，双方继续协同行动"
        await store.append_pair(
            1,
            user,
            assistant,
            char_name="角色",
            user_name="用户",
        )
        if (index + 1) % 10 == 0:
            await pipeline.run(1, BACKEND)

    records = await store.load_chat(1)
    episode_directory = store.session_dir(1) / "episodes"
    episode_paths = sorted(episode_directory.glob("episode-*.md"))
    assert [path.name for path in episode_paths] == [
        f"episode-{number:06d}.md" for number in range(1, 11)
    ]
    episode_texts = [
        await store.read_text(1, f"episodes/{path.name}")
        for path in episode_paths
    ]
    for number, text in enumerate(episode_texts, start=1):
        start = (number - 1) * 20 + 1
        end = number * 20
        assert f"<!-- messages: {start}-{end} -->" in text

    state = await store.load_state(1)
    assert state.last_story_state_message == 200
    assert state.last_memory_message == 200
    assert state.last_character_message == 200
    assert state.last_episode_message == 200
    assert state.last_summary_message == 200

    memory_document = await store.read_text(1, "memory.md")
    for heading in (
        "## Relationships",
        "## Key Events",
        "## Decisions",
        "## User Preferences",
    ):
        assert heading in memory_document
    assert "彼此信任的同伴" in memory_document

    summary = await store.read_text(1, "summary.md")
    assert summary.startswith("故事从用户与角色立下共同寻找银钥匙的约定开始")
    assert "山洞地下河畔" in summary
    assert "完成" not in summary
    assert all(
        marker not in summary.casefold()
        for marker in ("[open]", "[closed]", "[done]")
    )

    completion_count = len(completion_calls)
    embedding_count = len(embedding_calls)
    immutable_episodes = dict(zip((path.name for path in episode_paths), episode_texts))
    restarted_store = MarkdownMemoryStore(tmp_path)
    await MemoryPipeline(restarted_store).run(1, BACKEND)
    assert await restarted_store.load_state(1) == state
    assert len(completion_calls) == completion_count
    assert len(embedding_calls) == embedding_count
    assert {
        path.name: await restarted_store.read_text(1, f"episodes/{path.name}")
        for path in sorted(
            (restarted_store.session_dir(1) / "episodes").glob("episode-*.md")
        )
    } == immutable_episodes

    result = ContextBuilder(
        total_budget=settings.total_token_budget,
        min_recent_messages=settings.min_recent_messages,
        reply_token_reserve=settings.reply_token_reserve,
    ).build(
        ContextSources(
            character={"name": "角色"},
            story_state=await restarted_store.read_text(1, "story_state.md"),
            memory=memory_document,
            summary=summary,
            episodes=episode_texts,
            recent=[
                {"role": record.role, "content": record.content}
                for record in records[-10:]
            ],
            user_message="我们继续寻找那件东西",
        )
    )

    text = "\n".join(message["content"] for message in result.messages)
    assert REQUIRED_OPENING in text
    assert "山洞地下河畔" in text
    assert "彼此信任的同伴" in text
    assert "发现石门后的新鲜足迹" in text
    assert "银钥匙" in text
    assert episode_texts[0] in text
    assert estimate_messages_tokens(result.messages) == result.total_tokens
    assert (
        result.total_tokens + settings.reply_token_reserve
        <= settings.total_token_budget
    )
    assert [call["stage"] for call in completion_calls].count("story") == 10
    assert [call["stage"] for call in completion_calls].count("memory") == 10
    assert [call["stage"] for call in completion_calls].count("episode") == 10
    assert [call["stage"] for call in completion_calls].count("summary") == 10
    assert len(embedding_calls) == 10


@pytest.mark.asyncio
async def test_pipeline_restart_resumes_after_completed_story_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "story_state_interval_messages", 2)
    monkeypatch.setattr(settings, "memory_extract_interval_messages", 2)
    monkeypatch.setattr(settings, "episode_size_messages", 100)
    monkeypatch.setattr(settings, "rag_index_interval_messages", 100)
    monkeypatch.setattr(settings, "assets_interval_messages", 100)
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "在雨夜进入古堡",
        "城门在身后合拢",
        char_name="角色",
        user_name="用户",
    )
    story_calls = 0
    memory_calls = 0

    async def update_story(
        store_arg,
        session_id,
        records,
        complete,
        *,
        source,
    ):
        nonlocal story_calls
        story_calls += 1
        text = "## 当前场景\n- 古��门厅"
        await store_arg.write_text(session_id, "story_state.md", text)
        return StageUpdateResult(
            stage="story",
            completed=True,
            source=source,
            checkpoint=source.count,
            artifacts=(text_artifact("story_state.md", text),),
        )

    async def extract_memory(
        session_id,
        messages,
        backend,
        *,
        store,
        source,
    ):
        nonlocal memory_calls
        memory_calls += 1
        if memory_calls == 1:
            raise RuntimeError("memory stage interrupted")
        text = "## Facts\n- 用户已进入古堡"
        await store.write_text(session_id, "memory.md", text)
        return StageUpdateResult(
            stage="memory",
            completed=True,
            source=source,
            checkpoint=source.count,
            artifacts=(text_artifact("memory.md", text),),
        )

    monkeypatch.setattr(story_memory, "update_story_state", update_story)
    monkeypatch.setattr(memory, "extract_memory_and_characters", extract_memory)

    with pytest.raises(RuntimeError, match="memory stage interrupted"):
        await MemoryPipeline(store).run(1, BACKEND)

    interrupted = await store.load_state(1)
    assert interrupted.last_story_state_message == 2
    assert interrupted.last_memory_message == 0
    assert interrupted.last_error == "RuntimeError: memory stage interrupted"

    restarted_store = MarkdownMemoryStore(tmp_path)
    await MemoryPipeline(restarted_store).run(1, BACKEND)

    recovered = await restarted_store.load_state(1)
    assert story_calls == 1
    assert memory_calls == 2
    assert recovered.last_story_state_message == 2
    assert recovered.last_memory_message == 2
    assert recovered.last_character_message == 2
    assert recovered.last_error == ""


class _ScalarResult:
    def __init__(self, value: Any) -> None:
        self.value = value

    def scalar_one_or_none(self):
        return self.value


class _AcceptanceDB:
    def __init__(
        self,
        web_session: Session,
        feishu_session: Session,
        character: Character,
    ) -> None:
        self.sessions = {
            web_session.id: web_session,
            feishu_session.id: feishu_session,
        }
        self.feishu_session = feishu_session
        self.character = character

    async def get(self, model, object_id):
        if model is Session:
            return self.sessions.get(object_id)
        if model is Character and object_id == self.character.id:
            return self.character
        return None

    async def execute(self, statement):
        return _ScalarResult(self.feishu_session)


class _MemoryManager:
    def __init__(self) -> None:
        self.submitted: list[int] = []

    async def submit(self, session_id: int) -> None:
        self.submitted.append(session_id)


def _session(session_id: int, *, chat_id: str | None = None) -> Session:
    return Session(
        id=session_id,
        character_id=7,
        worldbook_ids="[]",
        feishu_chat_id=chat_id,
        user_name="用户",
        user_persona="",
        messages='[{"role":"assistant","content":"STALE_DB"}]',
        summary="",
        status="active",
        created_at=dt.datetime(2026, 7, 21),
    )


@pytest.mark.asyncio
async def test_web_and_http_feishu_write_identical_markdown_record_shapes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    manager = _MemoryManager()
    web_session = _session(1)
    feishu_session = _session(2, chat_id="chat-2")
    web_legacy_messages = web_session.messages
    feishu_legacy_messages = feishu_session.messages
    character = Character(
        id=7,
        name="灵汐",
        description="",
        personality="",
        scenario="",
        first_message="",
        example_dialogues="",
        system_prompt="",
    )
    db = _AcceptanceDB(web_session, feishu_session, character)

    async def override_db():
        yield db

    async def resolve_backend(backend_id, db_arg):
        return BACKEND

    async def load_domain_context(*args, **kwargs):
        return DomainContext()

    async def complete(**kwargs):
        return {
            "content": "正文",
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            },
        }

    async def record_usage(**kwargs):
        return None

    cards: list[tuple[str, dict[str, Any]]] = []

    async def send_card(chat_id, card):
        cards.append((chat_id, card))

    async def reject_unexpected_text_send(chat_id, text):
        raise AssertionError(
            f"unexpected Feishu text/network path: {chat_id}: {text}"
        )

    monkeypatch.setattr(memory, "md_store", store)
    monkeypatch.setattr(main, "memory_task_manager", manager)
    monkeypatch.setattr(sessions, "_resolve_backend", resolve_backend)
    monkeypatch.setattr(sessions, "_prepare_domain_context", load_domain_context)
    monkeypatch.setattr(sessions, "chat_completion", complete)
    monkeypatch.setattr(feishu, "chat_completion", complete)
    monkeypatch.setattr(sessions, "record_usage", record_usage)
    monkeypatch.setattr(feishu, "record_usage", record_usage)
    monkeypatch.setattr(feishu.feishu_client, "send_interactive_card", send_card)
    monkeypatch.setattr(
        feishu.feishu_client,
        "send_text_message",
        reject_unexpected_text_send,
    )
    main.app.dependency_overrides[get_db] = override_db

    transport = httpx.ASGITransport(app=main.app)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            web_response = await client.post(
                "/api/sessions/1/message",
                json={"content": "继续。"},
            )
            feishu_response = await client.post(
                "/api/feishu/webhook",
                json={
                    "header": {"event_type": "im.message.receive_v1"},
                    "event": {
                        "message": {
                            "chat_id": "chat-2",
                            "message_id": "message-2",
                            "message_type": "text",
                            "content": json.dumps(
                                {"text": "继续。"},
                                ensure_ascii=False,
                            ),
                        },
                        "sender": {
                            "sender_id": {"open_id": "sender-2"}
                        },
                    },
                },
            )
    finally:
        main.app.dependency_overrides.pop(get_db, None)

    assert web_response.status_code == 200
    assert feishu_response.status_code == 200
    assert cards
    web_records = await store.load_chat(1)
    feishu_records = await store.load_chat(2)
    def record_shape(records):
        return [
            (
                record.number,
                record.role,
                record.content,
                record.name,
                record.msg_type,
            )
            for record in records
        ]

    assert record_shape(web_records) == record_shape(feishu_records)
    assert manager.submitted == [1, 2]
    assert web_session.messages == web_legacy_messages
    assert feishu_session.messages == feishu_legacy_messages


@pytest.mark.asyncio
async def test_http_feishu_reset_leaves_v1_messages_as_read_only_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    active = _session(1, chat_id="chat-1")
    legacy_messages = active.messages
    await store.append_pair(
        1,
        "旧问题",
        "旧回答",
        char_name="灵汐",
        user_name="用户",
    )

    class CommandDB:
        async def get(self, model, object_id):
            if model is Character and object_id == active.character_id:
                return SimpleNamespace(name="灵汐")
            return None

        async def execute(self, statement):
            return _ScalarResult(active)

        async def commit(self):
            return None

    sent: list[str] = []

    async def send_text(chat_id: str, text: str) -> None:
        sent.append(text)

    monkeypatch.setattr(memory, "md_store", store)
    monkeypatch.setattr(feishu.feishu_client, "send_text_message", send_text)

    await feishu._handle_command(
        "/reset",
        "chat-1",
        "sender-1",
        CommandDB(),
    )

    assert await store.load_chat(1) == []
    assert active.messages == legacy_messages
    assert sent == ["Session reset. Chat history cleared."]


@pytest.mark.asyncio
async def test_http_feishu_first_message_writes_only_markdown_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    character = Character(
        id=7,
        name="灵汐",
        description="",
        personality="",
        scenario="",
        first_message="你好，{{user}}。",
        example_dialogues="",
        system_prompt="",
    )

    class CardDB:
        def __init__(self) -> None:
            self.session: Session | None = None

        async def get(self, model, object_id):
            if model is Character and object_id == character.id:
                return character
            return None

        def add(self, value: Session) -> None:
            self.session = value

        async def flush(self) -> None:
            assert self.session is not None
            self.session.id = 12

        async def commit(self) -> None:
            if self.session is not None and self.session.id is None:
                self.session.id = 12

        async def refresh(self, value: Session) -> None:
            return None

        async def rollback(self) -> None:
            return None

    db = CardDB()

    async def send_card(chat_id: str, card: dict[str, Any]) -> None:
        return None

    monkeypatch.setattr(memory, "md_store", store)
    monkeypatch.setattr(feishu.feishu_client, "send_interactive_card", send_card)

    await feishu._handle_card_action(
        {
            "action": {"option": "7", "value": {}},
            "open_chat_id": "chat-1",
        },
        db,
    )

    assert db.session is not None
    assert db.session.messages is None
    records = await store.load_chat(12)
    assert [(record.role, record.content, record.name) for record in records] == [
        ("assistant", "你好，User。", "灵汐")
    ]


@pytest.mark.asyncio
async def test_stream_failure_leaves_authoritative_chat_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "stream_guard_chars", 1)
    store = MarkdownMemoryStore(tmp_path)
    await store.append_pair(
        1,
        "原问题",
        "原回答",
        char_name="角色",
        user_name="用户",
    )
    chat_path = tmp_path / "1" / "chat.md"
    before = chat_path.read_bytes()

    async def fail_after_delta(messages):
        yield "正文已经开始"
        raise RuntimeError("provider disconnected")

    request = TurnRequest(
        session_id=1,
        content="新问题",
        character={"name": "角色"},
        user_name="用户",
    )
    events = [
        event
        async for event in ChatService(
            store,
            None,
            stream_completion=fail_after_delta,
        ).stream(request)
    ]

    assert events[-1].error
    assert chat_path.read_bytes() == before


@pytest.mark.asyncio
async def test_v1_session_lazily_creates_state_without_rewriting_memory(
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
        "旧问题",
        "旧回答",
        char_name="角色",
        user_name="用户",
    )
    legacy_memory = "# Session Memory\n\n- 必须保留的旧记忆\n"
    await store.write_text(1, "memory.md", legacy_memory)
    await store.write_text(1, ".last_extract_count", "2")
    state_path = tmp_path / "1" / "memory_state.md"
    assert not state_path.exists()

    await MemoryPipeline(store).run(1, BACKEND)

    assert state_path.exists()
    assert await store.read_text(1, "memory.md") == legacy_memory
    state = await store.load_state(1)
    assert state.last_memory_message == 2
    assert state.last_character_message == 2


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("/list", "(37条消息)"),
        ("/info", "消息数: 37"),
        ("/resume", "(37条消息)"),
    ],
)
def test_ws_session_views_use_lite_api_msg_count(
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    expected: str,
) -> None:
    from app.services import feishu_ws_worker as worker

    lite_session = {
        "id": 1,
        "character_id": 7,
        "feishu_chat_id": "chat-1",
        "status": "active",
        "user_name": "用户",
        "msg_count": 37,
        "last_summary": "最近消息",
        "messages": ["legacy field must not be used for counts"],
    }
    original = copy.deepcopy(lite_session)
    calls: list[str] = []
    sent: list[str] = []

    def api_get(path: str):
        calls.append(path)
        if path == "/api/characters":
            return [{"id": 7, "name": "灵汐"}]
        return [lite_session]

    monkeypatch.setattr(worker, "api_get", api_get)
    monkeypatch.setattr(worker, "send_text", lambda chat_id, text: sent.append(text))

    worker._handle_command(command, "chat-1", "sender-1")

    assert "/api/sessions?lite=true" in calls
    assert any(expected in text for text in sent)
    assert lite_session == original


@pytest.mark.parametrize(
    "command",
    ["/retry", "/undo", "/memory", "/summary", "/assets"],
)
def test_ws_stateful_commands_forward_to_session_api(
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    from app.services import feishu_ws_worker as worker

    lite_session = {
        "id": 9,
        "character_id": 7,
        "feishu_chat_id": "chat-1",
        "status": "active",
        "msg_count": 4,
    }
    original = copy.deepcopy(lite_session)
    posts: list[tuple[str, dict[str, Any]]] = []

    monkeypatch.setattr(worker, "api_get", lambda path: [lite_session])

    def api_post(path: str, data: dict[str, Any]):
        posts.append((path, data))
        return {"messages": [{"role": "assistant", "content": "已执行"}]}

    monkeypatch.setattr(worker, "api_post", api_post)
    monkeypatch.setattr(worker, "send_text", lambda *args: None)

    worker._handle_command(command, "chat-1", "sender-1")

    assert posts == [
        ("/api/sessions/9/message", {"content": command})
    ]
    assert lite_session == original


def test_ws_reset_uses_session_api_without_mutating_lite_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import feishu_ws_worker as worker

    lite_session = {
        "id": 9,
        "character_id": 7,
        "feishu_chat_id": "chat-1",
        "status": "active",
        "msg_count": 4,
    }
    original = copy.deepcopy(lite_session)
    deleted: list[str] = []
    posted: list[tuple[str, dict[str, Any]]] = []

    def api_get(path: str):
        if path == "/api/characters":
            return [{"id": 7, "name": "灵汐"}]
        return [lite_session]

    def api_post(path: str, data: dict[str, Any]):
        posted.append((path, data))
        return {"messages": []}

    monkeypatch.setattr(worker, "api_get", api_get)
    monkeypatch.setattr(worker, "api_delete", deleted.append)
    monkeypatch.setattr(worker, "api_post", api_post)
    monkeypatch.setattr(worker, "send_text", lambda *args: None)

    worker._handle_command("/reset", "chat-1", "sender-1")

    assert deleted == ["/api/sessions/9"]
    assert posted == [
        (
            "/api/sessions",
            {
                "character_id": 7,
                "feishu_chat_id": "chat-1",
                "worldbook_ids": [],
            },
        )
    ]
    assert lite_session == original


def test_ws_resume_by_id_uses_session_status_api_without_mutating_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import feishu_ws_worker as worker

    current = {
        "id": 9,
        "character_id": 7,
        "feishu_chat_id": "chat-1",
        "status": "active",
        "msg_count": 4,
    }
    target = {
        "id": 10,
        "character_id": 8,
        "feishu_chat_id": "chat-2",
        "status": "archived",
        "msg_count": 12,
    }
    sessions_payload = [current, target]
    original = copy.deepcopy(sessions_payload)
    patches: list[tuple[str, dict[str, Any]]] = []

    def api_get(path: str):
        if path == "/api/characters":
            return [
                {"id": 7, "name": "灵汐"},
                {"id": 8, "name": "星遥"},
            ]
        return sessions_payload

    def api_patch(path: str, data: dict[str, Any]):
        patches.append((path, data))
        return {}

    monkeypatch.setattr(worker, "api_get", api_get)
    monkeypatch.setattr(worker, "api_patch", api_patch)
    monkeypatch.setattr(worker, "send_text", lambda *args: None)

    worker._handle_command("/resume 10", "chat-1", "sender-1")

    assert patches == [
        ("/api/sessions/9/status", {"status": "archived"}),
        (
            "/api/sessions/10/status",
            {"status": "active", "feishu_chat_id": "chat-1"},
        ),
    ]
    assert sessions_payload == original


def test_ws_chars_reads_markdown_character_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import feishu_ws_worker as worker

    calls: list[int] = []

    async def list_names(session_id: int):
        calls.append(session_id)
        return ["灵汐"]

    monkeypatch.setattr(
        worker,
        "api_get",
        lambda path: [
            {
                "id": 9,
                "character_id": 7,
                "feishu_chat_id": "chat-1",
                "status": "active",
                "msg_count": 4,
            }
        ],
    )
    monkeypatch.setattr(memory, "list_character_names", list_names)
    monkeypatch.setattr(worker, "send_text", lambda *args: None)

    worker._handle_command("/chars list", "chat-1", "sender-1")

    assert calls == [9]


def test_readme_documents_md_memory_v2_contract() -> None:
    readme = (Path(__file__).parents[1] / "README.md").read_text(
        encoding="utf-8"
    )

    assert "## MD Memory V2" in readme
    assert "chat.md" in readme and "authoritative" in readme
    assert "memory_state.md" in readme
    assert "summary.md" in readme and "overall plot" in readme
    normalized = " ".join(readme.split())
    assert (
        "overall plot summary does not decide whether a storyline is complete; "
        "no completion decision is required or tracked"
        in normalized
    )
    for variable in (
        "MEMORY_V2_ENABLED",
        "STORY_STATE_INTERVAL_MESSAGES",
        "MEMORY_EXTRACT_INTERVAL_MESSAGES",
        "EPISODE_SIZE_MESSAGES",
        "RAG_INDEX_INTERVAL_MESSAGES",
        "ASSETS_INTERVAL_MESSAGES",
        "TOTAL_TOKEN_BUDGET",
        "REPLY_TOKEN_RESERVE",
        "MIN_RECENT_MESSAGES",
        "STREAM_GUARD_CHARS",
    ):
        assert variable in readme
    assert REQUIRED_OPENING in readme
    assert "adult narrative priming" in readme
    assert "no-refusal" in readme
