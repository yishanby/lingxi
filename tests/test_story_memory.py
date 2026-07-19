from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.services.md_store import ChatRecord, MarkdownMemoryStore
from app.services.story_memory import (
    create_due_episodes,
    render_records,
    update_story_state,
    update_summary_from_episodes,
)
from app.services.token_utils import estimate_tokens


def records(start: int, end: int | None = None) -> list[ChatRecord]:
    if end is None:
        start, end = 1, start
    return [
        ChatRecord(
            number,
            "user" if number % 2 else "assistant",
            f"message {number}",
            "2026-01-01 00:00",
            "name",
        )
        for number in range(start, end + 1)
    ]


def story_state(*open_threads: str) -> str:
    threads = "\n".join(f"- [open] {thread}" for thread in open_threads) or "- （无）"
    return (
        "# Story State\n\n"
        "## 时间与地点\n- 当前地点：门厅\n\n"
        "## 在场角色\n- 旅人\n\n"
        "## 当前场景\n- 正在交谈\n\n"
        f"## 未完成剧情线\n{threads}\n\n"
        "## 最近变化\n- 开始行动"
    )


def episode_body(label: str = "完成章节") -> str:
    return (
        f"## 剧情摘要\n{label}\n\n"
        "## 状态变化\n- 无\n\n"
        "## 承诺与伏笔\n- [open] 约定"
    )


def episode_document(number: int, start: int, end: int, label: str = "完成章节") -> str:
    return (
        f"# Episode {number:06d}\n\n"
        f"<!-- messages: {start}-{end} -->\n\n"
        f"{episode_body(label)}\n"
    )


def test_render_records_preserves_numbers_roles_multiline_and_unicode() -> None:
    source = [
        ChatRecord(7, "user", "第一行\n第二行 🙂", "2026-01-01 00:00", "用户"),
        ChatRecord(8, "assistant", "回复", "2026-01-01 00:01", "角色"),
    ]

    assert render_records(source) == "[7] user: 第一行\n第二行 🙂\n[8] assistant: 回复"


@pytest.mark.asyncio
async def test_story_state_preserves_open_threads_and_writes_complete_state(
    tmp_path: Path,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    existing = "## 未完成剧情线\n- [open] 找到钥匙"
    await store.write_text(1, "story_state.md", existing)
    expected = story_state("找到钥匙")
    calls: list[list[dict[str, str]]] = []

    async def fake_complete(messages: list[dict[str, str]]) -> str:
        calls.append(messages)
        assert existing in messages[-1]["content"]
        assert "[1] user: message 1" in messages[-1]["content"]
        return f"\n{expected}\n"

    result = await update_story_state(store, 1, records(2), fake_complete)

    assert result == expected
    assert "[open] 找到钥匙" in result
    assert await store.read_text(1, "story_state.md") == expected + "\n"
    assert calls[0][0]["role"] == "system"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid",
    [
        "",
        "## 未完成剧情线\n- [open] 找到钥匙",
        story_state().replace("## 当前场景", "## 其他场景"),
        story_state().replace("## 未完成剧情线", "## Open Threads"),
    ],
    ids=["empty", "incomplete", "missing-scene", "missing-open-section"],
)
async def test_invalid_story_state_output_does_not_replace_valid_file(
    tmp_path: Path,
    invalid: str,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    old = story_state("旧伏笔") + "\n"
    await store.write_text(1, "story_state.md", old)

    async def fake_complete(_messages: list[dict[str, str]]) -> str:
        return invalid

    with pytest.raises(ValueError, match="story state"):
        await update_story_state(store, 1, records(2), fake_complete)

    assert await store.read_text(1, "story_state.md") == old


@pytest.mark.asyncio
async def test_story_state_cannot_silently_drop_an_open_thread(tmp_path: Path) -> None:
    store = MarkdownMemoryStore(tmp_path)
    old = story_state("找到钥匙") + "\n"
    await store.write_text(1, "story_state.md", old)

    async def fake_complete(_messages: list[dict[str, str]]) -> str:
        return story_state()

    with pytest.raises(ValueError, match="open thread"):
        await update_story_state(store, 1, records(2), fake_complete)

    assert await store.read_text(1, "story_state.md") == old


@pytest.mark.asyncio
async def test_story_state_allows_explicitly_closed_thread(tmp_path: Path) -> None:
    store = MarkdownMemoryStore(tmp_path)
    await store.write_text(1, "story_state.md", story_state("找到钥匙") + "\n")
    dialogue = [
        ChatRecord(
            3,
            "user",
            "[closed] 找到钥匙",
            "2026-01-01 00:02",
            "name",
        )
    ]

    async def fake_complete(_messages: list[dict[str, str]]) -> str:
        return story_state()

    assert await update_story_state(store, 1, dialogue, fake_complete) == story_state()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("thread", "evidence"),
    [
        ("找到钥匙", "旅人终于在地毯下找到了钥匙。"),
        (
            "Find the silver key",
            "At last: the silver key was FOUND beneath the rug!",
        ),
    ],
    ids=["chinese-result", "english-punctuation-and-inflection"],
)
async def test_story_state_allows_natural_explicit_completion_evidence(
    tmp_path: Path,
    thread: str,
    evidence: str,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    await store.write_text(1, "story_state.md", story_state(thread) + "\n")
    dialogue = [
        ChatRecord(3, "assistant", evidence, "2026-01-01 00:02", "name")
    ]

    async def fake_complete(_messages: list[dict[str, str]]) -> str:
        return story_state()

    assert await update_story_state(store, 1, dialogue, fake_complete) == story_state()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("thread", "evidence", "closes"),
    [
        ("找到银色钥匙", "旅人终于在地毯下找到了银色钥匙。", True),
        ("找到银色钥匙", "旅人找到了藏在地毯下的银色钥匙。", True),
        ("找到银色钥匙", "据传言，旅人终于找到了银色钥匙。", False),
        ("找到银色钥匙", "有人声称旅人找到了银色钥匙。", False),
        ("找到银色钥匙", "据称旅人找到了银色钥匙。", False),
        ("找到银色钥匙", "旅人找到了晚餐，并讨论银色钥匙。", False),
        ("找到银色钥匙", "旅人找到了银色钥匙的画像。", False),
        ("找到银色钥匙", "旅人找到了银色钥匙复制品。", False),
        ("找到银色钥匙", "已经确认，没有在地下室找到银色钥匙。", False),
        ("找到银色钥匙", "已经发誓，明天会找到银色钥匙。", False),
        (
            "Find the silver key",
            "At last, the traveler found the silver key under the rug.",
            True,
        ),
        (
            "Find the silver key",
            "The traveler found, beneath the rug, the silver key.",
            True,
        ),
        (
            "Find the silver key",
            "Rumor has it: the traveler finally found the silver key.",
            False,
        ),
        (
            "Find the silver key",
            "Apparently, the traveler found the silver key.",
            False,
        ),
        (
            "Find the silver key",
            "A witness claimed the traveler found the silver key.",
            False,
        ),
        (
            "Find the silver key",
            "The traveler found dinner and discussed the silver key.",
            False,
        ),
        (
            "Find the silver key",
            "The traveler found a picture of the silver key.",
            False,
        ),
        (
            "Find the silver key",
            "The traveler found the silver key replica.",
            False,
        ),
        (
            "Find the silver key",
            "The traveler found dinner; then discussed the silver key.",
            False,
        ),
        (
            "Find the silver key",
            "It is confirmed: the traveler did not find the silver key.",
            False,
        ),
        (
            "Find the silver key",
            "The traveler swore: tomorrow, they will find the silver key.",
            False,
        ),
    ],
    ids=[
        "zh-direct-result",
        "zh-locative-target",
        "zh-hearsay",
        "zh-source-claim",
        "zh-reported-claim",
        "zh-unrelated-object",
        "zh-picture",
        "zh-replica",
        "zh-negated-result",
        "zh-future-intent",
        "en-direct-result",
        "en-comma-locative-target",
        "en-hearsay",
        "en-apparently",
        "en-witness-claim",
        "en-unrelated-object",
        "en-picture",
        "en-replica",
        "en-clause-boundary",
        "en-negated-result",
        "en-future-intent",
    ],
)
async def test_story_state_natural_completion_requires_direct_target_evidence(
    tmp_path: Path,
    thread: str,
    evidence: str,
    closes: bool,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    old = story_state(thread) + "\n"
    await store.write_text(1, "story_state.md", old)
    dialogue = [
        ChatRecord(3, "assistant", evidence, "2026-01-01 00:02", "name")
    ]

    async def fake_complete(_messages: list[dict[str, str]]) -> str:
        return story_state()

    if closes:
        assert await update_story_state(store, 1, dialogue, fake_complete) == story_state()
        assert await store.read_text(1, "story_state.md") == story_state() + "\n"
    else:
        with pytest.raises(ValueError, match="open thread"):
            await update_story_state(store, 1, dialogue, fake_complete)
        assert await store.read_text(1, "story_state.md") == old


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("thread", "evidence"),
    [
        ("找到钥匙", "旅人完成了晚餐，但仍未找到钥匙。"),
        ("找到钥匙", "旅人计划明天找到钥匙。"),
        ("找到钥匙", "下一步是找到钥匙。"),
        ("找到钥匙", "旅人谈起钥匙，也完成了另一项无关任务。"),
        ("找到钥匙", "旅人可能找到了钥匙。"),
        ("找到钥匙", "钥匙找到了吗？"),
        ("钥匙的下落", "旅人完成了晚餐，同时讨论钥匙的下落。"),
    ],
    ids=[
        "explicitly-unresolved",
        "future-intent",
        "next-step-intent",
        "unrelated-completion",
        "uncertain-result",
        "question",
        "generic-unrelated-completion",
    ],
)
async def test_story_state_natural_evidence_does_not_close_unresolved_thread(
    tmp_path: Path,
    thread: str,
    evidence: str,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    old = story_state(thread) + "\n"
    await store.write_text(1, "story_state.md", old)
    dialogue = [
        ChatRecord(3, "assistant", evidence, "2026-01-01 00:02", "name")
    ]

    async def fake_complete(_messages: list[dict[str, str]]) -> str:
        return story_state()

    with pytest.raises(ValueError, match="open thread"):
        await update_story_state(store, 1, dialogue, fake_complete)

    assert await store.read_text(1, "story_state.md") == old


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [RuntimeError("provider failed"), asyncio.CancelledError()],
    ids=["failure", "cancellation"],
)
async def test_story_state_completion_failure_preserves_old_file(
    tmp_path: Path,
    failure: BaseException,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    old = story_state("保留") + "\n"
    await store.write_text(1, "story_state.md", old)

    async def fake_complete(_messages: list[dict[str, str]]) -> str:
        raise failure

    with pytest.raises(type(failure)):
        await update_story_state(store, 1, records(2), fake_complete)

    assert await store.read_text(1, "story_state.md") == old


@pytest.mark.asyncio
async def test_episode_covers_exact_message_range(tmp_path: Path) -> None:
    store = MarkdownMemoryStore(tmp_path)
    prompts: list[str] = []

    async def fake_complete(messages: list[dict[str, str]]) -> str:
        prompts.append(messages[-1]["content"])
        return episode_body()

    checkpoint = await create_due_episodes(
        store,
        1,
        records(25),
        0,
        fake_complete,
        episode_size=20,
    )

    assert checkpoint == 20
    assert [path.name for path in (tmp_path / "1" / "episodes").iterdir()] == [
        "episode-000001.md"
    ]
    text = await store.read_text(1, "episodes/episode-000001.md")
    assert "<!-- messages: 1-20 -->" in text
    assert "[20] assistant: message 20" in prompts[0]
    assert "message 21" not in prompts[0]


@pytest.mark.asyncio
async def test_episode_nonzero_checkpoint_selects_numbers_not_list_indexes(
    tmp_path: Path,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    await store.write_text(
        1,
        "episodes/episode-000001.md",
        episode_document(1, 1, 20, "第一章"),
    )
    prompts: list[str] = []

    async def fake_complete(messages: list[dict[str, str]]) -> str:
        prompts.append(messages[-1]["content"])
        return episode_body("第二章")

    checkpoint = await create_due_episodes(
        store,
        1,
        records(21, 45),
        20,
        fake_complete,
        episode_size=20,
    )

    assert checkpoint == 40
    assert "[21] user: message 21" in prompts[0]
    assert "[40] assistant: message 40" in prompts[0]
    assert "message 20" not in prompts[0]
    assert "message 41" not in prompts[0]
    assert await store.read_text(1, "episodes/episode-000002.md") == episode_document(
        2, 21, 40, "第二章"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "existing",
    [
        {"episode-000001.md": "# Episode 000001\n\nmissing metadata\n"},
        {"episode-000001.md": episode_document(1, 2, 21, "conflict")},
        {
            "episode-000001.md": episode_document(1, 1, 20),
            "episode-000002.md": episode_document(1, 21, 40, "duplicate"),
        },
        {
            "episode-000001.md": episode_document(1, 1, 20),
            "episode-000003.md": episode_document(3, 41, 60, "gap"),
        },
        {
            "episode-000001.md": episode_document(1, 21, 40, "out of order"),
            "episode-000002.md": episode_document(2, 1, 20, "out of order"),
        },
    ],
    ids=["malformed", "conflicting-range", "duplicate", "gap", "out-of-order"],
)
async def test_episode_creation_validates_entire_existing_chain_before_completion(
    tmp_path: Path,
    existing: dict[str, str],
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    for filename, document in existing.items():
        await store.write_text(1, f"episodes/{filename}", document)
    calls = 0

    async def fake_complete(_messages: list[dict[str, str]]) -> str:
        nonlocal calls
        calls += 1
        return episode_body("must not be written")

    with pytest.raises(ValueError, match="existing episode"):
        await create_due_episodes(
            store,
            1,
            records(21, 40),
            20,
            fake_complete,
            episode_size=20,
        )

    assert calls == 0
    episode_dir = tmp_path / "1" / "episodes"
    assert {
        path.name: path.read_text(encoding="utf-8")
        for path in episode_dir.iterdir()
    } == existing


@pytest.mark.asyncio
async def test_existing_episode_is_idempotent_without_another_completion(
    tmp_path: Path,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    existing = episode_document(1, 1, 20, "不可变内容")
    await store.write_text(1, "episodes/episode-000001.md", existing)

    async def must_not_complete(_messages: list[dict[str, str]]) -> str:
        raise AssertionError("completion must not be called")

    checkpoint = await create_due_episodes(
        store,
        1,
        records(20),
        0,
        must_not_complete,
        episode_size=20,
    )

    assert checkpoint == 20
    assert await store.read_text(1, "episodes/episode-000001.md") == existing


@pytest.mark.asyncio
async def test_conflicting_existing_episode_is_never_overwritten(tmp_path: Path) -> None:
    store = MarkdownMemoryStore(tmp_path)
    conflict = episode_document(1, 2, 21, "其他范围")
    await store.write_text(1, "episodes/episode-000001.md", conflict)

    async def must_not_complete(_messages: list[dict[str, str]]) -> str:
        raise AssertionError("completion must not be called")

    with pytest.raises(ValueError, match="existing episode"):
        await create_due_episodes(
            store,
            1,
            records(21),
            0,
            must_not_complete,
            episode_size=20,
        )

    assert await store.read_text(1, "episodes/episode-000001.md") == conflict


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid",
    ["", "plain text", episode_body().replace("## 状态变化", "## 改变")],
    ids=["empty", "plain", "missing-heading"],
)
async def test_invalid_episode_output_does_not_create_file(
    tmp_path: Path,
    invalid: str,
) -> None:
    store = MarkdownMemoryStore(tmp_path)

    async def fake_complete(_messages: list[dict[str, str]]) -> str:
        return invalid

    with pytest.raises(ValueError, match="episode"):
        await create_due_episodes(
            store,
            1,
            records(20),
            0,
            fake_complete,
            episode_size=20,
        )

    assert not (tmp_path / "1" / "episodes" / "episode-000001.md").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "checkpoint", "episode_size"),
    [
        (records(1, 20), -1, 20),
        (records(1, 20), 1, 20),
        ([], 20, 20),
        (records(1, 20), 0, 0),
        (records(1, 20), 0, -1),
        (records(1, 10) + records(12, 21), 0, 20),
        (list(reversed(records(20))), 0, 20),
    ],
    ids=[
        "negative-checkpoint",
        "unaligned-checkpoint",
        "checkpoint-beyond-empty-history",
        "zero-size",
        "negative-size",
        "gap",
        "reordered",
    ],
)
async def test_episode_rejects_invalid_interval_or_record_sequence(
    tmp_path: Path,
    source: list[ChatRecord],
    checkpoint: int,
    episode_size: int,
) -> None:
    store = MarkdownMemoryStore(tmp_path)

    async def must_not_complete(_messages: list[dict[str, str]]) -> str:
        raise AssertionError("completion must not be called")

    with pytest.raises(ValueError):
        await create_due_episodes(
            store,
            1,
            source,
            checkpoint,
            must_not_complete,
            episode_size=episode_size,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [RuntimeError("provider failed"), asyncio.CancelledError()],
    ids=["failure", "cancellation"],
)
async def test_episode_completion_failure_does_not_create_file(
    tmp_path: Path,
    failure: BaseException,
) -> None:
    store = MarkdownMemoryStore(tmp_path)

    async def fake_complete(_messages: list[dict[str, str]]) -> str:
        raise failure

    with pytest.raises(type(failure)):
        await create_due_episodes(
            store,
            1,
            records(20),
            0,
            fake_complete,
            episode_size=20,
        )

    assert not (tmp_path / "1" / "episodes" / "episode-000001.md").exists()


@pytest.mark.asyncio
async def test_summary_uses_only_pending_episodes_and_returns_last_end(
    tmp_path: Path,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    await store.write_text(1, "episodes/episode-000001.md", episode_document(1, 1, 20, "第一章"))
    await store.write_text(1, "episodes/episode-000002.md", episode_document(2, 21, 40, "第二章"))
    await store.write_text(1, "episodes/episode-000003.md", episode_document(3, 41, 60, "第三章"))
    await store.write_text(1, "summary.md", "旧摘要\n")
    prompts: list[str] = []

    async def fake_complete(messages: list[dict[str, str]]) -> str:
        prompts.append(messages[-1]["content"])
        return "完整新摘要"

    checkpoint = await update_summary_from_episodes(
        store,
        1,
        20,
        fake_complete,
        max_tokens=100,
    )

    assert checkpoint == 60
    assert "旧摘要" in prompts[0]
    assert "第一章" not in prompts[0]
    assert "第二章" in prompts[0]
    assert "第三章" in prompts[0]
    assert await store.read_text(1, "summary.md") == "完整新摘要\n"


@pytest.mark.asyncio
async def test_summary_with_no_pending_episodes_does_not_call_or_write(
    tmp_path: Path,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    await store.write_text(1, "episodes/episode-000001.md", episode_document(1, 1, 20))
    old = "保持原样（无结尾换行）"
    await store.write_text(1, "summary.md", old)

    async def must_not_complete(_messages: list[dict[str, str]]) -> str:
        raise AssertionError("completion must not be called")

    assert await update_summary_from_episodes(
        store,
        1,
        20,
        must_not_complete,
        max_tokens=100,
    ) == 20
    assert await store.read_text(1, "summary.md") == old


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "documents",
    [
        [("episode-000001.md", "# Episode 000001\n\nno metadata\n")],
        [("episode-000001.md", episode_document(1, 1, 20).replace("1-20", "20-1"))],
        [
            ("episode-000001.md", episode_document(1, 1, 20)),
            ("episode-000002.md", episode_document(2, 20, 39)),
        ],
        [
            ("episode-000001.md", episode_document(1, 1, 20)),
            ("episode-000002.md", episode_document(2, 22, 41)),
        ],
        [
            ("episode-000001.md", episode_document(1, 21, 40)),
            ("episode-000002.md", episode_document(2, 1, 20)),
        ],
    ],
    ids=["no-metadata", "reversed-range", "overlap", "gap", "out-of-order"],
)
async def test_malformed_episode_set_does_not_advance_or_replace_summary(
    tmp_path: Path,
    documents: list[tuple[str, str]],
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    old = "旧摘要\n"
    await store.write_text(1, "summary.md", old)
    for filename, document in documents:
        await store.write_text(1, f"episodes/{filename}", document)

    async def must_not_complete(_messages: list[dict[str, str]]) -> str:
        raise AssertionError("completion must not be called")

    with pytest.raises(ValueError, match="episode"):
        await update_summary_from_episodes(
            store,
            1,
            0,
            must_not_complete,
            max_tokens=100,
        )

    assert await store.read_text(1, "summary.md") == old


@pytest.mark.asyncio
async def test_summary_is_truncated_to_token_bound_even_for_one_long_line(
    tmp_path: Path,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    await store.write_text(1, "episodes/episode-000001.md", episode_document(1, 1, 20))

    async def fake_complete(_messages: list[dict[str, str]]) -> str:
        return "很长的摘要内容" * 100

    checkpoint = await update_summary_from_episodes(
        store,
        1,
        0,
        fake_complete,
        max_tokens=12,
    )

    summary = (await store.read_text(1, "summary.md")).rstrip("\n")
    assert checkpoint == 20
    assert summary
    assert estimate_tokens(summary) <= 12


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    ["empty", "failure", "cancellation"],
)
async def test_invalid_or_failed_summary_preserves_old_file(
    tmp_path: Path,
    failure: str,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    await store.write_text(1, "episodes/episode-000001.md", episode_document(1, 1, 20))
    old = "旧摘要\n"
    await store.write_text(1, "summary.md", old)

    async def fake_complete(_messages: list[dict[str, str]]) -> str:
        if failure == "failure":
            raise RuntimeError("provider failed")
        if failure == "cancellation":
            raise asyncio.CancelledError()
        return " \n "

    error = ValueError if failure == "empty" else (
        RuntimeError if failure == "failure" else asyncio.CancelledError
    )
    with pytest.raises(error):
        await update_summary_from_episodes(
            store,
            1,
            0,
            fake_complete,
            max_tokens=100,
        )

    assert await store.read_text(1, "summary.md") == old


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("checkpoint", "max_tokens"),
    [(-1, 100), (True, 100), (0, 0), (0, -1), (0, True)],
    ids=[
        "negative-checkpoint",
        "boolean-checkpoint",
        "zero-budget",
        "negative-budget",
        "boolean-budget",
    ],
)
async def test_summary_rejects_invalid_checkpoint_or_token_budget(
    tmp_path: Path,
    checkpoint: int,
    max_tokens: int,
) -> None:
    store = MarkdownMemoryStore(tmp_path)

    async def must_not_complete(_messages: list[dict[str, str]]) -> str:
        raise AssertionError("completion must not be called")

    with pytest.raises(ValueError):
        await update_summary_from_episodes(
            store,
            1,
            checkpoint,
            must_not_complete,
            max_tokens=max_tokens,
        )
