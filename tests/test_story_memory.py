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


def story_state() -> str:
    return (
        "# Story State\n\n"
        "## 时间与地点\n- 当前地点：门厅\n\n"
        "## 在场角色\n- 旅人\n\n"
        "## 当前场景\n- 正在交谈\n\n"
        "## 最近变化\n- 开始行动"
    )


def legacy_story_state() -> str:
    return story_state().replace(
        "\n\n## 最近变化",
        "\n\n## 未完成剧情线\n- [open] 找到钥匙\n\n## 最近变化",
    )


def episode_body(label: str = "完成章节") -> str:
    return (
        f"## 剧情摘要\n{label}\n\n"
        "## 状态变化\n- 无\n\n"
        "## 承诺与伏笔\n- 约定在月圆前返回"
    )


def episode_document(number: int, start: int, end: int, label: str = "完成章节") -> str:
    return (
        f"# Episode {number:06d}\n\n"
        f"<!-- messages: {start}-{end} -->\n\n"
        f"{episode_body(label)}\n"
    )


_EXTRA_HEADING_CASES = [
    pytest.param("# Appendix", id="h1-arbitrary"),
    pytest.param("## Characters", id="h2-benign"),
    pytest.param("### Notes", id="h3-benign"),
    pytest.param("#### Unfinished Threads", id="h4-prior-morphology"),
    pytest.param("##### 【待完成】目标", id="h5-prior-morphology"),
    pytest.param("###### oPeN tHrEaDs ######", id="h6-prior-morphology"),
    pytest.param("Appendix\n===", id="setext-h1-benign"),
    pytest.param("Characters\n---", id="setext-h2-benign"),
    pytest.param("Story-Arc Progress\n===", id="setext-h1-prior-morphology"),
    pytest.param("剧情线进度\n---", id="setext-h2-prior-morphology"),
]

_CONTAINER_HEADING_CASES = [
    pytest.param("> # Appendix", id="blockquote-atx-h1"),
    pytest.param(">   ### Notes", id="blockquote-spaced-atx-h3"),
    pytest.param("- ### Appendix", id="unordered-list-atx-h3"),
    pytest.param("  - #### Appendix", id="indented-list-atx-h4"),
    pytest.param("1. ##### Appendix", id="ordered-list-atx-h5"),
    pytest.param("> > ###### Appendix", id="nested-blockquote-atx-h6"),
    pytest.param("> - ### Appendix", id="blockquote-list-atx-h3"),
    pytest.param("> Appendix\n> ===", id="blockquote-setext-h1"),
    pytest.param("- Appendix\n  ---", id="list-setext-h2"),
    pytest.param("> - Appendix\n>   ---", id="nested-list-setext-h2"),
    pytest.param("10. item\n    # Appendix", id="ordered-list-continuation-atx"),
    pytest.param(
        "- item\n  - nested\n    ### Notes",
        id="nested-list-continuation-atx",
    ),
    pytest.param("> ```\n# Appendix\n> ```", id="blockquote-fence-exit-atx"),
    pytest.param("``` invalid`\n# Appendix", id="invalid-backtick-info-atx"),
]


async def _assert_generated_heading_rejected(
    tmp_path: Path,
    target: str,
    heading: str,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    generated_suffix = f"\n\n{heading}\n- 不应写入"

    if target == "story":
        old = legacy_story_state() + "\n"
        await store.write_text(1, "story_state.md", old)

        async def fake_complete(_messages: list[dict[str, str]]) -> str:
            return story_state() + generated_suffix

        with pytest.raises(ValueError, match="story state"):
            await update_story_state(store, 1, records(2), fake_complete)

        assert await store.read_text(1, "story_state.md") == old
    elif target == "episode":
        old = episode_document(1, 1, 20, "旧章节")
        await store.write_text(1, "episodes/episode-000001.md", old)

        async def fake_complete(_messages: list[dict[str, str]]) -> str:
            return episode_body("新章节") + generated_suffix

        with pytest.raises(ValueError, match="episode"):
            await create_due_episodes(
                store,
                1,
                records(40),
                20,
                fake_complete,
                episode_size=20,
            )

        assert await store.read_text(1, "episodes/episode-000001.md") == old
        assert not (tmp_path / "1" / "episodes" / "episode-000002.md").exists()
    else:
        old = "# 旧格式摘要\n\n## 未完成伏笔\n- [open] 旧格式伏笔\n"
        await store.write_text(1, "summary.md", old)
        await store.write_text(
            1,
            "episodes/episode-000001.md",
            episode_document(1, 1, 20),
        )

        async def fake_complete(_messages: list[dict[str, str]]) -> str:
            return "新摘要正文。" + generated_suffix

        with pytest.raises(ValueError, match="summary"):
            await update_summary_from_episodes(
                store,
                1,
                0,
                fake_complete,
                max_tokens=200,
            )

        assert await store.read_text(1, "summary.md") == old


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["story", "episode", "summary"])
@pytest.mark.parametrize("heading", _EXTRA_HEADING_CASES)
async def test_generated_output_rejects_every_extra_real_heading(
    tmp_path: Path,
    target: str,
    heading: str,
) -> None:
    await _assert_generated_heading_rejected(tmp_path, target, heading)


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["story", "episode", "summary"])
@pytest.mark.parametrize("heading", _CONTAINER_HEADING_CASES)
async def test_generated_output_rejects_real_heading_in_markdown_container(
    tmp_path: Path,
    target: str,
    heading: str,
) -> None:
    await _assert_generated_heading_rejected(tmp_path, target, heading)


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["story", "episode", "summary"])
async def test_generated_narrative_plot_state_words_are_not_headings_and_are_allowed(
    tmp_path: Path,
    target: str,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    narrative = "Closed Threads and Completed Threads are phrases discussed in the journal."

    if target == "story":
        generated = story_state().replace("- 正在交谈", f"- 正在交谈\n\n{narrative}")

        async def fake_complete(_messages: list[dict[str, str]]) -> str:
            return generated

        assert await update_story_state(store, 1, records(2), fake_complete) == generated
        assert await store.read_text(1, "story_state.md") == generated + "\n"
    elif target == "episode":
        generated = episode_body().replace("完成章节", f"完成章节\n\n{narrative}")

        async def fake_complete(_messages: list[dict[str, str]]) -> str:
            return generated

        assert await create_due_episodes(
            store,
            1,
            records(20),
            0,
            fake_complete,
            episode_size=20,
        ) == 20
        assert narrative in await store.read_text(1, "episodes/episode-000001.md")
    else:
        await store.write_text(
            1,
            "episodes/episode-000001.md",
            episode_document(1, 1, 20),
        )
        generated = narrative

        async def fake_complete(_messages: list[dict[str, str]]) -> str:
            return generated

        assert await update_summary_from_episodes(
            store,
            1,
            0,
            fake_complete,
            max_tokens=200,
        ) == 20
        assert await store.read_text(1, "summary.md") == generated + "\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["story", "episode", "summary"])
@pytest.mark.parametrize("fence", ["```", "~~~"], ids=["backtick", "tilde"])
async def test_fenced_heading_literal_does_not_count_as_document_heading(
    tmp_path: Path,
    target: str,
    fence: str,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    code_block = f"{fence}markdown\n## Closed Threads\n- literal example\n{fence}"

    if target == "story":
        generated = story_state().replace(
            "- 当前地点：门厅",
            f"- 当前地点：门厅\n\n{code_block}",
        )

        async def fake_complete(_messages: list[dict[str, str]]) -> str:
            return generated

        assert await update_story_state(store, 1, records(2), fake_complete) == generated
        assert await store.read_text(1, "story_state.md") == generated + "\n"
    elif target == "episode":
        generated = episode_body().replace("完成章节", f"完成章节\n\n{code_block}")

        async def fake_complete(_messages: list[dict[str, str]]) -> str:
            return generated

        assert await create_due_episodes(
            store,
            1,
            records(20),
            0,
            fake_complete,
            episode_size=20,
        ) == 20
        assert code_block in await store.read_text(1, "episodes/episode-000001.md")
    else:
        await store.write_text(
            1,
            "episodes/episode-000001.md",
            episode_document(1, 1, 20),
        )
        generated = (
            "整体剧情连续叙事。\n\n"
            "- 保留早期事件\n"
            "- 保留人物关系\n\n"
            f"{code_block}"
        )

        async def fake_complete(_messages: list[dict[str, str]]) -> str:
            return generated

        assert await update_summary_from_episodes(
            store,
            1,
            0,
            fake_complete,
            max_tokens=200,
        ) == 20
        assert await store.read_text(1, "summary.md") == generated + "\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["story", "episode", "summary"])
@pytest.mark.parametrize(
    "code_block",
    [
        "> ```markdown\n> # Appendix\n> Notes\n> ---\n> ```",
        "- ```markdown\n  ### Notes\n  Appendix\n  ---\n  ```",
        "> - ~~~markdown\n>   #### Notes\n>   ~~~",
    ],
    ids=["blockquote", "list", "nested-blockquote-list"],
)
async def test_heading_literal_in_container_fence_is_allowed(
    tmp_path: Path,
    target: str,
    code_block: str,
) -> None:
    store = MarkdownMemoryStore(tmp_path)

    if target == "story":
        generated = story_state().replace(
            "- 当前地点：门厅",
            f"- 当前地点：门厅\n\n{code_block}",
        )

        async def fake_complete(_messages: list[dict[str, str]]) -> str:
            return generated

        assert await update_story_state(store, 1, records(2), fake_complete) == generated
        assert await store.read_text(1, "story_state.md") == generated + "\n"
    elif target == "episode":
        generated = episode_body().replace("完成章节", f"完成章节\n\n{code_block}")

        async def fake_complete(_messages: list[dict[str, str]]) -> str:
            return generated

        assert await create_due_episodes(
            store,
            1,
            records(20),
            0,
            fake_complete,
            episode_size=20,
        ) == 20
        assert code_block in await store.read_text(1, "episodes/episode-000001.md")
    else:
        await store.write_text(
            1,
            "episodes/episode-000001.md",
            episode_document(1, 1, 20),
        )
        generated = f"整体剧情连续叙事。\n\n{code_block}"

        async def fake_complete(_messages: list[dict[str, str]]) -> str:
            return generated

        assert await update_summary_from_episodes(
            store,
            1,
            0,
            fake_complete,
            max_tokens=200,
        ) == 20
        assert await store.read_text(1, "summary.md") == generated + "\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["story", "episode", "summary"])
async def test_list_followed_by_thematic_break_is_not_setext_heading(
    tmp_path: Path,
    target: str,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    content = "- Appendix\n---"

    if target == "story":
        generated = story_state().replace(
            "- 当前地点：门厅",
            f"- 当前地点：门厅\n\n{content}",
        )

        async def fake_complete(_messages: list[dict[str, str]]) -> str:
            return generated

        assert await update_story_state(store, 1, records(2), fake_complete) == generated
        assert await store.read_text(1, "story_state.md") == generated + "\n"
    elif target == "episode":
        generated = episode_body().replace("完成章节", f"完成章节\n\n{content}")

        async def fake_complete(_messages: list[dict[str, str]]) -> str:
            return generated

        assert await create_due_episodes(
            store,
            1,
            records(20),
            0,
            fake_complete,
            episode_size=20,
        ) == 20
        assert content in await store.read_text(1, "episodes/episode-000001.md")
    else:
        await store.write_text(
            1,
            "episodes/episode-000001.md",
            episode_document(1, 1, 20),
        )
        generated = f"整体剧情连续叙事。\n\n{content}"

        async def fake_complete(_messages: list[dict[str, str]]) -> str:
            return generated

        assert await update_summary_from_episodes(
            store,
            1,
            0,
            fake_complete,
            max_tokens=200,
        ) == 20
        assert await store.read_text(1, "summary.md") == generated + "\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["story", "episode"])
async def test_real_extra_h2_outside_fence_is_rejected(
    tmp_path: Path,
    target: str,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    extra = "\n\n## 附加事实\n- 不允许额外二级标题"

    if target == "story":

        async def fake_complete(_messages: list[dict[str, str]]) -> str:
            return story_state() + extra

        with pytest.raises(ValueError, match="story state"):
            await update_story_state(store, 1, records(2), fake_complete)
        assert not (tmp_path / "1" / "story_state.md").exists()
    else:

        async def fake_complete(_messages: list[dict[str, str]]) -> str:
            return episode_body() + extra

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


def test_render_records_preserves_numbers_roles_multiline_and_unicode() -> None:
    source = [
        ChatRecord(7, "user", "第一行\n第二行 🙂", "2026-01-01 00:00", "用户"),
        ChatRecord(8, "assistant", "回复", "2026-01-01 00:01", "角色"),
    ]

    assert render_records(source) == "[7] user: 第一行\n第二行 🙂\n[8] assistant: 回复"


@pytest.mark.asyncio
async def test_story_state_migrates_legacy_state_to_current_snapshot(
    tmp_path: Path,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    existing = legacy_story_state()
    await store.write_text(1, "story_state.md", existing)
    expected = story_state()
    calls: list[list[dict[str, str]]] = []

    async def fake_complete(messages: list[dict[str, str]]) -> str:
        calls.append(messages)
        assert existing in messages[-1]["content"]
        assert "[1] user: message 1" in messages[-1]["content"]
        assert "不判断剧情线是否完成" in messages[0]["content"]
        assert "不得输出 [open]" in messages[0]["content"]
        assert "只允许一个 # Story State" in messages[0]["content"]
        assert "不得输出任何其他真实 Markdown 标题" in messages[0]["content"]
        return f"\n{expected}\n"

    result = await update_story_state(store, 1, records(2), fake_complete)

    assert result == expected
    assert "未完成剧情线" not in result
    assert all(marker not in result for marker in ("[open]", "[closed]", "[done]"))
    assert await store.read_text(1, "story_state.md") == expected + "\n"
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid",
    [
        "",
        story_state().replace(
            "## 当前场景\n- 正在交谈\n\n",
            "",
        ),
        story_state() + "\n\n## 未完成剧情线\n- 找到钥匙",
        story_state() + "\n- [open] 找到钥匙",
        story_state() + "\n- [closed] 找到钥匙",
        story_state() + "\n- [done] 找到钥匙",
        story_state() + "\n\n## Plot State\n- resolved",
    ],
    ids=[
        "empty",
        "missing-scene",
        "legacy-plot-heading",
        "open-marker",
        "closed-marker",
        "done-marker",
        "plot-state-heading",
    ],
)
async def test_invalid_story_snapshot_does_not_replace_prior_file(
    tmp_path: Path,
    invalid: str,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    old = story_state() + "\n"
    await store.write_text(1, "story_state.md", old)

    async def fake_complete(_messages: list[dict[str, str]]) -> str:
        return invalid

    with pytest.raises(ValueError, match="story state"):
        await update_story_state(store, 1, records(2), fake_complete)

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
    old = legacy_story_state() + "\n"
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
    systems: list[str] = []

    async def fake_complete(messages: list[dict[str, str]]) -> str:
        systems.append(messages[0]["content"])
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
    assert "不判断剧情线是否完成" in systems[0]
    assert "只允许三个按顺序的二级标题" in systems[0]
    assert "不得输出任何其他真实 Markdown 标题" in systems[0]
    assert all(marker not in text for marker in ("[open]", "[closed]", "[done]"))


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
    [
        "",
        "plain text",
        episode_body().replace("## 状态变化", "## 改变"),
        episode_body().replace("- 约定", "- [open] 约定"),
        episode_body().replace("- 约定", "- [closed] 约定"),
        episode_body().replace("- 约定", "- [done] 约定"),
        episode_body() + "\n\n## 未完成剧情线\n- 找到钥匙",
    ],
    ids=[
        "empty",
        "plain",
        "missing-heading",
        "open-marker",
        "closed-marker",
        "done-marker",
        "plot-state-heading",
    ],
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
    first = episode_document(1, 1, 20, "序章救援")
    legacy_pending = episode_document(2, 21, 40, "第二章约定").replace(
        "## 承诺与伏笔\n- 约定在月圆前返回",
        "## 承诺与伏笔\n### 未完成伏笔\n- [open] 约定在月圆前返回",
    )
    await store.write_text(1, "episodes/episode-000001.md", first)
    await store.write_text(1, "episodes/episode-000002.md", legacy_pending)
    old = (
        "# 旧格式摘要\n\n## 未完成剧情线\n"
        "早期事件：旅人在序章救下守门人。\n- [open] 旧格式承诺\n"
    )
    await store.write_text(1, "summary.md", old)
    calls: list[list[dict[str, str]]] = []

    async def fake_complete(messages: list[dict[str, str]]) -> str:
        calls.append(messages)
        prompt = messages[-1]["content"]
        assert old in prompt
        assert "序章救援" not in prompt
        assert "第二章约定" in prompt
        assert "### 未完成伏笔" in prompt
        assert "[open] 约定在月圆前返回" in prompt
        return "早期事件：旅人在序章救下守门人。\n\n人物关系与后续约定均已记录。"

    checkpoint = await update_summary_from_episodes(
        store,
        1,
        20,
        fake_complete,
        max_tokens=200,
    )

    assert checkpoint == 40
    summary = await store.read_text(1, "summary.md")
    assert "早期事件：旅人在序章救下守门人" in summary
    assert all(marker not in summary for marker in ("[open]", "[closed]", "[done]"))
    assert "保留完整故事弧" in calls[0][0]["content"]
    assert "不判断剧情线或伏笔是否完成" in calls[0][0]["content"]
    assert "不得输出任何真实 Markdown 标题" in calls[0][0]["content"]


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
    ("ranges", "checkpoint"),
    [
        ([], -1),
        ([], 1),
        ([(1, 1, 20)], 10),
        ([(1, 1, 20)], 21),
        ([(1, 1, 20)], 100),
        ([(1, 1, 20), (2, 21, 40)], 10),
        ([(1, 1, 20), (2, 21, 40)], 30),
        ([(1, 1, 20), (2, 21, 40)], 41),
    ],
    ids=[
        "empty-negative",
        "empty-nonzero",
        "single-middle",
        "single-gap",
        "single-beyond",
        "multiple-first-middle",
        "multiple-second-middle",
        "multiple-beyond",
    ],
)
async def test_summary_checkpoint_must_be_zero_or_existing_episode_end(
    tmp_path: Path,
    ranges: list[tuple[int, int, int]],
    checkpoint: int,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    for number, start, end in ranges:
        await store.write_text(
            1,
            f"episodes/episode-{number:06d}.md",
            episode_document(number, start, end),
        )
    old = "旧摘要保持不变\n"
    await store.write_text(1, "summary.md", old)
    calls = 0

    async def must_not_complete(_messages: list[dict[str, str]]) -> str:
        nonlocal calls
        calls += 1
        raise AssertionError("completion must not be called")

    with pytest.raises(ValueError, match="summary checkpoint"):
        await update_summary_from_episodes(
            store,
            1,
            checkpoint,
            must_not_complete,
            max_tokens=100,
        )

    assert calls == 0
    assert await store.read_text(1, "summary.md") == old


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ranges", "checkpoint"),
    [
        ([], 0),
        ([(1, 1, 20)], 20),
        ([(1, 1, 20), (2, 21, 40)], 40),
    ],
    ids=["empty-zero", "single-final-end", "multiple-final-end"],
)
async def test_summary_accepts_zero_or_final_episode_end_without_work(
    tmp_path: Path,
    ranges: list[tuple[int, int, int]],
    checkpoint: int,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    for number, start, end in ranges:
        await store.write_text(
            1,
            f"episodes/episode-{number:06d}.md",
            episode_document(number, start, end),
        )
    old = "保持原样（无结尾换行）"
    await store.write_text(1, "summary.md", old)

    async def must_not_complete(_messages: list[dict[str, str]]) -> str:
        raise AssertionError("completion must not be called")

    assert await update_summary_from_episodes(
        store,
        1,
        checkpoint,
        must_not_complete,
        max_tokens=100,
    ) == checkpoint
    assert await store.read_text(1, "summary.md") == old


@pytest.mark.asyncio
async def test_summary_validates_episode_chain_before_checkpoint(
    tmp_path: Path,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    await store.write_text(
        1,
        "episodes/episode-000001.md",
        episode_document(1, 2, 21),
    )
    old = "旧摘要\n"
    await store.write_text(1, "summary.md", old)

    async def must_not_complete(_messages: list[dict[str, str]]) -> str:
        raise AssertionError("completion must not be called")

    with pytest.raises(ValueError, match="episode"):
        await update_summary_from_episodes(
            store,
            1,
            10,
            must_not_complete,
            max_tokens=100,
        )

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
async def test_summary_rejects_heading_beyond_truncation_boundary(
    tmp_path: Path,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    await store.write_text(1, "episodes/episode-000001.md", episode_document(1, 1, 20))
    old = "# 旧格式摘要\n\n- [open] 保留旧承诺\n"
    await store.write_text(1, "summary.md", old)

    async def fake_complete(_messages: list[dict[str, str]]) -> str:
        return "连续叙事内容。" * 100 + "\n\n## Characters\n- 旅人"

    with pytest.raises(ValueError, match="summary"):
        await update_summary_from_episodes(
            store,
            1,
            0,
            fake_complete,
            max_tokens=12,
        )

    assert await store.read_text(1, "summary.md") == old


@pytest.mark.asyncio
async def test_summary_rejects_heading_created_by_truncation(
    tmp_path: Path,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    await store.write_text(1, "episodes/episode-000001.md", episode_document(1, 1, 20))
    old = "# 旧格式摘要\n\n- [open] 保留旧承诺\n"
    await store.write_text(1, "summary.md", old)

    async def fake_complete(_messages: list[dict[str, str]]) -> str:
        return "abc\n---x"

    with pytest.raises(ValueError, match="summary"):
        await update_summary_from_episodes(
            store,
            1,
            0,
            fake_complete,
            max_tokens=1,
        )

    assert await store.read_text(1, "summary.md") == old


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid",
    [
        "连续叙事。\n\n- [open] 找到钥匙",
        "连续叙事。\n\n- [closed] 找到钥匙",
        "连续叙事。\n\n- [done] 找到钥匙",
    ],
    ids=[
        "open-marker",
        "closed-marker",
        "done-marker",
    ],
)
async def test_invalid_summary_output_preserves_legacy_summary(
    tmp_path: Path,
    invalid: str,
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    legacy_episode = episode_document(1, 1, 20).replace(
        "- 约定在月圆前返回",
        "- [open] 约定在月圆前返回",
    )
    await store.write_text(1, "episodes/episode-000001.md", legacy_episode)
    old = "# 旧格式摘要\n\n## 未完成伏笔\n早期事件仍需保留\n- [open] 旧格式伏笔\n"
    await store.write_text(1, "summary.md", old)

    async def fake_complete(messages: list[dict[str, str]]) -> str:
        assert old in messages[-1]["content"]
        assert "[open] 约定在月圆前返回" in messages[-1]["content"]
        return invalid

    with pytest.raises(ValueError, match="summary"):
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
