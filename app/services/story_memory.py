"""Current story state, immutable episodes, and episode-derived summaries."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.services.md_store import (
    ChatRecord,
    MarkdownMemoryStore,
    MarkdownMemoryTransaction,
)
from app.services.token_utils import estimate_tokens, truncate_to_tokens

Completion = Callable[[list[dict[str, str]]], Awaitable[str]]

STORY_STATE_SYSTEM = """你是长篇互动故事的当前场景快照维护器。输出完整的 Story State Markdown。
只根据对话证据记录当前场景，不得猜测或补造事实。
不判断剧情线是否完成，不维护剧情线状态。
不得输出 [open]、[closed] 或 [done] 标记，不得输出剧情线状态标题。
固定标题且按此顺序输出：时间与地点、在场角色、当前场景、最近变化。"""

EPISODE_SYSTEM = """你是长篇互动故事章节压缩器。只记录提供的精确消息范围内的事实。
输出完整 Markdown，固定标题且按此顺序输出：剧情摘要、状态变化、承诺与伏笔。
承诺与伏笔只作为事实记录，不判断剧情线是否完成。
不得输出 [open]、[closed] 或 [done] 标记，不得输出剧情线状态标题。"""

_SUMMARY_SYSTEM = """根据已有摘要和新增章节，更新可独立阅读的整体剧情回顾。
保留完整故事弧、人物关系、关键事件、状态变化、承诺与伏笔事实，以及早期事件。
不判断剧情线或伏笔是否完成，不得输出 [open]、[closed] 或 [done] 标记，不得输出剧情线状态标题。"""

_STORY_HEADINGS = (
    "时间与地点",
    "在场角色",
    "当前场景",
    "最近变化",
)
_EPISODE_HEADINGS = ("剧情摘要", "状态变化", "承诺与伏笔")
_CANONICAL_SECTION_HEADING = re.compile(
    r"^##[ \t]+(?P<name>[^\r\n]+?)[ \t]*$"
)
_MACHINE_MARKER = re.compile(r"\[(?:open|closed|done)\]", re.IGNORECASE)
_ATX_HEADING = re.compile(
    r"^[ \t]{0,3}(?P<marks>#{1,6})(?:[ \t]+(?P<title>.*?))?[ \t]*$"
)
_ATX_CLOSING_HASHES = re.compile(r"[ \t]+#+[ \t]*\Z")
_SETEXT_TITLE = re.compile(r"^[ \t]{0,3}(?P<title>\S(?:.*?\S)?)[ \t]*$")
_SETEXT_UNDERLINE = re.compile(
    r"^[ \t]{0,3}(?P<underline>=+|-+)[ \t]*$"
)
_FENCE_OPEN = re.compile(r"^[ ]{0,3}(?P<fence>`{3,}|~{3,})")
_ENGLISH_PLOT_TOPICS = frozenset(
    {
        "thread",
        "threads",
        "storyline",
        "storylines",
        "foreshadowing",
        "foreshadowings",
        "commitment",
        "commitments",
        "plotline",
        "plotlines",
    }
)
_ENGLISH_PLOT_STATES = frozenset(
    {
        "open",
        "closed",
        "completed",
        "completion",
        "resolved",
        "resolution",
        "unresolved",
        "status",
        "pending",
        "done",
        "active",
        "inactive",
    }
)
_CHINESE_PLOT_TOPICS = ("剧情线", "故事线", "伏笔", "承诺")
_CHINESE_PLOT_STATES = (
    "未完成",
    "已完成",
    "完成状态",
    "未解决",
    "已解决",
    "开放",
    "关闭",
    "状态",
    "待处理",
    "进行中",
)
_EPISODE_FILENAME = re.compile(r"episode-(?P<number>\d{6})\.md\Z")
_EPISODE_DOCUMENT = re.compile(
    r"\A# Episode (?P<number>\d{6})[ \t]*\r?\n\r?\n"
    r"<!-- messages: (?P<start>[1-9]\d*)-(?P<end>[1-9]\d*) -->[ \t]*\r?\n\r?\n"
    r"(?P<body>[\s\S]*?)\r?\n\Z"
)


@dataclass(frozen=True)
class _MarkdownHeading:
    title: str
    level: int
    start: int
    end: int
    source: str


@dataclass(frozen=True)
class _Episode:
    number: int
    start: int
    end: int
    text: str


def render_records(records: list[ChatRecord]) -> str:
    """Render records with their authoritative message numbers."""
    return "\n".join(
        f"[{record.number}] {record.role}: {record.content}" for record in records
    )


def _validate_sections(
    text: str,
    expected: tuple[str, ...],
    *,
    label: str,
    prefix: str,
) -> None:
    sections = [heading for heading in _markdown_headings(text) if heading.level == 2]
    if [heading.title for heading in sections] != list(expected):
        raise ValueError(f"{label} must contain every required heading in order")
    for heading, name in zip(sections, expected, strict=True):
        canonical = _CANONICAL_SECTION_HEADING.fullmatch(heading.source)
        if canonical is None or canonical.group("name") != name:
            raise ValueError(f"{label} must contain every required heading in order")
    if not sections or text[: sections[0].start].strip() != prefix:
        raise ValueError(f"{label} has an invalid or incomplete preamble")
    for index, heading in enumerate(sections):
        end = sections[index + 1].start if index + 1 < len(sections) else len(text)
        if not text[heading.end : end].strip():
            raise ValueError(f"{label} contains an empty section")


def _completion_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} completion returned empty or invalid output")
    return value.strip()


def _markdown_headings(text: str) -> list[_MarkdownHeading]:
    raw_lines = text.splitlines(keepends=True)
    lines = [line.rstrip("\r\n") for line in raw_lines]
    offsets: list[int] = []
    offset = 0
    for raw_line in raw_lines:
        offsets.append(offset)
        offset += len(raw_line)

    headings: list[_MarkdownHeading] = []
    fence: tuple[str, int] | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        if fence is not None:
            marker, minimum = fence
            stripped = line.lstrip(" ")
            if len(line) - len(stripped) <= 3 and re.fullmatch(
                rf"{re.escape(marker)}{{{minimum},}}[ \t]*",
                stripped,
            ):
                fence = None
            index += 1
            continue

        fence_match = _FENCE_OPEN.match(line)
        if fence_match is not None:
            opening = fence_match.group("fence")
            fence = (opening[0], len(opening))
            index += 1
            continue

        atx_match = _ATX_HEADING.fullmatch(line)
        if atx_match is not None:
            title = atx_match.group("title") or ""
            headings.append(
                _MarkdownHeading(
                    title=_ATX_CLOSING_HASHES.sub("", title).strip(),
                    level=len(atx_match.group("marks")),
                    start=offsets[index],
                    end=offsets[index] + len(raw_lines[index]),
                    source=line,
                )
            )
            index += 1
            continue

        title_match = _SETEXT_TITLE.fullmatch(line)
        underline_match = (
            _SETEXT_UNDERLINE.fullmatch(lines[index + 1])
            if index + 1 < len(lines)
            else None
        )
        if (
            title_match is not None
            and underline_match is not None
        ):
            underline = underline_match.group("underline")
            headings.append(
                _MarkdownHeading(
                    title=title_match.group("title").strip(),
                    level=1 if underline.startswith("=") else 2,
                    start=offsets[index],
                    end=offsets[index + 1] + len(raw_lines[index + 1]),
                    source=f"{line}\n{lines[index + 1]}",
                )
            )
            index += 2
            continue
        index += 1
    return headings


def _is_plot_state_title(title: str) -> bool:
    compact = re.sub(r"\s+", "", title)
    if any(topic in compact for topic in _CHINESE_PLOT_TOPICS) and any(
        state in compact for state in _CHINESE_PLOT_STATES
    ):
        return True

    words = re.findall(r"[a-z]+", title.casefold())
    word_set = set(words)
    line_words = {"line", "lines"}
    thread_words = {"thread", "threads"}
    has_specific_topic = bool(word_set & _ENGLISH_PLOT_TOPICS) or (
        bool(word_set & {"plot", "story"}) and bool(word_set & line_words)
    )
    has_state = bool(word_set & _ENGLISH_PLOT_STATES)
    bare_plot_section = (
        "plot" in word_set and bool(word_set & (line_words | thread_words))
    ) or bool(word_set & {"plotline", "plotlines"})
    return bare_plot_section or (
        has_state and (has_specific_topic or "plot" in word_set)
    )


def _reject_plot_classification(text: str, *, label: str) -> None:
    if _MACHINE_MARKER.search(text) or any(
        _is_plot_state_title(heading.title) for heading in _markdown_headings(text)
    ):
        raise ValueError(f"{label} must not classify plot state")


def _validate_story_state(updated: str) -> None:
    _validate_sections(
        updated,
        _STORY_HEADINGS,
        label="story state",
        prefix="# Story State",
    )
    _reject_plot_classification(updated, label="story state")


def _validate_records(records: list[ChatRecord]) -> None:
    previous = 0
    for index, record in enumerate(records):
        if type(record.number) is not int or record.number <= 0:
            raise ValueError("record numbers must be positive integers")
        if index and record.number != previous + 1:
            raise ValueError("record numbers must be ordered and contiguous")
        previous = record.number


def _validate_nonnegative_integer(value: int, *, label: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")


def _parse_episode(text: str, filename: str) -> _Episode:
    filename_match = _EPISODE_FILENAME.fullmatch(filename)
    document_match = _EPISODE_DOCUMENT.fullmatch(text)
    if filename_match is None or document_match is None:
        raise ValueError(f"episode has an invalid document structure: {filename}")

    filename_number = int(filename_match.group("number"))
    document_number = int(document_match.group("number"))
    start = int(document_match.group("start"))
    end = int(document_match.group("end"))
    if filename_number <= 0 or document_number != filename_number or start > end:
        raise ValueError(f"episode has inconsistent metadata: {filename}")

    body = document_match.group("body")
    _validate_sections(body, _EPISODE_HEADINGS, label="episode", prefix="")
    return _Episode(filename_number, start, end, text)


def _validate_episode_body(body: str) -> str:
    cleaned = _completion_text(body, label="episode")
    _validate_sections(cleaned, _EPISODE_HEADINGS, label="episode", prefix="")
    _reject_plot_classification(cleaned, label="episode")
    return cleaned


async def _load_existing_episode_chain(
    store: MarkdownMemoryStore,
    transaction: MarkdownMemoryTransaction,
    *,
    episode_size: int,
    checkpoint: int,
) -> dict[int, _Episode]:
    episode_directory = store.session_dir(transaction.session_id) / "episodes"
    paths = (
        sorted(episode_directory.glob("episode-*.md"))
        if episode_directory.exists()
        else []
    )
    episodes: dict[int, _Episode] = {}
    for expected_number, path in enumerate(paths, start=1):
        text = await transaction.read_text(f"episodes/{path.name}")
        try:
            episode = _parse_episode(text, path.name)
        except ValueError as exc:
            raise ValueError(f"existing episode is invalid: {path.name}") from exc

        expected_start = (expected_number - 1) * episode_size + 1
        expected_end = expected_number * episode_size
        if (
            episode.number != expected_number
            or episode.start != expected_start
            or episode.end != expected_end
        ):
            raise ValueError(
                "existing episode chain has conflicting, duplicate, out-of-order, "
                f"or gapped metadata: {path.name}"
            )
        episodes[episode.number] = episode

    required_episodes = checkpoint // episode_size
    if required_episodes > len(episodes):
        raise ValueError("existing episode chain does not cover the checkpoint")
    return episodes


async def update_story_state(
    store: MarkdownMemoryStore,
    session_id: int,
    records: list[ChatRecord],
    complete: Completion,
) -> str:
    """Replace current story state only after validating a complete result."""
    async with store.transaction(session_id) as transaction:
        existing = await transaction.read_text("story_state.md")
        messages = [
            {"role": "system", "content": STORY_STATE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"现有状态：\n{existing}\n\n"
                    f"新增对话：\n{render_records(records)}"
                ),
            },
        ]
        updated = _completion_text(
            await complete(messages),
            label="story state",
        )
        _validate_story_state(updated)
        await transaction.write_text("story_state.md", updated + "\n")
        return updated


async def create_due_episodes(
    store: MarkdownMemoryStore,
    session_id: int,
    records: list[ChatRecord],
    last_episode_message: int,
    complete: Completion,
    *,
    episode_size: int,
) -> int:
    """Create every due immutable episode and return its message checkpoint."""
    _validate_nonnegative_integer(last_episode_message, label="episode checkpoint")
    if type(episode_size) is not int or episode_size <= 0:
        raise ValueError("episode size must be a positive integer")
    if last_episode_message % episode_size:
        raise ValueError("episode checkpoint must align to episode size")
    _validate_records(records)

    if not records:
        if last_episode_message:
            raise ValueError("episode checkpoint exceeds available records")
        return 0
    if last_episode_message > records[-1].number:
        raise ValueError("episode checkpoint exceeds available records")
    if last_episode_message < records[0].number - 1:
        raise ValueError("records do not include the next episode message")

    by_number = {record.number: record for record in records}
    checkpoint = last_episode_message
    async with store.transaction(session_id) as transaction:
        existing_episodes = await _load_existing_episode_chain(
            store,
            transaction,
            episode_size=episode_size,
            checkpoint=last_episode_message,
        )
        while records[-1].number - checkpoint >= episode_size:
            start = checkpoint + 1
            end = checkpoint + episode_size
            try:
                batch = [by_number[number] for number in range(start, end + 1)]
            except KeyError as exc:
                raise ValueError("episode range is not present in records") from exc

            episode_number = checkpoint // episode_size + 1
            filename = f"episode-{episode_number:06d}.md"
            relative = f"episodes/{filename}"
            existing = existing_episodes.get(episode_number)
            if existing is not None:
                checkpoint = end
                continue

            body = _validate_episode_body(
                await complete(
                    [
                        {"role": "system", "content": EPISODE_SYSTEM},
                        {"role": "user", "content": render_records(batch)},
                    ]
                )
            )
            document = (
                f"# Episode {episode_number:06d}\n\n"
                f"<!-- messages: {start}-{end} -->\n\n"
                f"{body}\n"
            )
            await transaction.write_text(relative, document)
            existing_episodes[episode_number] = _Episode(
                episode_number,
                start,
                end,
                document,
            )
            checkpoint = end
    return checkpoint


def _truncate_summary(text: str, max_tokens: int) -> str:
    updated = truncate_to_tokens(text, max_tokens).rstrip()
    if estimate_tokens(updated) <= max_tokens:
        return updated

    low = 0
    high = len(updated)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_tokens(updated[:middle]) <= max_tokens:
            low = middle
        else:
            high = middle - 1
    return updated[:low].rstrip()


async def update_summary_from_episodes(
    store: MarkdownMemoryStore,
    session_id: int,
    last_summary_message: int,
    complete: Completion,
    *,
    max_tokens: int,
) -> int:
    """Update the rolling summary from a strictly validated episode sequence."""
    _validate_nonnegative_integer(last_summary_message, label="summary checkpoint")
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("summary token budget must be a positive integer")

    async with store.transaction(session_id) as transaction:
        episode_directory = store.session_dir(session_id) / "episodes"
        paths = (
            sorted(episode_directory.glob("episode-*.md"))
            if episode_directory.exists()
            else []
        )
        episodes: list[_Episode] = []
        previous_end = 0
        for expected_number, path in enumerate(paths, start=1):
            text = await transaction.read_text(f"episodes/{path.name}")
            episode = _parse_episode(text, path.name)
            if episode.number != expected_number:
                raise ValueError(f"episode filenames are not contiguous: {path.name}")
            if episode.start != previous_end + 1:
                raise ValueError(
                    "episode ranges overlap, are out of order, or have a gap: "
                    f"{path.name}"
                )
            episodes.append(episode)
            previous_end = episode.end

        pending = [
            episode for episode in episodes if episode.end > last_summary_message
        ]
        if not pending:
            return last_summary_message

        existing = await transaction.read_text("summary.md")
        messages = [
            {"role": "system", "content": _SUMMARY_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"已有摘要：\n{existing}\n\n新增章节：\n"
                    + "\n\n---\n\n".join(episode.text for episode in pending)
                ),
            },
        ]
        raw = _completion_text(await complete(messages), label="summary")
        _reject_plot_classification(raw, label="summary")
        updated = _truncate_summary(raw, max_tokens)
        if not updated:
            raise ValueError("summary completion was empty after truncation")
        await transaction.write_text("summary.md", updated + "\n")
        return pending[-1].end
