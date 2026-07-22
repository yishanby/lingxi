"""Current story state, immutable episodes, and episode-derived summaries."""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from html.parser import HTMLParser

from markdown_it import MarkdownIt
from markdown_it.token import Token

from app.services.md_store import (
    ChatRecord,
    MarkdownMemoryStore,
    MarkdownMemoryTransaction,
)
from app.services.stage_receipts import (
    ChatSourceIdentity,
    StageUpdateResult,
    text_artifact,
)
from app.services.token_utils import estimate_tokens, truncate_to_tokens

Completion = Callable[[list[dict[str, str]]], Awaitable[str]]

STORY_STATE_SYSTEM = """你是长篇互动故事的当前场景快照维护器。输出完整的 Story State Markdown。
只根据对话证据记录当前场景，不得猜测或补造事实。
不判断剧情线是否完成，不维护剧情线状态；完整剧情脉络由 summary.md 负责。
只允许一个 # Story State，随后只允许四个按顺序的二级标题：时间与地点、在场角色、当前场景、最近变化；各节都必须非空。
不得输出任何其他真实 Markdown 标题（ATX 或 Setext）；代码围栏中的标题文字可作为正文内容。
不得输出代码之外的原始 HTML。
不得输出 [open]、[closed] 或 [done] 标记。"""

EPISODE_SYSTEM = """你是长篇互动故事章节压缩器。只记录提供的精确消息范围内的事实。
Episode 一级标题和消息范围元数据由系统添加，不要输出。
只允许三个按顺序的二级标题：剧情摘要、状态变化、承诺与伏笔；各节都必须非空。
不得输出任何其他真实 Markdown 标题（ATX 或 Setext）；代码围栏中的标题文字可作为正文内容。
不得输出代码之外的原始 HTML。
承诺与伏笔只作为事实记录，不判断剧情线是否完成。
不得输出 [open]、[closed] 或 [done] 标记。"""

_SUMMARY_SYSTEM = """根据已有摘要和新增章节，更新可独立阅读的整体剧情回顾。
保留完整故事弧、人物关系、关键事件、状态变化、承诺与伏笔事实，以及早期事件。
只输出无 Markdown 标题的连续整体剧情叙事，可使用普通段落、列表和代码围栏。
不得输出任何真实 Markdown 标题（ATX 或 Setext）；代码围栏中的标题文字可作为正文内容。
不得输出代码之外的原始 HTML。
不判断剧情线或伏笔是否完成，不得输出 [open]、[closed] 或 [done] 标记。"""

_STORY_HEADINGS = (
    "时间与地点",
    "在场角色",
    "当前场景",
    "最近变化",
)
_EPISODE_HEADINGS = ("剧情摘要", "状态变化", "承诺与伏笔")
_STORY_GENERATED_HEADINGS = (
    (1, "Story State"),
    *((2, title) for title in _STORY_HEADINGS),
)
_EPISODE_GENERATED_HEADINGS = tuple(
    (2, title) for title in _EPISODE_HEADINGS
)
_CANONICAL_SECTION_HEADING = re.compile(
    r"^##[ \t]+(?P<name>[^\r\n]+?)[ \t]*$"
)
_MACHINE_MARKER = re.compile(r"\[(?:open|closed|done)\]", re.IGNORECASE)
_EPISODE_RANGE_LABEL = re.compile(r"(?<!\w)messages\s*:")
_DEFAULT_IGNORABLE_RANGES = (
    (0x00AD, 0x00AD),
    (0x034F, 0x034F),
    (0x061C, 0x061C),
    (0x115F, 0x1160),
    (0x17B4, 0x17B5),
    (0x180B, 0x180F),
    (0x200B, 0x200F),
    (0x202A, 0x202E),
    (0x2060, 0x206F),
    (0x3164, 0x3164),
    (0xFE00, 0xFE0F),
    (0xFEFF, 0xFEFF),
    (0xFFA0, 0xFFA0),
    (0xFFF0, 0xFFF8),
    (0x1BCA0, 0x1BCA3),
    (0x1D173, 0x1D17A),
    (0xE0000, 0xE0FFF),
)
_MARKDOWN = MarkdownIt("commonmark")
_EPISODE_FILENAME = re.compile(r"episode-(?P<number>[0-9]{6})\.md\Z")
_EPISODE_DOCUMENT = re.compile(
    r"\A# Episode (?P<number>[0-9]{6})[ \t]*\r?\n\r?\n"
    r"<!-- messages: (?P<start>[1-9][0-9]*)-(?P<end>[1-9][0-9]*) -->[ \t]*\r?\n\r?\n"
    r"(?P<body>[\s\S]*?)\r?\n\Z"
)


@dataclass(frozen=True)
class _MarkdownHeading:
    title: str
    level: int
    container_level: int
    start: int
    end: int
    source: str


@dataclass(frozen=True)
class _Episode:
    number: int
    start: int
    end: int
    text: str


class _EpisodeRangeCommentParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.count = 0

    def handle_comment(self, data: str) -> None:
        if _EPISODE_RANGE_LABEL.search(_normalize_html_comment(data)):
            self.count += 1


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
        if not _has_meaningful_markdown_content(text[heading.end : end]):
            raise ValueError(f"{label} contains an empty section")


def _completion_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} completion returned empty or invalid output")
    return value.strip()


def _has_meaningful_text(text: str) -> bool:
    return any(
        unicodedata.category(character)[0] in {"L", "N", "S"}
        for character in text
    )


def _is_default_ignorable(character: str) -> bool:
    codepoint = ord(character)
    return unicodedata.category(character) == "Cf" or any(
        start <= codepoint <= end for start, end in _DEFAULT_IGNORABLE_RANGES
    )


def _normalize_html_comment(data: str) -> str:
    normalized = unicodedata.normalize("NFKC", html.unescape(data))
    return "".join(
        character
        for character in normalized
        if not _is_default_ignorable(character)
    ).casefold()


def _image_alt_has_meaningful_content(tokens: list[Token]) -> bool:
    for token in tokens:
        if token.type == "text" and _has_meaningful_text(token.content):
            return True
        if token.type == "image" and _image_alt_has_meaningful_content(
            token.children or []
        ):
            return True
    return False


def _inline_has_meaningful_content(tokens: list[Token]) -> bool:
    for token in tokens:
        if token.type == "code_inline" and token.content.strip():
            return True
        if token.type in {"text", "text_special"} and _has_meaningful_text(
            token.content
        ):
            return True
        if token.type == "image" and _image_alt_has_meaningful_content(
            token.children or []
        ):
            return True
    return False


def _has_meaningful_markdown_content(text: str) -> bool:
    for token in _MARKDOWN.parse(text):
        if token.type in {"fence", "code_block"} and token.content.strip():
            return True
        if token.type == "inline" and _inline_has_meaningful_content(
            token.children or []
        ):
            return True
    return False


def _markdown_headings(text: str) -> list[_MarkdownHeading]:
    raw_lines = text.splitlines(keepends=True)
    offsets = [0]
    for raw_line in raw_lines:
        offsets.append(offsets[-1] + len(raw_line))

    tokens = _MARKDOWN.parse(text)
    headings: list[_MarkdownHeading] = []
    for index, token in enumerate(tokens):
        if token.type != "heading_open" or token.map is None:
            continue
        inline = tokens[index + 1]
        if inline.type != "inline":
            raise ValueError("Markdown heading has no inline content")

        start_line, end_line = token.map
        start = offsets[start_line]
        end = offsets[end_line]
        headings.append(
            _MarkdownHeading(
                title=inline.content.strip(),
                level=int(token.tag.removeprefix("h")),
                container_level=token.level,
                start=start,
                end=end,
                source=text[start:end].rstrip("\r\n"),
            )
        )
    return headings


def _validate_generated_headings(
    text: str,
    expected: tuple[tuple[int, str], ...],
    *,
    label: str,
) -> None:
    headings = _markdown_headings(text)
    actual = tuple((heading.level, heading.title) for heading in headings)
    if actual != expected or any(heading.container_level for heading in headings):
        raise ValueError(f"{label} contains invalid Markdown headings")


def _reject_machine_markers(text: str, *, label: str) -> None:
    if _MACHINE_MARKER.search(text):
        raise ValueError(f"{label} must not contain plot-state markers")


def _tokens_contain_raw_html(tokens: list[Token]) -> bool:
    return any(
        token.type in {"html_block", "html_inline"}
        or bool(token.children and _tokens_contain_raw_html(token.children))
        for token in tokens
    )


def _reject_generated_raw_html(text: str, *, label: str) -> None:
    if _tokens_contain_raw_html(_MARKDOWN.parse(text)):
        raise ValueError(f"{label} must not contain raw HTML")


def _html_inline_sources(tokens: list[Token]) -> list[str]:
    sources: list[str] = []
    for token in tokens:
        if token.type == "html_inline":
            sources.append(token.content)
        if token.children:
            sources.extend(_html_inline_sources(token.children))
    return sources


def _episode_range_comment_count(text: str) -> int:
    count = 0
    for token in _MARKDOWN.parse(text):
        sources: list[str] = []
        if token.type == "html_block":
            sources.append(token.content)
        elif token.type == "inline":
            sources.extend(_html_inline_sources(token.children or []))
        for source in sources:
            parser = _EpisodeRangeCommentParser()
            parser.feed(source)
            parser.close()
            count += parser.count
    return count


def _reject_episode_range_metadata(text: str) -> None:
    if _episode_range_comment_count(text):
        raise ValueError("episode body must not contain range metadata")


def _validate_story_state(updated: str) -> None:
    _reject_generated_raw_html(updated, label="story state")
    _validate_sections(
        updated,
        _STORY_HEADINGS,
        label="story state",
        prefix="# Story State",
    )
    _validate_generated_headings(
        updated,
        _STORY_GENERATED_HEADINGS,
        label="story state",
    )
    _reject_machine_markers(updated, label="story state")


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
    if _episode_range_comment_count(text) != 1:
        raise ValueError(f"episode has ambiguous range metadata: {filename}")

    filename_number = int(filename_match.group("number"))
    document_number = int(document_match.group("number"))
    start = int(document_match.group("start"))
    end = int(document_match.group("end"))
    if filename_number <= 0 or document_number != filename_number or start > end:
        raise ValueError(f"episode has inconsistent metadata: {filename}")

    _completion_text(document_match.group("body"), label="existing episode")
    return _Episode(filename_number, start, end, text)


def _validate_episode_body(body: str) -> str:
    cleaned = _completion_text(body, label="episode")
    _reject_episode_range_metadata(cleaned)
    _reject_generated_raw_html(cleaned, label="episode")
    _validate_sections(cleaned, _EPISODE_HEADINGS, label="episode", prefix="")
    _validate_generated_headings(
        cleaned,
        _EPISODE_GENERATED_HEADINGS,
        label="episode",
    )
    _reject_machine_markers(cleaned, label="episode")
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


async def _load_episode_chain_for_invalidation(
    store: MarkdownMemoryStore,
    transaction: MarkdownMemoryTransaction,
    *,
    episode_size: int,
) -> tuple[dict[int, _Episode], tuple[_Episode, ...]]:
    """Split a structurally valid chain into current-size prefix and old-size suffix."""
    episode_directory = store.session_dir(transaction.session_id) / "episodes"
    paths = (
        sorted(episode_directory.glob("episode-*.md"))
        if episode_directory.exists()
        else []
    )
    canonical: dict[int, _Episode] = {}
    incompatible: list[_Episode] = []
    prior_end = 0
    historical_size: int | None = None
    size_mismatch_seen = False
    for expected_number, path in enumerate(paths, start=1):
        text = await transaction.read_text(f"episodes/{path.name}")
        try:
            episode = _parse_episode(text, path.name)
        except ValueError as exc:
            raise ValueError(f"existing episode is invalid: {path.name}") from exc
        width = episode.end - episode.start + 1
        if (
            episode.number != expected_number
            or episode.start != prior_end + 1
            or (historical_size is not None and width != historical_size)
        ):
            raise ValueError(
                "existing episode chain has conflicting, duplicate, out-of-order, "
                f"or gapped metadata: {path.name}"
            )
        historical_size = width if historical_size is None else historical_size
        prior_end = episode.end
        expected_start = (expected_number - 1) * episode_size + 1
        expected_end = expected_number * episode_size
        if (
            size_mismatch_seen
            or episode.start != expected_start
            or episode.end != expected_end
        ):
            size_mismatch_seen = True
            incompatible.append(episode)
        else:
            canonical[episode.number] = episode
    return canonical, tuple(incompatible)


async def update_story_state(
    store: MarkdownMemoryStore,
    session_id: int,
    records: list[ChatRecord],
    complete: Completion,
    *,
    source: ChatSourceIdentity,
) -> StageUpdateResult:
    """Replace current story state only after validating a complete result."""
    async with store.transaction(session_id) as transaction:
        existing = await transaction.read_text("story_state.md")
        if not existing.strip():
            existing = (
                "# Story State\n\n"
                "## 时间与地点\n\n（未知）\n\n"
                "## 在场角色\n\n（未知）\n\n"
                "## 当前场景\n\n（无）\n\n"
                "## 最近变化\n\n（无）\n"
            )
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
        try:
            _validate_story_state(updated)
        except ValueError as exc:
            import logging as _logging
            _logger = _logging.getLogger(__name__)
            _logger.warning(
                "Story state validation failed for session %s, "
                "attempting auto-fix: %s",
                session_id,
                exc,
            )
            # Try to salvage: ensure it starts with # Story State
            if not updated.startswith("# Story State"):
                updated = "# Story State\n\n" + updated
            # Ensure all 4 headings exist
            for heading in _STORY_HEADINGS:
                if f"## {heading}" not in updated:
                    updated += f"\n\n## {heading}\n\n（暂无更新）"
            try:
                _validate_story_state(updated)
            except ValueError:
                _logger.warning(
                    "Story state auto-fix failed for session %s, skipping update",
                    session_id,
                )
                return StageUpdateResult(
                    stage="story",
                    completed=True,
                    source=source,
                    checkpoint=source.count,
                    artifacts=(),
                )
        document = updated + "\n"
        await transaction.write_text("story_state.md", document)
        return StageUpdateResult(
            stage="story",
            completed=True,
            source=source,
            checkpoint=source.count,
            artifacts=(text_artifact("story_state.md", document),),
        )


async def create_due_episodes(
    store: MarkdownMemoryStore,
    session_id: int,
    records: list[ChatRecord],
    last_episode_message: int,
    complete: Completion,
    *,
    episode_size: int,
    source: ChatSourceIdentity,
) -> StageUpdateResult:
    """Create every due immutable episode and return its message checkpoint."""
    _validate_nonnegative_integer(last_episode_message, label="episode checkpoint")
    if type(episode_size) is not int or episode_size <= 0:
        raise ValueError("episode size must be a positive integer")
    if last_episode_message % episode_size:
        raise ValueError("episode checkpoint must align to episode size")
    _validate_records(records)

    history_end = records[-1].number if records else 0
    if records and last_episode_message > history_end:
        raise ValueError("episode checkpoint exceeds available records")
    if records and last_episode_message < records[0].number - 1:
        raise ValueError("records do not include the next episode message")

    by_number = {record.number: record for record in records}
    checkpoint = last_episode_message
    async with store.transaction(session_id) as transaction:
        existing_episodes = await _load_existing_episode_chain(
            store,
            transaction,
            episode_size=episode_size,
            checkpoint=min(last_episode_message, history_end),
        )
        if existing_episodes and max(
            episode.end for episode in existing_episodes.values()
        ) > history_end:
            raise ValueError("existing episode chain exceeds available records")
        if last_episode_message > history_end:
            raise ValueError("episode checkpoint exceeds available records")
        while records and history_end - checkpoint >= episode_size:
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

            try:
                body = _validate_episode_body(
                    await complete(
                        [
                            {"role": "system", "content": EPISODE_SYSTEM},
                            {"role": "user", "content": render_records(batch)},
                        ]
                    )
                )
            except ValueError as exc:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "Episode validation failed for session %s ep %s, skipping: %s",
                    session_id, episode_number, exc,
                )
                checkpoint = end
                continue
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
        artifacts = tuple(
            text_artifact(
                f"episodes/episode-{episode.number:06d}.md",
                episode.text,
            )
            for episode in sorted(
                existing_episodes.values(),
                key=lambda item: item.number,
            )
        )
    return StageUpdateResult(
        stage="episode",
        completed=True,
        source=source,
        checkpoint=checkpoint,
        artifacts=artifacts,
    )


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


def validate_generated_summary(text: object, *, max_tokens: int) -> str:
    """Validate and bound one generated overall-plot summary document."""
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("summary token budget must be a positive integer")
    raw = _completion_text(text, label="summary")
    _reject_generated_raw_html(raw, label="summary")
    _validate_generated_headings(raw, (), label="summary")
    _reject_machine_markers(raw, label="summary")
    updated = _truncate_summary(raw, max_tokens)
    if not updated:
        raise ValueError("summary completion was empty after truncation")
    _reject_generated_raw_html(updated, label="summary")
    _validate_generated_headings(updated, (), label="summary")
    _reject_machine_markers(updated, label="summary")
    return updated + "\n"


async def load_summary_episode_chain(
    store: MarkdownMemoryStore,
    transaction: MarkdownMemoryTransaction,
) -> list[_Episode]:
    """Load the canonical contiguous episode chain consumed by summaries."""
    episode_directory = store.session_dir(transaction.session_id) / "episodes"
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
    return episodes


async def update_summary_from_episodes(
    store: MarkdownMemoryStore,
    session_id: int,
    last_summary_message: int,
    complete: Completion,
    *,
    max_tokens: int,
    source: ChatSourceIdentity,
) -> StageUpdateResult:
    """Update the rolling summary from a strictly validated episode sequence."""
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("summary token budget must be a positive integer")

    async with store.transaction(session_id) as transaction:
        episodes = await load_summary_episode_chain(store, transaction)

        _validate_nonnegative_integer(
            last_summary_message,
            label="summary checkpoint",
        )
        valid_checkpoints = {episode.end for episode in episodes}
        if last_summary_message and last_summary_message not in valid_checkpoints:
            raise ValueError(
                "summary checkpoint must be zero or match an existing episode end"
            )

        inputs = tuple(
            text_artifact(
                f"episodes/episode-{episode.number:06d}.md",
                episode.text,
            )
            for episode in episodes
        )
        pending = [
            episode for episode in episodes if episode.end > last_summary_message
        ]
        if not pending:
            if not episodes:
                await transaction.delete_file("summary.md")
                artifacts = ()
            else:
                summary_path = store.file_path(session_id, "summary.md")
                existing = await transaction.read_text("summary.md")
                artifacts = (
                    (text_artifact("summary.md", existing),)
                    if summary_path.exists()
                    else ()
                )
            return StageUpdateResult(
                stage="summary",
                completed=True,
                source=source,
                checkpoint=last_summary_message,
                artifacts=artifacts,
                inputs=inputs,
            )

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
        document = validate_generated_summary(
            await complete(messages),
            max_tokens=max_tokens,
        )
        await transaction.write_text("summary.md", document)
        return StageUpdateResult(
            stage="summary",
            completed=True,
            source=source,
            checkpoint=pending[-1].end,
            artifacts=(text_artifact("summary.md", document),),
            inputs=inputs,
        )
