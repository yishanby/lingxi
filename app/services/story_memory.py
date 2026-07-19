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

STORY_STATE_SYSTEM = """你是长篇互动故事状态维护器。输出完整的 Story State Markdown，而不是增量补丁。
必须保留没有明确完成的 [open] 剧情线；只有新增对话中有明确完成证据时才能移除。
只根据对话证据更新，不得猜测或补造事实。
固定标题且按此顺序输出：时间与地点、在场角色、当前场景、未完成剧情线、最近变化。"""

EPISODE_SYSTEM = """你是长篇互动故事章节压缩器。只总结提供的精确消息范围，不得引用范围外内容。
输出完整 Markdown，固定标题且按此顺序输出：剧情摘要、状态变化、承诺与伏笔。
所有尚未解决的承诺、任务和伏笔必须使用 [open]。"""

_SUMMARY_SYSTEM = """更新完整故事回顾，只根据已有摘要和新增章节写作。
保留人物关系、关键事件、状态变化和所有未完成伏笔；输出可独立阅读的完整 Markdown。"""

_STORY_HEADINGS = (
    "时间与地点",
    "在场角色",
    "当前场景",
    "未完成剧情线",
    "最近变化",
)
_EPISODE_HEADINGS = ("剧情摘要", "状态变化", "承诺与伏笔")
_HEADING = re.compile(r"^##[ \t]+(?P<name>[^\r\n]+?)[ \t]*\r?$", re.MULTILINE)
_OPEN_THREAD = re.compile(
    r"^[ \t]*-[ \t]*\[open\][ \t]+(?P<text>.+?)[ \t]*\r?$",
    re.MULTILINE,
)
_EPISODE_FILENAME = re.compile(r"episode-(?P<number>\d{6})\.md\Z")
_EPISODE_DOCUMENT = re.compile(
    r"\A# Episode (?P<number>\d{6})[ \t]*\r?\n\r?\n"
    r"<!-- messages: (?P<start>[1-9]\d*)-(?P<end>[1-9]\d*) -->[ \t]*\r?\n\r?\n"
    r"(?P<body>[\s\S]*?)\r?\n\Z"
)
_ENGLISH_WORD = re.compile(r"[a-z0-9]+")
_ENGLISH_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "for",
        "from",
        "in",
        "into",
        "of",
        "on",
        "the",
        "to",
        "with",
    }
)
_ENGLISH_RESULTS: dict[str, str] = {
    "find": r"\b(?:found|located|discovered)\b",
    "locate": r"\b(?:located|found|discovered)\b",
    "retrieve": r"\b(?:retrieved|recovered|obtained)\b",
    "get": r"\b(?:got|obtained|acquired|retrieved)\b",
    "obtain": r"\b(?:obtained|acquired|retrieved)\b",
    "rescue": r"\b(?:rescued|saved|freed)\b",
    "save": r"\b(?:saved|rescued|freed)\b",
    "defeat": r"\b(?:defeated|beat|vanquished)\b",
    "solve": r"\b(?:solved|resolved)\b",
    "complete": r"\b(?:completed|finished)\b",
    "finish": r"\b(?:finished|completed)\b",
    "return": r"\breturned\b",
    "deliver": r"\bdelivered\b",
    "unlock": r"\b(?:unlocked|opened)\b",
    "open": r"\bopened\b",
    "reach": r"\b(?:reached|arrived)\b",
    "arrive": r"\barrived\b",
    "escape": r"\bescaped\b",
    "fulfill": r"\bfulfilled\b",
}
_ENGLISH_GENERIC_RESULT = re.compile(
    r"\b(?:completed|finished|resolved|fulfilled|accomplished|succeeded)\b",
    re.IGNORECASE,
)
_ENGLISH_DISQUALIFIER = re.compile(
    r"(?:\bnot\b|\bnever\b|haven't|hasn't|hadn't|didn't|failed[ \t]+to|"
    r"\bwill\b|going[ \t]+to|plan(?:s|ned)?[ \t]+to|intend(?:s|ed)?[ \t]+to|"
    r"want(?:s|ed)?[ \t]+to|need(?:s|ed)?[ \t]+to|\bmust\b|\bcould\b|"
    r"\bmight\b|\bmay\b|\bperhaps\b|\bmaybe\b|\bpossibly\b)",
    re.IGNORECASE,
)
_ENGLISH_HEARSAY = re.compile(
    r"\b(?:rumou?r|reportedly|allegedly|apparently|hearsay)\b|"
    r"\baccording[ \t]+to\b|\bit[ \t]+is[ \t]+said\b|\bheard[ \t]+that\b|"
    r"\b(?:witness|source|someone|somebody)[ \t]+(?:claim(?:s|ed)?|say(?:s|said)?)\b",
    re.IGNORECASE,
)
_CHINESE_RESULTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("找到", "寻找", "寻回"), r"(?:找到了?|寻得了?|发现了?|寻回了?)"),
    (("拿到", "获得", "取得"), r"(?:拿到了?|获得了?|取得了?)"),
    (("解决", "解开", "破解"), r"(?:解决了?|解开了?|破解了?)"),
    (("营救", "救出", "解救"), r"(?:营救成功|救出了?|解救了?)"),
    (("击败", "打败"), r"(?:击败了?|打败了?)"),
    (("完成",), r"完成了?"),
    (("兑现", "履行"), r"(?:兑现了?|履行了?)"),
    (("达成",), r"达成了?"),
    (("归还", "交付", "交给"), r"(?:归还了?|交付了?|交给了?)"),
    (("开启", "打开"), r"(?:开启了?|打开了?)"),
    (("到达", "抵达"), r"(?:到达了?|抵达了?)"),
    (("逃离",), r"逃离了?"),
)
_CHINESE_GENERIC_RESULT = re.compile(r"(?:完成了?|解决了?|达成了?|兑现了?|成功了?)")
_CHINESE_DISQUALIFIER = re.compile(
    r"(?:未|没|没有|尚未|仍未|还未|还没|无法|不能|并未|"
    r"准备|计划|打算|将要|将|要|会|想要|希望|发誓|承诺|"
    r"明天|未来|可能|也许|或许|似乎|好像)"
)
_CHINESE_HEARSAY = re.compile(
    r"(?:据传|传言|传闻|听说|据说|据称|相传|"
    r"有人(?:说|声称)|消息称|目击者声称)"
)
_CHINESE_RESULT_MARKER = re.compile(
    r"(?:终于|已经|已|成功|最终|总算|确实|现已)"
)
_SENTENCE = re.compile(r"[^。！？!?；;\r\n]+")
_ENGLISH_FORWARD_EVENT_GAP = re.compile(
    r"\A\s*(?:(?:the|a|an)\s+)?\Z|"
    r"\A\s*,\s*(?:(?:beneath|under|inside|within|behind|beside|near|at|in|on|"
    r"from)\b[^,.;!?]{0,48}|(?:finally|successfully|there))\s*,\s*"
    r"(?:(?:the|a|an)\s+)?\Z",
    re.IGNORECASE,
)
_ENGLISH_REVERSE_EVENT_GAP = re.compile(
    r"\A\s*(?:(?:was|has[ \t]+been|had[ \t]+been)[ \t]+)?"
    r"(?:(?:finally|successfully)[ \t]+)?\Z",
    re.IGNORECASE,
)
_ENGLISH_REPRESENTATION_PREFIX = re.compile(
    r"\b(?:picture|image|photo|portrait|drawing|model|replica|copy|fake|"
    r"counterfeit)(?:[ \t]+of)?(?:[ \t]+the)?[ \t]*\Z",
    re.IGNORECASE,
)
_ENGLISH_REPRESENTATION_SUFFIX = re.compile(
    r"\A[ \t]+(?:replica|copy|model|image|picture|portrait|photo|drawing|"
    r"fake|counterfeit|facsimile)\b",
    re.IGNORECASE,
)
_CHINESE_FORWARD_EVENT_GAP = re.compile(
    r"\A(?:那把|这把|一把)?\Z|"
    r"\A(?:藏在|位于|放在|留在|埋在|掉在|躲在).{1,16}的"
    r"(?:那把|这把|一把)?\Z"
)
_CHINESE_REVERSE_EVENT_GAP = re.compile(
    r"\A(?:终于|已经|已|成功|最终|总算|确实|现已)*\Z"
)
_CHINESE_REPRESENTATION_PREFIX = re.compile(
    r"(?:假的|仿制的|复制的|画像中的|照片中的|图片中的|模型中的)\Z"
)
_CHINESE_REPRESENTATION_SUFFIX = re.compile(
    r"\A(?:的)?(?:画像|照片|图像|图片|肖像|复制品|副本|仿制品|模型|假货|赝品)"
)


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
    matches = list(_HEADING.finditer(text))
    if [match.group("name") for match in matches] != list(expected):
        raise ValueError(f"{label} must contain every required heading in order")
    if not matches or text[: matches[0].start()].strip() != prefix:
        raise ValueError(f"{label} has an invalid or incomplete preamble")
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        if not text[match.end() : end].strip():
            raise ValueError(f"{label} contains an empty section")


def _completion_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} completion returned empty or invalid output")
    return value.strip()


def _section(text: str, heading: str) -> str:
    match = re.search(rf"^##[ \t]+{re.escape(heading)}[ \t]*\r?$", text, re.MULTILINE)
    if match is None:
        return ""
    following = _HEADING.search(text, match.end())
    end = following.start() if following is not None else len(text)
    return text[match.end() : end]


def _open_threads(text: str) -> set[str]:
    return {
        match.group("text").strip()
        for match in _OPEN_THREAD.finditer(_section(text, "未完成剧情线"))
    }


def _sentence_is_question(content: str, match: re.Match[str]) -> bool:
    return match.end() < len(content) and content[match.end()] in "?？"


def _english_target(thread: str) -> re.Pattern[str] | None:
    words = _ENGLISH_WORD.findall(thread.casefold())
    keywords = [
        word
        for word in words
        if word not in _ENGLISH_STOP_WORDS and word not in _ENGLISH_RESULTS
    ]
    if not keywords:
        return None
    joiner = r"(?:[ \t-]+)"
    pattern = joiner.join(
        rf"\b{re.escape(keyword)}(?:s|es)?\b" for keyword in keywords
    )
    return re.compile(pattern, re.IGNORECASE)


def _english_target_is_representation(
    sentence: str,
    target: re.Match[str],
) -> bool:
    prefix = sentence[max(0, target.start() - 48) : target.start()]
    suffix = sentence[target.end() : target.end() + 24]
    return bool(
        _ENGLISH_REPRESENTATION_PREFIX.search(prefix)
        or _ENGLISH_REPRESENTATION_SUFFIX.search(suffix)
    )


def _english_event_is_bound(
    sentence: str,
    result: re.Match[str],
    target: re.Match[str],
) -> bool:
    if _english_target_is_representation(sentence, target):
        return False
    if result.end() <= target.start():
        gap = sentence[result.end() : target.start()]
        return _ENGLISH_FORWARD_EVENT_GAP.fullmatch(gap) is not None
    if target.end() <= result.start():
        gap = sentence[target.end() : result.start()]
        return _ENGLISH_REVERSE_EVENT_GAP.fullmatch(gap) is not None
    return False


def _english_natural_completion(thread: str, content: str) -> bool:
    words = _ENGLISH_WORD.findall(thread.casefold())
    actions = [word for word in words if word in _ENGLISH_RESULTS]
    target = _english_target(thread)
    if target is None:
        return False
    result = (
        re.compile(
            "|".join(_ENGLISH_RESULTS[action] for action in actions),
            re.IGNORECASE,
        )
        if actions
        else _ENGLISH_GENERIC_RESULT
    )

    for sentence_match in _SENTENCE.finditer(content):
        sentence = sentence_match.group(0)
        if _sentence_is_question(content, sentence_match):
            continue
        if _ENGLISH_HEARSAY.search(sentence) or _ENGLISH_DISQUALIFIER.search(sentence):
            continue
        if any(
            _english_event_is_bound(sentence, result_match, target_match)
            for result_match in result.finditer(sentence)
            for target_match in target.finditer(sentence)
        ):
            return True
    return False


def _compact_chinese(value: str) -> str:
    return "".join(re.findall(r"[\u3400-\u9fff]+", value))


def _chinese_target_is_representation(
    sentence: str,
    target: re.Match[str],
) -> bool:
    prefix = sentence[max(0, target.start() - 12) : target.start()]
    suffix = sentence[target.end() : target.end() + 8]
    return bool(
        _CHINESE_REPRESENTATION_PREFIX.search(prefix)
        or _CHINESE_REPRESENTATION_SUFFIX.search(suffix)
    )


def _chinese_event_is_bound(
    sentence: str,
    result: re.Match[str],
    target: re.Match[str],
) -> bool:
    if _chinese_target_is_representation(sentence, target):
        return False
    if result.end() <= target.start():
        gap = sentence[result.end() : target.start()]
        return _CHINESE_FORWARD_EVENT_GAP.fullmatch(gap) is not None
    if target.end() <= result.start():
        gap = sentence[target.end() : result.start()]
        return _CHINESE_REVERSE_EVENT_GAP.fullmatch(gap) is not None
    return False


def _chinese_natural_completion(thread: str, content: str) -> bool:
    selected: list[tuple[tuple[str, ...], str]] = [
        item
        for item in _CHINESE_RESULTS
        if any(action in thread for action in item[0])
    ]
    actions = tuple(action for item in selected for action in item[0])
    if actions:
        pieces = re.split("|".join(re.escape(action) for action in actions), thread)
        result = re.compile("|".join(item[1] for item in selected))
    else:
        pieces = [thread]
        result = _CHINESE_GENERIC_RESULT

    targets = []
    for piece in pieces:
        normalized = _compact_chinese(piece)
        normalized = re.sub(r"^(?:请|要|去|把|将|让|一个|那把|这把)+", "", normalized)
        normalized = re.sub(r"(?:任务|目标)$", "", normalized)
        if len(normalized) >= 2:
            targets.append(normalized)
    if len(targets) != 1:
        return False
    target = re.compile(re.escape(targets[0]))

    for sentence_match in _SENTENCE.finditer(content):
        raw_sentence = sentence_match.group(0)
        if _sentence_is_question(content, sentence_match):
            continue
        if _CHINESE_HEARSAY.search(raw_sentence):
            continue
        sentence = _compact_chinese(raw_sentence)
        if _CHINESE_DISQUALIFIER.search(sentence):
            continue
        for result_match in result.finditer(sentence):
            result_text = result_match.group(0)
            prefix = sentence[: result_match.start()]
            if "了" not in result_text and "成功" not in result_text:
                if not _CHINESE_RESULT_MARKER.search(prefix):
                    continue
            if any(
                _chinese_event_is_bound(sentence, result_match, target_match)
                for target_match in target.finditer(sentence)
            ):
                return True
    return False


def _has_explicit_closure(thread: str, records: list[ChatRecord]) -> bool:
    escaped = re.escape(thread)
    patterns = (
        rf"^[ \t]*(?:-[ \t]*)?\[(?:closed|done)\][ \t]+{escaped}[ \t]*$",
        rf"^[ \t]*(?:已完成|已解决|完成|解决)[：:][ \t]*{escaped}[ \t]*$",
    )
    for record in records:
        if any(
            re.search(pattern, record.content, re.IGNORECASE | re.MULTILINE)
            for pattern in patterns
        ):
            return True
        if _english_natural_completion(thread, record.content):
            return True
        if _chinese_natural_completion(thread, record.content):
            return True
    return False


def _validate_story_state(
    updated: str,
    existing: str,
    records: list[ChatRecord],
) -> None:
    _validate_sections(
        updated,
        _STORY_HEADINGS,
        label="story state",
        prefix="# Story State",
    )
    retained = _open_threads(updated)
    for thread in _open_threads(existing):
        if thread not in retained and not _has_explicit_closure(thread, records):
            raise ValueError(f"story state removed an unresolved open thread: {thread}")


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
        _validate_story_state(updated, existing, records)
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
        updated = _truncate_summary(raw, max_tokens)
        if not updated:
            raise ValueError("summary completion was empty after truncation")
        await transaction.write_text("summary.md", updated + "\n")
        return pending[-1].end
