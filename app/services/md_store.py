"""Safe, atomic Markdown storage primitives for session memory."""

from __future__ import annotations

import asyncio
import datetime as dt
import os
import re
import tempfile
import unicodedata
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from weakref import WeakValueDictionary

import aiofiles


@dataclass(frozen=True)
class ChatRecord:
    number: int
    role: str
    content: str
    timestamp: str
    name: str
    msg_type: str = "ic"


@dataclass(frozen=True)
class MemoryState:
    schema_version: int = 2
    last_memory_message: int = 0
    last_story_state_message: int = 0
    last_summary_message: int = 0
    last_episode_message: int = 0
    last_rag_message: int = 0
    last_character_message: int = 0
    last_assets_message: int = 0
    last_error: str = ""


_TIMESTAMP_PATTERN = r"(?:\d{4}-\d{2}-\d{2} \d{2}:\d{2}|unknown)"
_VALID_TIMESTAMP = re.compile(rf"{_TIMESTAMP_PATTERN}\Z")
_CHAT_HEADER = re.compile(
    rf"^## \[(?P<timestamp>{_TIMESTAMP_PATTERN})\] "
    r"(?P<name>(?:(?! \(\(OOC\)\)| <!-- (?:role|content-length):).)+?)"
    r"(?P<ooc> \(\(OOC\)\))?"
    r"(?: <!-- role:(?P<role>user|assistant|system) -->"
    r"(?: <!-- content-length:(?P<content_length>\d+) -->)?"
    r")?[ \t]*\r?$",
    re.MULTILINE,
)

_CHECKPOINT_FIELDS = {
    "last-memory-message": "last_memory_message",
    "last-story-state-message": "last_story_state_message",
    "last-summary-message": "last_summary_message",
    "last-episode-message": "last_episode_message",
    "last-rag-message": "last_rag_message",
    "last-character-message": "last_character_message",
    "last-assets-message": "last_assets_message",
}
_STATE_NUMBER = r"(?:0|[1-9]\d*)"
_MEMORY_STATE_PREFIX = re.compile(
    r"\A# Memory State\n\n"
    rf"<!-- schema-version: (?P<schema_version>{_STATE_NUMBER}) -->\n"
    rf"<!-- last-memory-message: (?P<last_memory_message>{_STATE_NUMBER}) -->\n"
    rf"<!-- last-story-state-message: (?P<last_story_state_message>{_STATE_NUMBER}) -->\n"
    rf"<!-- last-summary-message: (?P<last_summary_message>{_STATE_NUMBER}) -->\n"
    rf"<!-- last-episode-message: (?P<last_episode_message>{_STATE_NUMBER}) -->\n"
    rf"<!-- last-rag-message: (?P<last_rag_message>{_STATE_NUMBER}) -->\n"
    rf"<!-- last-character-message: (?P<last_character_message>{_STATE_NUMBER}) -->\n"
    rf"<!-- last-assets-message: (?P<last_assets_message>{_STATE_NUMBER}) -->\n\n"
    r"## Last Error\n\n"
    rf"<!-- last-error-length: (?P<last_error_length>{_STATE_NUMBER}) -->\n"
)

_SAFE_CHARACTER_NAME = re.compile(r"[\w\u3400-\u9fff .-]{1,120}")
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "CONIN$",
    "CONOUT$",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
    *(f"COM{number}" for number in "¹²³"),
    *(f"LPT{number}" for number in "¹²³"),
}
_WINDOWS_INVALID_CHARACTERS = frozenset('<>:"|?*')


def _contains_control_character(value: str) -> bool:
    return any(unicodedata.category(character) == "Cc" for character in value)


def _validate_session_id(session_id: int) -> None:
    if (
        isinstance(session_id, bool)
        or not isinstance(session_id, int)
        or session_id <= 0
    ):
        raise ValueError("session_id must be a positive integer")


def _is_windows_reserved(component: str) -> bool:
    stem = component.split(".", 1)[0].rstrip(" .").upper()
    return stem in _WINDOWS_RESERVED_NAMES


def _validate_path_component(component: str) -> None:
    if (
        not component
        or component in {".", ".."}
        or any(character in _WINDOWS_INVALID_CHARACTERS for character in component)
        or component.endswith((".", " "))
        or _contains_control_character(component)
        or _is_windows_reserved(component)
    ):
        raise ValueError("file path must be a safe relative path")


def _safe_relative_parts(relative: str | Path) -> tuple[str, ...]:
    if not isinstance(relative, (str, Path)):
        raise ValueError("file path must be a safe relative path")
    raw = str(relative)
    windows_path = PureWindowsPath(raw)
    posix_path = PurePosixPath(raw)
    if (
        not raw
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or posix_path.is_absolute()
    ):
        raise ValueError("file path must be a safe relative path")

    parts = tuple(re.split(r"[\\/]", raw))
    for component in parts:
        _validate_path_component(component)
    return parts


def _validate_chat_name(name: str) -> None:
    if (
        not isinstance(name, str)
        or not 1 <= len(name) <= 120
        or name != name.strip()
        or "<!--" in name
        or "-->" in name
        or "((OOC))" in name
        or _contains_control_character(name)
    ):
        raise ValueError("unsafe chat record name")


def _validate_record(record: ChatRecord) -> None:
    _validate_chat_name(record.name)
    if record.role not in {"user", "assistant", "system"}:
        raise ValueError("invalid chat record role")
    if record.msg_type not in {"ic", "ooc"}:
        raise ValueError("invalid chat record message type")
    if record.msg_type == "ooc" and record.role != "user":
        raise ValueError("invalid chat record role/message type combination")
    if not _VALID_TIMESTAMP.fullmatch(record.timestamp):
        raise ValueError("invalid chat record timestamp")
    if not isinstance(record.content, str):
        raise ValueError("invalid chat record content")


def _validate_memory_state(state: MemoryState) -> None:
    if type(state.schema_version) is not int or state.schema_version != 2:
        raise ValueError("invalid memory state schema version")
    for field_name in _CHECKPOINT_FIELDS.values():
        value = getattr(state, field_name)
        if type(value) is not int or value < 0:
            raise ValueError("invalid memory state checkpoint")
    if not isinstance(state.last_error, str):
        raise ValueError("invalid memory state error")


class MarkdownMemoryStore:
    """Filesystem-backed Markdown store with atomic session transactions.

    Public methods acquire the per-session transaction. Code that already holds a
    transaction must call methods on the yielded ``MarkdownMemoryTransaction`` to
    avoid nested acquisition of the same non-reentrant lock.
    """

    def __init__(self, base: Path = Path("data/memory")) -> None:
        self.base = Path(base).resolve()
        self._locks: WeakValueDictionary[int, asyncio.Lock] = WeakValueDictionary()
        self._locks_guard = asyncio.Lock()
        self._pipeline_locks: WeakValueDictionary[int, asyncio.Lock] = (
            WeakValueDictionary()
        )
        self._pipeline_locks_guard = asyncio.Lock()

    async def lock_for(self, session_id: int) -> asyncio.Lock:
        """Return the stable lock while any caller retains it."""
        _validate_session_id(session_id)
        async with self._locks_guard:
            lock = self._locks.get(session_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[session_id] = lock
            return lock

    async def pipeline_lock_for(self, session_id: int) -> asyncio.Lock:
        """Return a shared lock for whole pipeline operations.

        This lock is separate from the non-reentrant transaction lock so stages
        may safely call helpers that open their own short store transactions.
        """
        _validate_session_id(session_id)
        async with self._pipeline_locks_guard:
            lock = self._pipeline_locks.get(session_id)
            if lock is None:
                lock = asyncio.Lock()
                self._pipeline_locks[session_id] = lock
            return lock

    @asynccontextmanager
    async def transaction(
        self,
        session_id: int,
    ) -> AsyncIterator[MarkdownMemoryTransaction]:
        """Own one session lock and expose non-locking operations within it."""
        lock = await self.lock_for(session_id)
        async with lock:
            yield MarkdownMemoryTransaction(self, session_id)

    def session_dir(self, session_id: int) -> Path:
        _validate_session_id(session_id)
        path = (self.base / str(session_id)).resolve()
        try:
            relative = path.relative_to(self.base)
        except ValueError as exc:
            raise ValueError("session path escaped memory base") from exc
        if not relative.parts:
            raise ValueError("session path escaped memory base")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def file_path(self, session_id: int, relative: str | Path) -> Path:
        """Resolve a portable safe path and guarantee session containment."""
        parts = _safe_relative_parts(relative)
        directory = self.session_dir(session_id)
        path = directory.joinpath(*parts).resolve()
        try:
            path.relative_to(directory)
        except ValueError as exc:
            raise ValueError("file path must be a safe relative path") from exc
        if path == directory:
            raise ValueError("file path must be a safe relative path")
        return path

    def character_path(self, session_id: int, name: str) -> Path:
        """Return a validated safe path below the character directory."""
        if (
            not isinstance(name, str)
            or name != name.strip()
            or not _SAFE_CHARACTER_NAME.fullmatch(name)
            or name in {".", ".."}
            or name.endswith((".", " "))
        ):
            raise ValueError("unsafe character profile name")
        try:
            _validate_path_component(f"{name}.md")
        except ValueError as exc:
            raise ValueError("unsafe character profile name") from exc

        session_directory = self.session_dir(session_id)
        directory = (session_directory / "characters").resolve()
        try:
            directory.relative_to(session_directory)
        except ValueError as exc:
            raise ValueError("character path escaped profile directory") from exc
        directory.mkdir(parents=True, exist_ok=True)

        path = (directory / f"{name}.md").resolve()
        try:
            path.relative_to(directory)
        except ValueError as exc:
            raise ValueError("character path escaped profile directory") from exc
        if path.parent != directory:
            raise ValueError("character path escaped profile directory")
        return path

    async def _read_text_unlocked(
        self,
        session_id: int,
        relative: str | Path,
    ) -> str:
        path = self.file_path(session_id, relative)
        if not path.exists():
            return ""
        async with aiofiles.open(
            path,
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            return await handle.read()

    async def _write_text_unlocked(
        self,
        session_id: int,
        relative: str | Path,
        text: str,
    ) -> None:
        target = self.file_path(session_id, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, raw_temp = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
        os.close(file_descriptor)
        temp = Path(raw_temp)
        try:
            async with aiofiles.open(
                temp,
                "w",
                encoding="utf-8",
                newline="",
            ) as handle:
                await handle.write(text)
                await handle.flush()
            os.replace(temp, target)
        finally:
            temp.unlink(missing_ok=True)

    async def _append_records_unlocked(
        self,
        session_id: int,
        records: list[ChatRecord],
    ) -> None:
        existing = await self._read_text_unlocked(session_id, "chat.md")
        if existing:
            parse_chat_markdown(existing)
        rendered = render_chat_records(records)
        separator = "" if not existing or existing.endswith("\n") else "\n"
        await self._write_text_unlocked(
            session_id,
            "chat.md",
            f"{existing}{separator}{rendered}",
        )

    async def read_text(self, session_id: int, relative: str | Path) -> str:
        async with self.transaction(session_id) as transaction:
            return await transaction.read_text(relative)

    async def write_text(
        self,
        session_id: int,
        relative: str | Path,
        text: str,
    ) -> None:
        async with self.transaction(session_id) as transaction:
            await transaction.write_text(relative, text)

    async def append_record(
        self,
        session_id: int,
        role: str,
        content: str,
        *,
        name: str,
        msg_type: str = "ic",
    ) -> None:
        async with self.transaction(session_id) as transaction:
            await transaction.append_record(
                role,
                content,
                name=name,
                msg_type=msg_type,
            )

    async def append_pair(
        self,
        session_id: int,
        user_text: str,
        assistant_text: str,
        *,
        char_name: str,
        user_name: str,
        msg_type: str = "ic",
    ) -> None:
        async with self.transaction(session_id) as transaction:
            await transaction.append_pair(
                user_text,
                assistant_text,
                char_name=char_name,
                user_name=user_name,
                msg_type=msg_type,
            )

    async def load_chat(self, session_id: int) -> list[ChatRecord]:
        async with self.transaction(session_id) as transaction:
            return await transaction.load_chat()

    async def load_state(self, session_id: int) -> MemoryState:
        async with self.transaction(session_id) as transaction:
            return await transaction.load_state()

    async def save_state(self, session_id: int, state: MemoryState) -> None:
        async with self.transaction(session_id) as transaction:
            await transaction.save_state(state)


class MarkdownMemoryTransaction:
    """Non-locking session operations available only to a lock owner."""

    def __init__(self, store: MarkdownMemoryStore, session_id: int) -> None:
        self._store = store
        self.session_id = session_id

    async def read_text(self, relative: str | Path) -> str:
        return await self._store._read_text_unlocked(self.session_id, relative)

    async def write_text(self, relative: str | Path, text: str) -> None:
        await self._store._write_text_unlocked(self.session_id, relative, text)

    async def append_record(
        self,
        role: str,
        content: str,
        *,
        name: str,
        msg_type: str = "ic",
    ) -> None:
        now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M")
        record = ChatRecord(0, role, content, now, name, msg_type)
        await self._store._append_records_unlocked(self.session_id, [record])

    async def append_pair(
        self,
        user_text: str,
        assistant_text: str,
        *,
        char_name: str,
        user_name: str,
        msg_type: str = "ic",
    ) -> None:
        now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M")
        records = [
            ChatRecord(0, "user", user_text, now, user_name, msg_type),
            ChatRecord(0, "assistant", assistant_text, now, char_name),
        ]
        await self._store._append_records_unlocked(self.session_id, records)

    async def load_chat(self) -> list[ChatRecord]:
        text = await self._store._read_text_unlocked(self.session_id, "chat.md")
        return parse_chat_markdown(text)

    async def load_state(self) -> MemoryState:
        text = await self._store._read_text_unlocked(
            self.session_id,
            "memory_state.md",
        )
        return parse_memory_state(text)

    async def save_state(self, state: MemoryState) -> None:
        await self._store._write_text_unlocked(
            self.session_id,
            "memory_state.md",
            render_memory_state(state),
        )


def _content_start(text: str, header_end: int) -> int:
    if header_end == len(text):
        return header_end
    if text.startswith("\n", header_end):
        return header_end + 1
    raise ValueError("chat.md contains an invalid header line ending")


def _consume_framed_separator(text: str, content_end: int) -> int:
    if content_end == len(text):
        return content_end
    if text.startswith("\r\n", content_end):
        return content_end + 2
    if text.startswith("\n", content_end):
        return content_end + 1
    raise ValueError("chat.md contains invalid content framing")


def parse_chat_markdown(text: str) -> list[ChatRecord]:
    """Parse lossless framed records and compatible boundary-delimited V1."""
    if not text.strip():
        return []

    first = _CHAT_HEADER.search(text)
    if first is None or text[: first.start()].strip():
        raise ValueError("chat.md contains an invalid preamble or header")

    records: list[ChatRecord] = []
    cursor = first.start()
    while cursor < len(text):
        match = _CHAT_HEADER.match(text, cursor)
        if match is None:
            raise ValueError("chat.md contains invalid content framing")
        start = _content_start(text, match.end())
        content_length = match.group("content_length")

        if content_length is not None:
            end = start + int(content_length)
            if end > len(text):
                raise ValueError("chat.md contains invalid content framing")
            content = text[start:end]
            cursor = _consume_framed_separator(text, end)
            if cursor < len(text) and _CHAT_HEADER.match(text, cursor) is None:
                raise ValueError("chat.md contains invalid content framing")
        else:
            next_match = _CHAT_HEADER.search(text, start)
            end = next_match.start() if next_match is not None else len(text)
            content = text[start:end].strip()
            cursor = end

        role = match.group("role")
        if role is None:
            role = "system" if match.group("name") == "System" else "user"
        msg_type = "ooc" if match.group("ooc") else "ic"
        if msg_type == "ooc" and role != "user":
            raise ValueError("invalid chat record role/message type combination")
        records.append(
            ChatRecord(
                number=len(records) + 1,
                role=role,
                content=content,
                timestamp=match.group("timestamp"),
                name=match.group("name"),
                msg_type=msg_type,
            )
        )
    return records


def render_chat_records(records: list[ChatRecord]) -> str:
    """Render exact content with a readable character-count frame."""
    parts: list[str] = []
    for record in records:
        _validate_record(record)
        ooc = " ((OOC))" if record.role == "user" and record.msg_type == "ooc" else ""
        parts.append(
            f"## [{record.timestamp}] {record.name}{ooc} <!-- role:{record.role} --> "
            f"<!-- content-length:{len(record.content)} -->\n"
            f"{record.content}\n"
        )
    return "".join(parts)


def parse_memory_state(text: str) -> MemoryState:
    """Parse a complete V2 state, defaulting only absent/empty V1 state."""
    if not text.strip():
        return MemoryState()
    match = _MEMORY_STATE_PREFIX.match(text)
    if match is None:
        raise ValueError("invalid memory state structure")

    error_length = int(match.group("last_error_length"))
    error_start = match.end()
    if error_length == 0:
        if text[error_start:] != "(none)\n":
            raise ValueError("invalid memory state error frame")
        error = ""
    else:
        error_end = error_start + error_length
        if error_end > len(text) or text[error_end:] != "\n":
            raise ValueError("invalid memory state error frame")
        error = text[error_start:error_end]

    values: dict[str, int | str] = {
        "schema_version": int(match.group("schema_version")),
        **{
            field_name: int(match.group(field_name))
            for field_name in _CHECKPOINT_FIELDS.values()
        },
        "last_error": error,
    }
    state = MemoryState(**values)
    _validate_memory_state(state)
    return state


def render_memory_state(state: MemoryState) -> str:
    """Render all validated state fields as Markdown comments."""
    _validate_memory_state(state)
    displayed_error = state.last_error or "(none)"
    return (
        "# Memory State\n\n"
        f"<!-- schema-version: {state.schema_version} -->\n"
        f"<!-- last-memory-message: {state.last_memory_message} -->\n"
        f"<!-- last-story-state-message: {state.last_story_state_message} -->\n"
        f"<!-- last-summary-message: {state.last_summary_message} -->\n"
        f"<!-- last-episode-message: {state.last_episode_message} -->\n"
        f"<!-- last-rag-message: {state.last_rag_message} -->\n"
        f"<!-- last-character-message: {state.last_character_message} -->\n"
        f"<!-- last-assets-message: {state.last_assets_message} -->\n\n"
        "## Last Error\n\n"
        f"<!-- last-error-length: {len(state.last_error)} -->\n"
        f"{displayed_error}\n"
    )
