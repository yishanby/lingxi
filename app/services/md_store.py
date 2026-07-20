"""Safe, atomic Markdown storage primitives for session memory."""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import unicodedata
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from weakref import WeakValueDictionary

import aiofiles


logger = logging.getLogger(__name__)


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
    cleanup_required: bool = False
    rebuild_story_required: bool = False
    rebuild_memory_required: bool = False
    rebuild_episode_required: bool = False
    rebuild_summary_required: bool = False
    rebuild_rag_required: bool = False
    rebuild_assets_required: bool = False
    rebuild_from_message: int | None = None
    last_error: str = ""

    @property
    def rebuild_required(self) -> bool:
        return self.cleanup_required or any(
            (
                self.rebuild_story_required,
                self.rebuild_memory_required,
                self.rebuild_episode_required,
                self.rebuild_summary_required,
                self.rebuild_rag_required,
                self.rebuild_assets_required,
            )
        )

    def checkpoint_exceeds(self, total: int) -> bool:
        if type(total) is not int or total < 0:
            raise ValueError("total must be a nonnegative integer")
        return any(
            getattr(self, field_name) > total
            for field_name in _CHECKPOINT_FIELDS.values()
        )


@dataclass(frozen=True)
class _InvalidationPlan:
    message_number: int
    surviving_episode_end: int
    episode_files_to_delete: tuple[str, ...]


@dataclass(frozen=True)
class InvalidationIntent:
    schema_version: int
    operation_kind: str
    boundary: int
    old_chat_count: int
    old_chat_sha256: str
    target_chat_count: int
    target_chat_sha256: str


class InvalidationIntentError(ValueError):
    """A persisted invalidation journal cannot be trusted or resolved."""


class LegacyChatImportError(ValueError):
    """A DB-only V1 history cannot be safely imported into Markdown."""


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
_REBUILD_FIELDS = {
    "cleanup-required": "cleanup_required",
    "rebuild-story-required": "rebuild_story_required",
    "rebuild-memory-required": "rebuild_memory_required",
    "rebuild-episode-required": "rebuild_episode_required",
    "rebuild-summary-required": "rebuild_summary_required",
    "rebuild-rag-required": "rebuild_rag_required",
    "rebuild-assets-required": "rebuild_assets_required",
}
_STATE_NUMBER = r"(?:0|[1-9]\d*)"
_STATE_BOOLEAN = r"(?:true|false)"
_STATE_BOUNDARY = rf"(?:none|{_STATE_NUMBER})"
_MEMORY_STATE_PREFIX = re.compile(
    r"\A# Memory State\n\n"
    rf"<!-- schema-version: (?P<schema_version>{_STATE_NUMBER}) -->\n"
    rf"<!-- last-memory-message: (?P<last_memory_message>{_STATE_NUMBER}) -->\n"
    rf"<!-- last-story-state-message: (?P<last_story_state_message>{_STATE_NUMBER}) -->\n"
    rf"<!-- last-summary-message: (?P<last_summary_message>{_STATE_NUMBER}) -->\n"
    rf"<!-- last-episode-message: (?P<last_episode_message>{_STATE_NUMBER}) -->\n"
    rf"<!-- last-rag-message: (?P<last_rag_message>{_STATE_NUMBER}) -->\n"
    rf"<!-- last-character-message: (?P<last_character_message>{_STATE_NUMBER}) -->\n"
    rf"<!-- last-assets-message: (?P<last_assets_message>{_STATE_NUMBER}) -->\n"
    r"(?:"
    rf"<!-- cleanup-required: (?P<cleanup_required>{_STATE_BOOLEAN}) -->\n"
    rf"<!-- rebuild-story-required: (?P<rebuild_story_required>{_STATE_BOOLEAN}) -->\n"
    rf"<!-- rebuild-memory-required: (?P<rebuild_memory_required>{_STATE_BOOLEAN}) -->\n"
    rf"<!-- rebuild-episode-required: (?P<rebuild_episode_required>{_STATE_BOOLEAN}) -->\n"
    rf"<!-- rebuild-summary-required: (?P<rebuild_summary_required>{_STATE_BOOLEAN}) -->\n"
    rf"<!-- rebuild-rag-required: (?P<rebuild_rag_required>{_STATE_BOOLEAN}) -->\n"
    rf"<!-- rebuild-assets-required: (?P<rebuild_assets_required>{_STATE_BOOLEAN}) -->\n"
    rf"(?:<!-- rebuild-from-message: (?P<rebuild_from_message>{_STATE_BOUNDARY}) -->\n)?"
    r")?\n"
    r"## Last Error\n\n"
    rf"<!-- last-error-length: (?P<last_error_length>{_STATE_NUMBER}) -->\n"
)
_INTENT_SHA256 = r"[0-9a-f]{64}"
_INVALIDATION_INTENT_MAX_CHARS = 4096
_INVALIDATION_INTENT_MAX_BYTES = 4096
_INVALIDATION_INTENT_DOCUMENT = re.compile(
    r"\A# Invalidation Intent\n\n"
    r"<!-- schema-version: (?P<schema_version>1) -->\n"
    r"<!-- operation-kind: (?P<operation_kind>truncate|replace-final-pair) -->\n"
    rf"<!-- boundary-message: (?P<boundary>{_STATE_NUMBER}) -->\n"
    rf"<!-- old-chat-count: (?P<old_chat_count>{_STATE_NUMBER}) -->\n"
    rf"<!-- old-chat-sha256: (?P<old_chat_sha256>{_INTENT_SHA256}) -->\n"
    rf"<!-- target-chat-count: (?P<target_chat_count>{_STATE_NUMBER}) -->\n"
    rf"<!-- target-chat-sha256: (?P<target_chat_sha256>{_INTENT_SHA256}) -->\n\Z"
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
_CRITICAL_TOP_LEVEL_FILES = frozenset(
    {
        "assets.md",
        "assets_summary.txt",
        "chat.md",
        "memory.md",
        "story_state.md",
        "summary.md",
    }
)


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


def _is_critical_memory_file(parts: tuple[str, ...]) -> bool:
    folded = tuple(part.casefold() for part in parts)
    if len(folded) == 1:
        return folded[0] in _CRITICAL_TOP_LEVEL_FILES
    return (
        len(folded) == 2
        and folded[0] == "characters"
        and folded[1].endswith(".md")
    )


def _backup_slot_for_name(
    target_name: str,
    candidate_name: str,
    *,
    windows: bool | None = None,
) -> int | None:
    use_windows_semantics = os.name == "nt" if windows is None else windows
    flags = re.IGNORECASE if use_windows_semantics else 0
    match = re.fullmatch(
        rf"{re.escape(target_name)}\.bak\.([1-9][0-9]*)",
        candidate_name,
        flags=flags,
    )
    return int(match.group(1)) if match is not None else None


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


def _legacy_timestamp(value: object) -> str:
    if isinstance(value, str) and _VALID_TIMESTAMP.fullmatch(value):
        return value
    parsed: dt.datetime | None = None
    if isinstance(value, dt.datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = dt.datetime.fromisoformat(
                value.strip().replace("Z", "+00:00")
            )
        except ValueError:
            pass
    return parsed.strftime("%Y-%m-%d %H:%M") if parsed is not None else "unknown"


def _validate_memory_state(state: MemoryState) -> None:
    if type(state.schema_version) is not int or state.schema_version != 2:
        raise ValueError("invalid memory state schema version")
    for field_name in _CHECKPOINT_FIELDS.values():
        value = getattr(state, field_name)
        if type(value) is not int or value < 0:
            raise ValueError("invalid memory state checkpoint")
    for field_name in _REBUILD_FIELDS.values():
        if type(getattr(state, field_name)) is not bool:
            raise ValueError("invalid memory state rebuild marker")
    if (
        state.rebuild_from_message is not None
        and (
            type(state.rebuild_from_message) is not int
            or state.rebuild_from_message < 0
        )
    ):
        raise ValueError("invalid memory state rebuild boundary")
    if not isinstance(state.last_error, str):
        raise ValueError("invalid memory state error")


class MarkdownMemoryStore:
    """Filesystem-backed Markdown store with atomic session transactions.

    Public methods acquire the per-session transaction. Code that already holds a
    transaction must call methods on the yielded ``MarkdownMemoryTransaction`` to
    avoid nested acquisition of the same non-reentrant lock.
    """

    def __init__(
        self,
        base: Path = Path("data/memory"),
        *,
        backup_count: int = 3,
    ) -> None:
        if type(backup_count) is not int or backup_count < 0:
            raise ValueError("backup_count must be a nonnegative integer")
        self.base = Path(base).resolve()
        self.backup_count = backup_count
        self._locks: WeakValueDictionary[int, asyncio.Lock] = WeakValueDictionary()
        self._locks_guard = asyncio.Lock()
        self._pipeline_locks: WeakValueDictionary[int, asyncio.Lock] = (
            WeakValueDictionary()
        )
        self._pipeline_locks_guard = asyncio.Lock()
        self._turn_locks: WeakValueDictionary[int, asyncio.Lock] = (
            WeakValueDictionary()
        )
        self._turn_locks_guard = asyncio.Lock()

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

    async def turn_lock_for(self, session_id: int) -> asyncio.Lock:
        """Return the shared lock for a whole chat turn operation.

        This is intentionally distinct from the transaction lock: a turn owns
        it while loading context, calling the provider, and atomically appending
        the completed pair, while each store access may take a short transaction.
        """
        _validate_session_id(session_id)
        async with self._turn_locks_guard:
            lock = self._turn_locks.get(session_id)
            if lock is None:
                lock = asyncio.Lock()
                self._turn_locks[session_id] = lock
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
        intended = self.base / str(session_id)
        if self._is_path_alias(intended):
            raise ValueError("session path escaped memory base")
        path = intended.resolve()
        try:
            relative = path.relative_to(self.base)
        except ValueError as exc:
            raise ValueError("session path escaped memory base") from exc
        if not relative.parts or path != intended:
            raise ValueError("session path escaped memory base")
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _is_path_alias(path: Path) -> bool:
        """Return whether one lexical entry is a symlink or directory junction."""
        if os.path.islink(path):
            return True
        is_junction = getattr(os.path, "isjunction", None)
        return bool(is_junction is not None and is_junction(path))

    def _lexical_file_path(
        self,
        session_id: int,
        relative: str | Path,
        *,
        allow_final_alias: bool,
    ) -> Path:
        parts = _safe_relative_parts(relative)
        directory = self.session_dir(session_id)
        path = directory.joinpath(*parts)
        parent = path.parent
        try:
            parent.relative_to(directory)
        except ValueError as exc:
            raise ValueError("file path must be a safe relative path") from exc
        if path == directory or parent.resolve() != parent:
            raise ValueError("file path must be a safe relative path")
        cursor = directory
        for component in parent.relative_to(directory).parts:
            cursor /= component
            if self._is_path_alias(cursor):
                raise ValueError("file path must be a safe relative path")
        if not allow_final_alias and self._is_path_alias(path):
            raise ValueError("file path must be a safe relative path")
        return path

    def file_path(self, session_id: int, relative: str | Path) -> Path:
        """Return a safe lexical file path without following aliases."""
        return self._lexical_file_path(
            session_id,
            relative,
            allow_final_alias=False,
        )

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

        try:
            path = self.file_path(
                session_id,
                f"characters/{name}.md",
            )
            path.parent.mkdir(parents=True, exist_ok=True)
        except ValueError as exc:
            raise ValueError("character path escaped profile directory") from exc
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
        parts = _safe_relative_parts(relative)
        target = self.file_path(session_id, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        backups: list[tuple[int, Path]] | None = None
        if target.exists() and _is_critical_memory_file(parts):
            backups = self._discover_backups_unlocked(
                session_id,
                parts,
                target,
            )
            if await self._text_file_matches(target, text):
                excess = [
                    (number, path)
                    for number, path in backups
                    if number > self.backup_count
                ]
                if excess:
                    self._prune_excess_backups_unlocked(excess)
                return
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
            if backups is not None:
                if self.backup_count > 0 or backups:
                    await self._replace_with_backups_unlocked(
                        session_id,
                        parts,
                        target,
                        temp,
                        backups,
                    )
                else:
                    os.replace(temp, target)
            else:
                os.replace(temp, target)
        finally:
            temp.unlink(missing_ok=True)

    @staticmethod
    async def _text_file_matches(path: Path, text: str) -> bool:
        expected = text.encode("utf-8")
        if path.stat().st_size != len(expected):
            return False
        async with aiofiles.open(path, "rb") as handle:
            return await handle.read() == expected

    def _prune_excess_backups_unlocked(
        self,
        excess: list[tuple[int, Path]],
    ) -> None:
        staging_paths: list[Path] = []
        moved: list[tuple[Path, Path]] = []
        try:
            for _, original in excess:
                stage = self._temporary_backup_path(
                    original.parent,
                    "backup-prune",
                )
                staging_paths.append(stage)
                os.replace(original, stage)
                moved.append((original, stage))
        except BaseException:
            try:
                rollback_errors: list[BaseException] = []
                for original, stage in reversed(moved):
                    try:
                        os.replace(stage, original)
                    except BaseException as exc:
                        rollback_errors.append(exc)
                if rollback_errors:
                    raise RuntimeError(
                        "failed to roll back backup pruning"
                    ) from rollback_errors[0]
            finally:
                self._cleanup_backup_staging(staging_paths)
            raise
        else:
            try:
                self._cleanup_backup_staging(staging_paths)
            except OSError:
                logger.warning(
                    "backup prune cleanup failed after commit",
                    exc_info=True,
                )

    def _backup_path(
        self,
        session_id: int,
        parts: tuple[str, ...],
        number: int,
    ) -> Path:
        relative = Path(*parts[:-1], f"{parts[-1]}.bak.{number}")
        return self.file_path(session_id, relative)

    def _discover_backups_unlocked(
        self,
        session_id: int,
        parts: tuple[str, ...],
        target: Path,
    ) -> list[tuple[int, Path]]:
        backups: dict[int, Path] = {}
        for candidate in target.parent.iterdir():
            number = _backup_slot_for_name(target.name, candidate.name)
            if number is None:
                continue
            if number in backups:
                raise ValueError("ambiguous backup slot")
            relative = Path(*parts[:-1], candidate.name)
            path = self.file_path(session_id, relative)
            if path.is_dir():
                raise ValueError("backup path must be a file")
            backups[number] = path
        return sorted(backups.items())

    @staticmethod
    def _temporary_backup_path(parent: Path, name: str) -> Path:
        file_descriptor, raw_temp = tempfile.mkstemp(
            prefix=f".{name}.",
            suffix=".tmp",
            dir=parent,
        )
        os.close(file_descriptor)
        return Path(raw_temp)

    async def _copy_backup_stage(
        self,
        source: Path,
        name: str,
        staging_paths: list[Path],
    ) -> Path:
        stage = self._temporary_backup_path(source.parent, name)
        staging_paths.append(stage)
        await asyncio.to_thread(shutil.copyfile, source, stage)
        return stage

    async def _link_or_copy_backup_stage(
        self,
        source: Path,
        name: str,
        staging_paths: list[Path],
    ) -> Path:
        stage = self._temporary_backup_path(source.parent, name)
        staging_paths.append(stage)
        os.unlink(stage)
        try:
            os.link(source, stage)
        except OSError:
            await asyncio.to_thread(shutil.copyfile, source, stage)
        return stage

    async def _replace_with_backups_unlocked(
        self,
        session_id: int,
        parts: tuple[str, ...],
        target: Path,
        new_temp: Path,
        backups: list[tuple[int, Path]],
    ) -> None:
        staging_paths: list[Path] = []
        rollback_stages: dict[Path, Path] = {}
        desired_stages: dict[int, Path] = {}
        original_by_slot = dict(backups)
        committed_slots: list[int] = []
        deleted_originals: list[Path] = []
        live_replaced = False
        try:
            old_live_stage = await self._copy_backup_stage(
                target,
                "backup-live",
                staging_paths,
            )
            if self.backup_count > 0:
                desired_stages[1] = await self._link_or_copy_backup_stage(
                    old_live_stage,
                    "backup-desired",
                    staging_paths,
                )

            for number, original in backups:
                rollback = await self._link_or_copy_backup_stage(
                    original,
                    "backup-rollback",
                    staging_paths,
                )
                rollback_stages[original] = rollback
                if number < self.backup_count:
                    desired = await self._link_or_copy_backup_stage(
                        rollback,
                        "backup-desired",
                        staging_paths,
                    )
                    desired_stages[number + 1] = desired

            os.replace(new_temp, target)
            live_replaced = True

            for number, desired in sorted(desired_stages.items()):
                os.replace(
                    desired,
                    self._backup_path(
                        session_id,
                        parts,
                        number,
                    ),
                )
                committed_slots.append(number)

            for number, original in backups:
                if number in desired_stages:
                    continue
                original.unlink()
                deleted_originals.append(original)
        except BaseException:
            try:
                if live_replaced:
                    self._rollback_backup_replace(
                        session_id,
                        parts,
                        target,
                        old_live_stage,
                        original_by_slot,
                        rollback_stages,
                        committed_slots,
                        deleted_originals,
                    )
            finally:
                self._cleanup_backup_staging(staging_paths)
            raise
        else:
            try:
                self._cleanup_backup_staging(staging_paths)
            except OSError:
                logger.warning(
                    "backup staging cleanup failed after commit",
                    exc_info=True,
                )

    def _rollback_backup_replace(
        self,
        session_id: int,
        parts: tuple[str, ...],
        target: Path,
        old_live_stage: Path,
        original_by_slot: dict[int, Path],
        rollback_stages: dict[Path, Path],
        committed_slots: list[int],
        deleted_originals: list[Path],
    ) -> None:
        rollback_errors: list[BaseException] = []
        try:
            os.replace(old_live_stage, target)
        except BaseException as exc:
            rollback_errors.append(exc)

        for number in committed_slots:
            original = original_by_slot.get(number)
            try:
                if original is None:
                    self._backup_path(
                        session_id,
                        parts,
                        number,
                    ).unlink(missing_ok=True)
                else:
                    os.replace(rollback_stages[original], original)
            except BaseException as exc:
                rollback_errors.append(exc)

        for original in deleted_originals:
            try:
                os.replace(rollback_stages[original], original)
            except BaseException as exc:
                rollback_errors.append(exc)

        if rollback_errors:
            raise RuntimeError("failed to roll back backup transaction") from rollback_errors[0]

    @classmethod
    def _cleanup_backup_staging(cls, paths: list[Path]) -> None:
        cleanup_errors: list[OSError] = []
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                cleanup_errors.append(exc)
                cls._scrub_backup_stage(path)

        if cleanup_errors:
            raise OSError("failed to clean backup staging files") from cleanup_errors[0]

    @classmethod
    def _scrub_backup_stage(cls, path: Path) -> None:
        empty = cls._temporary_backup_path(path.parent, "backup-empty")
        try:
            os.replace(empty, path)
        finally:
            if empty.exists():
                os.unlink(empty)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    async def _delete_file_unlocked(
        self,
        session_id: int,
        relative: str | Path,
    ) -> None:
        target = self._lexical_file_path(
            session_id,
            relative,
            allow_final_alias=True,
        )
        if self._is_path_alias(target):
            if target.is_dir():
                raise ValueError("deletion target must be a file")
            target.unlink(missing_ok=True)
            return
        if not target.exists():
            return
        if target.is_dir():
            raise ValueError("deletion target must be a file")
        target.unlink()

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

    async def delete_file(
        self,
        session_id: int,
        relative: str | Path,
    ) -> None:
        async with self.transaction(session_id) as transaction:
            await transaction.delete_file(relative)

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

    async def import_legacy_chat(
        self,
        session_id: int,
        legacy_messages: str | None,
        *,
        char_name: str,
        user_name: str,
    ) -> list[ChatRecord]:
        """Atomically seed a missing chat.md from the read-only V1 DB fallback."""
        async with self.transaction(session_id) as transaction:
            chat_path = self.file_path(session_id, "chat.md")
            if chat_path.exists():
                return await transaction.load_chat()

            if not legacy_messages:
                return []
            try:
                messages = json.loads(legacy_messages)
            except (TypeError, json.JSONDecodeError) as exc:
                raise LegacyChatImportError(
                    "legacy chat history is invalid JSON"
                ) from exc
            if not isinstance(messages, list):
                raise LegacyChatImportError("legacy chat history must be a list")

            records: list[ChatRecord] = []
            for message in messages:
                if not isinstance(message, Mapping):
                    raise LegacyChatImportError(
                        "legacy chat history must contain objects"
                    )
                role = message.get("role")
                content = message.get("content")
                if role not in {"user", "assistant", "system"} or not isinstance(
                    content, str
                ):
                    raise LegacyChatImportError(
                        "legacy chat history contains an invalid message"
                    )
                raw_name = message.get("name")
                name = (
                    raw_name
                    if isinstance(raw_name, str) and raw_name
                    else user_name
                    if role == "user"
                    else char_name
                    if role == "assistant"
                    else "System"
                )
                raw_type = message.get("msg_type", "ic")
                msg_type = (
                    raw_type
                    if role == "user" and raw_type in {"ic", "ooc"}
                    else "ic"
                )
                records.append(
                    ChatRecord(
                        number=0,
                        role=role,
                        content=content,
                        timestamp=_legacy_timestamp(message.get("timestamp")),
                        name=name,
                        msg_type=msg_type,
                    )
                )

            if records:
                try:
                    document = render_chat_records(records)
                except ValueError as exc:
                    raise LegacyChatImportError(
                        "legacy chat history contains invalid metadata"
                    ) from exc
                await transaction.write_text("chat.md", document)
                return await transaction.load_chat()
            return []

    async def load_chat(self, session_id: int) -> list[ChatRecord]:
        async with self.transaction(session_id) as transaction:
            return await transaction.load_chat()

    async def load_state(self, session_id: int) -> MemoryState:
        async with self.transaction(session_id) as transaction:
            return await transaction.load_state()

    async def save_state(self, session_id: int, state: MemoryState) -> None:
        async with self.transaction(session_id) as transaction:
            await transaction.save_state(state)

    async def recover_invalidation(self, session_id: int) -> None:
        """Resolve journaled file cleanup under the pipeline lock, without LLMs."""
        async with self.transaction(session_id) as transaction:
            records = await transaction.load_chat()
            total = records[-1].number if records else 0
            intent = await transaction.load_invalidation_intent()
            state = await transaction.load_state()
            if (
                intent is None
                and not state.cleanup_required
                and not state.checkpoint_exceeds(total)
                and not (total == 0 and state.rebuild_required)
                and (state.rebuild_required or state.rebuild_from_message is None)
            ):
                return
        pipeline_lock = await self.pipeline_lock_for(session_id)
        async with pipeline_lock:
            async with self.transaction(session_id) as transaction:
                records = await transaction.load_chat()
                total = records[-1].number if records else 0
                await transaction.recover_pending_invalidation(total)

    async def truncate_chat(
        self,
        session_id: int,
        *,
        remove_count: int,
    ) -> list[ChatRecord]:
        """Commit authoritative chat, then make derived cleanup recoverable.

        Lock order is pipeline then transaction. ChatService already owns the
        turn lock before entering here, so concurrent turns cannot interleave.
        A failure after the chat replacement means the truncation committed;
        callers must reload instead of repeating it while startup recovery
        finishes any marker-backed cleanup.
        """
        if type(remove_count) is not int or remove_count < 1:
            raise ValueError("remove_count must be a positive integer")
        pipeline_lock = await self.pipeline_lock_for(session_id)
        async with pipeline_lock:
            async with self.transaction(session_id) as transaction:
                records = await transaction.load_chat()
                total = records[-1].number if records else 0
                await transaction.recover_pending_invalidation(total)
                if remove_count > len(records):
                    raise ValueError("remove_count exceeds available chat records")
                retained = records[:-remove_count]
                plan = await transaction.plan_invalidation(len(retained))
                intent = build_invalidation_intent(
                    "truncate",
                    boundary=len(retained),
                    old_records=records,
                    target_records=retained,
                )
                await transaction.save_invalidation_intent(intent)
                try:
                    await transaction.write_text(
                        "chat.md",
                        render_chat_records(retained),
                    )
                except Exception:
                    await transaction.clear_invalidation_intent()
                    raise
                await transaction.start_invalidation(plan)
                await transaction.finish_invalidation_cleanup(plan)
                if not retained:
                    await transaction.reconcile_empty_history()
                await transaction.clear_invalidation_intent()
                return retained

    async def invalidate_after(
        self,
        session_id: int,
        message_number: int,
    ) -> None:
        """Invalidate derived artifacts after an authoritative message boundary."""
        pipeline_lock = await self.pipeline_lock_for(session_id)
        async with pipeline_lock:
            async with self.transaction(session_id) as transaction:
                records = await transaction.load_chat()
                total = records[-1].number if records else 0
                await transaction.recover_pending_invalidation(total)
                if (
                    type(message_number) is not int
                    or message_number < 0
                    or message_number > total
                ):
                    raise ValueError(
                        "message_number must be within authoritative chat history"
                    )
                plan = await transaction.plan_invalidation(message_number)
                await transaction.start_invalidation(plan)
                await transaction.finish_invalidation_cleanup(plan)
                if total == 0:
                    await transaction.reconcile_empty_history()

    async def replace_final_pair(
        self,
        session_id: int,
        *,
        expected_user: ChatRecord,
        expected_assistant: ChatRecord,
        assistant_content: str,
    ) -> list[ChatRecord]:
        """Replace one snapshotted final pair after invalidating its derivations."""
        if not isinstance(assistant_content, str):
            raise ValueError("assistant_content must be text")
        pipeline_lock = await self.pipeline_lock_for(session_id)
        async with pipeline_lock:
            async with self.transaction(session_id) as transaction:
                records = await transaction.load_chat()
                total = records[-1].number if records else 0
                await transaction.recover_pending_invalidation(total)
                if (
                    len(records) < 2
                    or records[-2] != expected_user
                    or records[-1] != expected_assistant
                    or expected_user.role != "user"
                    or expected_assistant.role != "assistant"
                ):
                    raise ValueError("final chat pair changed before replacement")
                prefix = records[:-2]
                replacement = replace(
                    expected_assistant,
                    content=assistant_content,
                )
                updated = [*prefix, expected_user, replacement]
                plan = await transaction.plan_invalidation(len(prefix))
                intent = build_invalidation_intent(
                    "replace-final-pair",
                    boundary=len(prefix),
                    old_records=records,
                    target_records=updated,
                )
                await transaction.save_invalidation_intent(intent)
                try:
                    await transaction.write_text(
                        "chat.md",
                        render_chat_records(updated),
                    )
                except Exception:
                    await transaction.clear_invalidation_intent()
                    raise
                await transaction.start_invalidation(plan)
                await transaction.finish_invalidation_cleanup(plan)
                await transaction.clear_invalidation_intent()
                return updated


class MarkdownMemoryTransaction:
    """Non-locking session operations available only to a lock owner."""

    def __init__(self, store: MarkdownMemoryStore, session_id: int) -> None:
        self._store = store
        self.session_id = session_id

    async def read_text(self, relative: str | Path) -> str:
        return await self._store._read_text_unlocked(self.session_id, relative)

    async def write_text(self, relative: str | Path, text: str) -> None:
        await self._store._write_text_unlocked(self.session_id, relative, text)

    async def delete_file(self, relative: str | Path) -> None:
        await self._store._delete_file_unlocked(self.session_id, relative)

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

    async def load_invalidation_intent(self) -> InvalidationIntent | None:
        path = self._store.file_path(
            self.session_id,
            "invalidation_intent.md",
        )
        try:
            async with aiofiles.open(path, "rb") as handle:
                payload = await handle.read(_INVALIDATION_INTENT_MAX_BYTES + 1)
        except FileNotFoundError:
            return None
        if len(payload) > _INVALIDATION_INTENT_MAX_BYTES:
            raise InvalidationIntentError("invalid invalidation intent")
        try:
            text = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            raise InvalidationIntentError("invalid invalidation intent") from None
        return parse_invalidation_intent(text)

    async def save_invalidation_intent(self, intent: InvalidationIntent) -> None:
        await self.write_text(
            "invalidation_intent.md",
            render_invalidation_intent(intent),
        )

    async def clear_invalidation_intent(self) -> None:
        await self.delete_file("invalidation_intent.md")

    async def resolve_invalidation_intent(self) -> bool:
        """Resolve one journal by canonical old/target identity, old first."""
        intent = await self.load_invalidation_intent()
        if intent is None:
            return False
        records = await self.load_chat()
        count = len(records)
        digest = _chat_sha256(records)
        if (
            count == intent.old_chat_count
            and digest == intent.old_chat_sha256
        ):
            await self.clear_invalidation_intent()
            return True
        if not (
            count == intent.target_chat_count
            and digest == intent.target_chat_sha256
        ):
            raise InvalidationIntentError(
                "invalidation intent does not match authoritative chat"
            )

        plan = await self.plan_invalidation(intent.boundary)
        await self.start_invalidation(plan)
        await self.finish_invalidation_cleanup(plan)
        if not records:
            await self.reconcile_empty_history()
        await self.clear_invalidation_intent()
        return True

    async def recover_pending_invalidation(self, total: int) -> MemoryState:
        """Resolve journal or legacy cleanup without inferring retry boundaries."""
        if type(total) is not int or total < 0:
            raise ValueError("total must be a nonnegative integer")
        await self.resolve_invalidation_intent()
        current = await self.load_state()
        if not current.rebuild_required and current.rebuild_from_message is not None:
            current = replace(current, rebuild_from_message=None)
            await self.save_state(current)
        exceeds = current.checkpoint_exceeds(total)
        if total == 0 and current.rebuild_required and not current.cleanup_required:
            await self.reconcile_empty_history()
            return await self.load_state()
        if not current.cleanup_required and not exceeds:
            return current
        boundary = (
            current.rebuild_from_message if current.rebuild_required else None
        )
        if boundary is None:
            if not exceeds:
                raise InvalidationIntentError(
                    "pending invalidation has no persisted rebuild boundary"
                )
            boundary = total
        if boundary > total:
            raise InvalidationIntentError(
                "persisted rebuild boundary exceeds chat history"
            )
        plan = await self.plan_invalidation(boundary)
        if exceeds:
            await self.start_invalidation(plan)
            current = await self.load_state()
        if current.cleanup_required:
            await self.finish_invalidation_cleanup(plan)
            current = await self.load_state()
        if total == 0 and current.rebuild_required:
            await self.reconcile_empty_history()
        return await self.load_state()

    async def plan_invalidation(self, message_number: int) -> _InvalidationPlan:
        """Validate every cleanup input and return a read-only mutation plan."""
        from app.config import settings
        from app.services import story_memory

        if type(message_number) is not int or message_number < 0:
            raise ValueError("message_number must be a nonnegative integer")
        for relative in (
            "chat.md",
            "memory_state.md",
            "story_state.md",
            "summary.md",
            "rag/index.json",
        ):
            self._store.file_path(self.session_id, relative)
        episodes, incompatible_episodes = (
            await story_memory._load_episode_chain_for_invalidation(
            self._store,
            self,
            episode_size=settings.episode_size_messages,
            )
        )
        surviving_episode_end = max(
            (
                episode.end
                for episode in episodes.values()
                if episode.end <= message_number
            ),
            default=0,
        )
        files_to_delete = tuple(
            f"episodes/episode-{episode.number:06d}.md"
            for episode in sorted(
                (*episodes.values(), *incompatible_episodes),
                key=lambda episode: episode.number,
                reverse=True,
            )
            if episode in incompatible_episodes or episode.end > message_number
        )
        for relative in files_to_delete:
            self._store.file_path(self.session_id, relative)
        return _InvalidationPlan(
            message_number=message_number,
            surviving_episode_end=surviving_episode_end,
            episode_files_to_delete=files_to_delete,
        )

    async def start_invalidation(self, plan: _InvalidationPlan) -> None:
        """Persist recovery markers after authoritative chat has committed."""
        await self.save_state(
            MemoryState(
                last_episode_message=plan.surviving_episode_end,
                cleanup_required=True,
                rebuild_story_required=True,
                rebuild_memory_required=True,
                rebuild_episode_required=True,
                rebuild_summary_required=True,
                rebuild_rag_required=True,
                rebuild_assets_required=True,
                rebuild_from_message=plan.message_number,
            )
        )

    async def finish_invalidation_cleanup(self, plan: _InvalidationPlan) -> None:
        """Idempotently clean disposable derivations, then clear cleanup pending."""
        from app.services import rag

        await rag.invalidate_after(
            self.session_id,
            plan.message_number,
            transaction=self,
        )
        await self.write_text("story_state.md", "")
        await self.write_text("summary.md", "")
        for relative in plan.episode_files_to_delete:
            await self.delete_file(relative)
        state = await self.load_state()
        updated = replace(state, cleanup_required=False)
        if not updated.rebuild_required:
            updated = replace(updated, rebuild_from_message=None)
        await self.save_state(updated)

    async def reconcile_empty_history(self) -> None:
        """Idempotently remove every active derivation for authoritative emptiness."""
        from app.services import rag

        if await self.load_chat():
            raise InvalidationIntentError(
                "empty-history reconciliation requires empty authoritative chat"
            )
        session = self._store.session_dir(self.session_id)
        derived_directories: list[tuple[Path, str]] = []
        for directory_name, pattern in (
            ("characters", "*.md"),
            ("episodes", "episode-*.md"),
        ):
            intended = session / directory_name
            if not intended.exists():
                continue
            directory = intended.resolve()
            if directory != intended or directory.parent != session:
                raise ValueError("derived artifact directory escaped session directory")
            derived_directories.append((directory, pattern))
        for relative in ("memory.md", "assets.md", "assets_summary.txt"):
            await self.delete_file(relative)
        for directory, pattern in derived_directories:
            for path in sorted(directory.glob(pattern), reverse=True):
                resolved = path.resolve()
                if resolved.parent != directory:
                    raise ValueError("derived artifact escaped session directory")
                await self.delete_file(path.relative_to(session))
        await self.write_text("story_state.md", "")
        await self.write_text("summary.md", "")
        await rag.save_index(
            self.session_id,
            rag.empty_index(),
            transaction=self,
        )
        await self.save_state(MemoryState())

    async def invalidate_after(self, message_number: int) -> None:
        """Invalidate derivations when authoritative chat is already committed."""
        plan = await self.plan_invalidation(message_number)
        await self.start_invalidation(plan)
        await self.finish_invalidation_cleanup(plan)


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


def _chat_sha256(records: list[ChatRecord]) -> str:
    canonical = render_chat_records(records).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _validate_invalidation_intent(intent: InvalidationIntent) -> None:
    if type(intent.schema_version) is not int or intent.schema_version != 1:
        raise InvalidationIntentError("invalid invalidation intent")
    if intent.operation_kind not in {"truncate", "replace-final-pair"}:
        raise InvalidationIntentError("invalid invalidation intent")
    for value in (intent.boundary, intent.old_chat_count, intent.target_chat_count):
        if type(value) is not int or value < 0:
            raise InvalidationIntentError("invalid invalidation intent")
    for digest in (intent.old_chat_sha256, intent.target_chat_sha256):
        if not isinstance(digest, str) or not re.fullmatch(_INTENT_SHA256, digest):
            raise InvalidationIntentError("invalid invalidation intent")
    if intent.operation_kind == "truncate":
        valid_shape = (
            intent.old_chat_count > intent.target_chat_count
            and intent.target_chat_count == intent.boundary
        )
    else:
        valid_shape = (
            intent.old_chat_count == intent.target_chat_count
            and intent.target_chat_count == intent.boundary + 2
        )
    if not valid_shape:
        raise InvalidationIntentError("invalid invalidation intent")


def build_invalidation_intent(
    operation_kind: str,
    *,
    boundary: int,
    old_records: list[ChatRecord],
    target_records: list[ChatRecord],
) -> InvalidationIntent:
    """Build a content-free intent from canonical store chat bytes."""
    intent = InvalidationIntent(
        schema_version=1,
        operation_kind=operation_kind,
        boundary=boundary,
        old_chat_count=len(old_records),
        old_chat_sha256=_chat_sha256(old_records),
        target_chat_count=len(target_records),
        target_chat_sha256=_chat_sha256(target_records),
    )
    _validate_invalidation_intent(intent)
    return intent


def parse_invalidation_intent(text: str) -> InvalidationIntent:
    """Parse one complete strict invalidation journal document."""
    try:
        oversized = (
            len(text) > _INVALIDATION_INTENT_MAX_CHARS
            or len(text.encode("utf-8")) > _INVALIDATION_INTENT_MAX_BYTES
        )
    except (AttributeError, TypeError, UnicodeEncodeError):
        raise InvalidationIntentError("invalid invalidation intent") from None
    if oversized:
        raise InvalidationIntentError("invalid invalidation intent")
    match = _INVALIDATION_INTENT_DOCUMENT.fullmatch(text)
    if match is None:
        raise InvalidationIntentError("invalid invalidation intent")
    try:
        intent = InvalidationIntent(
            schema_version=int(match.group("schema_version")),
            operation_kind=match.group("operation_kind"),
            boundary=int(match.group("boundary")),
            old_chat_count=int(match.group("old_chat_count")),
            old_chat_sha256=match.group("old_chat_sha256"),
            target_chat_count=int(match.group("target_chat_count")),
            target_chat_sha256=match.group("target_chat_sha256"),
        )
    except (TypeError, ValueError):
        raise InvalidationIntentError("invalid invalidation intent") from None
    _validate_invalidation_intent(intent)
    return intent


def render_invalidation_intent(intent: InvalidationIntent) -> str:
    """Render a validated content-free invalidation journal."""
    _validate_invalidation_intent(intent)
    return (
        "# Invalidation Intent\n\n"
        f"<!-- schema-version: {intent.schema_version} -->\n"
        f"<!-- operation-kind: {intent.operation_kind} -->\n"
        f"<!-- boundary-message: {intent.boundary} -->\n"
        f"<!-- old-chat-count: {intent.old_chat_count} -->\n"
        f"<!-- old-chat-sha256: {intent.old_chat_sha256} -->\n"
        f"<!-- target-chat-count: {intent.target_chat_count} -->\n"
        f"<!-- target-chat-sha256: {intent.target_chat_sha256} -->\n"
    )


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

    boundary = match.group("rebuild_from_message")
    values: dict[str, bool | int | str | None] = {
        "schema_version": int(match.group("schema_version")),
        **{
            field_name: int(match.group(field_name))
            for field_name in _CHECKPOINT_FIELDS.values()
        },
        **{
            field_name: match.group(field_name) == "true"
            for field_name in _REBUILD_FIELDS.values()
        },
        "rebuild_from_message": (
            None if boundary in (None, "none") else int(boundary)
        ),
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
        f"<!-- last-assets-message: {state.last_assets_message} -->\n"
        f"<!-- cleanup-required: {str(state.cleanup_required).lower()} -->\n"
        f"<!-- rebuild-story-required: {str(state.rebuild_story_required).lower()} -->\n"
        f"<!-- rebuild-memory-required: {str(state.rebuild_memory_required).lower()} -->\n"
        f"<!-- rebuild-episode-required: {str(state.rebuild_episode_required).lower()} -->\n"
        f"<!-- rebuild-summary-required: {str(state.rebuild_summary_required).lower()} -->\n"
        f"<!-- rebuild-rag-required: {str(state.rebuild_rag_required).lower()} -->\n"
        f"<!-- rebuild-assets-required: {str(state.rebuild_assets_required).lower()} -->\n"
        f"<!-- rebuild-from-message: {state.rebuild_from_message if state.rebuild_from_message is not None else 'none'} -->\n\n"
        "## Last Error\n\n"
        f"<!-- last-error-length: {len(state.last_error)} -->\n"
        f"{displayed_error}\n"
    )
