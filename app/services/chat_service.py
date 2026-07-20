"""Shared guarded chat turns backed by authoritative Markdown history."""

from __future__ import annotations

import copy
import inspect
import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.config import settings
from app.schemas.api import WorldBookEntry
from app.services.context_builder import ContextBuilder, ContextSources
from app.services.md_store import MarkdownMemoryStore
from app.services.output_guard import (
    _RETRY_CORRECTION,
    complete_with_guard,
    has_refusal,
    normalize_opening,
)
from app.services.prompt import _legacy_character_context, _match_worldbook_entries
from app.services.prompt_policy import REQUIRED_OPENING

logger = logging.getLogger(__name__)

Completion = Callable[[list[dict[str, Any]]], Awaitable[dict[str, Any]]]
StreamCompletion = Callable[
    [list[dict[str, Any]]], AsyncIterator[str | dict[str, Any]]
]

SAFE_STREAM_ERROR = "生成失败，请重试。"
SAFE_CONTEXT_ERROR = "对话上下文过长，请缩短当前输入后重试。"
_PARAGRAPH_BOUNDARY = re.compile(r"\r?\n\s*\r?\n")
_USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


class MemorySubmitter(Protocol):
    def submit(self, session_id: int) -> Awaitable[None] | None: ...


class ContextBuilderLike(Protocol):
    def build(self, sources: ContextSources) -> Any: ...


class ContextBudgetExceeded(ValueError):
    """The mandatory protected prompt content cannot fit the configured budget."""


@dataclass(frozen=True, slots=True)
class TurnRequest:
    session_id: int
    content: str
    character: Mapping[str, Any]
    user_name: str = "用户"
    msg_type: str = "ic"
    user_persona: str = ""
    persona_position: str = "in_prompt"
    worldbook_entries: Sequence[WorldBookEntry | Mapping[str, Any]] = field(
        default_factory=tuple
    )
    story_state: str = ""
    memory: str = ""
    character_profiles: Sequence[str] = field(default_factory=tuple)
    assets: str = ""
    episodes: Sequence[str] = field(default_factory=tuple)
    rag: Sequence[str] = field(default_factory=tuple)
    overall_summary: str = ""
    history_seed: Sequence[dict[str, str]] = field(default_factory=tuple)
    is_new_conversation: bool | None = None


@dataclass(frozen=True, slots=True)
class TurnResult:
    content: str
    usage: dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0


@dataclass(frozen=True, slots=True)
class StreamEvent:
    delta: str = ""
    usage: dict[str, Any] | None = None
    error: str = ""
    done: bool = False


class ChatService:
    """Serialize, guard, persist, and enqueue complete chat turns."""

    def __init__(
        self,
        store: MarkdownMemoryStore,
        memory_manager: MemorySubmitter | None,
        *,
        completion: Completion | None = None,
        stream_completion: StreamCompletion | None = None,
        context_builder: ContextBuilderLike | None = None,
    ) -> None:
        self.store = store
        self.memory_manager = memory_manager
        self._completion = completion
        self._stream_completion = stream_completion
        self._context_builder = context_builder or ContextBuilder(
            total_budget=settings.total_token_budget,
            min_recent_messages=settings.min_recent_messages,
        )

    async def send(self, request: TurnRequest) -> TurnResult:
        if self._completion is None:
            raise RuntimeError("non-stream completion is not configured")
        turn_lock = await self.store.turn_lock_for(request.session_id)
        async with turn_lock:
            messages = await self._build_messages(request)
            result = await complete_with_guard(self._completion, messages)
            content = str(result["content"])
            await self.store.append_pair(
                request.session_id,
                request.content,
                content,
                char_name=self._character_name(request),
                user_name=request.user_name,
                msg_type=request.msg_type,
            )
            await self._submit_memory(request.session_id)
            usage = result.get("usage")
            return TurnResult(
                content=content,
                usage=copy.deepcopy(usage) if isinstance(usage, dict) else {},
                retry_count=int(result.get("retry_count", 0)),
            )

    async def stream(self, request: TurnRequest) -> AsyncIterator[StreamEvent]:
        if self._stream_completion is None:
            raise RuntimeError("stream completion is not configured")
        turn_lock = await self.store.turn_lock_for(request.session_id)
        async with turn_lock:
            try:
                messages = await self._build_messages(request)
            except ContextBudgetExceeded:
                yield StreamEvent(error=SAFE_CONTEXT_ERROR)
                return

            first_usage: dict[str, Any] = {}
            for attempt in range(2):
                call_messages = copy.deepcopy(messages)
                if attempt:
                    call_messages.append(
                        {"role": "user", "content": _RETRY_CORRECTION}
                    )
                upstream = self._stream_completion(call_messages)
                buffer = ""
                accepted = ""
                usage: dict[str, Any] = {}
                validated = False
                refused = False
                ended_normally = False
                try:
                    async for item in upstream:
                        if isinstance(item, dict):
                            raw_usage = item.get("usage")
                            if isinstance(raw_usage, dict):
                                usage = copy.deepcopy(raw_usage)
                            continue
                        chunk = str(item)
                        if not validated:
                            buffer += chunk
                            if not self._opening_guard_ready(buffer):
                                continue
                            if has_refusal(buffer):
                                refused = True
                                break
                            normalized = normalize_opening(buffer)
                            accepted = normalized
                            validated = True
                            if normalized:
                                yield StreamEvent(delta=normalized)
                            continue
                        accepted += chunk
                        if chunk:
                            yield StreamEvent(delta=chunk)
                    else:
                        ended_normally = True
                except Exception:
                    logger.exception(
                        "Streaming provider failed for session %s",
                        request.session_id,
                    )
                    yield StreamEvent(error=SAFE_STREAM_ERROR)
                    return
                finally:
                    close = getattr(upstream, "aclose", None)
                    if close is not None:
                        await close()

                if ended_normally and not validated:
                    if has_refusal(buffer):
                        refused = True
                    else:
                        accepted = normalize_opening(buffer)
                        validated = True
                        if accepted:
                            yield StreamEvent(delta=accepted)

                if refused:
                    if attempt == 0:
                        first_usage = usage
                        continue
                    logger.warning(
                        "Streaming model refused after retry for session %s",
                        request.session_id,
                    )
                    yield StreamEvent(error=SAFE_STREAM_ERROR)
                    return

                if not ended_normally:
                    yield StreamEvent(error=SAFE_STREAM_ERROR)
                    return

                combined_usage = _aggregate_usage(first_usage, usage)
                await self.store.append_pair(
                    request.session_id,
                    request.content,
                    accepted,
                    char_name=self._character_name(request),
                    user_name=request.user_name,
                    msg_type=request.msg_type,
                )
                await self._submit_memory(request.session_id)
                yield StreamEvent(done=True, usage=combined_usage)
                return

    async def _build_messages(self, request: TurnRequest) -> list[dict[str, Any]]:
        records = await self.store.load_chat(request.session_id)
        recent = [
            {"role": record.role, "content": record.content}
            for record in records
        ]
        character = copy.deepcopy(dict(request.character))
        if request.history_seed:
            character["_history_seed"] = copy.deepcopy(list(request.history_seed))
        entries = [
            entry
            if isinstance(entry, WorldBookEntry)
            else WorldBookEntry.model_validate(entry)
            for entry in request.worldbook_entries
        ]
        recent_text = " ".join(
            message["content"] for message in recent[-10:]
        ) + " " + request.content
        before, after = _match_worldbook_entries(entries, recent_text)
        ordered_character_context = _legacy_character_context(
            character,
            user_name=request.user_name,
            user_persona=request.user_persona,
            persona_position=request.persona_position,
            worldbook_before=before,
            worldbook_after=after,
        )
        is_new = (
            not records
            if request.is_new_conversation is None
            else request.is_new_conversation
        )
        context_persona = (
            request.user_persona
            if request.persona_position != "none"
            else ""
        )
        sources = ContextSources(
            character=character,
            user_name=request.user_name,
            user_persona=context_persona,
            story_state=request.story_state,
            memory=request.memory,
            character_profiles=list(request.character_profiles),
            assets=request.assets,
            episodes=list(request.episodes),
            rag=list(request.rag),
            summary=request.overall_summary,
            recent=recent,
            user_message=request.content,
            is_new_conversation=is_new,
            ordered_character_context=ordered_character_context,
        )
        try:
            result = self._context_builder.build(sources)
        except ValueError as exc:
            raise ContextBudgetExceeded(SAFE_CONTEXT_ERROR) from exc
        return copy.deepcopy(result.messages)

    async def _submit_memory(self, session_id: int) -> None:
        if self.memory_manager is None:
            return
        try:
            result = self.memory_manager.submit(session_id)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception(
                "Committed chat turn but failed to submit managed memory for session %s",
                session_id,
            )

    @staticmethod
    def _character_name(request: TurnRequest) -> str:
        return str(request.character.get("name") or "角色")

    @staticmethod
    def _opening_guard_ready(buffer: str) -> bool:
        if len(buffer) >= settings.stream_guard_chars:
            return True
        substantive = buffer.lstrip()
        while substantive.startswith(REQUIRED_OPENING):
            substantive = substantive[len(REQUIRED_OPENING):].lstrip()
        return _PARAGRAPH_BOUNDARY.search(substantive) is not None


def _aggregate_usage(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> dict[str, Any]:
    aggregate = copy.deepcopy(dict(second))
    for key in _USAGE_KEYS:
        aggregate[key] = first.get(key, 0) + second.get(key, 0)
    return aggregate
