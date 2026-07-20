"""Shared guarded chat turns backed by authoritative Markdown history."""

from __future__ import annotations

import copy
import inspect
import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from app.config import settings
from app.schemas.api import WorldBookEntry
from app.services.context_builder import ContextBuilder, ContextSources
from app.services.md_store import ChatRecord, MarkdownMemoryStore
from app.services.output_guard import (
    _RETRY_CORRECTION,
    RefusalOpeningState,
    classify_refusal_opening,
    complete_with_guard,
    has_substantive_output,
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
class DomainContext:
    """Lazy domain inputs used to build ContextSources under the turn lock."""

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


DomainContextLoader = Callable[
    [tuple[ChatRecord, ...]],
    Awaitable[DomainContext] | DomainContext,
]


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
        domain_context_loader: DomainContextLoader | None = None,
    ) -> None:
        self.store = store
        self.memory_manager = memory_manager
        self._completion = completion
        self._stream_completion = stream_completion
        self._domain_context_loader = domain_context_loader
        self._context_builder = context_builder or ContextBuilder(
            total_budget=settings.total_token_budget,
            min_recent_messages=settings.min_recent_messages,
        )

    async def send(self, request: TurnRequest) -> TurnResult:
        if self._completion is None:
            raise RuntimeError("non-stream completion is not configured")
        turn_lock = await self.store.turn_lock_for(request.session_id)
        async with turn_lock:
            await self.store.recover_invalidation(request.session_id)
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

    async def undo(self, session_id: int) -> list[ChatRecord]:
        """Remove one complete final pair and enqueue regeneration once."""
        turn_lock = await self.store.turn_lock_for(session_id)
        async with turn_lock:
            await self.store.recover_invalidation(session_id)
            records = await self.store.load_chat(session_id)
            self._final_pair(records)
            retained = await self.store.truncate_chat(
                session_id,
                remove_count=2,
            )
            await self._submit_memory(session_id)
            return retained

    async def retry(self, request: TurnRequest) -> TurnResult:
        """Generate first, then atomically replace one complete final pair."""
        if self._completion is None:
            raise RuntimeError("non-stream completion is not configured")
        turn_lock = await self.store.turn_lock_for(request.session_id)
        async with turn_lock:
            await self.store.recover_invalidation(request.session_id)
            records = await self.store.load_chat(request.session_id)
            original_user, original_assistant = self._final_pair(records)
            retry_request = replace(
                request,
                content=original_user.content,
                user_name=original_user.name,
                msg_type=original_user.msg_type,
            )
            messages = await self._build_messages(
                retry_request,
                records=records[:-2],
                discard_invalidated_context=True,
            )
            result = await complete_with_guard(self._completion, messages)
            content = str(result["content"])
            await self.store.replace_final_pair(
                request.session_id,
                expected_user=original_user,
                expected_assistant=original_assistant,
                assistant_content=content,
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
                await self.store.recover_invalidation(request.session_id)
                messages = await self._build_messages(request)
            except ContextBudgetExceeded:
                yield StreamEvent(error=SAFE_CONTEXT_ERROR)
                return
            except Exception as exc:
                logger.warning(
                    "Chat context preparation failed for session %s (%s)",
                    request.session_id,
                    type(exc).__name__,
                )
                yield StreamEvent(error=SAFE_STREAM_ERROR)
                return

            first_usage: dict[str, Any] = {}
            for attempt in range(2):
                call_messages = copy.deepcopy(messages)
                if attempt:
                    call_messages.append(
                        {"role": "user", "content": _RETRY_CORRECTION}
                    )
                try:
                    upstream = self._stream_completion(call_messages)
                except Exception as exc:
                    logger.warning(
                        "Streaming provider failed for session %s (%s)",
                        request.session_id,
                        type(exc).__name__,
                    )
                    yield StreamEvent(error=SAFE_STREAM_ERROR)
                    return
                buffer = ""
                accepted = ""
                usage: dict[str, Any] = {}
                validated = False
                refused = False
                invalid = False
                ended_normally = False
                provider_failed = False
                closed_cleanly = False
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
                            if not has_substantive_output(buffer):
                                continue
                            classification = classify_refusal_opening(buffer)
                            if classification is RefusalOpeningState.REFUSAL:
                                refused = True
                                break
                            if (
                                classification
                                is RefusalOpeningState.POSSIBLE_REFUSAL_PREFIX
                            ):
                                continue
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
                except Exception as exc:
                    logger.warning(
                        "Streaming provider failed for session %s (%s)",
                        request.session_id,
                        type(exc).__name__,
                    )
                    provider_failed = True
                finally:
                    closed_cleanly = await self._close_upstream(
                        upstream,
                        request.session_id,
                    )

                if provider_failed:
                    yield StreamEvent(error=SAFE_STREAM_ERROR)
                    return

                if not closed_cleanly:
                    yield StreamEvent(error=SAFE_STREAM_ERROR)
                    return

                if ended_normally and not validated:
                    classification = classify_refusal_opening(
                        buffer, final=True
                    )
                    if classification is RefusalOpeningState.REFUSAL:
                        refused = True
                    elif (
                        not has_substantive_output(buffer)
                        or classification
                        is RefusalOpeningState.POSSIBLE_REFUSAL_PREFIX
                    ):
                        invalid = True
                    else:
                        accepted = normalize_opening(buffer)
                        validated = True
                        if accepted:
                            yield StreamEvent(delta=accepted)

                if refused or invalid:
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

    @staticmethod
    async def _close_upstream(
        upstream: AsyncIterator[str | dict[str, Any]],
        session_id: int,
    ) -> bool:
        try:
            close = getattr(upstream, "aclose", None)
            if close is not None:
                await close()
        except Exception as exc:
            logger.warning(
                "Streaming provider cleanup failed for session %s (%s)",
                session_id,
                type(exc).__name__,
            )
            return False
        return True

    async def _build_messages(
        self,
        request: TurnRequest,
        *,
        records: Sequence[ChatRecord] | None = None,
        discard_invalidated_context: bool = False,
    ) -> list[dict[str, Any]]:
        if records is None:
            records = await self.store.load_chat(request.session_id)
        domain = await self._load_domain_context(request, records)
        if discard_invalidated_context:
            domain = replace(
                domain,
                story_state="",
                character_profiles=(),
                assets="",
                episodes=(),
                rag=(),
                overall_summary="",
            )
        recent = [
            {"role": record.role, "content": record.content}
            for record in records
        ]
        character = copy.deepcopy(dict(request.character))
        is_new = (
            not records
            if request.is_new_conversation is None
            else request.is_new_conversation
        )
        if domain.history_seed and is_new:
            character["_history_seed"] = copy.deepcopy(list(domain.history_seed))
        entries = [
            entry
            if isinstance(entry, WorldBookEntry)
            else WorldBookEntry.model_validate(entry)
            for entry in domain.worldbook_entries
        ]
        recent_text = " ".join(
            message["content"] for message in recent[-10:]
        ) + " " + request.content
        before, after = _match_worldbook_entries(entries, recent_text)
        ordered_character_context = _legacy_character_context(
            character,
            user_name=request.user_name,
            user_persona=domain.user_persona,
            persona_position=domain.persona_position,
            worldbook_before=before,
            worldbook_after=after,
        )
        context_persona = (
            domain.user_persona
            if domain.persona_position != "none"
            else ""
        )
        sources = ContextSources(
            character=character,
            user_name=request.user_name,
            user_persona=context_persona,
            story_state=domain.story_state,
            memory=domain.memory,
            character_profiles=list(domain.character_profiles),
            assets=domain.assets,
            episodes=list(domain.episodes),
            rag=list(domain.rag),
            summary=domain.overall_summary,
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

    async def _load_domain_context(
        self,
        request: TurnRequest,
        records: Sequence[ChatRecord],
    ) -> DomainContext:
        if self._domain_context_loader is None:
            return DomainContext(
                user_persona=request.user_persona,
                persona_position=request.persona_position,
                worldbook_entries=request.worldbook_entries,
                story_state=request.story_state,
                memory=request.memory,
                character_profiles=request.character_profiles,
                assets=request.assets,
                episodes=request.episodes,
                rag=request.rag,
                overall_summary=request.overall_summary,
                history_seed=request.history_seed,
            )
        loaded = self._domain_context_loader(tuple(records))
        if inspect.isawaitable(loaded):
            loaded = await loaded
        if not isinstance(loaded, DomainContext):
            raise TypeError("domain context loader returned an invalid value")
        return loaded

    async def _submit_memory(self, session_id: int) -> None:
        if self.memory_manager is None:
            return
        try:
            result = self.memory_manager.submit(session_id)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            logger.warning(
                "Committed chat turn but failed to submit managed memory for session %s (%s)",
                session_id,
                type(exc).__name__,
            )

    @staticmethod
    def _character_name(request: TurnRequest) -> str:
        return str(request.character.get("name") or "角色")

    @staticmethod
    def _final_pair(records: Sequence[ChatRecord]) -> tuple[ChatRecord, ChatRecord]:
        if (
            len(records) < 2
            or records[-2].role != "user"
            or records[-1].role != "assistant"
        ):
            raise ValueError("chat must end with a complete user/assistant pair")
        return records[-2], records[-1]

    @staticmethod
    def _opening_guard_ready(buffer: str) -> bool:
        threshold = max(1, settings.stream_guard_chars)
        if len(buffer) >= threshold:
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
