"""Checkpointed orchestration for Markdown-backed session memory."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

from app.config import settings
from app.services import llm, memory, rag, story_memory
from app.services.md_store import (
    ChatRecord,
    InvalidationIntentError,
    MarkdownMemoryStore,
    MemoryState,
)


class MemoryPipeline:
    """Run recoverable memory stages against one authoritative chat snapshot."""

    def __init__(self, store: MarkdownMemoryStore) -> None:
        self.store = store

    def _complete_text(
        self,
        backend: dict[str, Any],
    ) -> Callable[[list[dict[str, str]]], Awaitable[str]]:
        async def complete(messages: list[dict[str, str]]) -> str:
            result = await llm.chat_completion(
                provider=backend["provider"],
                api_key=backend["api_key"],
                model=backend["model"],
                base_url=backend["base_url"],
                messages=messages,
                params=dict(backend.get("params", {})),
            )
            return result["content"]

        return complete

    async def update_story(
        self,
        session_id: int,
        records: list[ChatRecord],
        backend: dict[str, Any],
    ) -> str:
        return await story_memory.update_story_state(
            self.store,
            session_id,
            records,
            self._complete_text(backend),
        )

    async def _save_changes(self, session_id: int, **changes: Any) -> None:
        async with self.store.transaction(session_id) as transaction:
            state = await transaction.load_state()
            await transaction.save_state(replace(state, **changes))

    async def _load_state(self, session_id: int) -> MemoryState:
        async with self.store.transaction(session_id) as transaction:
            return await transaction.load_state()

    async def _recover_invalidation(
        self,
        session_id: int,
        total: int,
    ) -> MemoryState:
        """Resume cleanup or normalize checkpoints after a committed rollback."""
        async with self.store.transaction(session_id) as transaction:
            return await transaction.recover_pending_invalidation(total)

    @staticmethod
    def _pending_records(
        records: list[ChatRecord],
        checkpoint: int,
    ) -> list[ChatRecord]:
        return [record for record in records if record.number > checkpoint]

    @staticmethod
    def _messages(records: list[ChatRecord]) -> list[dict[str, str]]:
        return [
            {"role": record.role, "content": record.content}
            for record in records
        ]

    async def run(self, session_id: int, backend: dict[str, Any]) -> None:
        lock = await self.store.pipeline_lock_for(session_id)
        async with lock:
            await self._run_serialized(session_id, backend)

    async def _run_serialized(
        self,
        session_id: int,
        backend: dict[str, Any],
    ) -> None:
        async with self.store.transaction(session_id) as transaction:
            records = await transaction.load_chat()

        total = records[-1].number if records else 0
        try:
            state = await self._recover_invalidation(
                session_id,
                total,
            )
            state = await memory.migrate_legacy_extract_checkpoint(
                self.store,
                session_id,
                total,
            )
            if (
                state.rebuild_story_required or
                total - state.last_story_state_message
                >= settings.story_state_interval_messages
            ):
                pending = self._pending_records(
                    records,
                    state.last_story_state_message,
                )
                await self.update_story(session_id, pending, backend)
                await self._save_changes(
                    session_id,
                    last_story_state_message=total,
                    rebuild_story_required=False,
                    last_error="",
                )

            state = await self._load_state(session_id)
            if (
                state.rebuild_memory_required or
                total - state.last_memory_message
                >= settings.memory_extract_interval_messages
            ):
                pending = self._pending_records(records, state.last_memory_message)
                await memory.extract_memory_and_characters(
                    session_id,
                    self._messages(pending),
                    backend,
                )
                await self._save_changes(
                    session_id,
                    last_memory_message=total,
                    last_character_message=total,
                    rebuild_memory_required=False,
                    last_error="",
                )

            state = await self._load_state(session_id)
            force_episode = state.rebuild_episode_required
            if (
                force_episode
                or total - state.last_episode_message
                >= settings.episode_size_messages
            ):
                episode_checkpoint = await story_memory.create_due_episodes(
                    self.store,
                    session_id,
                    records,
                    state.last_episode_message,
                    self._complete_text(backend),
                    episode_size=settings.episode_size_messages,
                )
                await self._save_changes(
                    session_id,
                    last_episode_message=episode_checkpoint,
                    rebuild_episode_required=False,
                    last_error="",
                )

            state = await self._load_state(session_id)
            force_summary = state.rebuild_summary_required
            if force_summary or state.last_episode_message > state.last_summary_message:
                summary_checkpoint = await story_memory.update_summary_from_episodes(
                    self.store,
                    session_id,
                    state.last_summary_message,
                    self._complete_text(backend),
                    max_tokens=settings.summary_max_tokens,
                )
                await self._save_changes(
                    session_id,
                    last_summary_message=summary_checkpoint,
                    rebuild_summary_required=False,
                    last_error="",
                )

            state = await self._load_state(session_id)
            if (
                state.rebuild_rag_required or
                total - state.last_rag_message
                >= settings.rag_index_interval_messages
            ):
                await rag.build_index(
                    session_id,
                    embedding_base_url=settings.rag_embedding_base_url,
                    embedding_api_key=settings.rag_embedding_api_key,
                    embedding_model=settings.rag_embedding_model,
                )
                await self._save_changes(
                    session_id,
                    last_rag_message=total,
                    rebuild_rag_required=False,
                    last_error="",
                )

            state = await self._load_state(session_id)
            if (
                state.rebuild_assets_required or
                total - state.last_assets_message
                >= settings.assets_interval_messages
            ):
                pending = self._pending_records(records, state.last_assets_message)
                await memory.update_assets(
                    session_id,
                    self._messages(pending),
                    backend,
                )
                await self._save_changes(
                    session_id,
                    last_assets_message=total,
                    rebuild_assets_required=False,
                    last_error="",
                )
        except InvalidationIntentError:
            raise
        except Exception as exc:
            await self._save_changes(
                session_id,
                last_error=f"{type(exc).__name__}: {exc}",
            )
            raise
