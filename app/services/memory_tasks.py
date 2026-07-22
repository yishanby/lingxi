"""Lifecycle-managed background execution for the memory pipeline."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from app.config import settings
from app.services import memory
from app.services.md_store import MemoryState
from app.services.memory_pipeline import MemoryPipeline

logger = logging.getLogger(__name__)

BackendResolver = Callable[[int], Awaitable[dict[str, Any]]]


class MemoryTaskManager:
    """Deduplicate session work and run it through one managed worker."""

    def __init__(
        self,
        pipeline: MemoryPipeline,
        backend_resolver: BackendResolver,
    ) -> None:
        self.pipeline = pipeline
        self.backend_resolver = backend_resolver
        self.queue: asyncio.Queue[int] = asyncio.Queue()
        self.pending: set[int] = set()
        self.running: set[int] = set()
        self.rerun: set[int] = set()
        self.worker: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self.worker is not None and not self.worker.done():
            return
        self.worker = asyncio.create_task(
            self._run_worker(),
            name="memory-v2-worker",
        )

    async def stop(self) -> None:
        worker = self.worker
        if worker is None:
            return
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        self.worker = None

    async def submit(self, session_id: int) -> None:
        if session_id in self.pending:
            if session_id in self.running:
                self.rerun.add(session_id)
            return
        self.pending.add(session_id)
        await self.queue.put(session_id)

    @staticmethod
    def _has_pending(
        total: int,
        state: MemoryState,
        *,
        has_invalidation_intent: bool = False,
    ) -> bool:
        if (
            has_invalidation_intent
            or state.rebuild_required
            or state.checkpoint_exceeds(total)
        ):
            return True
        return any(
            (
                total - state.last_story_state_message
                >= settings.story_state_interval_messages,
                total - state.last_memory_message
                >= settings.memory_extract_interval_messages,
                total - state.last_episode_message
                >= settings.episode_size_messages,
                state.last_episode_message > state.last_summary_message,
                total - state.last_rag_message
                >= settings.rag_index_interval_messages,
                total - state.last_assets_message
                >= settings.assets_interval_messages,
            )
        )

    async def scan_pending_sessions(self) -> None:
        """Submit every persisted numeric session with checkpointed work due."""
        # Disabled on startup: dozens of backlogged sessions flood the LLM
        # proxy queue.  Pipeline will run on-demand when new messages arrive.
        return
        for directory in sorted(store.base.iterdir(), key=lambda path: path.name):
            if not directory.is_dir() or not directory.name.isdecimal():
                continue
            session_id = int(directory.name)
            if session_id <= 0 or str(session_id) != directory.name:
                continue
            try:
                async with store.transaction(session_id) as transaction:
                    records = await transaction.load_chat()
                    intent = await transaction.load_invalidation_intent()
                total = records[-1].number if records else 0
                state = await memory.migrate_legacy_extract_checkpoint(
                    store,
                    session_id,
                    total,
                )
                if self._has_pending(
                    total,
                    state,
                    has_invalidation_intent=intent is not None,
                ):
                    await self.submit(session_id)
            except Exception:
                logger.exception(
                    "Failed to scan memory state for session %s",
                    session_id,
                )

    async def _run_worker(self) -> None:
        while True:
            session_id = await self.queue.get()
            cancelled = False
            self.running.add(session_id)
            try:
                # Delay pipeline execution to avoid competing with
                # user requests for the LLM proxy (which is serial).
                await asyncio.sleep(15)
                backend = await self.backend_resolver(session_id)
                await self.pipeline.run(session_id, backend)
            except asyncio.CancelledError:
                cancelled = True
                raise
            except Exception:
                logger.exception(
                    "Memory pipeline failed for session %s",
                    session_id,
                )
            finally:
                self.running.discard(session_id)
                if not cancelled and session_id in self.rerun:
                    self.rerun.discard(session_id)
                    self.queue.put_nowait(session_id)
                else:
                    self.rerun.discard(session_id)
                    self.pending.discard(session_id)
                self.queue.task_done()
