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
from app.services.stage_receipts import (
    RECEIPT_MAX_BYTES,
    RECEIPT_STAGES,
    ArtifactIdentity,
    ChatSourceIdentity,
    StageName,
    StageReceipt,
    StageUpdateResult,
    chat_source_identity,
    parse_receipt,
    receipt_path,
    text_artifact,
    validate_stage_result,
    write_receipt,
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
        *,
        source: ChatSourceIdentity,
    ) -> StageUpdateResult:
        return await story_memory.update_story_state(
            self.store,
            session_id,
            records,
            self._complete_text(backend),
            source=source,
        )

    async def _save_changes(
        self,
        session_id: int,
        *,
        expected_source: ChatSourceIdentity | None = None,
        **changes: Any,
    ) -> None:
        async with self.store.transaction(session_id) as transaction:
            state = await transaction.load_state()
            updated = replace(state, **changes)
            if not updated.rebuild_required:
                updated = replace(updated, rebuild_from_message=None)
            if expected_source is not None:
                current = chat_source_identity(await transaction.load_chat())
                if current != expected_source:
                    raise RuntimeError("source changed before state save")
            await transaction.save_state(updated)

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

    async def _require_stage_result(
        self,
        session_id: int,
        result: object,
        *,
        stage: StageName,
        source: ChatSourceIdentity,
        expected_checkpoint: int | None,
        persist_receipt: bool = False,
    ) -> StageUpdateResult:
        validate_stage_result(result, stage)  # type: ignore[arg-type]
        assert isinstance(result, StageUpdateResult)
        if result.source != source or (
            expected_checkpoint is not None
            and result.checkpoint != expected_checkpoint
        ):
            raise RuntimeError(f"{stage} stage did not return a completed result")
        async with self.store.transaction(session_id) as transaction:
            current = chat_source_identity(await transaction.load_chat())
            if current != source:
                raise RuntimeError(f"{stage} stage source changed before completion")
            if persist_receipt:
                await self._verify_owned_artifacts(
                    transaction,
                    stage,
                    result.checkpoint,
                    result.artifacts,
                    result.inputs,
                )
                await self._verify_stage_inputs(
                    transaction,
                    stage,
                    result.checkpoint,
                    result.inputs,
                )
                await write_receipt(transaction, result)
        return result

    def _owned_artifact_paths(self, session_id: int, stage: StageName) -> set[str]:
        session = self.store.session_dir(session_id)
        if stage == "story":
            candidates = ("story_state.md",)
        elif stage == "memory":
            candidates = ("memory.md",)
            directory = session / "characters"
            return {
                *(
                    {"memory.md"}
                    if self.store.file_path(session_id, "memory.md").is_file()
                    else set()
                ),
                *(
                    {f"characters/{path.name}" for path in directory.glob("*.md")}
                    if directory.exists()
                    else set()
                ),
            }
        elif stage == "summary":
            candidates = ("summary.md",)
        elif stage == "assets":
            candidates = ("assets.md", "assets_summary.txt")
        else:
            raise ValueError("stage does not use owned rebuild artifacts")
        return {
            relative
            for relative in candidates
            if self.store.file_path(session_id, relative).is_file()
        }

    async def _verify_owned_artifacts(
        self,
        transaction: Any,
        stage: StageName,
        checkpoint: int,
        artifacts: tuple[ArtifactIdentity, ...],
        inputs: tuple[ArtifactIdentity, ...],
    ) -> None:
        artifact_paths = {artifact.path for artifact in artifacts}
        valid_shape = False
        if stage == "story":
            valid_shape = artifact_paths == {"story_state.md"}
        elif stage == "memory":
            profile_paths = artifact_paths - {"memory.md"}
            valid_shape = "memory.md" in artifact_paths and all(
                path.startswith("characters/")
                and path.count("/") == 1
                and path.removeprefix("characters/").endswith(".md")
                and path.removeprefix("characters/") != ".md"
                for path in profile_paths
            )
        elif stage == "assets":
            valid_shape = artifact_paths in (
                set(),
                {"assets.md", "assets_summary.txt"},
            )
        elif stage == "summary":
            valid_shape = (
                artifact_paths == {"summary.md"}
                if checkpoint > 0 or inputs
                else artifact_paths == set()
            )
        if not valid_shape:
            raise RuntimeError(f"{stage} stage did not reconcile required artifacts")

        expected_paths = self._owned_artifact_paths(transaction.session_id, stage)
        if artifact_paths != expected_paths:
            raise RuntimeError(f"{stage} stage did not reconcile required artifacts")
        for artifact in artifacts:
            text = await transaction.read_text(artifact.path)
            if text_artifact(artifact.path, text) != artifact:
                raise RuntimeError(f"{stage} stage artifact verification failed")

    async def _verify_stage_inputs(
        self,
        transaction: Any,
        stage: StageName,
        checkpoint: int,
        inputs: tuple[ArtifactIdentity, ...],
    ) -> None:
        expected: tuple[ArtifactIdentity, ...] = ()
        if stage == "summary":
            try:
                episodes = await story_memory.load_summary_episode_chain(
                    self.store,
                    transaction,
                )
            except ValueError as exc:
                raise RuntimeError("summary stage input verification failed") from exc
            frontier = episodes[-1].end if episodes else 0
            if frontier != checkpoint:
                raise RuntimeError("summary stage input verification failed")
            expected = tuple(
                text_artifact(
                    f"episodes/episode-{episode.number:06d}.md",
                    episode.text,
                )
                for episode in episodes
            )
        if inputs != expected:
            raise RuntimeError(f"{stage} stage input verification failed")

    async def _load_verified_receipt(
        self,
        session_id: int,
        stage: StageName,
        source: ChatSourceIdentity,
        expected_checkpoint: int | None,
    ) -> StageUpdateResult | None:
        if stage not in RECEIPT_STAGES:
            return None
        relative = receipt_path(stage)
        try:
            path = self.store.file_path(session_id, relative)
        except ValueError:
            await self.store.delete_file(session_id, relative)
            return None
        if not path.is_file():
            return None
        if path.stat().st_size > RECEIPT_MAX_BYTES:
            await self.store.delete_file(session_id, relative)
            return None
        async with self.store.transaction(session_id) as transaction:
            try:
                receipt: StageReceipt = parse_receipt(
                    await transaction.read_text(relative)
                )
            except ValueError:
                await transaction.delete_file(relative)
                return None
            if (
                receipt.stage != stage
                or receipt.source != source
                or (
                    expected_checkpoint is not None
                    and receipt.checkpoint != expected_checkpoint
                )
            ):
                await transaction.delete_file(relative)
                return None
            try:
                await self._verify_owned_artifacts(
                    transaction,
                    stage,
                    receipt.checkpoint,
                    receipt.artifacts,
                    receipt.inputs,
                )
                await self._verify_stage_inputs(
                    transaction,
                    stage,
                    receipt.checkpoint,
                    receipt.inputs,
                )
            except RuntimeError:
                await transaction.delete_file(relative)
                return None
            return StageUpdateResult(
                stage=receipt.stage,
                completed=True,
                source=receipt.source,
                checkpoint=receipt.checkpoint,
                artifacts=receipt.artifacts,
                inputs=receipt.inputs,
            )

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
        source = chat_source_identity(records)
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
                force_story = state.rebuild_story_required
                pending = self._pending_records(
                    records,
                    state.last_story_state_message,
                )
                result = (
                    await self._load_verified_receipt(
                        session_id,
                        "story",
                        source,
                        total,
                    )
                    if force_story
                    else None
                )
                if result is None:
                    result = await self.update_story(
                        session_id,
                        pending,
                        backend,
                        source=source,
                    )
                await self._require_stage_result(
                    session_id,
                    result,
                    stage="story",
                    source=source,
                    expected_checkpoint=total,
                    persist_receipt=force_story,
                )
                await self._save_changes(
                    session_id,
                    expected_source=source,
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
                force_memory = state.rebuild_memory_required
                result = (
                    await self._load_verified_receipt(
                        session_id,
                        "memory",
                        source,
                        total,
                    )
                    if force_memory
                    else None
                )
                if result is None and force_memory:
                    result = await memory.rebuild_memory_and_profiles(
                        session_id,
                        self._messages(records),
                        backend,
                        store=self.store,
                        source=source,
                    )
                elif result is None:
                    pending = self._pending_records(
                        records,
                        state.last_memory_message,
                    )
                    result = await memory.extract_memory_and_characters(
                        session_id,
                        self._messages(pending),
                        backend,
                        store=self.store,
                        source=source,
                    )
                await self._require_stage_result(
                    session_id,
                    result,
                    stage="memory",
                    source=source,
                    expected_checkpoint=total,
                    persist_receipt=force_memory,
                )
                await self._save_changes(
                    session_id,
                    expected_source=source,
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
                result = await story_memory.create_due_episodes(
                    self.store,
                    session_id,
                    records,
                    state.last_episode_message,
                    self._complete_text(backend),
                    episode_size=settings.episode_size_messages,
                    source=source,
                )
                result = await self._require_stage_result(
                    session_id,
                    result,
                    stage="episode",
                    source=source,
                    expected_checkpoint=None,
                )
                await self._save_changes(
                    session_id,
                    expected_source=source,
                    last_episode_message=result.checkpoint,
                    rebuild_episode_required=False,
                    last_error="",
                )

            state = await self._load_state(session_id)
            force_summary = state.rebuild_summary_required
            if force_summary or state.last_episode_message > state.last_summary_message:
                result = (
                    await self._load_verified_receipt(
                        session_id,
                        "summary",
                        source,
                        state.last_episode_message,
                    )
                    if force_summary
                    else None
                )
                if result is None:
                    result = await story_memory.update_summary_from_episodes(
                        self.store,
                        session_id,
                        state.last_summary_message,
                        self._complete_text(backend),
                        max_tokens=settings.summary_max_tokens,
                        source=source,
                    )
                result = await self._require_stage_result(
                    session_id,
                    result,
                    stage="summary",
                    source=source,
                    expected_checkpoint=state.last_episode_message,
                    persist_receipt=force_summary,
                )
                await self._save_changes(
                    session_id,
                    expected_source=source,
                    last_summary_message=result.checkpoint,
                    rebuild_summary_required=False,
                    last_error="",
                )

            state = await self._load_state(session_id)
            if (
                state.rebuild_rag_required or
                total - state.last_rag_message
                >= settings.rag_index_interval_messages
            ):
                result = await rag.build_index(
                    session_id,
                    store=self.store,
                    source=source,
                    force_rebuild=state.rebuild_rag_required,
                    embedding_base_url=settings.rag_embedding_base_url,
                    embedding_api_key=settings.rag_embedding_api_key,
                    embedding_model=settings.rag_embedding_model,
                )
                await self._require_stage_result(
                    session_id,
                    result,
                    stage="rag",
                    source=source,
                    expected_checkpoint=total,
                )
                await self._save_changes(
                    session_id,
                    expected_source=source,
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
                force_assets = state.rebuild_assets_required
                result = (
                    await self._load_verified_receipt(
                        session_id,
                        "assets",
                        source,
                        total,
                    )
                    if force_assets
                    else None
                )
                if result is None and force_assets:
                    result = await memory.rebuild_assets(
                        session_id,
                        self._messages(records),
                        backend,
                        store=self.store,
                        source=source,
                    )
                elif result is None:
                    pending = self._pending_records(
                        records,
                        state.last_assets_message,
                    )
                    result = await memory.update_assets(
                        session_id,
                        self._messages(pending),
                        backend,
                        store=self.store,
                        source=source,
                    )
                await self._require_stage_result(
                    session_id,
                    result,
                    stage="assets",
                    source=source,
                    expected_checkpoint=total,
                    persist_receipt=force_assets,
                )
                await self._save_changes(
                    session_id,
                    expected_source=source,
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
