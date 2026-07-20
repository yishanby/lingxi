"""Shared DB + Markdown protocol for creating V2 sessions."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Sequence

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.tables import Session
from app.services.md_store import ChatRecord, MarkdownMemoryStore


logger = logging.getLogger(__name__)


class SessionInitializationError(RuntimeError):
    """A new session could not be initialized consistently."""


_CANONICAL_SESSION_DIRECTORY = re.compile(r"[1-9][0-9]*\Z")


def _markdown_session_floor(store: MarkdownMemoryStore) -> int:
    floor = 0
    try:
        candidates = list(store.base.iterdir())
    except OSError:
        return floor
    for candidate in candidates:
        if _CANONICAL_SESSION_DIRECTORY.fullmatch(candidate.name) is None:
            continue
        try:
            if (
                not candidate.is_dir()
                or candidate.is_symlink()
                or candidate.resolve() != candidate
            ):
                continue
        except OSError:
            continue
        floor = max(floor, int(candidate.name))
    return floor


async def _raise_sqlite_sequence_floor(db: AsyncSession, floor: int) -> None:
    if floor < 0:
        raise ValueError("session sequence floor must be nonnegative")
    await db.execute(
        text(
            "UPDATE sqlite_sequence "
            "SET seq=CASE WHEN seq < :floor THEN :floor ELSE seq END "
            "WHERE name='sessions'"
        ),
        {"floor": floor},
    )
    await db.execute(
        text(
            "INSERT INTO sqlite_sequence(name, seq) "
            "SELECT 'sessions', :floor "
            "WHERE NOT EXISTS ("
            "SELECT 1 FROM sqlite_sequence WHERE name='sessions')"
        ),
        {"floor": floor},
    )


async def _rollback_reservation(db: AsyncSession) -> None:
    try:
        await db.rollback()
    except Exception as exc:
        logger.warning(
            "Session ID reservation rollback failed (%s)",
            type(exc).__name__,
        )


async def _reserve_session_id(
    db: AsyncSession,
    store: MarkdownMemoryStore,
) -> int | None:
    """Persist an ID reservation before any DB row or Markdown write."""
    if not isinstance(db, AsyncSession):
        return None
    if db.new or db.dirty or db.deleted:
        raise SessionInitializationError(
            "cannot reserve a session id with pending database changes"
        )

    try:
        await db.rollback()
        await _raise_sqlite_sequence_floor(db, _markdown_session_floor(store))
        await db.execute(
            text(
                "UPDATE sqlite_sequence SET seq=seq+1 "
                "WHERE name='sessions'"
            )
        )
        reserved_id = int(
            (
                await db.execute(
                    text(
                        "SELECT seq FROM sqlite_sequence "
                        "WHERE name='sessions'"
                    )
                )
            ).scalar_one()
        )
        await db.commit()
        return reserved_id
    except (Exception, asyncio.CancelledError) as exc:
        cleanup = asyncio.create_task(_rollback_reservation(db))
        cancelled_during_cleanup = await _finish_compensation(cleanup)
        if isinstance(exc, asyncio.CancelledError) or cancelled_during_cleanup:
            raise asyncio.CancelledError from exc
        logger.warning(
            "Session ID reservation failed (%s)",
            type(exc).__name__,
        )
        raise SessionInitializationError from exc


async def _compensate_creation(
    db: AsyncSession,
    store: MarkdownMemoryStore,
    session_id: int | None,
    *,
    owns_chat: bool,
) -> None:
    try:
        await db.rollback()
    except Exception as exc:
        logger.warning(
            "Session creation rollback failed (%s)",
            type(exc).__name__,
        )

    if isinstance(session_id, int) and session_id > 0:
        try:
            persisted = await db.get(Session, session_id)
            if persisted is not None:
                await db.delete(persisted)
                await db.commit()
        except Exception as exc:
            logger.warning(
                "Session creation row cleanup failed (%s)",
                type(exc).__name__,
            )
            try:
                await db.rollback()
            except Exception as rollback_exc:
                logger.warning(
                    "Session creation cleanup rollback failed (%s)",
                    type(rollback_exc).__name__,
                )

        if owns_chat:
            try:
                await store.delete_file(session_id, "chat.md")
            except Exception as exc:
                logger.warning(
                    "Session creation Markdown cleanup failed (%s)",
                    type(exc).__name__,
                )


async def _finish_compensation(task: asyncio.Task[None]) -> bool:
    """Wait for cleanup despite cancellation; report any new cancellation."""
    cancelled = False
    while True:
        try:
            await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            cancelled = True
            if task.done():
                break
    task.result()
    return cancelled


async def create_session_with_markdown(
    db: AsyncSession,
    store: MarkdownMemoryStore,
    session: Session,
    initial_records: Sequence[ChatRecord],
) -> Session:
    """Create one DB row and one create-only authoritative chat document."""
    reserved_id = await _reserve_session_id(db, store)
    if reserved_id is not None:
        session.id = reserved_id
    owns_chat = False
    try:
        db.add(session)
        await db.flush()
        await store.create_chat(session.id, initial_records)
        owns_chat = True
        await db.commit()
        await db.refresh(session)
        return session
    except (Exception, asyncio.CancelledError) as exc:
        cleanup = asyncio.create_task(
            _compensate_creation(
                db,
                store,
                session.id,
                owns_chat=owns_chat,
            )
        )
        cancelled_during_cleanup = await _finish_compensation(cleanup)
        if isinstance(exc, asyncio.CancelledError) or cancelled_during_cleanup:
            raise asyncio.CancelledError from exc
        logger.warning(
            "Session initialization failed (%s)",
            type(exc).__name__,
        )
        raise SessionInitializationError from exc
