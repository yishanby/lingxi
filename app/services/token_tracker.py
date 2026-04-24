"""Token usage tracking service."""

from __future__ import annotations

import datetime
import logging
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session
from app.models.tables import TokenUsage

logger = logging.getLogger(__name__)


async def record_usage(
    *,
    session_id: int | None = None,
    user_id: str | None = None,
    character_name: str = "",
    model: str = "",
    usage: dict[str, int],
) -> None:
    """Save a token usage record to the database."""
    try:
        async with async_session() as db:
            record = TokenUsage(
                session_id=session_id,
                user_id=user_id,
                character_name=character_name,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                model=model,
            )
            db.add(record)
            await db.commit()
    except Exception:
        logger.exception("Failed to record token usage")


async def get_usage_stats(
    user_id: str | None = None,
    days: int = 30,
) -> dict[str, Any]:
    """Return aggregated token usage stats."""
    since = datetime.datetime.utcnow() - datetime.timedelta(days=days)
    async with async_session() as db:
        base = select(
            func.coalesce(func.sum(TokenUsage.prompt_tokens), 0).label("prompt_tokens"),
            func.coalesce(func.sum(TokenUsage.completion_tokens), 0).label("completion_tokens"),
            func.coalesce(func.sum(TokenUsage.total_tokens), 0).label("total_tokens"),
            func.count(TokenUsage.id).label("request_count"),
        ).where(TokenUsage.created_at >= since)

        if user_id:
            base = base.where(TokenUsage.user_id == user_id)

        row = (await db.execute(base)).one()
        totals = {
            "prompt_tokens": row.prompt_tokens,
            "completion_tokens": row.completion_tokens,
            "total_tokens": row.total_tokens,
            "request_count": row.request_count,
            "days": days,
        }

        # By character
        char_q = (
            select(
                TokenUsage.character_name,
                func.coalesce(func.sum(TokenUsage.total_tokens), 0).label("total_tokens"),
                func.count(TokenUsage.id).label("request_count"),
            )
            .where(TokenUsage.created_at >= since)
            .group_by(TokenUsage.character_name)
            .order_by(func.sum(TokenUsage.total_tokens).desc())
        )
        if user_id:
            char_q = char_q.where(TokenUsage.user_id == user_id)

        char_rows = (await db.execute(char_q)).all()
        by_character = [
            {"character_name": r.character_name or "(unknown)", "total_tokens": r.total_tokens, "request_count": r.request_count}
            for r in char_rows
        ]

        # By model
        model_q = (
            select(
                TokenUsage.model,
                func.coalesce(func.sum(TokenUsage.total_tokens), 0).label("total_tokens"),
                func.count(TokenUsage.id).label("request_count"),
            )
            .where(TokenUsage.created_at >= since)
            .group_by(TokenUsage.model)
            .order_by(func.sum(TokenUsage.total_tokens).desc())
        )
        if user_id:
            model_q = model_q.where(TokenUsage.user_id == user_id)

        model_rows = (await db.execute(model_q)).all()
        by_model = [
            {"model": r.model or "(unknown)", "total_tokens": r.total_tokens, "request_count": r.request_count}
            for r in model_rows
        ]

        return {**totals, "by_character": by_character, "by_model": by_model}
