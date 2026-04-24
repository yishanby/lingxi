"""Token usage statistics endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.services.token_tracker import get_usage_stats

router = APIRouter(prefix="/api/stats", tags=["stats"])


@router.get("/tokens")
async def token_stats(
    user_id: str | None = Query(None),
    days: int = Query(30, ge=1, le=365),
):
    """Return aggregated token usage statistics."""
    return await get_usage_stats(user_id=user_id, days=days)
