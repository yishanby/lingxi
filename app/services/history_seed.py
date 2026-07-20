"""History seed extraction – pull example pairs from prior sessions for few-shot."""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.tables import Session
from app.services import memory
from app.services.token_utils import estimate_tokens

logger = logging.getLogger(__name__)

# Max token budget for seed examples
SEED_MAX_TOKENS = 2000
# Number of exchange pairs to extract
SEED_PAIRS = 3


async def get_history_seed(
    db: AsyncSession,
    character_id: int,
    exclude_session_id: int | None = None,
) -> list[dict[str, str]]:
    """Extract 2-3 user/assistant exchange pairs from prior sessions of this character.

    Returns a list of {"role": ..., "content": ...} dicts suitable for injecting
    into the prompt as few-shot examples.
    """
    # Newest candidates first; each candidate still contributes pairs in its
    # original chronological order.
    stmt = select(Session).where(
        Session.character_id == character_id,
    ).order_by(Session.created_at.desc()).limit(10)

    result = await db.execute(stmt)
    sessions = result.scalars().all()

    seed_messages: list[dict[str, str]] = []
    total_tokens = 0

    pairs_found = 0
    for sess in sessions:
        if exclude_session_id and sess.id == exclude_session_id:
            continue

        chat_path = memory.md_store.file_path(sess.id, "chat.md")
        if chat_path.exists():
            try:
                records = await memory.md_store.load_chat(sess.id)
            except Exception as exc:
                logger.warning(
                    "Skipping unreadable Markdown history seed for session %s (%s)",
                    sess.id,
                    type(exc).__name__,
                )
                continue
            messages = [
                {"role": record.role, "content": record.content}
                for record in records
            ]
        else:
            messages = json.loads(sess.messages) if sess.messages else []
        if len(messages) < 4:
            continue

        # Skip the first message (usually a greeting), take pairs from early conversation
        # where the model's RP behavior is most established
        i = 2  # start after first exchange
        while i < len(messages) - 1 and pairs_found < SEED_PAIRS:
            user_msg = messages[i]
            asst_msg = messages[i + 1] if i + 1 < len(messages) else None

            if not asst_msg:
                break

            if user_msg.get("role") == "user" and asst_msg.get("role") == "assistant":
                user_tokens = estimate_tokens(user_msg["content"])
                asst_tokens = estimate_tokens(asst_msg["content"])

                # Skip very long messages to keep seed compact
                if user_tokens > 200 or asst_tokens > 500:
                    i += 2
                    continue

                if total_tokens + user_tokens + asst_tokens > SEED_MAX_TOKENS:
                    break

                seed_messages.append({"role": "user", "content": user_msg["content"]})
                seed_messages.append({"role": "assistant", "content": asst_msg["content"]})
                total_tokens += user_tokens + asst_tokens
                pairs_found += 1
                i += 2
            else:
                i += 1

        if pairs_found >= SEED_PAIRS or total_tokens >= SEED_MAX_TOKENS:
            break

    if seed_messages:
        logger.info(
            "Extracted %d seed messages (%d tokens) for character %d",
            len(seed_messages), total_tokens, character_id,
        )

    return seed_messages
