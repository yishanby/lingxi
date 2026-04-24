"""Rolling summary – compresses older chat history to keep context manageable."""

from __future__ import annotations

import json
import logging
from typing import Any

from app.config import settings
from app.database import async_session
from app.services.llm import chat_completion

logger = logging.getLogger(__name__)

SUMMARIZE_SYSTEM_PROMPT = (
    "你是一个专业的故事摘要助手。请根据提供的对话内容生成一份简洁的中文摘要。\n"
    "要求：\n"
    "1. 保留关键剧情发展和重要事件\n"
    "2. 保留人物关系和互动要点\n"
    "3. 保留重要的决定和转折点\n"
    "4. 保留情感变化和氛围转折\n"
    "5. 使用第三人称叙述\n"
    f"6. 摘要总长度控制在{settings.summary_max_chars}字以内\n"
    "如果提供了之前的摘要，请将新内容与之前的摘要合并为一份完整的摘要。"
)


async def maybe_summarize(session_id: int, backend: dict[str, Any]) -> None:
    """Check whether a summary pass is needed and run it if so.

    Uses its own database session so the caller's transaction is unaffected.
    Failures are logged but never propagated.
    """
    try:
        async with async_session() as db:
            from app.models.tables import Session

            session = await db.get(Session, session_id)
            if session is None:
                return

            messages: list[dict[str, Any]] = json.loads(session.messages) if session.messages else []
            total = len(messages)
            summary_up_to = session.summary_up_to or 0
            window = settings.summary_recent_window
            threshold = settings.summary_trigger_threshold

            unsummarised = total - summary_up_to - window
            if unsummarised < threshold:
                return

            # Range to compress: [summary_up_to, total - window)
            to_compress = messages[summary_up_to : total - window]
            if not to_compress:
                return

            # Build the summarisation request
            user_parts: list[str] = []
            if session.summary:
                user_parts.append(f"【之前的摘要】\n{session.summary}")
            user_parts.append("【需要压缩的新对话】")
            for m in to_compress:
                role_label = "用户" if m.get("role") == "user" else "角色"
                user_parts.append(f"{role_label}: {m.get('content', '')}")

            llm_messages = [
                {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT},
                {"role": "user", "content": "\n".join(user_parts)},
            ]

            result = await chat_completion(
                provider=backend["provider"],
                api_key=backend["api_key"],
                model=backend["model"],
                base_url=backend["base_url"],
                messages=llm_messages,
                params=backend.get("params", {}),
            )

            new_summary = result["content"]

            # Truncate if LLM exceeded the limit
            if len(new_summary) > settings.summary_max_chars:
                new_summary = new_summary[: settings.summary_max_chars]

            session.summary = new_summary
            session.summary_up_to = total - window
            await db.commit()

            logger.info(
                "Summarised session %d: compressed messages [%d:%d], summary_up_to=%d",
                session_id, summary_up_to, total - window, session.summary_up_to,
            )
    except Exception:
        logger.exception("Summary generation failed for session %d", session_id)
