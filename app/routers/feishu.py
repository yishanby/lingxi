"""Feishu webhook / event subscription router."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.tables import Character, Session
from app.services import memory as memory_service
from app.services import feishu_client
from app.services.chat_service import ChatService, ContextBudgetExceeded
from app.services.llm import chat_completion
from app.services.output_guard import OutputGuardError
from app.services.token_tracker import record_usage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/feishu", tags=["feishu"])


@router.post("/webhook")
async def feishu_webhook(request: Request, db: AsyncSession = Depends(get_db)):
    """Main entry point for Feishu event subscriptions (HTTP mode).

    Handles:
    - URL verification challenge
    - im.chat.member.bot.added_v1 → send character selection card
    - im.message.receive_v1 → route chat messages
    - card.action.trigger → handle interactive card actions
    """
    body = await request.json()

    # ── URL Verification ─────────────────────────────────────────────────
    if body.get("type") == "url_verification":
        return {"challenge": body.get("challenge", "")}

    # ── Event callback v2 ────────────────────────────────────────────────
    header = body.get("header", {})
    event_type = header.get("event_type", "")
    event = body.get("event", {})

    # Deduplicate (Feishu may retry)
    # In production you'd track event IDs; here we just process.

    if event_type == "im.chat.member.bot.added_v1":
        await _handle_bot_added(event, db)
    elif event_type == "im.message.receive_v1":
        await _handle_message(event, db)

    # ── Card action callback (separate schema) ───────────────────────────
    if body.get("action"):
        await _handle_card_action(body, db)

    return {"code": 0, "msg": "ok"}


# ── Bot added to group ───────────────────────────────────────────────────────

async def _handle_bot_added(event: dict[str, Any], db: AsyncSession) -> None:
    chat_id = event.get("chat_id", "")
    if not chat_id:
        return

    # Fetch all characters for the selection card
    result = await db.execute(select(Character).order_by(Character.name))
    characters = result.scalars().all()
    if not characters:
        await feishu_client.send_text_message(
            chat_id,
            "No characters available yet. Create one via the web UI first!",
        )
        return

    char_list = [{"id": c.id, "name": c.name} for c in characters]
    card = feishu_client.build_character_selection_card(char_list)
    await feishu_client.send_interactive_card(chat_id, card)


# ── Incoming message ─────────────────────────────────────────────────────────

async def _handle_message(event: dict[str, Any], db: AsyncSession) -> None:
    message = event.get("message", {})
    chat_id = message.get("chat_id", "")
    sender = event.get("sender", {}).get("sender_id", {}).get("open_id", "")
    if message.get("message_type", "") != "text":
        return

    content_json = message.get("content", "{}")
    try:
        text = json.loads(content_json).get("text", "").strip()
    except json.JSONDecodeError:
        text = content_json.strip()
    if not text:
        return
    if text.startswith("/"):
        await _handle_command(text, chat_id, sender, db)
        return

    result = await db.execute(
        select(Session).where(
            Session.feishu_chat_id == chat_id, Session.status == "active"
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        await feishu_client.send_text_message(
            chat_id,
            "No active session. Add me to a group or use /switch to select a character.",
        )
        return

    char = await db.get(Character, session.character_id)
    if not char:
        await feishu_client.send_text_message(chat_id, "Character not found.")
        return

    from app.routers import sessions as sessions_router

    try:
        backend = await sessions_router._resolve_backend(session.backend_id, db)
        turn_request = await sessions_router._prepare_turn_request(
            session=session,
            char=char,
            content=text,
            msg_type="ic",
            db=db,
        )

        async def complete(messages):
            return await chat_completion(
                provider=backend["provider"],
                api_key=backend["api_key"],
                model=backend["model"],
                base_url=backend["base_url"],
                messages=messages,
                params=dict(backend["params"]),
            )

        service = ChatService(
            memory_service.md_store,
            sessions_router._managed_memory_manager(),
            completion=complete,
        )
        turn = await service.send(turn_request)
    except ContextBudgetExceeded:
        await feishu_client.send_text_message(
            chat_id, "对话上下文过长，请缩短当前输入后重试。"
        )
        return
    except OutputGuardError:
        await feishu_client.send_text_message(chat_id, "模型拒绝生成，请重试。")
        return
    except Exception:
        logger.warning("LLM call failed in Feishu handler")
        await feishu_client.send_text_message(chat_id, "生成失败，请稍后重试。")
        return

    try:
        await record_usage(
            session_id=session.id,
            user_id=sender,
            character_name=char.name,
            model=backend["model"],
            usage=turn.usage,
        )
    except Exception:
        logger.exception("Failed to record Feishu token usage for session %s", session.id)

    p_tok = turn.usage.get("prompt_tokens", 0)
    c_tok = turn.usage.get("completion_tokens", 0)
    t_tok = turn.usage.get("total_tokens", 0)
    usage_footer = (
        f"\n\n---\n💡 Token: {t_tok:,} "
        f"(输入 {p_tok:,} / 输出 {c_tok:,})"
    )
    card = feishu_client.build_character_reply_card(
        char.name, turn.content + usage_footer
    )
    await feishu_client.send_interactive_card(chat_id, card)

# ── Slash commands ───────────────────────────────────────────────────────────

async def _handle_command(
    text: str, chat_id: str, sender: str, db: AsyncSession
) -> None:
    cmd = text.split()[0].lower()

    if cmd == "/reset":
        result = await db.execute(
            select(Session).where(
                Session.feishu_chat_id == chat_id, Session.status == "active"
            )
        )
        session = result.scalar_one_or_none()
        if session:
            turn_lock = await memory_service.md_store.turn_lock_for(session.id)
            async with turn_lock:
                await memory_service.md_store.write_text(session.id, "chat.md", "")
            # Keep the V1 column empty for compatibility; Markdown is authoritative.
            session.messages = "[]"
            await db.commit()
            await feishu_client.send_text_message(chat_id, "Session reset. Chat history cleared.")
        else:
            await feishu_client.send_text_message(chat_id, "No active session to reset.")

    elif cmd == "/switch":
        # Archive current session and show character picker
        result = await db.execute(
            select(Session).where(
                Session.feishu_chat_id == chat_id, Session.status == "active"
            )
        )
        session = result.scalar_one_or_none()
        if session:
            session.status = "archived"
            await db.commit()

        chars_result = await db.execute(select(Character).order_by(Character.name))
        characters = chars_result.scalars().all()
        if characters:
            char_list = [{"id": c.id, "name": c.name} for c in characters]
            card = feishu_client.build_character_selection_card(char_list)
            await feishu_client.send_interactive_card(chat_id, card)
        else:
            await feishu_client.send_text_message(chat_id, "No characters available.")

    elif cmd == "/info":
        result = await db.execute(
            select(Session).where(
                Session.feishu_chat_id == chat_id, Session.status == "active"
            )
        )
        session = result.scalar_one_or_none()
        if session:
            char = await db.get(Character, session.character_id)
            name = char.name if char else "Unknown"
            msg_count = len(await memory_service.md_store.load_chat(session.id))
            await feishu_client.send_text_message(
                chat_id, f"Character: {name}\nMessages: {msg_count}\nStatus: {session.status}"
            )
        else:
            await feishu_client.send_text_message(chat_id, "No active session.")

    elif cmd == "/stats":
        from app.services.token_tracker import get_usage_stats

        # Parse optional days argument: /stats 7
        parts = text.split()
        days = 30
        if len(parts) > 1:
            try:
                days = int(parts[1])
            except ValueError:
                pass

        stats = await get_usage_stats(user_id=sender, days=days)
        total = stats["total_tokens"]
        prompt = stats["prompt_tokens"]
        completion = stats["completion_tokens"]
        count = stats["request_count"]

        lines = [
            f"📊 Token Usage (last {days} days)",
            f"Total: {total:,} tokens ({count} requests)",
            f"  Prompt: {prompt:,} | Completion: {completion:,}",
        ]
        if stats["by_character"]:
            lines.append("\nBy character:")
            for c in stats["by_character"][:5]:
                lines.append(f"  {c['character_name']}: {c['total_tokens']:,} ({c['request_count']} req)")
        if stats["by_model"]:
            lines.append("\nBy model:")
            for m in stats["by_model"][:5]:
                lines.append(f"  {m['model']}: {m['total_tokens']:,} ({m['request_count']} req)")

        await feishu_client.send_text_message(chat_id, "\n".join(lines))

    else:
        await feishu_client.send_text_message(
            chat_id, "Available commands: /reset, /switch, /info, /stats"
        )


# ── Card action (character selection) ────────────────────────────────────────

async def _handle_card_action(body: dict[str, Any], db: AsyncSession) -> None:
    action = body.get("action", {})
    action_value = action.get("value", {})
    option = action.get("option")

    if not option:
        return

    # The option is the character ID from the select_static
    try:
        char_id = int(option)
    except (ValueError, TypeError):
        return

    # Figure out which chat this came from
    open_chat_id = body.get("open_chat_id", "")
    if not open_chat_id:
        return

    char = await db.get(Character, char_id)
    if not char:
        await feishu_client.send_text_message(open_chat_id, "Character not found.")
        return

    # Create new session
    session = Session(
        character_id=char_id,
        worldbook_ids="[]",
        feishu_chat_id=open_chat_id,
        messages="[]",
        status="active",
    )
    db.add(session)
    await db.commit()

    # Send first message if character has one
    if char.first_message:
        first_msg = char.first_message.replace("{{user}}", "User").replace("{{char}}", char.name)
        card = feishu_client.build_character_reply_card(char.name, first_msg)
        await feishu_client.send_interactive_card(open_chat_id, card)

        # Markdown is authoritative; the V1 column remains compatibility-only.
        await memory_service.md_store.append_record(
            session.id,
            "assistant",
            first_msg,
            name=char.name,
        )
        session.messages = json.dumps(
            [{"role": "assistant", "content": first_msg}],
            ensure_ascii=False,
        )
        await db.commit()
    else:
        await feishu_client.send_text_message(
            open_chat_id,
            f"Session started with {char.name}! Send a message to begin.",
        )
