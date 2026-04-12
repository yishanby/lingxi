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
from app.schemas.api import MessageItem, SessionCreate, WorldBookEntry
from app.services import feishu_client
from app.services.llm import LLMError, chat_completion
from app.services.prompt import assemble_prompt

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
    message_id = message.get("message_id", "")
    sender = event.get("sender", {}).get("sender_id", {}).get("open_id", "")
    msg_type = message.get("message_type", "")

    if msg_type != "text":
        return  # only handle text for now

    content_json = message.get("content", "{}")
    try:
        text = json.loads(content_json).get("text", "").strip()
    except json.JSONDecodeError:
        text = content_json.strip()

    if not text:
        return

    # ── Commands ─────────────────────────────────────────────────────────
    if text.startswith("/"):
        await _handle_command(text, chat_id, sender, db)
        return

    # ── Find active session for this chat ────────────────────────────────
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

    # ── Load character ───────────────────────────────────────────────────
    char = await db.get(Character, session.character_id)
    if not char:
        await feishu_client.send_text_message(chat_id, "Character not found.")
        return

    # ── Build prompt and call LLM ────────────────────────────────────────
    messages = [
        MessageItem(**m)
        for m in (json.loads(session.messages) if session.messages else [])
    ]

    char_dict = {
        "name": char.name,
        "description": char.description,
        "personality": char.personality,
        "scenario": char.scenario,
        "first_message": char.first_message,
        "example_dialogues": char.example_dialogues,
        "system_prompt": char.system_prompt,
    }

    # Worldbook entries
    from app.models.tables import WorldBook

    wb_ids = json.loads(session.worldbook_ids) if session.worldbook_ids else []
    all_entries: list[WorldBookEntry] = []
    for wid in wb_ids:
        wb = await db.get(WorldBook, wid)
        if wb:
            for e in json.loads(wb.entries) if wb.entries else []:
                all_entries.append(WorldBookEntry(**e))

    llm_messages = assemble_prompt(
        character=char_dict,
        worldbook_entries=all_entries,
        chat_history=messages,
        user_message=text,
    )

    # Resolve backend
    from app.routers.sessions import _resolve_backend

    try:
        backend = await _resolve_backend(None, db)
        reply = await chat_completion(
            provider=backend["provider"],
            api_key=backend["api_key"],
            model=backend["model"],
            base_url=backend["base_url"],
            messages=llm_messages,
            params=backend["params"],
        )
    except (LLMError, Exception) as exc:
        logger.exception("LLM call failed in Feishu handler")
        await feishu_client.send_text_message(chat_id, f"LLM error: {exc}")
        return

    # Persist messages
    import datetime

    now = datetime.datetime.utcnow().isoformat()
    messages.append(MessageItem(role="user", content=text, timestamp=now))
    messages.append(MessageItem(role="assistant", content=reply, timestamp=now))
    session.messages = json.dumps(
        [m.model_dump(mode="json") for m in messages], ensure_ascii=False
    )
    await db.commit()

    # Reply with a card showing character name
    card = feishu_client.build_character_reply_card(char.name, reply)
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
            msg_count = len(json.loads(session.messages) if session.messages else [])
            await feishu_client.send_text_message(
                chat_id, f"Character: {name}\nMessages: {msg_count}\nStatus: {session.status}"
            )
        else:
            await feishu_client.send_text_message(chat_id, "No active session.")

    else:
        await feishu_client.send_text_message(
            chat_id, "Available commands: /reset, /switch, /info"
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

        # Persist first message
        import datetime

        now = datetime.datetime.utcnow().isoformat()
        session.messages = json.dumps(
            [{"role": "assistant", "content": first_msg, "timestamp": now}],
            ensure_ascii=False,
        )
        await db.commit()
    else:
        await feishu_client.send_text_message(
            open_chat_id,
            f"Session started with {char.name}! Send a message to begin.",
        )
