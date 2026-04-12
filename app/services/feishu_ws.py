"""Feishu WebSocket long-connection client using lark-oapi SDK.

Replaces the HTTP webhook approach — no public URL needed.
Events arrive via WebSocket; responses go out via REST API.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import datetime
from typing import Any, Optional

import lark_oapi as lark
from lark_oapi.event.dispatcher_handler import EventDispatcherHandler
from lark_oapi.ws import Client as WsClient

from app.config import settings
from app.services import feishu_client

logger = logging.getLogger(__name__)

_ws_client: Optional[WsClient] = None
_ws_thread: Optional[threading.Thread] = None

# We need a reference to the DB session factory for event handlers
_async_session_factory = None


def set_session_factory(factory):
    """Called from main.py on startup to inject the async session factory."""
    global _async_session_factory
    _async_session_factory = factory


def _get_event_handler() -> EventDispatcherHandler:
    """Build the SDK event dispatcher with our handlers registered."""
    handler = (
        EventDispatcherHandler.builder(
            settings.feishu_encrypt_key or "",
            settings.feishu_verification_token or "",
        )
        .register_p2_im_chat_member_bot_added_v1(_on_bot_added)
        .register_p2_im_message_receive_v1(_on_message)
        .register_p2_card_action_trigger(_on_card_action)
        .build()
    )
    return handler


def _run_async(coro):
    """Run an async coroutine from a sync callback (SDK callbacks are sync)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── Event Handlers (sync, called by SDK) ─────────────────────────────────────

def _on_bot_added(data) -> None:
    """Bot added to a group chat — send character selection card."""
    try:
        chat_id = data.event.chat_id
        if not chat_id:
            logger.warning("bot_added event missing chat_id")
            return
        _run_async(_handle_bot_added_async(chat_id))
    except Exception:
        logger.exception("Error handling bot_added event")


def _on_message(data) -> None:
    """Incoming message — route to RP handler or command."""
    try:
        msg = data.event.message
        sender = data.event.sender
        chat_id = msg.chat_id
        message_id = msg.message_id
        msg_type = msg.message_type
        sender_id = sender.sender_id.open_id if sender and sender.sender_id else ""

        # Skip non-text and bot's own messages
        if msg_type != "text":
            return

        content_str = msg.content or "{}"
        try:
            text = json.loads(content_str).get("text", "").strip()
        except (json.JSONDecodeError, AttributeError):
            text = content_str.strip()

        if not text:
            return

        # Remove @bot mention prefix if present
        if text.startswith("@"):
            parts = text.split(" ", 1)
            text = parts[1].strip() if len(parts) > 1 else ""
            if not text:
                return

        _run_async(_handle_message_async(chat_id, message_id, sender_id, text))
    except Exception:
        logger.exception("Error handling message event")


def _on_card_action(data):
    """Interactive card action (character selection)."""
    try:
        from lark_oapi.api.im.v1 import P2CardActionTriggerResponse
        action = data.event.action
        option = action.option if hasattr(action, 'option') else None
        action_value = action.value if hasattr(action, 'value') else {}

        open_chat_id = data.event.context.open_chat_id if hasattr(data.event, 'context') else ""

        if option and open_chat_id:
            _run_async(_handle_card_action_async(open_chat_id, option))

        # Return empty response
        return None
    except Exception:
        logger.exception("Error handling card action")
        return None


# ── Async handlers (actual business logic) ───────────────────────────────────

async def _handle_bot_added_async(chat_id: str) -> None:
    from sqlalchemy import select
    from app.models.tables import Character

    async with _async_session_factory() as db:
        result = await db.execute(select(Character).order_by(Character.name))
        characters = result.scalars().all()
        if not characters:
            await feishu_client.send_text_message(
                chat_id, "还没有角色卡。请先在 Web UI 中创建或导入角色！"
            )
            return
        char_list = [{"id": c.id, "name": c.name} for c in characters]
        card = feishu_client.build_character_selection_card(char_list)
        await feishu_client.send_interactive_card(chat_id, card)


async def _handle_message_async(
    chat_id: str, message_id: str, sender_id: str, text: str
) -> None:
    from sqlalchemy import select
    from app.models.tables import Character, Session, WorldBook, Persona
    from app.schemas.api import MessageItem, WorldBookEntry
    from app.services.llm import chat_completion, LLMError
    from app.services.prompt import assemble_prompt
    from app.routers.sessions import _resolve_backend

    # Commands
    if text.startswith("/"):
        await _handle_command(text, chat_id, sender_id)
        return

    async with _async_session_factory() as db:
        # Find active session for this chat
        result = await db.execute(
            select(Session).where(
                Session.feishu_chat_id == chat_id, Session.status == "active"
            )
        )
        session = result.scalar_one_or_none()
        if not session:
            await feishu_client.send_text_message(
                chat_id, "当前没有活跃会话。请使用 /switch 选择角色，或将我添加到新群组。"
            )
            return

        char = await db.get(Character, session.character_id)
        if not char:
            await feishu_client.send_text_message(chat_id, "角色未找到。")
            return

        # Build prompt
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
        wb_ids = json.loads(session.worldbook_ids) if session.worldbook_ids else []
        all_entries: list[WorldBookEntry] = []
        for wid in wb_ids:
            wb = await db.get(WorldBook, wid)
            if wb:
                for e in json.loads(wb.entries) if wb.entries else []:
                    all_entries.append(WorldBookEntry(**e))

        # Resolve persona position
        user_name = session.user_name or "用户"
        user_persona = session.user_persona or ""
        persona_position = "in_prompt"
        if session.persona_id:
            persona = await db.get(Persona, session.persona_id)
            if persona:
                persona_position = persona.description_position or "in_prompt"

        llm_messages = assemble_prompt(
            character=char_dict,
            worldbook_entries=all_entries,
            chat_history=messages,
            user_message=text,
            user_name=user_name,
            user_persona=user_persona,
            persona_position=persona_position,
        )

        # Call LLM
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
            logger.exception("LLM call failed in Feishu WS handler")
            await feishu_client.send_text_message(chat_id, f"LLM 错误: {exc}")
            return

        # Persist messages
        now = datetime.datetime.utcnow().isoformat()
        messages.append(MessageItem(role="user", content=text, timestamp=now))
        messages.append(MessageItem(role="assistant", content=reply, timestamp=now))
        session.messages = json.dumps(
            [m.model_dump(mode="json") for m in messages], ensure_ascii=False
        )
        await db.commit()

        # Reply
        card = feishu_client.build_character_reply_card(char.name, reply)
        await feishu_client.send_interactive_card(chat_id, card)


async def _handle_command(text: str, chat_id: str, sender_id: str) -> None:
    from sqlalchemy import select
    from app.models.tables import Character, Session

    cmd = text.split()[0].lower()

    async with _async_session_factory() as db:
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
                await feishu_client.send_text_message(chat_id, "会话已重置，聊天记录已清空。")
            else:
                await feishu_client.send_text_message(chat_id, "没有活跃会话可重置。")

        elif cmd == "/switch":
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
                await feishu_client.send_text_message(chat_id, "没有可用角色。")

        elif cmd == "/info":
            result = await db.execute(
                select(Session).where(
                    Session.feishu_chat_id == chat_id, Session.status == "active"
                )
            )
            session = result.scalar_one_or_none()
            if session:
                from app.models.tables import Character
                char = await db.get(Character, session.character_id)
                name = char.name if char else "未知"
                msg_count = len(json.loads(session.messages) if session.messages else [])
                user = session.user_name or "用户"
                await feishu_client.send_text_message(
                    chat_id,
                    f"角色: {name}\n主角: {user}\n消息数: {msg_count}\n状态: {session.status}",
                )
            else:
                await feishu_client.send_text_message(chat_id, "没有活跃会话。")

        else:
            await feishu_client.send_text_message(
                chat_id, "可用命令: /reset /switch /info"
            )


async def _handle_card_action_async(chat_id: str, option: str) -> None:
    from app.models.tables import Character, Session, Persona
    import json as _json

    try:
        char_id = int(option)
    except (ValueError, TypeError):
        return

    async with _async_session_factory() as db:
        char = await db.get(Character, char_id)
        if not char:
            await feishu_client.send_text_message(chat_id, "角色未找到。")
            return

        # Try to find a linked or default persona
        from sqlalchemy import select
        persona_id = None
        user_name = "用户"
        user_persona = ""

        result = await db.execute(select(Persona))
        all_personas = result.scalars().all()
        for p in all_personas:
            linked = json.loads(p.linked_character_ids) if p.linked_character_ids else []
            if char_id in linked:
                persona_id = p.id
                user_name = p.name
                user_persona = p.description
                break
        if not persona_id:
            for p in all_personas:
                if p.is_default:
                    persona_id = p.id
                    user_name = p.name
                    user_persona = p.description
                    break

        # Get worldbook IDs
        char_linked_wbs = json.loads(char.linked_worldbook_ids) if char.linked_worldbook_ids else []

        # Build initial messages
        initial_messages = []
        if char.first_message:
            first_msg = char.first_message.replace("{{user}}", user_name).replace("{{char}}", char.name)
            now = datetime.datetime.utcnow().isoformat()
            initial_messages.append({"role": "assistant", "content": first_msg, "timestamp": now})

        session = Session(
            character_id=char_id,
            worldbook_ids=json.dumps(char_linked_wbs),
            feishu_chat_id=chat_id,
            persona_id=persona_id,
            user_name=user_name,
            user_persona=user_persona,
            messages=json.dumps(initial_messages, ensure_ascii=False),
            status="active",
        )
        db.add(session)
        await db.commit()

        if initial_messages:
            card = feishu_client.build_character_reply_card(char.name, initial_messages[0]["content"])
            await feishu_client.send_interactive_card(chat_id, card)
        else:
            await feishu_client.send_text_message(
                chat_id, f"已开始与 {char.name} 的对话！发送消息开始吧。"
            )


# ── Lifecycle ────────────────────────────────────────────────────────────────

def start_ws_client():
    """Start the Feishu WebSocket client in a background thread."""
    global _ws_client, _ws_thread

    if not settings.feishu_app_id or not settings.feishu_app_secret:
        logger.warning("Feishu app_id/app_secret not configured — WebSocket client not started")
        return

    event_handler = _get_event_handler()

    _ws_client = WsClient(
        app_id=settings.feishu_app_id,
        app_secret=settings.feishu_app_secret,
        event_handler=event_handler,
        log_level=lark.LogLevel.INFO,
        auto_reconnect=True,
    )

    def _run():
        logger.info("🔌 Starting Feishu WebSocket client...")
        _ws_client.start()

    _ws_thread = threading.Thread(target=_run, daemon=True, name="feishu-ws")
    _ws_thread.start()
    logger.info("🔌 Feishu WebSocket client thread started")


def stop_ws_client():
    """Stop the WebSocket client (best effort)."""
    global _ws_client
    if _ws_client:
        logger.info("Stopping Feishu WebSocket client...")
        _ws_client = None
