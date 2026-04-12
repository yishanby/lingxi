"""Feishu WebSocket long-connection — runs as a separate process.

Started by main.py on startup. Communicates with the FastAPI app via its REST API.
This avoids event-loop conflicts between lark-oapi SDK and uvicorn.
"""

from __future__ import annotations

import asyncio
import json
import logging
import datetime
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import httpx
import lark_oapi as lark
from lark_oapi.event.dispatcher_handler import EventDispatcherHandler
from lark_oapi.ws import Client as WsClient

from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s – %(message)s",
)
logger = logging.getLogger("feishu_ws_worker")

API_BASE = f"http://127.0.0.1:{settings.app_port}"


# ── Feishu REST API helpers (reuse from feishu_client) ───────────────────────

_FEISHU_BASE = "https://open.feishu.cn/open-apis"
_tenant_token: str = ""
_tenant_token_expires: float = 0.0


async def get_tenant_access_token() -> str:
    global _tenant_token, _tenant_token_expires
    import time

    if _tenant_token and time.time() < _tenant_token_expires - 60:
        return _tenant_token

    url = f"{_FEISHU_BASE}/auth/v3/tenant_access_token/internal"
    body = {"app_id": settings.feishu_app_id, "app_secret": settings.feishu_app_secret}
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(url, json=body)
        resp.raise_for_status()
        data = resp.json()

    if data.get("code") != 0:
        raise RuntimeError(f"Feishu token error: {data}")

    _tenant_token = data["tenant_access_token"]
    _tenant_token_expires = time.time() + data.get("expire", 7200)
    return _tenant_token


async def _feishu_headers() -> dict[str, str]:
    token = await get_tenant_access_token()
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


async def send_text(chat_id: str, text: str):
    url = f"{_FEISHU_BASE}/im/v1/messages"
    body = {
        "receive_id_type": "chat_id",
        "receive_id": chat_id,
        "msg_type": "text",
        "content": f'{{"text":"{_escape(text)}"}}',
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, headers=await _feishu_headers(), json=body)
        return resp.json()


async def send_card(chat_id: str, card: dict):
    url = f"{_FEISHU_BASE}/im/v1/messages"
    body = {
        "receive_id_type": "chat_id",
        "receive_id": chat_id,
        "msg_type": "interactive",
        "content": json.dumps(card, ensure_ascii=False),
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, headers=await _feishu_headers(), json=body)
        return resp.json()


def build_character_selection_card(characters: list[dict]) -> dict:
    options = [
        {"text": {"tag": "plain_text", "content": c["name"]}, "value": str(c["id"])}
        for c in characters[:20]
    ]
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": "🎭 选择角色"},
            "template": "purple",
        },
        "elements": [
            {
                "tag": "div",
                "text": {"tag": "lark_md", "content": "选择一个角色开始 RP 对话："},
            },
            {
                "tag": "action",
                "actions": [
                    {
                        "tag": "select_static",
                        "placeholder": {"tag": "plain_text", "content": "选择角色..."},
                        "options": options,
                        "value": {"action": "select_character"},
                    },
                ],
            },
        ],
    }


def build_reply_card(char_name: str, content: str) -> dict:
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": char_name},
            "template": "violet",
        },
        "elements": [
            {"tag": "div", "text": {"tag": "lark_md", "content": content}},
        ],
    }


# ── Internal API calls (to our own FastAPI backend) ──────────────────────────

async def api_get(path: str):
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{API_BASE}{path}")
        resp.raise_for_status()
        return resp.json()


async def api_post(path: str, data: dict):
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{API_BASE}{path}", json=data)
        resp.raise_for_status()
        return resp.json()


# ── Event Handlers ───────────────────────────────────────────────────────────

def _on_bot_added(data) -> None:
    try:
        chat_id = data.event.chat_id
        if not chat_id:
            return
        asyncio.get_event_loop().run_until_complete(_handle_bot_added(chat_id))
    except Exception:
        logger.exception("Error in bot_added handler")


def _on_message(data) -> None:
    try:
        msg = data.event.message
        sender = data.event.sender
        chat_id = msg.chat_id
        msg_type = msg.message_type
        sender_id = sender.sender_id.open_id if sender and sender.sender_id else ""

        if msg_type != "text":
            return

        content_str = msg.content or "{}"
        try:
            text = json.loads(content_str).get("text", "").strip()
        except (json.JSONDecodeError, AttributeError):
            text = content_str.strip()

        if not text:
            return

        # Remove @bot mention
        if text.startswith("@"):
            parts = text.split(" ", 1)
            text = parts[1].strip() if len(parts) > 1 else ""
            if not text:
                return

        asyncio.get_event_loop().run_until_complete(
            _handle_message(chat_id, sender_id, text)
        )
    except Exception:
        logger.exception("Error in message handler")


def _on_card_action(data):
    try:
        action = data.event.action
        option = action.option if hasattr(action, "option") else None
        open_chat_id = ""
        if hasattr(data.event, "context") and data.event.context:
            open_chat_id = getattr(data.event.context, "open_chat_id", "")

        if option and open_chat_id:
            asyncio.get_event_loop().run_until_complete(
                _handle_card_action(open_chat_id, option)
            )
        return None
    except Exception:
        logger.exception("Error in card_action handler")
        return None


# ── Async business logic ────────────────────────────────────────────────────

async def _handle_bot_added(chat_id: str) -> None:
    try:
        characters = await api_get("/api/characters")
        if not characters:
            await send_text(chat_id, "还没有角色卡。请先在 Web UI 中创建或导入角色！")
            return
        char_list = [{"id": c["id"], "name": c["name"]} for c in characters]
        card = build_character_selection_card(char_list)
        await send_card(chat_id, card)
    except Exception:
        logger.exception("handle_bot_added failed")


async def _handle_message(chat_id: str, sender_id: str, text: str) -> None:
    # Commands
    if text.startswith("/"):
        await _handle_command(text, chat_id, sender_id)
        return

    try:
        # Find active session for this chat
        sessions = await api_get("/api/sessions")
        session = None
        for s in sessions:
            if s.get("feishu_chat_id") == chat_id and s.get("status") == "active":
                session = s
                break

        if not session:
            await send_text(chat_id, "当前没有活跃会话。请使用 /switch 选择角色。")
            return

        # Send message through our API
        result = await api_post(
            f"/api/sessions/{session['id']}/message",
            {"content": text},
        )

        # Get the latest assistant message
        messages = result.get("messages", [])
        if messages:
            last_msg = messages[-1]
            if last_msg.get("role") == "assistant":
                # Get character name
                characters = await api_get("/api/characters")
                char_name = "角色"
                for c in characters:
                    if c["id"] == session.get("character_id"):
                        char_name = c["name"]
                        break
                card = build_reply_card(char_name, last_msg["content"])
                await send_card(chat_id, card)

    except Exception as exc:
        logger.exception("handle_message failed")
        await send_text(chat_id, f"处理消息出错: {exc}")


async def _handle_command(text: str, chat_id: str, sender_id: str) -> None:
    cmd = text.split()[0].lower()

    if cmd == "/reset":
        sessions = await api_get("/api/sessions")
        for s in sessions:
            if s.get("feishu_chat_id") == chat_id and s.get("status") == "active":
                # Reset via direct API call — we need a reset endpoint or delete + recreate
                # For now just notify
                await send_text(chat_id, "会话已重置。（功能完善中）")
                return
        await send_text(chat_id, "没有活跃会话。")

    elif cmd == "/switch":
        characters = await api_get("/api/characters")
        if characters:
            char_list = [{"id": c["id"], "name": c["name"]} for c in characters]
            card = build_character_selection_card(char_list)
            await send_card(chat_id, card)
        else:
            await send_text(chat_id, "没有可用角色。")

    elif cmd == "/info":
        sessions = await api_get("/api/sessions")
        for s in sessions:
            if s.get("feishu_chat_id") == chat_id and s.get("status") == "active":
                msg_count = len(s.get("messages", []))
                user = s.get("user_name", "用户")
                await send_text(
                    chat_id,
                    f"角色ID: {s.get('character_id')}\n主角: {user}\n消息数: {msg_count}",
                )
                return
        await send_text(chat_id, "没有活跃会话。")

    else:
        await send_text(chat_id, "可用命令: /reset /switch /info")


async def _handle_card_action(chat_id: str, option: str) -> None:
    try:
        char_id = int(option)
    except (ValueError, TypeError):
        return

    try:
        # Create session via API
        result = await api_post(
            "/api/sessions",
            {
                "character_id": char_id,
                "feishu_chat_id": chat_id,
                "worldbook_ids": [],
            },
        )

        # If session has messages (first_message), send as card
        messages = result.get("messages", [])
        char_name = f"角色#{char_id}"
        # Look up character name
        try:
            characters = await api_get("/api/characters")
            for c in characters:
                if c["id"] == char_id:
                    char_name = c["name"]
                    break
        except Exception:
            pass

        if messages:
            card = build_reply_card(char_name, messages[0]["content"])
            await send_card(chat_id, card)
        else:
            await send_text(chat_id, f"已开始与 {char_name} 的对话！发送消息开始吧。")
    except Exception:
        logger.exception("handle_card_action failed")
        await send_text(chat_id, "创建会话失败。")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not settings.feishu_app_id or not settings.feishu_app_secret:
        logger.error("FEISHU_APP_ID and FEISHU_APP_SECRET must be set in .env")
        sys.exit(1)

    logger.info(f"🔌 Connecting to Feishu as app {settings.feishu_app_id}...")

    event_handler = (
        EventDispatcherHandler.builder(
            settings.feishu_encrypt_key or "",
            settings.feishu_verification_token or "",
        )
        .register_p2_im_chat_member_bot_added_v1(_on_bot_added)
        .register_p2_im_message_receive_v1(_on_message)
        .register_p2_card_action_trigger(_on_card_action)
        .build()
    )

    ws_client = WsClient(
        app_id=settings.feishu_app_id,
        app_secret=settings.feishu_app_secret,
        event_handler=event_handler,
        log_level=lark.LogLevel.INFO,
        auto_reconnect=True,
    )

    logger.info("🔌 Feishu WebSocket client starting...")
    ws_client.start()


if __name__ == "__main__":
    main()
