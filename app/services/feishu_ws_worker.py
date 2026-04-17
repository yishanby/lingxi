"""Feishu WebSocket long-connection — runs as a separate process.

Started by main.py on startup. Communicates with the FastAPI app via its REST API.
All handlers are SYNCHRONOUS — SDK callbacks run inside the WS event loop,
so we use synchronous httpx/requests calls everywhere.
"""

from __future__ import annotations

import json
import logging
import sys
import os
import time
import threading
from collections import OrderedDict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import httpx
import lark_oapi as lark
from lark_oapi.event.dispatcher_handler import EventDispatcherHandler
from lark_oapi.ws import Client as WsClient

from app.config import settings

# ── Event deduplication ──────────────────────────────────────────────────────
_seen_events: OrderedDict[str, float] = OrderedDict()
_seen_events_lock = threading.Lock()
_MAX_SEEN = 500


def _is_duplicate_event(event_id: str) -> bool:
    """Return True if we've already processed this event_id."""
    if not event_id:
        return False
    with _seen_events_lock:
        if event_id in _seen_events:
            return True
        _seen_events[event_id] = time.time()
        # Prune old entries
        while len(_seen_events) > _MAX_SEEN:
            _seen_events.popitem(last=False)
        return False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s – %(message)s",
)
logger = logging.getLogger("feishu_ws_worker")

API_BASE = f"http://127.0.0.1:{settings.app_port}"

# ── Feishu REST API helpers (SYNCHRONOUS) ────────────────────────────────────

_FEISHU_BASE = "https://open.feishu.cn/open-apis"
_tenant_token: str = ""
_tenant_token_expires: float = 0.0


def get_tenant_access_token() -> str:
    global _tenant_token, _tenant_token_expires

    if _tenant_token and time.time() < _tenant_token_expires - 60:
        return _tenant_token

    url = f"{_FEISHU_BASE}/auth/v3/tenant_access_token/internal"
    body = {"app_id": settings.feishu_app_id, "app_secret": settings.feishu_app_secret}
    with httpx.Client(timeout=10) as client:
        resp = client.post(url, json=body)
        resp.raise_for_status()
        data = resp.json()

    if data.get("code") != 0:
        raise RuntimeError(f"Feishu token error: {data}")

    _tenant_token = data["tenant_access_token"]
    _tenant_token_expires = time.time() + data.get("expire", 7200)
    return _tenant_token


def _feishu_headers() -> dict[str, str]:
    token = get_tenant_access_token()
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def send_text(chat_id: str, text: str):
    url = f"{_FEISHU_BASE}/im/v1/messages?receive_id_type=chat_id"
    body = {
        "receive_id": chat_id,
        "msg_type": "text",
        "content": f'{{"text":"{_escape(text)}"}}',
    }
    with httpx.Client(timeout=30) as client:
        resp = client.post(url, headers=_feishu_headers(), json=body)
        r = resp.json()
        if r.get("code") != 0:
            logger.error(f"send_text failed: {r}")
        return r


def send_card(chat_id: str, card: dict):
    url = f"{_FEISHU_BASE}/im/v1/messages?receive_id_type=chat_id"
    body = {
        "receive_id": chat_id,
        "msg_type": "interactive",
        "content": json.dumps(card, ensure_ascii=False),
    }
    with httpx.Client(timeout=30) as client:
        resp = client.post(url, headers=_feishu_headers(), json=body)
        r = resp.json()
        if r.get("code") != 0:
            logger.error(f"send_card failed: {r}")
        return r


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


# ── Streaming Card (CardKit) ────────────────────────────────────────────────

_CARDKIT_BASE = "https://open.feishu.cn/open-apis/cardkit/v1/cards"

# Persistent HTTP client for CardKit API — avoids repeated TLS handshakes (~300ms saved per call)
_cardkit_client = httpx.Client(timeout=10, limits=httpx.Limits(max_keepalive_connections=5))


def create_streaming_card(chat_id: str, char_name: str) -> tuple[str, str] | None:
    """Create a CardKit streaming card. Returns (card_id, message_id) or None."""
    try:
        token = get_tenant_access_token()
        card_json = {
            "schema": "2.0",
            "config": {
                "streaming_mode": True,
                "summary": {"content": "[生成中...]"},
                "streaming_config": {
                    "print_frequency_ms": {"default": 50},
                    "print_step": {"default": 1},
                },
            },
            "header": {
                "title": {"tag": "plain_text", "content": char_name},
                "template": "violet",
            },
            "body": {
                "elements": [
                    {"tag": "markdown", "content": "┃", "element_id": "content"}
                ]
            },
        }

        # Step 1: Create card
        resp = _cardkit_client.post(
            _CARDKIT_BASE,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"type": "card_json", "data": json.dumps(card_json, ensure_ascii=False)},
        )
        data = resp.json()
        if data.get("code") != 0 or not data.get("data", {}).get("card_id"):
            logger.error(f"Create streaming card failed: {data}")
            return None
        card_id = data["data"]["card_id"]

        # Step 2: Send card as message
        card_content = json.dumps({"type": "card", "data": {"card_id": card_id}})
        msg_resp = _cardkit_client.post(
            f"{_FEISHU_BASE}/im/v1/messages?receive_id_type=chat_id",
            headers=_feishu_headers(),
            json={
                "receive_id": chat_id,
                "msg_type": "interactive",
                "content": card_content,
                },
            )
        msg_data = msg_resp.json()
        if msg_data.get("code") != 0:
            logger.error(f"Send streaming card failed: {msg_data}")
            return None
        message_id = msg_data["data"]["message_id"]
        logger.info(f"Streaming card created: card_id={card_id}, msg_id={message_id}")
        return card_id, message_id
    except Exception:
        logger.exception("create_streaming_card failed")
        return None


def update_streaming_card(card_id: str, text: str, sequence: int) -> None:
    """Update the content element of a streaming card."""
    try:
        token = get_tenant_access_token()
        _cardkit_client.put(
            f"{_CARDKIT_BASE}/{card_id}/elements/content/content",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={
                "content": text,
                "sequence": sequence,
                "uuid": f"s_{card_id}_{sequence}",
            },
        )
    except Exception as e:
        logger.warning(f"update_streaming_card failed: {e}")


def close_streaming_card(card_id: str, final_text: str, sequence: int) -> None:
    """Close the streaming card (set streaming_mode=false)."""
    try:
        token = get_tenant_access_token()
        # Final content update
        update_streaming_card(card_id, final_text, sequence)
        # Close streaming mode
        _cardkit_client.patch(
            f"{_CARDKIT_BASE}/{card_id}/settings",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={
                "settings": json.dumps({"config": {
                    "streaming_mode": False,
                    "summary": {"content": (final_text or "")[:50]},
                }}),
                "sequence": sequence + 1,
                "uuid": f"c_{card_id}_{sequence + 1}",
            },
            )
        logger.info(f"Streaming card closed: card_id={card_id}")
    except Exception:
        logger.exception("close_streaming_card failed")


def _async_card_updater(card_id: str):
    """Background thread that processes card updates.

    Uses a 'latest value' pattern: if multiple updates queue while one is
    in-flight, only the most recent text is sent next. This prevents queue
    buildup when Feishu API latency (~600ms) exceeds the send interval.
    """
    _lock = threading.Lock()
    _pending_text = [None]  # latest text waiting to be sent (None = nothing)
    _final = [None]         # final text to send + close (None = not yet)
    _seq = [1]
    _stop = threading.Event()
    _has_work = threading.Event()

    def _worker():
        while not _stop.is_set():
            _has_work.wait(timeout=0.5)
            _has_work.clear()

            # Grab whatever is pending
            with _lock:
                text = _pending_text[0]
                _pending_text[0] = None
                final_text = _final[0]

            if final_text is not None:
                # Final: update content + close
                try:
                    _seq[0] += 1
                    close_streaming_card(card_id, final_text, _seq[0])
                except Exception:
                    logger.exception("Card updater: close failed")
                return
            elif text is not None:
                try:
                    _seq[0] += 1
                    update_streaming_card(card_id, text, _seq[0])
                    logger.debug(f"Card update seq={_seq[0]} len={len(text)}")
                except Exception:
                    logger.exception("Card updater: update failed")

    worker_thread = threading.Thread(target=_worker, daemon=True)
    worker_thread.start()

    def enqueue_update(text: str):
        with _lock:
            _pending_text[0] = text
        _has_work.set()

    def enqueue_final(text: str):
        with _lock:
            _final[0] = text
        _has_work.set()

    return enqueue_update, enqueue_final, worker_thread


def _stream_llm_to_card(session_id: int, user_text: str, card_id: str) -> str:
    """Call the streaming endpoint and update the card as chunks arrive.

    Card updates run in a background thread. When the Feishu API is slow,
    intermediate updates are dropped (only the latest snapshot is sent).
    This ensures the card always catches up quickly.
    """
    full_text = ""
    last_enqueue_time = 0.0
    ENQUEUE_INTERVAL = 0.3  # how often we push a snapshot to the updater

    enqueue_update, enqueue_final, worker_thread = _async_card_updater(card_id)
    logger.info(f"Starting stream: session={session_id}, card={card_id}")

    try:
        with httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
            with client.stream(
                "POST",
                f"{API_BASE}/api/sessions/{session_id}/message/stream",
                json={"content": user_text},
            ) as response:
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        if "error" in data:
                            logger.error(f"Stream error: {data['error']}")
                            break
                        delta = data.get("delta", "")
                        if delta:
                            full_text += delta
                            now = time.time()
                            if now - last_enqueue_time >= ENQUEUE_INTERVAL:
                                enqueue_update(full_text)
                                last_enqueue_time = now
                    except (json.JSONDecodeError, KeyError):
                        continue
    except Exception:
        logger.exception("_stream_llm_to_card failed")

    logger.info(f"Stream done: {len(full_text)} chars, closing card")
    # Final close (waits for at most one in-flight update then closes)
    enqueue_final(full_text)
    worker_thread.join(timeout=10)
    return full_text


# ── Internal API calls (SYNCHRONOUS, to our own FastAPI backend) ─────────────

def api_get(path: str):
    with httpx.Client(timeout=30) as client:
        resp = client.get(f"{API_BASE}{path}")
        resp.raise_for_status()
        return resp.json()


def api_post(path: str, data: dict):
    with httpx.Client(timeout=120) as client:
        resp = client.post(f"{API_BASE}{path}", json=data)
        resp.raise_for_status()
        return resp.json()


def api_delete(path: str):
    with httpx.Client(timeout=30) as client:
        resp = client.delete(f"{API_BASE}{path}")
        resp.raise_for_status()


def api_patch(path: str, data: dict):
    with httpx.Client(timeout=30) as client:
        resp = client.patch(f"{API_BASE}{path}", json=data)
        resp.raise_for_status()
        return resp.json()


# ── Event Handlers (ALL SYNCHRONOUS) ─────────────────────────────────────────

def _on_bot_added(data) -> None:
    try:
        chat_id = data.event.chat_id
        if not chat_id:
            logger.warning("bot_added event missing chat_id")
            return
        logger.info(f"Bot added to chat: {chat_id}")
        _handle_bot_added(chat_id)
    except Exception:
        logger.exception("Error in bot_added handler")


def _on_message(data) -> None:
    try:
        # Deduplicate events
        event_id = getattr(data.header, 'event_id', '') if hasattr(data, 'header') else ''
        if _is_duplicate_event(event_id):
            logger.debug(f"Duplicate event {event_id}, skipping")
            return

        msg = data.event.message
        sender = data.event.sender
        chat_id = msg.chat_id
        msg_type = msg.message_type
        sender_id = sender.sender_id.open_id if sender and sender.sender_id else ""
        sender_type = sender.sender_type if sender else ""

        # Ignore messages sent by bots (including ourselves)
        if sender_type == "app":
            logger.debug(f"Ignoring bot message in {chat_id}")
            return

        if msg_type != "text":
            logger.debug(f"Ignoring non-text message type: {msg_type}")
            return

        content_str = msg.content or "{}"
        try:
            text = json.loads(content_str).get("text", "").strip()
        except (json.JSONDecodeError, AttributeError):
            text = content_str.strip()

        if not text:
            return

        # Remove @bot mentions (Feishu uses @_user_N placeholders)
        mentions = getattr(msg, 'mentions', None) or []
        for mention in mentions:
            key = getattr(mention, 'key', '')
            if key:
                text = text.replace(key, '').strip()

        if not text:
            return

        # Ignore messages older than 30 seconds (stale events from reconnection)
        msg_create_time = getattr(msg, 'create_time', '') or ''
        if msg_create_time:
            try:
                msg_ts = int(msg_create_time) / 1000  # ms -> s
                age = time.time() - msg_ts
                if age > 30:
                    logger.debug(f"Ignoring stale message (age={age:.0f}s) in {chat_id}")
                    return
            except (ValueError, TypeError):
                pass

        # Deduplicate by message_id too
        msg_id = getattr(msg, 'message_id', '') or ''
        if msg_id and _is_duplicate_event(f"msg:{msg_id}"):
            logger.debug(f"Duplicate message_id {msg_id}, skipping")
            return

        logger.info(f"Message from {sender_id} in {chat_id}: {text[:50]}")

        # Run in a thread so we don't block the WS event loop (LLM calls take 10-30s)
        t = threading.Thread(
            target=_handle_message,
            args=(chat_id, sender_id, text),
            daemon=True,
        )
        t.start()
    except Exception:
        logger.exception("Error in message handler")


def _on_p2p_entered(data):
    """User opened the bot's 1-on-1 chat window. Do nothing — avoid spamming cards."""
    try:
        chat_id = data.event.chat_id
        logger.debug(f"User entered P2P chat: {chat_id} (ignored)")
    except Exception:
        pass


def _on_card_action(data):
    try:
        action = data.event.action
        option = action.option if hasattr(action, "option") else None
        open_chat_id = ""
        if hasattr(data.event, "context") and data.event.context:
            open_chat_id = getattr(data.event.context, "open_chat_id", "")

        logger.info(f"Card action: option={option}, chat={open_chat_id}")

        if option and open_chat_id:
            _handle_card_action(open_chat_id, option)
        return None
    except Exception:
        logger.exception("Error in card_action handler")
        return None


# ── Business logic (ALL SYNCHRONOUS) ─────────────────────────────────────────

def _handle_bot_added(chat_id: str) -> None:
    try:
        characters = api_get("/api/characters")
        if not characters:
            send_text(chat_id, "还没有角色卡。请先在 Web UI 中创建或导入角色！")
            return
        char_list = [{"id": c["id"], "name": c["name"]} for c in characters]
        card = build_character_selection_card(char_list)
        send_card(chat_id, card)
        logger.info(f"Sent character selection card to {chat_id}")
    except Exception:
        logger.exception("handle_bot_added failed")


def _handle_message(chat_id: str, sender_id: str, text: str) -> None:
    # Commands
    if text.startswith("/"):
        _handle_command(text, chat_id, sender_id)
        return

    try:
        # Find active session for this chat
        sessions = api_get("/api/sessions?lite=true")
        session = None
        for s in sessions:
            if s.get("feishu_chat_id") == chat_id and s.get("status") == "active":
                session = s
                break

        if not session:
            send_text(chat_id, "当前没有活跃会话。请使用 /switch 选择角色，或将机器人拉进新群。")
            return

        # Look up character name
        char_name = "角色"
        try:
            characters = api_get("/api/characters")
            for c in characters:
                if c["id"] == session.get("character_id"):
                    char_name = c["name"]
                    break
        except Exception:
            pass

        # Try streaming card approach
        card_info = create_streaming_card(chat_id, char_name)
        if card_info:
            card_id, message_id = card_info
            reply_text = _stream_llm_to_card(session["id"], text, card_id)
            if reply_text:
                logger.info(f"Streamed reply from {char_name} in {chat_id} ({len(reply_text)} chars)")
                _check_memory_size_hint(chat_id, session["id"])
            else:
                logger.warning(f"Empty streaming reply in {chat_id}")
        else:
            # Fallback to non-streaming
            logger.info(f"Streaming card failed, falling back to non-streaming")
            result = api_post(
                f"/api/sessions/{session['id']}/message",
                {"content": text},
            )
            messages = result.get("messages", [])
            if messages:
                last_msg = messages[-1]
                if last_msg.get("role") == "assistant":
                    card = build_reply_card(char_name, last_msg["content"])
                    send_card(chat_id, card)
                    logger.info(f"Sent reply from {char_name} in {chat_id}")
                    _check_memory_size_hint(chat_id, session["id"])

    except Exception as exc:
        logger.exception("handle_message failed")
        send_text(chat_id, f"处理消息出错: {exc}")


# Memory size threshold for compact hint (chars)
_MEMORY_COMPACT_HINT_THRESHOLD = 10000
_memory_hint_sent: set[int] = set()  # Track which sessions got the hint this run


def _check_memory_size_hint(chat_id: str, session_id: int) -> None:
    """Send a one-time hint if memory.md is getting large."""
    if session_id in _memory_hint_sent:
        return
    try:
        from pathlib import Path
        memory_path = Path(f"data/memory/{session_id}/memory.md")
        if not memory_path.exists():
            return
        size = memory_path.stat().st_size
        if size > _MEMORY_COMPACT_HINT_THRESHOLD:
            _memory_hint_sent.add(session_id)
            send_text(chat_id, f"💡 记忆已较长（{size}字），可能占用较多context。建议使用 /memory compact 精练。")
    except Exception:
        pass


def _handle_command(text: str, chat_id: str, sender_id: str) -> None:
    cmd = text.split()[0].lower()

    if cmd == "/help":
        help_text = (
            "📖 可用命令：\n\n"
            "💬 会话管理\n"
            "/list — 列出本群所有会话（含已归档）\n"
            "/switch — 选择/切换角色（归档当前会话）\n"
            "/resume — 恢复已归档的会话\n"
            "/reset — 重置当前会话（重新开始对话）\n"
            "/info — 查看当前会话信息\n\n"
            "✏️ 对话操作\n"
            "/retry — 重新生成上一条AI回复\n"
            "/undo — 撤销上一轮对话\n\n"
            "🧠 记忆管理\n"
            "/memory — 查看当前记忆\n"
            "/memory edit <内容> — 替换全部记忆\n"
            "/memory delete <关键词> — 删除含关键词的记忆\n"
            "/memory extract — 从最近对话提取记忆\n"
            "/memory rebuild — 从整个对话重建记忆（耗时较长）\n"
            "/memory rebuild preview — 查看重建进度\n"
            "/memory apply — 应用重建/压缩结果到记忆\n"
            "/memory compact — 压缩记忆（备份后精练）\n"
            "/memory compact preview — 查看压缩预览\n"
            "/memory clear — 清空记忆\n"
            "/remember <内容> — 手动添加一条记忆\n"
            "/summary — 查看前情提要\n"
            "/summary update — 强制更新摘要\n"
            "/summary clear — 清空摘要\n\n"
            "💰 资产追踪\n"
            "/assets — 查看当前资产记录\n\n"
            "/help — 显示此帮助信息"
        )
        send_text(chat_id, help_text)
        return

    if cmd == "/list":
        try:
            sessions = api_get("/api/sessions?lite=true")
            # Filter to this chat only
            chat_sessions = [s for s in sessions if s.get("feishu_chat_id") == chat_id]
            if not chat_sessions:
                send_text(chat_id, "本群还没有任何会话。")
                return

            # Look up character names
            try:
                characters = api_get("/api/characters")
                char_map = {c["id"]: c["name"] for c in characters}
            except Exception:
                char_map = {}

            # Find the most recent active session (highest id)
            active_sessions = [s for s in chat_sessions if s.get("status") == "active"]
            current_id = max((s["id"] for s in active_sessions), default=None) if active_sessions else None

            lines = ["📋 本群会话列表：\n"]
            for s in chat_sessions:
                sid = s["id"]
                char_name = char_map.get(s.get("character_id"), f"角色#{s.get('character_id')}")
                is_current = sid == current_id
                status_icon = "🟢" if is_current else "⚪"
                msg_count = s.get("msg_count", 0)
                current_tag = " ← 当前" if is_current else ""

                # Last message summary
                summary = s.get("last_summary", "")
                summary_line = ""
                if summary:
                    display = summary[:30] + ("…" if len(summary) > 30 else "")
                    summary_line = f"\n   💬 {display}"

                lines.append(f"{status_icon} #{sid} {char_name} ({msg_count}条消息){current_tag}{summary_line}")

            lines.append("\n使用 /resume <编号> 恢复会话，如 /resume 3")
            send_text(chat_id, "\n".join(lines))
        except Exception as exc:
            logger.exception("list command failed")
            send_text(chat_id, f"获取会话列表失败: {exc}")
        return

    if cmd == "/reset":
        sessions = api_get("/api/sessions?lite=true")
        for s in sessions:
            if s.get("feishu_chat_id") == chat_id and s.get("status") == "active":
                # Delete the old session
                try:
                    api_delete(f"/api/sessions/{s['id']}")
                except Exception:
                    pass
                # Look up character to recreate
                char_id = s.get("character_id")
                char_name = "角色"
                try:
                    characters = api_get("/api/characters")
                    for c in characters:
                        if c["id"] == char_id:
                            char_name = c["name"]
                            break
                except Exception:
                    pass
                # Create a fresh session
                try:
                    result = api_post("/api/sessions", {
                        "character_id": char_id,
                        "feishu_chat_id": chat_id,
                        "worldbook_ids": [],
                    })
                    messages = result.get("messages", [])
                    if messages:
                        card = build_reply_card(char_name, messages[0]["content"])
                        send_card(chat_id, card)
                    else:
                        send_text(chat_id, f"会话已重置！与 {char_name} 的对话重新开始。")
                except Exception:
                    send_text(chat_id, "重置成功，但创建新会话失败。请用 /switch 选择角色。")
                return
        send_text(chat_id, "没有活跃会话。")

    elif cmd == "/switch":
        # Archive the current active session first
        try:
            sessions = api_get("/api/sessions?lite=true")
            for s in sessions:
                if s.get("feishu_chat_id") == chat_id and s.get("status") == "active":
                    try:
                        api_patch(f"/api/sessions/{s['id']}/status", {"status": "archived"})
                        logger.info(f"Archived session {s['id']} before switch")
                    except Exception:
                        logger.exception(f"Failed to archive session {s['id']}")
        except Exception:
            pass

        characters = api_get("/api/characters")
        if characters:
            char_list = [{"id": c["id"], "name": c["name"]} for c in characters]
            card = build_character_selection_card(char_list)
            send_card(chat_id, card)
        else:
            send_text(chat_id, "没有可用角色。")

    elif cmd == "/info":
        sessions = api_get("/api/sessions?lite=true")
        for s in sessions:
            if s.get("feishu_chat_id") == chat_id and s.get("status") == "active":
                msg_count = len(s.get("messages", []))
                user = s.get("user_name", "用户")
                send_text(
                    chat_id,
                    f"角色ID: {s.get('character_id')}\n主角: {user}\n消息数: {msg_count}",
                )
                return
        send_text(chat_id, "没有活跃会话。")

    elif cmd == "/resume":
        try:
            parts = text.split()
            sessions = api_get("/api/sessions?lite=true")
            chat_sessions = [s for s in sessions if s.get("feishu_chat_id") == chat_id]

            if len(parts) < 2:
                # Show list of sessions to pick from
                if not chat_sessions:
                    send_text(chat_id, "本群没有任何会话。")
                    return
                try:
                    characters = api_get("/api/characters")
                    char_map = {c["id"]: c["name"] for c in characters}
                except Exception:
                    char_map = {}

                lines = ["📋 选择要恢复的会话：\n"]
                for s in chat_sessions:
                    sid = s["id"]
                    char_name = char_map.get(s.get("character_id"), f"角色#{s.get('character_id')}")
                    status = s.get("status", "unknown")
                    status_icon = "🟢" if status == "active" else "⚪"
                    msg_count = len(s.get("messages", []))
                    lines.append(f"{status_icon} #{sid} {char_name} ({msg_count}条消息) [{status}]")
                lines.append("\n输入 /resume <编号> 恢复，如 /resume 3")
                send_text(chat_id, "\n".join(lines))
                return

            # /resume <number> — activate that session
            try:
                target_id = int(parts[1])
            except ValueError:
                send_text(chat_id, "用法: /resume <会话编号>，如 /resume 3")
                return

            # Check target exists and belongs to this chat
            target_session = None
            for s in chat_sessions:
                if s["id"] == target_id:
                    target_session = s
                    break

            if not target_session:
                send_text(chat_id, f"未找到本群的会话 #{target_id}。使用 /list 查看可用会话。")
                return

            if target_session.get("status") == "active":
                send_text(chat_id, f"会话 #{target_id} 已经是活跃状态。")
                return

            # Archive current active session
            for s in chat_sessions:
                if s.get("status") == "active":
                    try:
                        api_patch(f"/api/sessions/{s['id']}/status", {"status": "archived"})
                        logger.info(f"Archived session {s['id']} for resume")
                    except Exception:
                        logger.exception(f"Failed to archive session {s['id']}")

            # Activate target session
            api_patch(f"/api/sessions/{target_id}/status", {"status": "active"})

            # Look up character name
            char_name = f"会话#{target_id}"
            try:
                characters = api_get("/api/characters")
                for c in characters:
                    if c["id"] == target_session.get("character_id"):
                        char_name = c["name"]
                        break
            except Exception:
                pass

            send_text(chat_id, f"✅ 已恢复会话 #{target_id}（{char_name}）。继续发消息即可。")

        except Exception as exc:
            logger.exception("resume command failed")
            send_text(chat_id, f"恢复会话失败: {exc}")
        return

    else:
        # Forward to backend API (handles /memory, /remember, /undo, /retry, etc.)
        try:
            sessions = api_get("/api/sessions?lite=true")
            session = None
            for s in sessions:
                if s.get("feishu_chat_id") == chat_id and s.get("status") == "active":
                    session = s
                    break
            if not session:
                send_text(chat_id, "没有活跃会话。请先用 /switch 选择角色。")
                return

            result = api_post(f"/api/sessions/{session['id']}/message", {"content": text})
            messages = result.get("messages", [])
            if messages and messages[-1].get("role") == "assistant":
                reply = messages[-1]["content"]
                send_text(chat_id, reply)
            else:
                send_text(chat_id, "命令已执行。")
        except Exception as exc:
            logger.exception(f"Forward command {cmd} failed")
            send_text(chat_id, f"未知命令。输入 /help 查看所有可用命令。")


def _handle_card_action(chat_id: str, option: str) -> None:
    try:
        char_id = int(option)
    except (ValueError, TypeError):
        return

    try:
        # Create session via API
        result = api_post(
            "/api/sessions",
            {
                "character_id": char_id,
                "feishu_chat_id": chat_id,
                "worldbook_ids": [],
            },
        )

        # Look up character name
        char_name = f"角色#{char_id}"
        try:
            characters = api_get("/api/characters")
            for c in characters:
                if c["id"] == char_id:
                    char_name = c["name"]
                    break
        except Exception:
            pass

        # If session has messages (first_message), send as card
        messages = result.get("messages", [])
        if messages:
            card = build_reply_card(char_name, messages[0]["content"])
            send_card(chat_id, card)
        else:
            send_text(chat_id, f"已开始与 {char_name} 的对话！发送消息开始吧。")

        logger.info(f"Session created for {char_name} in {chat_id}")
    except Exception:
        logger.exception("handle_card_action failed")
        send_text(chat_id, "创建会话失败。")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not settings.feishu_app_id or not settings.feishu_app_secret:
        logger.error("FEISHU_APP_ID and FEISHU_APP_SECRET must be set in .env")
        sys.exit(1)

    logger.info(f"🔌 Connecting to Feishu as app {settings.feishu_app_id}...")
    logger.info(f"📡 Backend API at {API_BASE}")

    event_handler = (
        EventDispatcherHandler.builder(
            settings.feishu_encrypt_key or "",
            settings.feishu_verification_token or "",
        )
        .register_p2_im_chat_member_bot_added_v1(_on_bot_added)
        .register_p2_im_chat_access_event_bot_p2p_chat_entered_v1(_on_p2p_entered)
        .register_p2_im_message_receive_v1(_on_message)
        .register_p2_card_action_trigger(_on_card_action)
        .build()
    )

    ws_client = WsClient(
        app_id=settings.feishu_app_id,
        app_secret=settings.feishu_app_secret,
        event_handler=event_handler,
        log_level=lark.LogLevel.DEBUG,
        auto_reconnect=True,
    )

    logger.info("🔌 Feishu WebSocket client starting...")
    ws_client.start()


if __name__ == "__main__":
    main()
