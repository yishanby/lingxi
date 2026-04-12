"""Feishu (Lark) API client – outbound calls to Feishu Open Platform."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_FEISHU_BASE = "https://open.feishu.cn/open-apis"

# ── Token cache ──────────────────────────────────────────────────────────────

_tenant_token: str = ""
_tenant_token_expires: float = 0.0


async def get_tenant_access_token() -> str:
    """Obtain (and cache) a tenant_access_token from Feishu."""
    global _tenant_token, _tenant_token_expires

    if _tenant_token and time.time() < _tenant_token_expires - 60:
        return _tenant_token

    url = f"{_FEISHU_BASE}/auth/v3/tenant_access_token/internal"
    body = {
        "app_id": settings.feishu_app_id,
        "app_secret": settings.feishu_app_secret,
    }
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(url, json=body)
        resp.raise_for_status()
        data = resp.json()

    if data.get("code") != 0:
        raise RuntimeError(f"Feishu token error: {data}")

    _tenant_token = data["tenant_access_token"]
    _tenant_token_expires = time.time() + data.get("expire", 7200)
    return _tenant_token


async def _headers() -> dict[str, str]:
    token = await get_tenant_access_token()
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


# ── Message sending ──────────────────────────────────────────────────────────

async def send_text_message(chat_id: str, text: str) -> dict[str, Any]:
    """Send a plain-text message to a Feishu chat."""
    url = f"{_FEISHU_BASE}/im/v1/messages"
    body = {
        "receive_id_type": "chat_id",
        "receive_id": chat_id,
        "msg_type": "text",
        "content": f'{{"text":"{_escape(text)}"}}',
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, headers=await _headers(), json=body)
        return resp.json()


async def send_interactive_card(chat_id: str, card: dict[str, Any]) -> dict[str, Any]:
    """Send an interactive message card to a Feishu chat."""
    import json as _json

    url = f"{_FEISHU_BASE}/im/v1/messages"
    body = {
        "receive_id_type": "chat_id",
        "receive_id": chat_id,
        "msg_type": "interactive",
        "content": _json.dumps(card, ensure_ascii=False),
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, headers=await _headers(), json=body)
        return resp.json()


async def reply_message(message_id: str, text: str) -> dict[str, Any]:
    """Reply to a specific message in Feishu."""
    url = f"{_FEISHU_BASE}/im/v1/messages/{message_id}/reply"
    body = {
        "msg_type": "text",
        "content": f'{{"text":"{_escape(text)}"}}',
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, headers=await _headers(), json=body)
        return resp.json()


# ── Card builders ────────────────────────────────────────────────────────────

def build_character_selection_card(
    characters: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build an interactive card that lets users pick a character."""
    options = [
        {"text": {"tag": "plain_text", "content": c["name"]}, "value": str(c["id"])}
        for c in characters[:20]  # Feishu limits options
    ]
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": "🎭 Choose a Character"},
            "template": "purple",
        },
        "elements": [
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": "Select a character to start the roleplay session:",
                },
            },
            {
                "tag": "action",
                "actions": [
                    {
                        "tag": "select_static",
                        "placeholder": {
                            "tag": "plain_text",
                            "content": "Choose character...",
                        },
                        "options": options,
                        "value": {"action": "select_character"},
                    },
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "Confirm"},
                        "type": "primary",
                        "value": {"action": "confirm_character"},
                    },
                ],
            },
        ],
    }


def build_character_reply_card(
    char_name: str, content: str
) -> dict[str, Any]:
    """Build a card that shows the character's reply with their name as title."""
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": char_name},
            "template": "violet",
        },
        "elements": [
            {
                "tag": "div",
                "text": {"tag": "lark_md", "content": content},
            },
        ],
    }


def _escape(text: str) -> str:
    """Escape characters for Feishu JSON string values."""
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
