"""LLM backend abstraction – calls OpenAI-compatible and Anthropic APIs."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class LLMError(Exception):
    pass


async def chat_completion(
    *,
    provider: str,
    api_key: str,
    model: str,
    base_url: str,
    messages: list[dict[str, str]],
    params: dict[str, Any] | None = None,
) -> str:
    """Send a chat-completion request and return the assistant reply text.

    Supports:
    - ``openai`` / ``custom`` – OpenAI-compatible ``/chat/completions``
    - ``anthropic`` – Anthropic Messages API ``/v1/messages``
    """
    params = params or {}
    timeout = httpx.Timeout(120.0, connect=10.0)

    if provider in ("openai", "custom"):
        return await _openai_compatible(
            api_key=api_key,
            model=model,
            base_url=base_url.rstrip("/"),
            messages=messages,
            params=params,
            timeout=timeout,
        )
    elif provider == "anthropic":
        return await _anthropic(
            api_key=api_key,
            model=model,
            base_url=base_url.rstrip("/") if base_url else "https://api.anthropic.com",
            messages=messages,
            params=params,
            timeout=timeout,
        )
    else:
        raise LLMError(f"Unknown provider: {provider}")


# ── OpenAI-compatible ────────────────────────────────────────────────────────

async def _openai_compatible(
    *,
    api_key: str,
    model: str,
    base_url: str,
    messages: list[dict[str, str]],
    params: dict[str, Any],
    timeout: httpx.Timeout,
) -> str:
    url = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body: dict[str, Any] = {"model": model, "messages": messages, **params}

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, headers=headers, json=body)
        if resp.status_code != 200:
            raise LLMError(f"OpenAI API error {resp.status_code}: {resp.text}")
        data = resp.json()
        return data["choices"][0]["message"]["content"]


# ── Anthropic Messages API ───────────────────────────────────────────────────

async def _anthropic(
    *,
    api_key: str,
    model: str,
    base_url: str,
    messages: list[dict[str, str]],
    params: dict[str, Any],
    timeout: httpx.Timeout,
) -> str:
    url = f"{base_url}/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }

    # Anthropic requires system as a top-level param, not in messages
    system_text = ""
    api_messages: list[dict[str, str]] = []
    for m in messages:
        if m["role"] == "system":
            system_text += ("\n\n" if system_text else "") + m["content"]
        else:
            api_messages.append(m)

    # Ensure messages alternate user/assistant; merge consecutive same-role
    api_messages = _merge_consecutive_roles(api_messages)

    body: dict[str, Any] = {
        "model": model,
        "messages": api_messages,
        "max_tokens": params.pop("max_tokens", 4096),
        **params,
    }
    if system_text:
        body["system"] = system_text

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, headers=headers, json=body)
        if resp.status_code != 200:
            raise LLMError(f"Anthropic API error {resp.status_code}: {resp.text}")
        data = resp.json()
        return data["content"][0]["text"]


def _merge_consecutive_roles(
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Merge consecutive messages of the same role (required by Anthropic)."""
    if not messages:
        return messages
    merged: list[dict[str, str]] = [messages[0].copy()]
    for m in messages[1:]:
        if m["role"] == merged[-1]["role"]:
            merged[-1]["content"] += "\n\n" + m["content"]
        else:
            merged.append(m.copy())
    # Anthropic requires the first message to be from user
    if merged and merged[0]["role"] != "user":
        merged.insert(0, {"role": "user", "content": "[Start]"})
    return merged
