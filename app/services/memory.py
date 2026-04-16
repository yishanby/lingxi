"""MD-based memory system for conversation persistence and LLM-driven memory extraction."""

from __future__ import annotations

import asyncio
import datetime
import logging
import os
import re
from pathlib import Path
from typing import Any

import aiofiles

from app.services.llm import chat_completion

logger = logging.getLogger(__name__)

# Base directory for memory data
MEMORY_BASE = Path("data/memory")

MEMORY_TEMPLATE = """# Session Memory

## Characters
- (none yet)

## Relationships
- (none yet)

## Key Events
- (none yet)

## Important Decisions
- (none yet)

## User Preferences (from OOC)
- (none yet)
"""

EXTRACTION_FULL_SYSTEM_PROMPT = (
    "You are a memory extraction system for a roleplay conversation. "
    "Given the FULL conversation history and existing memory, produce a COMPLETE updated memory document. "
    "Use these exact markdown headers: Characters, Relationships, Key Events, Decisions, User Preferences. "
    "Be thorough but concise. Use bullet points. This output will REPLACE the entire memory file."
)

EXTRACTION_SYSTEM_PROMPT = (
    "You are a memory extraction system for a roleplay conversation. "
    "Given the recent conversation, extract and update the following categories. "
    "Output ONLY the new/changed items in markdown format with these exact headers: "
    "Characters, Relationships, Key Events, Decisions, User Preferences. "
    "Be concise. Use bullet points. If a category has no new items, omit it entirely. "
    "Do NOT repeat items that are already in the existing memory."
)


def _session_dir(session_id: int) -> Path:
    return MEMORY_BASE / str(session_id)


def _ensure_dir(session_id: int) -> Path:
    d = _session_dir(session_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── 1. append_chat_md ───────────────────────────────────────────────────────

async def append_chat_md(
    session_id: int,
    role: str,
    content: str,
    char_name: str = "Assistant",
    user_name: str = "User",
    msg_type: str = "ic",
) -> None:
    """Append a message to chat.md with timestamp."""
    d = _ensure_dir(session_id)
    path = d / "chat.md"

    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M")

    if role == "user":
        if msg_type == "ooc":
            header = f"## [{now}] {user_name} ((OOC)) <!-- role:user -->"
        else:
            header = f"## [{now}] {user_name} <!-- role:user -->"
    elif role == "assistant":
        header = f"## [{now}] {char_name} <!-- role:assistant -->"
    else:
        header = f"## [{now}] System <!-- role:system -->"

    entry = f"\n{header}\n{content}\n"

    async with aiofiles.open(path, mode="a", encoding="utf-8") as f:
        await f.write(entry)


# ── 2. load_chat_md ─────────────────────────────────────────────────────────

async def load_chat_md(session_id: int) -> list[dict[str, Any]]:
    """Parse chat.md back into message dicts."""
    path = _session_dir(session_id) / "chat.md"
    if not path.exists():
        return []

    async with aiofiles.open(path, mode="r", encoding="utf-8") as f:
        text = await f.read()

    messages: list[dict[str, Any]] = []
    # Pattern: ## [timestamp] Name (optional ((OOC))) (optional <!-- role:xxx -->)
    pattern = re.compile(
        r"^## \[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\] (.+?)(?:\s+\(\(OOC\)\))?(?:\s+<!-- role:(\w+) -->)?\s*$",
        re.MULTILINE,
    )

    splits = pattern.split(text)
    # splits: [preamble, ts1, name1, role1, content1, ts2, name2, role2, content2, ...]
    i = 1
    while i + 3 < len(splits):
        timestamp = splits[i]
        name = splits[i + 1]
        role_tag = splits[i + 2]  # from <!-- role:xxx --> or None
        content = splits[i + 3].strip()

        # Determine role: use tag if present, else fall back to heuristic
        if role_tag:
            role = role_tag
        elif name == "System":
            role = "system"
        else:
            role = "user"  # can't distinguish without tag

        # Detect OOC from the header line
        is_ooc = "((OOC))" in (splits[i + 1] if splits[i + 1] else "")

        msg = {
            "role": role,
            "content": content,
            "timestamp": timestamp,
            "name": name,
            "msg_type": "ooc" if is_ooc else "ic",
        }
        messages.append(msg)
        i += 4

    return messages


# ── 3. save_memory ──────────────────────────────────────────────────────────

async def save_memory(session_id: int, memory_text: str) -> None:
    """Write memory.md."""
    d = _ensure_dir(session_id)
    async with aiofiles.open(d / "memory.md", mode="w", encoding="utf-8") as f:
        await f.write(memory_text)


# ── 4. load_memory ──────────────────────────────────────────────────────────

async def load_memory(session_id: int) -> str:
    """Read memory.md, return empty string if not found."""
    path = _session_dir(session_id) / "memory.md"
    if not path.exists():
        return ""
    async with aiofiles.open(path, mode="r", encoding="utf-8") as f:
        return await f.read()


# ── 5. save_context_snapshot ────────────────────────────────────────────────

async def save_context_snapshot(session_id: int, snapshot: str) -> None:
    """Write context.md."""
    d = _ensure_dir(session_id)
    async with aiofiles.open(d / "context.md", mode="w", encoding="utf-8") as f:
        await f.write(snapshot)


# ── 6. load_context_snapshot ────────────────────────────────────────────────

async def load_context_snapshot(session_id: int) -> str:
    """Read context.md, return empty string if not found."""
    path = _session_dir(session_id) / "context.md"
    if not path.exists():
        return ""
    async with aiofiles.open(path, mode="r", encoding="utf-8") as f:
        return await f.read()


# ── 7. extract_memory ──────────────────────────────────────────────────────

async def extract_memory(
    session_id: int,
    recent_messages: list[dict[str, str]],
    backend_config: dict[str, Any],
) -> None:
    """Call LLM to extract key facts from recent messages and UPDATE memory.md."""
    try:
        existing_memory = await load_memory(session_id)

        # Build the extraction prompt
        conversation_text = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}"
            for m in recent_messages[-20:]  # last 20 messages for context
        )

        user_prompt = (
            f"## Existing Memory\n{existing_memory}\n\n"
            f"## Recent Conversation\n{conversation_text}\n\n"
            "Extract new/changed items only. Do not repeat existing memory items."
        )

        llm_messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        extraction = await chat_completion(
            provider=backend_config["provider"],
            api_key=backend_config["api_key"],
            model=backend_config["model"],
            base_url=backend_config["base_url"],
            messages=llm_messages,
            params=backend_config.get("params", {}),
        )

        # Merge: append new extractions to existing memory
        if existing_memory:
            updated = _merge_memory(existing_memory, extraction)
        else:
            updated = MEMORY_TEMPLATE.rstrip() + "\n\n" + extraction

        await save_memory(session_id, updated)
        logger.info(f"Memory extracted for session {session_id}")

    except Exception as exc:
        logger.error(f"Memory extraction failed for session {session_id}: {exc}")


def _merge_memory(existing: str, new_extraction: str) -> str:
    """Append new extraction results under an update separator."""
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    return f"{existing.rstrip()}\n\n---\n### Update [{now}]\n{new_extraction}\n"


# ── 7b. extract_memory_full (synchronous, replaces memory) ─────────────────

async def extract_memory_full(
    session_id: int,
    all_messages: list[dict[str, str]],
    backend_config: dict[str, Any],
    custom_instruction: str | None = None,
) -> str:
    """Extract memory from FULL conversation. Replaces memory.md entirely. Returns the new memory text."""
    existing_memory = await load_memory(session_id)

    conversation_text = "\n".join(
        f"{m.get('role', 'user')}: {m.get('content', '')}"
        for m in all_messages
    )

    system_prompt = EXTRACTION_FULL_SYSTEM_PROMPT
    if custom_instruction:
        system_prompt += f"\n\nAdditional instruction from user: {custom_instruction}"

    user_prompt = (
        f"## Existing Memory\n{existing_memory}\n\n"
        f"## Full Conversation ({len(all_messages)} messages)\n{conversation_text}\n\n"
        "Produce the complete updated memory document."
    )

    llm_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    result = await chat_completion(
        provider=backend_config["provider"],
        api_key=backend_config["api_key"],
        model=backend_config["model"],
        base_url=backend_config["base_url"],
        messages=llm_messages,
        params=backend_config.get("params", {}),
    )

    # Don't save — return only. User decides via /memory edit.
    logger.info(f"Full memory extraction for session {session_id} (preview only)")
    return result


# ── 8. should_extract_memory ───────────────────────────────────────────────

async def should_extract_memory(session_id: int, message_count: int) -> bool:
    """Return True every 10 messages."""
    return message_count > 0 and message_count % 10 == 0


# ── 9. inject_memory_into_prompt ───────────────────────────────────────────

async def inject_memory_into_prompt(session_id: int, system_parts: list[str]) -> None:
    """Load memory.md and prepend relevant memories into the system prompt parts list."""
    memory = await load_memory(session_id)
    if memory and memory.strip():
        system_parts.append(f"[Long-term Memory]\n{memory}")


# ── Helpers for /undo and /retry ───────────────────────────────────────────

async def remove_last_chat_md_entries(session_id: int, count: int = 2) -> None:
    """Remove the last `count` entries from chat.md."""
    path = _session_dir(session_id) / "chat.md"
    if not path.exists():
        return

    async with aiofiles.open(path, mode="r", encoding="utf-8") as f:
        text = await f.read()

    # Split on message headers
    pattern = re.compile(r"(?=\n## \[)")
    parts = pattern.split(text)

    if len(parts) <= count:
        # Remove everything
        async with aiofiles.open(path, mode="w", encoding="utf-8") as f:
            await f.write("")
        return

    # Keep all but the last `count` parts
    trimmed = "".join(parts[:-count])
    async with aiofiles.open(path, mode="w", encoding="utf-8") as f:
        await f.write(trimmed)


# ── Rolling Summary ────────────────────────────────────────────────────────

SUMMARY_SYSTEM_PROMPT = (
    "你是一个故事回顾助手。根据已有的故事摘要和新的对话内容，生成一份更新后的故事回顾。\n\n"
    "要求：\n"
    "- 以故事回顾的方式书写，不要写成干巴巴的列表\n"
    "- 保持故事连贯性：记录角色、关系、情节发展、情感状态\n"
    "- 关注对角色扮演续写最重要的信息\n"
    "- 使用与对话相同的语言\n"
    "- 控制在2000字符以内\n"
    "- 直接输出回顾内容，不要加任何前缀说明"
)


async def load_summary(session_id: int) -> str:
    """Read summary.md, return empty string if not found."""
    path = _session_dir(session_id) / "summary.md"
    if not path.exists():
        return ""
    async with aiofiles.open(path, mode="r", encoding="utf-8") as f:
        return await f.read()


async def save_summary(session_id: int, text: str) -> None:
    """Write summary.md (overwrite)."""
    d = _ensure_dir(session_id)
    async with aiofiles.open(d / "summary.md", mode="w", encoding="utf-8") as f:
        await f.write(text)


def should_update_summary(
    messages: list[dict[str, Any]],
    max_history_chars: int,
) -> bool:
    """Return True when total message chars exceed max_history_chars."""
    total = sum(len(m.get("content", "")) for m in messages)
    return total > max_history_chars


async def update_rolling_summary(
    session_id: int,
    messages: list[dict[str, Any]],
    backend_config: dict[str, Any],
    max_history_chars: int,
) -> None:
    """Summarize messages that will be truncated and save to summary.md."""
    try:
        existing_summary = await load_summary(session_id)

        # Figure out which messages will be kept (from the end)
        kept_chars = 0
        cut_index = len(messages)
        for i in range(len(messages) - 1, -1, -1):
            msg_len = len(messages[i].get("content", ""))
            if kept_chars + msg_len > max_history_chars:
                cut_index = i + 1
                break
            kept_chars += msg_len

        truncated_messages = messages[:cut_index]
        if not truncated_messages:
            return

        # Build text of truncated messages
        truncated_text = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}"
            for m in truncated_messages
        )

        user_prompt_parts = []
        if existing_summary:
            user_prompt_parts.append(f"## 已有故事摘要\n{existing_summary}")
        user_prompt_parts.append(f"## 需要总结的新对话\n{truncated_text}")
        user_prompt_parts.append("请生成更新后的完整故事回顾。")

        llm_messages = [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(user_prompt_parts)},
        ]

        new_summary = await chat_completion(
            provider=backend_config["provider"],
            api_key=backend_config["api_key"],
            model=backend_config["model"],
            base_url=backend_config["base_url"],
            messages=llm_messages,
            params=backend_config.get("params", {}),
        )

        await save_summary(session_id, new_summary)
        logger.info(f"Rolling summary updated for session {session_id}")

    except Exception as exc:
        logger.error(f"Rolling summary update failed for session {session_id}: {exc}")


async def append_manual_memory(session_id: int, text: str) -> None:
    """Append a manual memory entry to memory.md."""
    d = _ensure_dir(session_id)
    path = d / "memory.md"

    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    entry = f"\n\n### Manual Note [{now}]\n- {text}\n"

    if not path.exists():
        content = MEMORY_TEMPLATE + entry
        async with aiofiles.open(path, mode="w", encoding="utf-8") as f:
            await f.write(content)
    else:
        async with aiofiles.open(path, mode="a", encoding="utf-8") as f:
            await f.write(entry)
