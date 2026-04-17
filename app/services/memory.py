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


# ── 7a. Asset Tracking ─────────────────────────────────────────────────────

ASSET_EXTRACTION_PROMPT = (
    "You are an asset tracking system for a roleplay conversation. "
    "Given the recent conversation and existing asset records, detect any changes to "
    "the character's assets (money, property, investments, items of value, debts, income, expenses). "
    "Output a COMPLETE updated asset record in this exact markdown format:\n"
    "```\n"
    "# 资产总览\n"
    "- 现金：¥XXX\n"
    "- [资产名称]：[描述/估值]\n"
    "...\n\n"
    "# 近期变动\n"
    "- [日期] [描述] [+/-金额]\n"
    "...（最近10条）\n"
    "```\n"
    "If NO asset changes occurred, output exactly: NO_CHANGE\n"
    "Keep the 近期变动 section to the last 10 entries max. "
    "Use Chinese. Be precise with numbers."
)

ASSET_SUMMARY_PROMPT = (
    "Summarize the following asset record into ONE concise line (under 80 chars) in Chinese. "
    "Format: 总资产约¥XX | 现金¥XX | 主要资产：A、B\n"
    "Output ONLY the summary line, nothing else."
)

# Asset-related keywords for Layer 1 disclosure
_ASSET_KEYWORDS = re.compile(
    r"钱|买|卖|投资|资产|花费|收入|预算|贷款|租金|花了|赚|亏|账|支出|成本|"
    r"价格|估值|财务|经费|薪|工资|报酬|利润|分红|股|债|房产|地产|物业|酒店|"
    r"公司|企业|生意|商业|交易|合同|签约|付款|转账|存款|取款"
)


def is_asset_relevant(text: str) -> bool:
    """Check if user message is asset-related (triggers Layer 1 disclosure)."""
    return bool(_ASSET_KEYWORDS.search(text))


async def load_assets(session_id: int) -> str:
    """Load full asset record from assets.md."""
    p = _session_dir(session_id) / "assets.md"
    if not p.exists():
        return ""
    async with aiofiles.open(p, mode="r", encoding="utf-8") as f:
        return await f.read()


async def save_assets(session_id: int, content: str) -> None:
    """Save full asset record to assets.md."""
    d = _ensure_dir(session_id)
    async with aiofiles.open(d / "assets.md", mode="w", encoding="utf-8") as f:
        await f.write(content)


async def load_assets_summary(session_id: int) -> str:
    """Load one-line asset summary from assets_summary.txt."""
    p = _session_dir(session_id) / "assets_summary.txt"
    if not p.exists():
        return ""
    async with aiofiles.open(p, mode="r", encoding="utf-8") as f:
        return (await f.read()).strip()


async def save_assets_summary(session_id: int, summary: str) -> None:
    """Save one-line asset summary."""
    d = _ensure_dir(session_id)
    async with aiofiles.open(d / "assets_summary.txt", mode="w", encoding="utf-8") as f:
        await f.write(summary.strip())


async def update_assets(
    session_id: int,
    recent_messages: list[dict[str, str]],
    backend_config: dict[str, Any],
) -> bool:
    """Detect asset changes from recent messages and update assets.md + summary.
    Returns True if assets were updated."""
    try:
        existing_assets = await load_assets(session_id)

        conversation_text = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}"
            for m in recent_messages[-20:]
        )

        user_prompt = (
            f"## Existing Assets\n{existing_assets or '(无记录)'}\n\n"
            f"## Recent Conversation\n{conversation_text}\n\n"
            "Detect any asset changes and output the updated record."
        )

        llm_messages = [
            {"role": "system", "content": ASSET_EXTRACTION_PROMPT},
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

        if "NO_CHANGE" in result.strip():
            logger.info(f"No asset changes for session {session_id}")
            return False

        # Save updated assets
        await save_assets(session_id, result.strip())

        # Regenerate summary
        summary_messages = [
            {"role": "system", "content": ASSET_SUMMARY_PROMPT},
            {"role": "user", "content": result.strip()},
        ]
        summary = await chat_completion(
            provider=backend_config["provider"],
            api_key=backend_config["api_key"],
            model=backend_config["model"],
            base_url=backend_config["base_url"],
            messages=summary_messages,
            params=backend_config.get("params", {}),
        )
        await save_assets_summary(session_id, summary.strip())
        logger.info(f"Assets updated for session {session_id}")
        return True

    except Exception as exc:
        logger.error(f"Asset update failed for session {session_id}: {exc}")
        return False


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

    # Truncate to avoid exceeding API request size limits
    max_chars = 80000
    if len(conversation_text) > max_chars:
        conversation_text = conversation_text[-max_chars:]
        # Find first complete line
        nl = conversation_text.find("\n")
        if nl > 0:
            conversation_text = conversation_text[nl + 1:]

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


# ── 7c. extract_memory_rebuild (chunked, processes entire chat.md) ─────────

async def extract_memory_rebuild(
    session_id: int,
    all_messages: list[dict[str, str]],
    backend_config: dict[str, Any],
    custom_instruction: str | None = None,
) -> str:
    """Process entire conversation in chunks and rebuild memory from scratch.
    
    Splits the conversation into ~60K char chunks, processes each sequentially,
    accumulating memory as it goes. Finally produces a clean consolidated memory.
    """
    CHUNK_CHARS = 60000

    # Build all conversation lines
    lines = [
        f"{m.get('role', 'user')}: {m.get('content', '')}"
        for m in all_messages
    ]

    # Split into chunks by character count
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_size = 0
    for line in lines:
        line_len = len(line) + 1  # +1 for newline
        if current_size + line_len > CHUNK_CHARS and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_size = 0
        current_chunk.append(line)
        current_size += line_len
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    logger.info(f"Memory rebuild for session {session_id}: {len(all_messages)} messages, {len(chunks)} chunks")

    accumulated_memory = ""

    system_prompt = EXTRACTION_FULL_SYSTEM_PROMPT
    if custom_instruction:
        system_prompt += f"\n\nAdditional instruction from user: {custom_instruction}"

    for i, chunk in enumerate(chunks):
        user_prompt = (
            f"## Current Memory\n{accumulated_memory}\n\n"
            f"## Conversation (part {i+1}/{len(chunks)})\n{chunk}\n\n"
            "Update the memory document with any new important information from this conversation segment. "
            "Keep existing important items. Remove outdated info. Output the complete updated memory."
        )

        llm_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        accumulated_memory = await chat_completion(
            provider=backend_config["provider"],
            api_key=backend_config["api_key"],
            model=backend_config["model"],
            base_url=backend_config["base_url"],
            messages=llm_messages,
            params=backend_config.get("params", {}),
        )
        logger.info(f"Memory rebuild chunk {i+1}/{len(chunks)} done, memory size: {len(accumulated_memory)}")

        # Persist after each chunk so progress is not lost
        preview_path = _session_dir(session_id) / "memory_rebuild_preview.md"
        async with aiofiles.open(preview_path, mode="w", encoding="utf-8") as f:
            await f.write(accumulated_memory)

    logger.info(f"Memory rebuild complete for session {session_id}, saved to preview")
    return accumulated_memory


# ── 7d. compact_memory ─────────────────────────────────────────────────────

COMPACT_SYSTEM_PROMPT = (
    "你是一个记忆压缩系统。给定一份角色扮演对话的记忆文档，请将其压缩精炼，保留最重要的信息。\n\n"
    "规则：\n"
    "1. 保留核心人物关系、性格特点、重要情节转折点\n"
    "2. 合并重复或相似的条目\n"
    "3. 删除已过时/不再相关的细节\n"
    "4. 保持原有的markdown标题结构（Characters, Relationships, Key Events等）\n"
    "5. 压缩后的内容应该是原来的 1/3 到 1/2 左右\n"
    "6. 用简洁的语言，每个要点一行\n"
    "7. 直接输出压缩后的完整记忆文档"
)


async def compact_memory(
    session_id: int,
    backend_config: dict[str, Any],
) -> str:
    """压缩memory.md，备份旧文件后用LLM精练。"""
    import shutil

    memory = await load_memory(session_id)
    if not memory or not memory.strip():
        raise ValueError("记忆为空，无需压缩")

    # Backup with timestamp
    d = _ensure_dir(session_id)
    now = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_path = d / f"memory_backup_{now}.md"
    shutil.copy2(d / "memory.md", backup_path)
    logger.info(f"Memory backed up to {backup_path}")

    llm_messages = [
        {"role": "system", "content": COMPACT_SYSTEM_PROMPT},
        {"role": "user", "content": f"以下是需要压缩的记忆文档（{len(memory)}字）：\n\n{memory}"},
    ]

    compacted = await chat_completion(
        provider=backend_config["provider"],
        api_key=backend_config["api_key"],
        model=backend_config["model"],
        base_url=backend_config["base_url"],
        messages=llm_messages,
        params=backend_config.get("params", {}),
    )

    # Save to preview file, not overwriting memory.md
    preview_path = d / "memory_compact_preview.md"
    async with aiofiles.open(preview_path, mode="w", encoding="utf-8") as f:
        await f.write(compacted)
    logger.info(f"Memory compacted for session {session_id}: {len(memory)} -> {len(compacted)} chars (preview)")
    return compacted, len(memory), len(compacted), backup_path.name


# ── 8. should_extract_memory ───────────────────────────────────────────────

_EXTRACT_INTERVAL = 10  # trigger every N new messages


def _last_extract_path(session_id: int) -> Path:
    return _memory_dir(session_id) / ".last_extract_count"


async def _read_last_extract_count(session_id: int) -> int:
    p = _last_extract_path(session_id)
    if p.exists():
        try:
            async with aiofiles.open(p, mode="r", encoding="utf-8") as f:
                return int((await f.read()).strip())
        except (ValueError, OSError):
            pass
    return 0


async def _save_last_extract_count(session_id: int, count: int) -> None:
    p = _last_extract_path(session_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(p, mode="w", encoding="utf-8") as f:
        await f.write(str(count))


async def should_extract_memory(session_id: int, message_count: int) -> bool:
    """Return True when message_count has grown by >= _EXTRACT_INTERVAL since last extract."""
    if message_count <= 0:
        return False
    last = await _read_last_extract_count(session_id)
    if message_count - last >= _EXTRACT_INTERVAL:
        await _save_last_extract_count(session_id, message_count)
        return True
    return False


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

        # Truncate to avoid exceeding API request size limits
        max_text_chars = 80000
        if len(truncated_text) > max_text_chars:
            truncated_text = truncated_text[-max_text_chars:]
            nl = truncated_text.find("\n")
            if nl > 0:
                truncated_text = truncated_text[nl + 1:]

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
