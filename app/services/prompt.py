"""Prompt assembly – builds the messages list sent to the LLM."""

from __future__ import annotations

import logging
import re
from typing import Any

from app.schemas.api import MessageItem, WorldBookEntry
from app.services.token_utils import estimate_tokens, truncate_to_tokens

logger = logging.getLogger(__name__)


def _match_worldbook_entries(
    entries: list[WorldBookEntry],
    recent_text: str,
) -> tuple[list[str], list[str]]:
    """Return (before_char, after_char) content lists for keyword-matched entries."""
    before: list[str] = []
    after: list[str] = []
    lower = recent_text.lower()
    for entry in entries:
        if not entry.enabled:
            continue
        keywords = [k.strip().lower() for k in entry.keyword.split(",") if k.strip()]
        if any(kw in lower for kw in keywords):
            if entry.position == "after_char":
                after.append(entry.content)
            else:
                before.append(entry.content)
    return before, after


def assemble_prompt(
    *,
    character: dict[str, Any],
    worldbook_entries: list[WorldBookEntry],
    chat_history: list[MessageItem],
    user_message: str,
    user_name: str = "用户",
    user_persona: str = "",
    persona_position: str = "in_prompt",  # in_prompt / after_scenario / none
    memory_context: str = "",
    summary_context: str = "",
    assets_summary: str = "",
    assets_full: str = "",
    character_profiles: str = "",
) -> list[dict[str, str]]:
    """Build the full messages payload for an LLM chat-completion call.

    Uses a token budget system to keep total input under control:
      - Layer 0 (fixed): system prompt, character card, worldbook
      - Layer 1 (recall): memory, assets, character profiles
      - Layer 2 (conversation): summary + recent messages
    """
    from app.config import settings

    messages: list[dict[str, str]] = []

    # ── Layer 0: System prompt (fixed) ───────────────────────────────────
    system_parts: list[str] = []
    if character.get("system_prompt"):
        system_parts.append(character["system_prompt"].replace("{{user}}", user_name).replace("{{char}}", character["name"]))
    if character.get("personality"):
        system_parts.append(f"Personality: {character['personality'].replace('{{user}}', user_name).replace('{{char}}', character['name'])}")
    if character.get("scenario"):
        system_parts.append(f"Scenario: {character['scenario'].replace('{{user}}', user_name).replace('{{char}}', character['name'])}")

    # User persona
    if user_persona and persona_position in ("after_scenario", "in_prompt"):
        system_parts.append(f"[User Character: {user_name}]\n{user_persona.replace('{{user}}', user_name).replace('{{char}}', character['name'])}")

    # Worldbook entries
    recent_text = " ".join(
        m.content for m in chat_history[-10:]
    ) + " " + user_message
    wb_before, wb_after = _match_worldbook_entries(worldbook_entries, recent_text)

    if wb_before:
        system_parts.append("[World Info]\n" + "\n".join(wb_before))

    if character.get("description"):
        desc = character["description"].replace("{{user}}", user_name).replace("{{char}}", character["name"])
        system_parts.append(f"[Character: {character['name']}]\n{desc}")

    if wb_after:
        system_parts.append("[Additional World Info]\n" + "\n".join(wb_after))

    # ── Layer 1: Recall (memory, assets, profiles) ───────────────────────
    layer1_parts: list[str] = []
    if memory_context:
        layer1_parts.append(f"[Long-term Memory]\n{memory_context}")
    if assets_summary:
        layer1_parts.append(f"[资产概览] {assets_summary}")
    if assets_full:
        layer1_parts.append(f"[资产详情]\n{assets_full}")
    if character_profiles:
        layer1_parts.append(f"[角色详情档案]\n{character_profiles}")

    # Truncate layer 1 to budget
    layer1_text = "\n\n".join(layer1_parts)
    layer1_text = truncate_to_tokens(layer1_text, settings.layer1_budget)
    if layer1_text:
        system_parts.append(layer1_text)

    # ── Layer 2: Summary (truncated) ─────────────────────────────────────
    if summary_context:
        truncated_summary = truncate_to_tokens(summary_context, settings.summary_max_tokens)
        system_parts.append(f"[Story So Far]\n{truncated_summary}")

    # Assemble system message
    if system_parts:
        messages.append({"role": "system", "content": "\n\n".join(system_parts)})

    # ── Example dialogues ────────────────────��───────────────────────────
    if character.get("example_dialogues"):
        examples = _parse_example_dialogues(
            character["example_dialogues"], character["name"], user_name
        )
        messages.extend(examples)

    # ── Layer 2: Chat history (token-budget-aware) ───────────────────────
    # Calculate remaining budget for conversation messages
    current_tokens = sum(estimate_tokens(m["content"]) for m in messages)
    user_msg_tokens = estimate_tokens(user_message)
    remaining_budget = settings.total_token_budget - current_tokens - user_msg_tokens

    # Take messages from newest to oldest until budget exhausted
    selected: list[MessageItem] = []
    used_tokens = 0
    for m in reversed(chat_history):
        msg_tokens = estimate_tokens(m.content)
        if used_tokens + msg_tokens > remaining_budget and len(selected) >= settings.min_recent_messages:
            break
        selected.append(m)
        used_tokens += msg_tokens
        # Always include at least min_recent_messages
        if used_tokens > remaining_budget and len(selected) >= settings.min_recent_messages:
            break

    selected.reverse()

    for m in selected:
        messages.append({"role": m.role, "content": m.content})

    # ── Current user message ─────────────────────────────────────────────
    messages.append({"role": "user", "content": user_message})

    # ── Final budget check: trim from oldest history if still over ───────
    total_tokens = sum(estimate_tokens(m["content"]) for m in messages)
    if total_tokens > settings.total_token_budget:
        _trim_to_budget(messages, settings.total_token_budget, settings.min_recent_messages)

    logger.debug(
        "Prompt assembled: %d messages, ~%d tokens",
        len(messages),
        sum(estimate_tokens(m["content"]) for m in messages),
    )

    return messages


def _trim_to_budget(
    messages: list[dict[str, str]],
    budget: int,
    min_recent: int,
) -> None:
    """Remove oldest non-system, non-final messages until under budget."""
    while sum(estimate_tokens(m["content"]) for m in messages) > budget:
        # Find the first removable message (not system, not the last `min_recent + 1` messages)
        # +1 for the current user message at the end
        protected_tail = min_recent + 1
        removable_start = None
        for i, m in enumerate(messages):
            if m["role"] != "system":
                removable_start = i
                break
        if removable_start is None or len(messages) - removable_start <= protected_tail:
            break  # can't remove any more
        messages.pop(removable_start)


def _parse_example_dialogues(
    text: str, char_name: str, user_name: str = "用户"
) -> list[dict[str, str]]:
    """Parse SillyTavern example dialogue format into user/assistant message pairs."""
    messages: list[dict[str, str]] = []
    blocks = re.split(r"<START>", text, flags=re.IGNORECASE)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            line = line.replace("{{char}}", char_name).replace("{{user}}", user_name)
            if line.startswith(f"{char_name}:"):
                messages.append(
                    {"role": "assistant", "content": line[len(char_name) + 1 :].strip()}
                )
            elif line.startswith("User:"):
                messages.append({"role": "user", "content": line[5:].strip()})
    return messages
