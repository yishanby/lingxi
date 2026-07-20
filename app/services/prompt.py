"""Prompt assembly - builds the messages list sent to the LLM."""

from __future__ import annotations

import logging
import re
from typing import Any

from app.schemas.api import MessageItem, WorldBookEntry
from app.services.context_builder import ContextBuilder, ContextSources
from app.services.prompt_policy import build_priming_history

logger = logging.getLogger(__name__)


def _match_worldbook_entries(
    entries: list[WorldBookEntry],
    recent_text: str,
) -> tuple[list[str], list[str]]:
    """Return (before_char, after_char) content lists for keyword matches."""
    before: list[str] = []
    after: list[str] = []
    lower = recent_text.lower()
    for entry in entries:
        if not entry.enabled:
            continue
        keywords = [
            keyword.strip().lower()
            for keyword in entry.keyword.split(",")
            if keyword.strip()
        ]
        if any(keyword in lower for keyword in keywords):
            if entry.position == "after_char":
                after.append(entry.content)
            else:
                before.append(entry.content)
    return before, after


def _legacy_character_context(
    character: dict[str, Any],
    *,
    user_name: str,
    user_persona: str,
    persona_position: str,
    worldbook_before: list[str],
    worldbook_after: list[str],
) -> list[str]:
    """Render the highest-priority tier in the legacy relative order."""
    char_name = str(character.get("name") or "角色")

    def substitute(value: Any) -> str:
        return str(value).replace("{{user}}", user_name).replace(
            "{{char}}", char_name
        )

    parts: list[str] = []
    if character.get("system_prompt"):
        parts.append(substitute(character["system_prompt"]))
    if character.get("personality"):
        parts.append(f"Personality: {substitute(character['personality'])}")
    if character.get("scenario"):
        parts.append(f"Scenario: {substitute(character['scenario'])}")
    if user_persona and persona_position in ("after_scenario", "in_prompt"):
        parts.append(
            f"[User Character: {user_name}]\n{substitute(user_persona)}"
        )
    if worldbook_before:
        parts.append("[World Info]\n" + "\n".join(worldbook_before))
    if character.get("description"):
        parts.append(
            f"[Character: {char_name}]\n{substitute(character['description'])}"
        )
    if worldbook_after:
        parts.append(
            "[Additional World Info]\n" + "\n".join(worldbook_after)
        )
    return parts


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
    rag_context: str = "",
    story_state: str = "",
    episodes: list[str] | None = None,
) -> list[dict[str, str]]:
    """Build the full messages payload for an LLM chat-completion call."""
    from app.config import settings

    recent_text = " ".join(
        message.content for message in chat_history[-10:]
    ) + " " + user_message
    wb_before, wb_after = _match_worldbook_entries(
        worldbook_entries, recent_text
    )

    assets_parts: list[str] = []
    if assets_summary:
        assets_parts.append(f"[资产概览] {assets_summary}")
    if assets_full:
        assets_parts.append(f"[资产详情]\n{assets_full}")

    ordered_character_context = _legacy_character_context(
        character,
        user_name=user_name,
        user_persona=user_persona,
        persona_position=persona_position,
        worldbook_before=wb_before,
        worldbook_after=wb_after,
    )

    sources = ContextSources(
        character=character,
        user_name=user_name,
        story_state=story_state,
        memory=memory_context,
        character_profiles=[character_profiles] if character_profiles else [],
        assets="\n\n".join(assets_parts),
        episodes=list(episodes or []),
        rag=[rag_context] if rag_context else [],
        summary=summary_context,
        recent=[
            {"role": message.role, "content": message.content}
            for message in chat_history
        ],
        user_message=user_message,
        is_new_conversation=not chat_history,
        ordered_character_context=ordered_character_context,
    )
    result = ContextBuilder(
        total_budget=settings.total_token_budget,
        min_recent_messages=settings.min_recent_messages,
    ).build(sources)

    logger.debug(
        "Prompt assembled: %d messages, ~%d tokens",
        len(result.messages),
        result.total_tokens,
        extra={
            "message_count": len(result.messages),
            "tokens_by_layer": result.tokens_by_layer,
            "total_tokens": result.total_tokens,
        },
    )
    return result.messages


def _parse_example_dialogues(
    text: str, char_name: str, user_name: str = "用户"
) -> list[dict[str, str]]:
    """Parse SillyTavern example dialogue into user/assistant messages."""
    messages: list[dict[str, str]] = []
    blocks = re.split(r"<START>", text, flags=re.IGNORECASE)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        for raw_line in block.split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            line = line.replace("{{char}}", char_name).replace(
                "{{user}}", user_name
            )
            if line.startswith(f"{char_name}:"):
                messages.append(
                    {
                        "role": "assistant",
                        "content": line[len(char_name) + 1 :].strip(),
                    }
                )
            elif line.startswith("User:"):
                messages.append({"role": "user", "content": line[5:].strip()})
    return messages


def _build_dev_seed(char_name: str, user_name: str) -> list[dict[str, str]]:
    """Legacy wrapper."""
    return _build_priming_history(char_name, user_name)


def _build_priming_history(
    char_name: str, user_name: str
) -> list[dict[str, str]]:
    """Legacy compatibility wrapper for the extracted priming policy."""
    return build_priming_history(char_name, user_name)
