"""Prompt assembly – builds the messages list sent to the LLM."""

from __future__ import annotations

import json
import re
from typing import Any

from app.schemas.api import MessageItem, WorldBookEntry


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
) -> list[dict[str, str]]:
    """Build the full messages payload for an LLM chat-completion call.

    Order:
      1. System prompt (character system_prompt + personality + scenario + user persona)
      2. Worldbook entries (position=before_char)
      3. Character description block
      4. Worldbook entries (position=after_char)
      5. Example dialogues (as few-shot user/assistant pairs)
      6. Chat history
      7. Current user message
    """
    messages: list[dict[str, str]] = []

    # ── 1. System prompt ─────────────────────────────────────────────────
    system_parts: list[str] = []
    if character.get("system_prompt"):
        system_parts.append(character["system_prompt"].replace("{{user}}", user_name).replace("{{char}}", character["name"]))
    if character.get("personality"):
        system_parts.append(f"Personality: {character['personality'].replace('{{user}}', user_name).replace('{{char}}', character['name'])}")
    if character.get("scenario"):
        system_parts.append(f"Scenario: {character['scenario'].replace('{{user}}', user_name).replace('{{char}}', character['name'])}")

    # User persona (position: after_scenario)
    if user_persona and persona_position == "after_scenario":
        system_parts.append(f"[User Character: {user_name}]\n{user_persona.replace('{{user}}', user_name).replace('{{char}}', character['name'])}")
    # User persona / protagonist definition (position: in_prompt)
    if user_persona and persona_position == "in_prompt":
        system_parts.append(f"[User Character: {user_name}]\n{user_persona.replace('{{user}}', user_name).replace('{{char}}', character['name'])}")

    # ── 2 & 4. Worldbook entries ─────────────────────────────────────────
    recent_text = " ".join(
        m.content for m in chat_history[-10:]
    ) + " " + user_message
    wb_before, wb_after = _match_worldbook_entries(worldbook_entries, recent_text)

    if wb_before:
        system_parts.append("[World Info]\n" + "\n".join(wb_before))

    # ── 3. Description ───────────────────────────────────────────────────
    if character.get("description"):
        desc = character["description"].replace("{{user}}", user_name).replace("{{char}}", character["name"])
        system_parts.append(f"[Character: {character['name']}]\n{desc}")

    if wb_after:
        system_parts.append("[Additional World Info]\n" + "\n".join(wb_after))

    if system_parts:
        messages.append({"role": "system", "content": "\n\n".join(system_parts)})

    # ── 5. Example dialogues (simple {{user}}/{{char}} parsing) ──────────
    if character.get("example_dialogues"):
        examples = _parse_example_dialogues(
            character["example_dialogues"], character["name"], user_name
        )
        messages.extend(examples)

    # ── 6. Chat history ──────────────────────────────────────────────────
    for m in chat_history:
        messages.append({"role": m.role, "content": m.content})

    # ── 7. Current user message ──────────────────────────────────────────
    messages.append({"role": "user", "content": user_message})

    return messages


def _parse_example_dialogues(
    text: str, char_name: str, user_name: str = "用户"
) -> list[dict[str, str]]:
    """Parse SillyTavern example dialogue format into user/assistant message pairs.

    Format:
      <START>
      {{user}}: Hello
      {{char}}: Hi there!
    """
    messages: list[dict[str, str]] = []
    # Split on <START> blocks
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
