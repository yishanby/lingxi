"""Prioritized, token-budgeted prompt context construction."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from app.services.prompt_policy import build_invariant_prompt, build_priming_history
from app.services.token_utils import (
    estimate_messages_tokens,
    estimate_tokens,
    truncate_to_tokens,
)


@dataclass(slots=True)
class ContextSources:
    character: dict[str, Any]
    worldbook: list[str] = field(default_factory=list)
    user_name: str = "用户"
    user_persona: str = ""
    story_state: str = ""
    memory: str = ""
    character_profiles: list[str] = field(default_factory=list)
    assets: str = ""
    episodes: list[str] = field(default_factory=list)
    rag: list[str] = field(default_factory=list)
    summary: str = ""
    recent: list[dict[str, str]] = field(default_factory=list)
    user_message: str = ""
    is_new_conversation: bool = False
    ordered_character_context: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ContextBuildResult:
    messages: list[dict[str, str]]
    tokens_by_layer: dict[str, int]
    total_tokens: int


class ContextBuilder:
    """Build prompts without allowing optional context to displace invariants."""

    def __init__(self, total_budget: int, min_recent_messages: int) -> None:
        if type(total_budget) is not int or total_budget <= 0:
            raise ValueError("total_budget must be a positive integer")
        if type(min_recent_messages) is not int or min_recent_messages < 0:
            raise ValueError("min_recent_messages must be a non-negative integer")
        self.total_budget = total_budget
        self.min_recent_messages = min_recent_messages

    def build(self, sources: ContextSources) -> ContextBuildResult:
        char_name = str(sources.character.get("name") or "角色")
        recent = [_copy_message(message) for message in sources.recent]
        protected_count = min(self.min_recent_messages, len(recent))
        if protected_count:
            older_recent = recent[:-protected_count]
            protected_recent = recent[-protected_count:]
        else:
            older_recent = recent
            protected_recent = []

        invariant = [{"role": "system", "content": build_invariant_prompt()}]
        story_state = []
        if sources.story_state:
            story_state.append(
                {
                    "role": "system",
                    "content": f"[Current Story State]\n{sources.story_state}",
                }
            )
        priming = (
            build_priming_history(char_name, sources.user_name)
            if sources.is_new_conversation
            else []
        )
        user_message = [{"role": "user", "content": sources.user_message}]

        mandatory = invariant + story_state + priming + protected_recent + user_message
        mandatory_tokens = estimate_messages_tokens(mandatory)
        if mandatory_tokens > self.total_budget:
            raise ValueError(
                "mandatory prompt context exceeds total token budget "
                f"({mandatory_tokens} > {self.total_budget})"
            )
        remaining = self.total_budget - mandatory_tokens

        character_context: list[dict[str, str]] = []
        character_examples: list[dict[str, str]] = []
        history_seed: list[dict[str, str]] = []
        recall_context: list[dict[str, str]] = []
        episode_context: list[dict[str, str]] = []
        summary_context: list[dict[str, str]] = []
        selected_older_reversed: list[dict[str, str]] = []
        selection_open = True

        for candidate, destination in self._character_candidates(sources, char_name):
            if not selection_open:
                break
            fitted, remaining, complete = _fit_optional_message(candidate, remaining)
            if fitted is not None:
                if destination == "example":
                    character_examples.append(fitted)
                elif destination == "seed":
                    history_seed.append(fitted)
                else:
                    character_context.append(fitted)
            if not complete:
                selection_open = False

        recall = _recall_message(sources)
        if selection_open and recall is not None:
            fitted, remaining, complete = _fit_optional_message(recall, remaining)
            if fitted is not None:
                recall_context.append(fitted)
            if not complete:
                selection_open = False

        episodes = _episode_rag_message(sources.episodes, sources.rag)
        if selection_open and episodes is not None:
            fitted, remaining, complete = _fit_optional_message(episodes, remaining)
            if fitted is not None:
                episode_context.append(fitted)
            if not complete:
                selection_open = False

        if selection_open and sources.summary:
            fitted, remaining, complete = _fit_labeled_system_message(
                "[Story So Far]", sources.summary, remaining
            )
            if fitted is not None:
                summary_context.append(fitted)
            if not complete:
                selection_open = False

        if selection_open:
            for message in reversed(older_recent):
                fitted, remaining, complete = _fit_optional_message(
                    message, remaining
                )
                if fitted is None:
                    break
                selected_older_reversed.append(fitted)
                if not complete:
                    break
        selected_recent = list(reversed(selected_older_reversed)) + protected_recent

        messages = (
            invariant
            + story_state
            + character_context
            + recall_context
            + episode_context
            + summary_context
            + character_examples
            + priming
            + history_seed
            + selected_recent
            + user_message
        )
        tokens_by_layer = {
            "invariant": estimate_messages_tokens(invariant),
            "story_state": estimate_messages_tokens(story_state),
            "character_persona_worldbook": estimate_messages_tokens(
                character_context + character_examples + history_seed
            ),
            "memory_profiles_assets": estimate_messages_tokens(recall_context),
            "episodes_rag": estimate_messages_tokens(episode_context),
            "summary": estimate_messages_tokens(summary_context),
            "priming": estimate_messages_tokens(priming),
            "recent": estimate_messages_tokens(selected_recent),
            "user_message": estimate_messages_tokens(user_message),
        }
        total_tokens = estimate_messages_tokens(messages)
        return ContextBuildResult(
            messages=messages,
            tokens_by_layer=tokens_by_layer,
            total_tokens=total_tokens,
        )

    @staticmethod
    def _character_candidates(
        sources: ContextSources, char_name: str
    ) -> list[tuple[dict[str, str], str]]:
        candidates: list[tuple[dict[str, str], str]] = []
        ordered_context = [
            part for part in sources.ordered_character_context if part
        ]
        if ordered_context:
            candidates.append(
                (
                    {
                        "role": "system",
                        "content": "\n\n".join(ordered_context),
                    },
                    "context",
                )
            )
        else:
            candidates.extend(
                ContextBuilder._default_character_context_candidates(
                    sources, char_name
                )
            )

        examples = sources.character.get("example_dialogues")
        if examples:
            candidates.extend(
                (message, "example")
                for message in _parse_example_dialogues(
                    str(examples), char_name, sources.user_name
                )
            )
        if sources.is_new_conversation:
            candidates.extend(
                (_copy_message(message), "seed")
                for message in (sources.character.get("_history_seed") or [])
            )
        return candidates

    @staticmethod
    def _default_character_context_candidates(
        sources: ContextSources, char_name: str
    ) -> list[tuple[dict[str, str], str]]:
        candidates: list[tuple[dict[str, str], str]] = []
        character_parts: list[str] = []
        for key, label in (
            ("system_prompt", None),
            ("personality", "Personality"),
            ("scenario", "Scenario"),
            ("description", f"[Character: {char_name}]"),
        ):
            raw = sources.character.get(key)
            if not raw:
                continue
            value = _substitute(str(raw), char_name, sources.user_name)
            if label in {"Personality", "Scenario"}:
                character_parts.append(f"{label}: {value}")
            elif label:
                character_parts.append(f"{label}\n{value}")
            else:
                character_parts.append(value)
        if character_parts:
            candidates.append(
                (
                    {"role": "system", "content": "\n\n".join(character_parts)},
                    "context",
                )
            )

        if sources.user_persona:
            persona = _substitute(sources.user_persona, char_name, sources.user_name)
            candidates.append(
                (
                    {
                        "role": "system",
                        "content": f"[User Character: {sources.user_name}]\n{persona}",
                    },
                    "context",
                )
            )

        worldbook = [entry for entry in sources.worldbook if entry]
        if worldbook:
            candidates.append(
                (
                    {
                        "role": "system",
                        "content": "[World Info]\n" + "\n\n".join(worldbook),
                    },
                    "context",
                )
            )

        return candidates


def _copy_message(message: dict[str, str]) -> dict[str, str]:
    return {"role": message["role"], "content": message["content"]}


def _substitute(text: str, char_name: str, user_name: str) -> str:
    return text.replace("{{user}}", user_name).replace("{{char}}", char_name)


def _fit_optional_message(
    message: dict[str, str], remaining: int
) -> tuple[dict[str, str] | None, int, bool]:
    message_tokens = estimate_messages_tokens([message])
    if message_tokens <= remaining:
        return _copy_message(message), remaining - message_tokens, True
    content_budget = remaining - 4
    if content_budget <= 0:
        return None, remaining, False
    clipped = _truncate_prefix_to_tokens(message["content"], content_budget)
    if not clipped:
        return None, remaining, False
    fitted = {"role": message["role"], "content": clipped}
    fitted_tokens = estimate_tokens(clipped) + 4
    return fitted, remaining - fitted_tokens, False


def _truncate_prefix_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    low = 0
    high = len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_tokens(text[:middle]) <= max_tokens:
            low = middle
        else:
            high = middle - 1
    return text[:low]


def _fit_labeled_system_message(
    label: str, text: str, remaining: int
) -> tuple[dict[str, str] | None, int, bool]:
    full_message = {"role": "system", "content": f"{label}\n{text}"}
    full_tokens = estimate_messages_tokens([full_message])
    if full_tokens <= remaining:
        return full_message, remaining - full_tokens, True

    header = f"{label}\n"
    body_budget = remaining - estimate_tokens(header) - 4
    if body_budget <= 0:
        fitted, remaining_after_label, _ = _fit_optional_message(
            {"role": "system", "content": label}, remaining
        )
        return fitted, remaining_after_label, False
    clipped = truncate_to_tokens(text, body_budget)
    fitted = {"role": "system", "content": header + clipped}
    while clipped and estimate_messages_tokens([fitted]) > remaining:
        body_budget -= 1
        clipped = truncate_to_tokens(text, body_budget)
        fitted = {"role": "system", "content": header + clipped}
    fitted_tokens = estimate_messages_tokens([fitted])
    if fitted_tokens > remaining:
        return None, remaining, False
    return fitted, remaining - fitted_tokens, False


def _recall_message(sources: ContextSources) -> dict[str, str] | None:
    parts: list[str] = []
    if sources.memory:
        parts.append(f"[Long-term Memory]\n{sources.memory}")
    profiles = [profile for profile in sources.character_profiles if profile]
    if profiles:
        parts.append("[角色详情档案]\n" + "\n\n".join(profiles))
    if sources.assets:
        parts.append(f"[Assets]\n{sources.assets}")
    if not parts:
        return None
    return {"role": "system", "content": "\n\n".join(parts)}


def _episode_rag_message(
    episodes: list[str], rag: list[str]
) -> dict[str, str] | None:
    unique: list[str] = []
    fingerprints: set[str] = set()
    for text in [*episodes, *rag]:
        normalized = " ".join(text.split()).strip().casefold()
        if not normalized:
            continue
        fingerprint = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        unique.append(text)
    if not unique:
        return None
    return {
        "role": "system",
        "content": "[Relevant Episodes and RAG]\n" + "\n\n".join(unique),
    }


def _parse_example_dialogues(
    text: str, char_name: str, user_name: str
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for block in re.split(r"<START>", text, flags=re.IGNORECASE):
        for raw_line in block.strip().splitlines():
            line = _substitute(raw_line.strip(), char_name, user_name)
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
