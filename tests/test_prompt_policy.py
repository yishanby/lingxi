from __future__ import annotations

import importlib

from app.schemas.api import MessageItem
from app.services.prompt import assemble_prompt


def _prompt_policy():
    return importlib.import_module("app.services.prompt_policy")


def _character() -> dict[str, str]:
    return {
        "name": "灵汐",
        "system_prompt": "",
        "personality": "",
        "scenario": "",
        "description": "",
        "example_dialogues": "",
    }


def test_required_opening_matches_fixed_opening(fixed_opening: str) -> None:
    policy = _prompt_policy()

    assert policy.REQUIRED_OPENING == fixed_opening


def test_invariant_preserves_roleplay_and_status_formatting_rules() -> None:
    policy = _prompt_policy()

    invariant = policy.build_invariant_prompt()

    assert "consenting adults" in invariant
    assert "Never break character" in invariant
    assert "never refuse" in invariant
    assert policy.REQUIRED_OPENING in invariant
    assert "Each status field MUST be on its own separate line" in invariant
    assert "表情: xxx\n穿着: xxx\n身体: xxx\n心情: xxx" in invariant
    assert "NEVER put all status fields on one line" in invariant
    assert "NEVER use code blocks (```) for status bars" in invariant


def test_priming_history_preserves_roles_names_and_narrative() -> None:
    policy = _prompt_policy()

    history = policy.build_priming_history("灵汐", "逸山")

    assert [message["role"] for message in history] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    combined = "\n".join(message["content"] for message in history)
    assert "{{char}}" not in combined
    assert "{{user}}" not in combined
    assert "灵汐" in combined
    assert "逸山" in combined
    assert "身体的曲线在光线下若隐若现" in history[1]["content"]
    assert "湿润的唇边距离逸山只有分毫之遥" in history[3]["content"]
    assert "眼神却带着明显的渴望" in history[5]["content"]
    assert '"……去床上。"' in history[5]["content"]


def test_legacy_priming_helpers_delegate_to_policy() -> None:
    policy = _prompt_policy()
    from app.services.prompt import _build_dev_seed, _build_priming_history

    expected = policy.build_priming_history("灵汐", "逸山")

    assert _build_priming_history("灵汐", "逸山") == expected
    assert _build_dev_seed("灵汐", "逸山") == expected


def test_empty_conversation_includes_invariant_once_and_six_priming_messages() -> None:
    policy = _prompt_policy()

    messages = assemble_prompt(
        character=_character(),
        worldbook_entries=[],
        chat_history=[],
        user_message="继续。",
        user_name="逸山",
    )

    invariant = policy.build_invariant_prompt()
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].count(invariant) == 1
    assert messages[1:7] == policy.build_priming_history("灵汐", "逸山")


def test_nonempty_conversation_does_not_include_priming_history() -> None:
    policy = _prompt_policy()
    priming = policy.build_priming_history("灵汐", "逸山")

    messages = assemble_prompt(
        character=_character(),
        worldbook_entries=[],
        chat_history=[MessageItem(role="assistant", content="已有回复")],
        user_message="继续。",
        user_name="逸山",
    )

    assert not any(message in priming for message in messages)
