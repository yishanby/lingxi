from __future__ import annotations

import logging

import pytest

from app.config import settings
from app.schemas.api import MessageItem, WorldBookEntry
from app.services.context_builder import ContextBuilder, ContextSources
from app.services.token_utils import estimate_messages_tokens, estimate_tokens
from app.services.prompt_policy import (
    REQUIRED_OPENING,
    build_invariant_prompt,
    build_priming_history,
)
from app.services.prompt import assemble_prompt


def _character(**overrides: str) -> dict[str, str]:
    character = {
        "name": "灵汐",
        "system_prompt": "",
        "personality": "",
        "scenario": "",
        "description": "",
        "example_dialogues": "",
    }
    character.update(overrides)
    return character


def test_estimate_messages_tokens_includes_message_framing_overhead() -> None:
    messages = [
        {"role": "system", "content": "规则"},
        {"role": "user", "content": "continue"},
    ]

    assert estimate_messages_tokens(messages) == sum(
        estimate_tokens(message["content"]) + 4 for message in messages
    )


def test_build_prioritizes_protected_context_within_total_budget() -> None:
    early_fact = "早期关键事实：灵汐在月蚀之夜救过逸山。"
    summary_fact = "故事摘要开端：两人因一封匿名信重逢。"
    current_state = "当前故事状态：两人正在旧城钟楼顶层对峙。"
    current_user_message = "现在告诉我钟声响起后发生了什么。"
    sources = ContextSources(
        character=_character(description=early_fact + "\n" + "角色设定。" * 1200),
        worldbook=["月蚀会唤醒沉睡的城门。\n" + "世界设定。" * 1200],
        story_state=current_state,
        memory="逸山仍保留着灵汐交给他的银钥匙。",
        episodes=["共同片段：钟楼的门曾在午夜自行打开。" * 50],
        rag=["共同片段：钟楼的门曾在午夜自行打开。" * 50],
        summary=summary_fact * 100,
        recent=[
            {"role": "user", "content": "我们已经到钟楼了吗？"},
            {"role": "assistant", "content": "是的，门就在你们身后合上。"},
        ],
        user_message=current_user_message,
    )

    result = ContextBuilder(total_budget=800, min_recent_messages=2).build(sources)

    combined = "\n".join(message["content"] for message in result.messages)
    assert REQUIRED_OPENING in combined
    assert current_state in combined
    assert early_fact in combined
    assert result.messages[-1] == {"role": "user", "content": current_user_message}
    assert result.messages[-3:-1] == sources.recent
    assert result.total_tokens == estimate_messages_tokens(result.messages)
    assert sum(result.tokens_by_layer.values()) == result.total_tokens
    assert result.total_tokens <= 800


def test_build_deduplicates_identical_episode_and_rag_content() -> None:
    duplicate = "钟楼的门曾在午夜自行打开。"
    sources = ContextSources(
        character=_character(),
        episodes=[duplicate],
        rag=[duplicate],
        user_message="继续。",
    )

    result = ContextBuilder(total_budget=800, min_recent_messages=0).build(sources)

    combined = "\n".join(message["content"] for message in result.messages)
    assert combined.count(duplicate) == 1


def test_build_clips_single_line_optional_history_with_framing_overhead() -> None:
    huge_optional_message = "这是不能整行装入预算的旧回复。" * 200
    sources = ContextSources(
        character=_character(),
        recent=[{"role": "assistant", "content": huge_optional_message}],
        user_message="继续。",
    )

    result = ContextBuilder(total_budget=330, min_recent_messages=0).build(sources)

    assert result.total_tokens == estimate_messages_tokens(result.messages)
    assert result.total_tokens <= 330
    assert result.messages[-1] == {"role": "user", "content": "继续。"}


def test_assemble_prompt_logs_layer_tokens_without_writing_plaintext_debug(
    tmp_path, monkeypatch, caplog
) -> None:
    (tmp_path / "data").mkdir()
    monkeypatch.chdir(tmp_path)
    secret_user_message = "不要把这句明文写进诊断文件。"

    with caplog.at_level(logging.DEBUG, logger="app.services.prompt"):
        messages = assemble_prompt(
            character=_character(),
            worldbook_entries=[],
            chat_history=[],
            user_message=secret_user_message,
        )

    assert messages[-1] == {"role": "user", "content": secret_user_message}
    assert not (tmp_path / "data" / "last_prompt_debug.json").exists()
    prompt_record = next(
        record
        for record in caplog.records
        if record.name == "app.services.prompt"
        and hasattr(record, "tokens_by_layer")
    )
    assert prompt_record.tokens_by_layer
    assert secret_user_message not in prompt_record.getMessage()


def test_assemble_prompt_preserves_priming_then_history_seed_order() -> None:
    seed = {"role": "assistant", "content": "跨会话历史种子。"}
    character = _character()
    character["_history_seed"] = [seed]

    messages = assemble_prompt(
        character=character,
        worldbook_entries=[],
        chat_history=[],
        user_message="继续。",
        user_name="逸山",
    )

    assert messages[1:7] == build_priming_history("灵汐", "逸山")
    assert messages[7] == seed


def test_build_raises_when_mandatory_context_exceeds_budget() -> None:
    sources = ContextSources(
        character=_character(),
        story_state="必须保留的当前状态。",
        recent=[
            {"role": "user", "content": "必须保留的最近用户消息。"},
            {"role": "assistant", "content": "必须保留的最近助手消息。"},
        ],
        user_message="必须保留的当前消息。",
    )

    with pytest.raises(ValueError, match="mandatory prompt context"):
        ContextBuilder(total_budget=300, min_recent_messages=2).build(sources)


def test_build_deduplicates_case_and_unicode_whitespace_variants() -> None:
    sources = ContextSources(
        character=_character(),
        episodes=["  Moon\u2003Gate  钟楼  "],
        rag=["moon gate\n钟楼"],
        user_message="继续。",
    )

    result = ContextBuilder(total_budget=800, min_recent_messages=0).build(sources)

    combined = "\n".join(message["content"] for message in result.messages)
    assert "Moon\u2003Gate" in combined
    assert "moon gate\n钟楼" not in combined


def test_build_keeps_higher_priority_context_before_optional_recent() -> None:
    character_fact = "最高优先角色事实。"
    character_message = {
        "role": "system",
        "content": f"[Character: 灵汐]\n{character_fact}",
    }
    mandatory = [
        {"role": "system", "content": build_invariant_prompt()},
        {"role": "user", "content": "继续。"},
    ]
    budget = estimate_messages_tokens(mandatory + [character_message])
    sources = ContextSources(
        character=_character(description=character_fact),
        memory="低一层的记忆。",
        episodes=["更低一层的片段。"],
        summary="更低一层的摘要。",
        recent=[{"role": "assistant", "content": "最低优先的旧消息。"}],
        user_message="继续。",
    )

    result = ContextBuilder(total_budget=budget, min_recent_messages=0).build(sources)

    combined = "\n".join(message["content"] for message in result.messages)
    assert character_fact in combined
    assert "低一层的记忆。" not in combined
    assert "更低一层的片段。" not in combined
    assert "更低一层的摘要。" not in combined
    assert "最低优先的旧消息。" not in combined


def test_optional_recent_selection_is_newest_first_but_output_is_chronological() -> None:
    selected = [
        {"role": "user", "content": "中间消息。"},
        {"role": "assistant", "content": "最新消息。"},
    ]
    mandatory = [
        {"role": "system", "content": build_invariant_prompt()},
        {"role": "user", "content": "当前消息。"},
    ]
    budget = estimate_messages_tokens(mandatory + selected)
    sources = ContextSources(
        character=_character(),
        recent=[
            {"role": "assistant", "content": "最旧消息。"},
            *selected,
        ],
        user_message="当前消息。",
    )

    result = ContextBuilder(total_budget=budget, min_recent_messages=0).build(sources)

    assert result.messages[-3:-1] == selected
    assert not any(
        message["content"] == "最旧消息。" for message in result.messages
    )


def test_clipped_high_priority_layer_consumes_budget_before_summary() -> None:
    sources = ContextSources(
        character={
            "name": "C",
            "description": "HIGH_PRIORITY_FACT " + "character-detail " * 500,
        },
        summary="LOW_PRIORITY_SUMMARY " * 100,
        user_message="continue",
    )

    result = ContextBuilder(total_budget=366, min_recent_messages=0).build(sources)

    combined = "\n".join(message["content"] for message in result.messages)
    assert "HIGH_PRIORITY_FACT" in combined
    assert "LOW_PRIORITY_SUMMARY" not in combined
    assert result.total_tokens == estimate_messages_tokens(result.messages)
    assert sum(result.tokens_by_layer.values()) == result.total_tokens
    assert result.total_tokens <= 366


def test_assemble_prompt_maps_story_state_and_episodes() -> None:
    story_state = "CURRENT_STORY_STATE: the gate is open."
    episode = "EPISODE_FACT: the key was hidden in the clock."

    messages = assemble_prompt(
        character=_character(),
        worldbook_entries=[],
        chat_history=[],
        user_message="continue",
        story_state=story_state,
        episodes=[episode],
    )

    combined = "\n".join(message["content"] for message in messages)
    assert story_state in combined
    assert episode in combined
    assert messages[-1] == {"role": "user", "content": "continue"}
    assert estimate_messages_tokens(messages) <= settings.total_token_budget


@pytest.mark.parametrize(
    ("persona_position", "includes_persona"),
    [
        ("after_scenario", True),
        ("in_prompt", True),
        ("none", False),
    ],
)
def test_assemble_prompt_preserves_persona_and_worldbook_position_order(
    persona_position: str,
    includes_persona: bool,
) -> None:
    messages = assemble_prompt(
        character=_character(
            scenario="SCENARIO_MARKER",
            description="DESCRIPTION_MARKER",
        ),
        worldbook_entries=[
            WorldBookEntry(
                keyword="trigger",
                content="BEFORE_CHARACTER_MARKER",
                position="before_char",
            ),
            WorldBookEntry(
                keyword="trigger",
                content="AFTER_CHARACTER_MARKER",
                position="after_char",
            ),
        ],
        chat_history=[MessageItem(role="assistant", content="existing")],
        user_message="trigger",
        user_persona="PERSONA_MARKER",
        persona_position=persona_position,
    )

    combined = "\n".join(message["content"] for message in messages)
    scenario_index = combined.index("SCENARIO_MARKER")
    before_index = combined.index("BEFORE_CHARACTER_MARKER")
    description_index = combined.index("DESCRIPTION_MARKER")
    after_index = combined.index("AFTER_CHARACTER_MARKER")
    if includes_persona:
        persona_index = combined.index("PERSONA_MARKER")
        assert scenario_index < persona_index < before_index
    else:
        assert "PERSONA_MARKER" not in combined
        assert scenario_index < before_index
    assert before_index < description_index < after_index


def test_optional_episode_summary_and_recent_share_deduplication() -> None:
    duplicate = "SHARED_OPTIONAL_FACT"
    sources = ContextSources(
        character={"name": "C"},
        episodes=[duplicate],
        summary=duplicate,
        recent=[{"role": "assistant", "content": duplicate}],
        user_message="continue",
    )

    result = ContextBuilder(total_budget=2000, min_recent_messages=0).build(sources)

    combined = "\n".join(message["content"] for message in result.messages)
    assert combined.count(duplicate) == 1


def test_protected_recent_tail_wins_deduplication_over_optional_sources() -> None:
    duplicate = "PROTECTED_SHARED_FACT"
    protected = {"role": "assistant", "content": duplicate}
    sources = ContextSources(
        character={
            "name": "C",
            "_history_seed": [{"role": "assistant", "content": duplicate}],
        },
        episodes=[duplicate],
        summary=duplicate,
        recent=[protected],
        user_message="continue",
        is_new_conversation=True,
    )

    result = ContextBuilder(total_budget=3000, min_recent_messages=1).build(sources)

    combined = "\n".join(message["content"] for message in result.messages)
    assert combined.count(duplicate) == 1
    assert result.messages[-2] == protected


def test_multiline_summary_uses_full_200_token_optional_budget() -> None:
    optional_budget = 200
    mandatory = [
        {"role": "system", "content": build_invariant_prompt()},
        {"role": "user", "content": "continue"},
    ]
    sources = ContextSources(
        character={"name": "C"},
        summary=(
            "SUMMARY_FIRST_LINE\n"
            "SECOND_LINE_FACT " + "detail " * 1000
        ),
        recent=[{"role": "assistant", "content": "LOWER_RECENT_FACT"}],
        user_message="continue",
    )
    budget = estimate_messages_tokens(mandatory) + optional_budget

    result = ContextBuilder(total_budget=budget, min_recent_messages=0).build(sources)

    combined = "\n".join(message["content"] for message in result.messages)
    assert "SUMMARY_FIRST_LINE" in combined
    assert "SECOND_LINE_FACT" in combined
    assert "LOWER_RECENT_FACT" not in combined
    assert result.tokens_by_layer["summary"] == optional_budget
    assert result.total_tokens == budget


def test_emits_core_before_story_then_recalled_context() -> None:
    sources = ContextSources(
        character={"name": "C", "description": "CORE_SETTING_MARKER"},
        story_state="CURRENT_STORY_MARKER",
        memory="LONG_MEMORY_MARKER",
        episodes=["EPISODE_MARKER"],
        summary="SUMMARY_MARKER",
        recent=[{"role": "assistant", "content": "RECENT_MARKER"}],
        user_message="CURRENT_USER_MARKER",
    )

    result = ContextBuilder(total_budget=3000, min_recent_messages=1).build(sources)

    combined = "\n".join(message["content"] for message in result.messages)
    assert (
        combined.index(REQUIRED_OPENING)
        < combined.index("CORE_SETTING_MARKER")
        < combined.index("CURRENT_STORY_MARKER")
        < combined.index("LONG_MEMORY_MARKER")
        < combined.index("EPISODE_MARKER")
        < combined.index("SUMMARY_MARKER")
        < combined.index("RECENT_MARKER")
        < combined.index("CURRENT_USER_MARKER")
    )


def test_history_seed_has_separate_exact_token_diagnostics() -> None:
    seed = {"role": "assistant", "content": "UNIQUE_HISTORY_SEED"}
    sources = ContextSources(
        character={"name": "C", "_history_seed": [seed]},
        user_message="continue",
        is_new_conversation=True,
    )

    result = ContextBuilder(total_budget=3000, min_recent_messages=0).build(sources)

    assert result.tokens_by_layer["history_seed"] == estimate_messages_tokens(
        [seed]
    )
    assert sum(result.tokens_by_layer.values()) == result.total_tokens
