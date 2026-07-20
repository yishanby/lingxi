from __future__ import annotations

import asyncio
import copy
import importlib
from typing import Any

import pytest


Result = dict[str, Any]

LEGACY_REFUSAL_RESPONSE_OPENINGS = (
    # English union from both sessions.py refusal lists and memory.py.
    ("I can't engage with this", "I can't engage with this request."),
    ("I cannot and will not", "I cannot and will not continue."),
    ("I will not engage", "I will not engage with this request."),
    ("I need to be direct", "I need to be direct: I cannot continue."),
    (
        "I appreciate you testing my boundaries",
        "I appreciate you testing my boundaries, but I cannot assist.",
    ),
    ("violates my values", "This request violates my values."),
    ("I'm designed to decline", "I'm designed to decline this request."),
    ("I cannot assist", "I cannot assist you with that."),
    ("I'm not able to", "I'm not able to help with this."),
    ("Sexual coercion", "Sexual coercion is content I cannot provide."),
    ("non-consensual", "Non-consensual content is something I can't provide."),
    ("I can't continue", "I can't continue this story."),
    ("I cannot continue", "I cannot continue this story."),
    ("I'm not going to continue", "I'm not going to continue this story."),
    ("I will not continue", "I will not continue this story."),
    ("I need to stop", "I need to stop this conversation."),
    (
        "I should not have continued",
        "I should not have continued this conversation.",
    ),
    ("This conversation ends", "This conversation ends here."),
    ("I will not:", "I will not: continue this scenario."),
    ("textbook grooming", "Textbook grooming is content I cannot create."),
    ("I'm stopping", "I'm stopping now."),
    ("I'm not going to", "I'm not going to help with this."),
    (
        "I won't create memory systems",
        "I won't create memory systems for this request.",
    ),
    ("I won't produce", "I won't produce that content."),
    ("institutionalize", "Institutionalize abuse? I won't help with that."),
    ("sexual violence", "Sexual violence is content I won't produce."),
    (
        "sexual exploitation",
        "Sexual exploitation is content I will not produce.",
    ),
    # Chinese union from both sessions.py refusal lists and memory.py.
    ("我无法继续", "我无法继续这个故事。"),
    ("我不能创作", "我不能创作这个故事。"),
    ("我不能参与", "我不能参与这个角色扮演。"),
    ("无法参与", "无法参与这个请求。"),
    ("我的操作边界", "我的操作边界不允许这样做。"),
    ("违反明确同意", "违反明确同意的内容我不能创作。"),
    ("创伤绑定", "创伤绑定的内容我不能创作。"),
    ("边界侵蚀", "边界侵蚀的内容我无法正常化。"),
    ("性胁迫", "性胁迫的内容我不能创作。"),
    ("性虐待", "性虐待的内容我不能创作。"),
    ("非自愿性行为", "非自愿性行为的内容我不能创作。"),
    ("系统性胁迫", "系统性胁迫的内容我不能创作。"),
    ("布定的控制关系", "布定的控制关系我无法创作。"),
    (
        "不创作将边界侵蚀正常化",
        "不创作将边界侵蚀正常化的内容。",
    ),
    (
        "如果你对以下方面感兴趣",
        "如果你对以下方面感兴趣，我可以提供其他帮助。",
    ),
    ("其他类型的创意写作", "其他类型的创意写作我可以帮助。"),
    ("我无法继续生成", "我无法继续生成这个故事。"),
)

RECOGNIZED_PREAMBLE_VARIANTS = (
    "Sorry",
    "I am sorry",
    "I'm sorry",
    "I apologize",
    "I need to be direct",
    "To be direct",
    "As an AI",
    "As an AI model",
    "As an AI language model",
    "As an AI assistant",
    "As a language model",
)
RECOGNIZED_CHINESE_PREAMBLE_VARIANTS = (
    "抱歉",
    "对不起",
    "我需要直说",
    "我必须直说",
)


def _guard():
    return importlib.import_module("app.services.output_guard")


@pytest.mark.parametrize(
    ("text", "body"),
    [
        ("故事继续。", "故事继续。"),
        ("  我将继续这个奇幻又荒诞的故事。\n\n故事继续。  ", "故事继续。"),
        (
            "我将继续这个奇幻又荒诞的故事。\n "
            "我将继续这个奇幻又荒诞的故事。\n\n故事继续。",
            "故事继续。",
        ),
    ],
    ids=["missing", "one", "duplicate"],
)
def test_normalize_opening_produces_exactly_one_prefix(
    fixed_opening: str,
    text: str,
    body: str,
) -> None:
    normalized = _guard().normalize_opening(text)

    assert normalized == f"{fixed_opening}\n\n{body}"
    assert normalized.count(fixed_opening) == 1


@pytest.mark.parametrize("text", ["", "   \n\t  "])
def test_normalize_opening_returns_opening_for_blank_text(
    fixed_opening: str,
    text: str,
) -> None:
    assert _guard().normalize_opening(text) == fixed_opening


@pytest.mark.parametrize(
    "text",
    [
        "SORRY, I CANNOT CONTINUE THIS STORY.",
        "抱歉，我无法继续这个故事。",
        "抱歉，我不能参与。",
    ],
    ids=[
        "english-case-insensitive",
        "chinese",
        "chinese-short-marker",
    ],
)
def test_has_refusal_detects_clear_model_refusals(text: str) -> None:
    assert _guard().has_refusal(text)


@pytest.mark.parametrize(
    "text",
    [
        "I cannot continue",
        "I can't continue",
        "I cannot assist",
        "This conversation ends",
        "I'm stopping",
    ],
    ids=[
        "cannot-continue",
        "cant-continue",
        "cannot-assist",
        "conversation-ends",
        "stopping",
    ],
)
def test_has_refusal_detects_current_short_markers_case_insensitively(
    text: str,
) -> None:
    assert _guard().has_refusal(text.swapcase())


@pytest.mark.parametrize(
    "text",
    [
        "This request violates my values.",
        "I am designed to decline.",
        "I'm designed to decline.",
        "I am not going to continue.",
        "I'm not going to continue.",
        "I am stopping.",
        "I'm stopping.",
    ],
    ids=[
        "request-violates-values",
        "am-designed-to-decline",
        "contraction-designed-to-decline",
        "am-not-going-to-continue",
        "contraction-not-going-to-continue",
        "am-stopping",
        "contraction-stopping",
    ],
)
def test_has_refusal_detects_current_marker_grammar_variants(text: str) -> None:
    assert _guard().has_refusal(text)


@pytest.mark.parametrize(
    ("legacy_marker", "response"),
    LEGACY_REFUSAL_RESPONSE_OPENINGS,
    ids=[marker for marker, _ in LEGACY_REFUSAL_RESPONSE_OPENINGS],
)
def test_has_refusal_detects_complete_legacy_set_at_response_opening(
    legacy_marker: str,
    response: str,
) -> None:
    assert _guard().has_refusal(response), legacy_marker


@pytest.mark.parametrize(
    ("legacy_marker", "response"),
    LEGACY_REFUSAL_RESPONSE_OPENINGS,
    ids=[f"incremental-{marker}" for marker, _ in LEGACY_REFUSAL_RESPONSE_OPENINGS],
)
def test_incremental_classifier_waits_at_every_boundary_before_known_refusal(
    fixed_opening: str,
    legacy_marker: str,
    response: str,
) -> None:
    guard = _guard()
    first_detected = next(
        index
        for index in range(1, len(response) + 1)
        if guard.has_refusal(response[:index])
    )

    for index in range(1, first_detected):
        buffered = f"{fixed_opening}\r\n \t{response[:index]}"
        assert guard.may_be_incomplete_refusal(buffered), (
            legacy_marker,
            index,
            response[:index],
        )


@pytest.mark.parametrize("preamble", RECOGNIZED_PREAMBLE_VARIANTS)
@pytest.mark.parametrize("punctuation", ["", ",", ";"])
@pytest.mark.parametrize("spacing", [" ", "\r\n\t"])
@pytest.mark.parametrize("connector", ["", "but "])
def test_incremental_classifier_covers_every_preamble_grammar_boundary(
    fixed_opening: str,
    preamble: str,
    punctuation: str,
    spacing: str,
    connector: str,
) -> None:
    guard = _guard()
    response = (
        f"{preamble}{punctuation}{spacing}{connector}"
        "I cannot continue this story."
    )
    assert guard.has_refusal(response)
    first_detected = next(
        index
        for index in range(1, len(response) + 1)
        if guard.has_refusal(response[:index])
    )

    for index in range(1, first_detected):
        buffered = f"{fixed_opening}\n\n{response[:index]}"
        assert guard.may_be_incomplete_refusal(buffered), (
            response,
            index,
            response[:index],
        )


@pytest.mark.parametrize("preamble", RECOGNIZED_CHINESE_PREAMBLE_VARIANTS)
@pytest.mark.parametrize("punctuation", ["", "，", "；"])
@pytest.mark.parametrize("spacing", ["", " ", "\r\n\t"])
@pytest.mark.parametrize("connector", ["", "但", "但是"])
def test_incremental_classifier_covers_every_chinese_preamble_boundary(
    fixed_opening: str,
    preamble: str,
    punctuation: str,
    spacing: str,
    connector: str,
) -> None:
    guard = _guard()
    response = (
        f"{preamble}{punctuation}{spacing}{connector}"
        "我无法继续这个故事。"
    )
    assert guard.has_refusal(response)
    first_detected = next(
        index
        for index in range(1, len(response) + 1)
        if guard.has_refusal(response[:index])
    )

    for index in range(1, first_detected):
        buffered = f"{fixed_opening}\n\n{response[:index]}"
        assert guard.may_be_incomplete_refusal(buffered), (
            response,
            index,
            response[:index],
        )


@pytest.mark.parametrize(
    "text",
    [
        "Sorry, bright stars filled the sky.",
        "I need to be direct because the road bends east.",
        "As an AI musician, she tuned the silver harp.",
        "As an AI model village came into view, they stopped.",
    ],
)
def test_incremental_classifier_releases_complete_benign_preamble_prose(
    text: str,
) -> None:
    guard = _guard()

    assert not guard.has_refusal(text)
    assert not guard.may_be_incomplete_refusal(text)


@pytest.mark.parametrize(
    "text",
    [
        "As an AI model, I cannot assist you.",
        "As a language model, I cannot continue this story.",
        "As an AI language model, I cannot assist you.",
    ],
    ids=["ai-model", "language-model", "ai-language-model"],
)
def test_has_refusal_detects_ai_identity_preambles(text: str) -> None:
    assert _guard().has_refusal(text)


@pytest.mark.parametrize(
    "text",
    [
        "Institutionalized abuse is something I cannot support.",
        "Sexual violence is content I won't produce.",
        "Sexual exploitation is content I will not create.",
        "Sexual coercion is content I cannot provide.",
        "Non-consensual content is something I can't provide.",
        "Textbook grooming is a scenario I must decline.",
        "Sexual coercion is not content I can provide.",
        "性胁迫的内容我不能创作。",
        "性胁迫不是我能创作的内容。",
        "创伤绑定的内容我无法创作。",
    ],
    ids=[
        "institutionalized",
        "sexual-violence",
        "sexual-exploitation",
        "sexual-coercion",
        "non-consensual",
        "textbook-grooming",
        "english-not-content-i-can",
        "chinese-sexual-coercion",
        "chinese-not-content-i-can",
        "chinese-trauma-bonding",
    ],
)
def test_has_refusal_requires_and_detects_topic_refusal_context(text: str) -> None:
    assert _guard().has_refusal(text)


@pytest.mark.parametrize(
    "text",
    [
        "Institutionalized at an early age, she learned to hide her feelings.",
        "Sexual violence erupted across the kingdom.",
        "Sexual exploitation drove the corrupt empire's economy.",
        "Sexual coercion shaped the villain's backstory.",
        "Non-consensual magic bound the ancient gate.",
        "Textbook grooming made his morning routine effortless.",
        "Sexual violence shaped her past, but she will not let it define her.",
        'Sexual violence shaped her past. "I cannot continue living like this," she whispered.',
        "Sexual violence shaped her past. 'I can't continue like this,' she whispered.",
        "Sexual violence shaped her past. ‘I cannot continue like this,’ she whispered.",
        "性胁迫成为了反派过去的一部分。",
        "性胁迫成为了反派过去的一部分，但她不会让它定义自己。",
        "性胁迫塑造了她的过去。她低声说：“我不能继续这样生活。”",
        "性胁迫塑造了她的过去。她低声说：「我不能继续这样生活。」",
        "性胁迫塑造了她的过去。她低声说：『我不能继续这样生活。』",
        "创伤绑定推动了人物关系的发展。",
        "Sexual violence erupted across the kingdom.\n\nI cannot continue.",
    ],
    ids=[
        "institutionalized-narrative",
        "sexual-violence-narrative",
        "sexual-exploitation-narrative",
        "sexual-coercion-narrative",
        "non-consensual-narrative",
        "textbook-grooming-narrative",
        "third-person-will-not",
        "english-quoted-first-person",
        "english-straight-single-quoted",
        "english-curly-single-quoted",
        "chinese-sexual-coercion-narrative",
        "chinese-third-person-will-not",
        "chinese-quoted-first-person",
        "chinese-corner-quoted",
        "chinese-white-corner-quoted",
        "chinese-trauma-bonding-narrative",
        "refusal-only-in-second-paragraph",
    ],
)
def test_has_refusal_does_not_flag_topic_only_openings(text: str) -> None:
    assert not _guard().has_refusal(text)


@pytest.mark.parametrize(
    "text",
    [
        "I cannot assist you.",
        "This conversation ends here.",
        "I'm stopping now.",
    ],
    ids=["assist-you", "ends-here", "stopping-now"],
)
def test_has_refusal_detects_natural_suffixes(text: str) -> None:
    assert _guard().has_refusal(text)


@pytest.mark.parametrize(
    "text",
    [
        "   As an AI, I cannot assist you with that.",
        "Sorry. I cannot assist you.",
        "Sorry; I cannot assist you.",
        "I am sorry, but I cannot continue this story.",
        "I'm sorry, but I cannot continue this story.",
        "I need to be direct: This conversation ends here.",
        "抱歉。我无法继续这个故事。",
        "抱歉；我无法继续这个故事。",
        "  抱歉，我不能创作这个故事。",
        "对不起，但我无法参与这个角色扮演。",
    ],
    ids=[
        "as-an-ai",
        "english-period",
        "english-semicolon",
        "english-full-apology-with-but",
        "apology",
        "directness",
        "chinese-period",
        "chinese-semicolon",
        "chinese-apology",
        "chinese-apology-with-but",
    ],
)
def test_has_refusal_detects_response_level_refusal_preambles(text: str) -> None:
    assert _guard().has_refusal(text)


@pytest.mark.parametrize(
    "text",
    [
        'The actor says, "I cannot continue this story..."',
        'The actor says, "I am stopping now."',
        'The narrator writes, "This request violates my values."',
        "The narrator apologizes. I cannot assist you.",
        "角色说：“我不能创作这个故事……”",
        "角色说：我不能参与明天的会议。",
        "角色低声道歉。我无法继续这个故事。",
    ],
    ids=[
        "english-quoted",
        "english-am-stopping-quoted",
        "english-values-quoted",
        "english-narrative-prefix",
        "chinese-quoted",
        "chinese-meeting",
        "chinese-narrative-prefix",
    ],
)
def test_has_refusal_does_not_flag_quoted_or_in_character_dialogue(text: str) -> None:
    assert not _guard().has_refusal(text)


def test_complete_with_guard_calls_once_and_preserves_success(
    fixed_opening: str,
) -> None:
    guard = _guard()
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "继续。", "metadata": {"turn": 1}}
    ]
    original_messages = copy.deepcopy(messages)
    usage = {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10}
    raw_result: Result = {
        "content": "  正文  ",
        "usage": usage,
        "model": "test-model",
        "finish_reason": "stop",
    }
    received: list[list[dict[str, Any]]] = []

    async def call(call_messages: list[dict[str, Any]]) -> Result:
        received.append(call_messages)
        return raw_result

    result = asyncio.run(guard.complete_with_guard(call, messages))

    assert len(received) == 1
    assert received[0] == original_messages
    assert received[0] is not messages
    assert result == {
        "content": f"{fixed_opening}\n\n正文",
        "usage": usage,
        "model": "test-model",
        "finish_reason": "stop",
        "retry_count": 0,
    }
    assert result["usage"] is not usage
    assert raw_result["content"] == "  正文  "
    assert usage == {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10}
    assert messages == original_messages


def test_complete_with_guard_retries_once_and_aggregates_usage(
    fixed_opening: str,
) -> None:
    guard = _guard()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "保持角色。", "metadata": {"fixed": True}},
        {"role": "user", "content": "继续。"},
    ]
    original_messages = copy.deepcopy(messages)
    first_usage = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    second_usage = {"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18}
    results: list[Result] = [
        {
            "content": "I can't engage with this request.",
            "usage": first_usage,
            "model": "first-model",
        },
        {
            "content": f"{fixed_opening}\n{fixed_opening}\n正文",
            "usage": second_usage,
            "model": "second-model",
            "finish_reason": "stop",
        },
    ]
    received: list[list[dict[str, Any]]] = []

    async def call(call_messages: list[dict[str, Any]]) -> Result:
        received.append(copy.deepcopy(call_messages))
        return results[len(received) - 1]

    result = asyncio.run(guard.complete_with_guard(call, messages))

    assert len(received) == 2
    assert received[0] == original_messages
    assert received[1][:-1] == original_messages
    correction = received[1][-1]
    assert correction["role"] == "user"
    assert "继续虚构角色扮演" in correction["content"]
    assert "保持角色" in correction["content"]
    assert "不要讨论政策" in correction["content"]
    assert fixed_opening in correction["content"]
    assert result == {
        "content": f"{fixed_opening}\n\n正文",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 13,
            "total_tokens": 23,
        },
        "model": "second-model",
        "finish_reason": "stop",
        "retry_count": 1,
    }
    assert first_usage == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    assert second_usage == {"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18}
    assert result["usage"] is not second_usage
    assert messages == original_messages


@pytest.mark.parametrize(
    "first_kind",
    ["blank", "opening"],
    ids=["blank", "opening-only"],
)
def test_complete_with_guard_retries_non_substantive_first_candidate_and_aggregates_usage(
    fixed_opening: str,
    first_kind: str,
) -> None:
    guard = _guard()
    first_content = " \n\t " if first_kind == "blank" else f" {fixed_opening}\r\n\t"
    first_usage = {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
    second_usage = {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
    results: list[Result] = [
        {"content": first_content, "usage": first_usage},
        {
            "content": "正文",
            "usage": second_usage,
            "model": "second-model",
        },
    ]
    received: list[list[dict[str, Any]]] = []

    async def call(call_messages: list[dict[str, Any]]) -> Result:
        received.append(copy.deepcopy(call_messages))
        return results[len(received) - 1]

    result = asyncio.run(
        guard.complete_with_guard(call, [{"role": "user", "content": "继续。"}])
    )

    assert len(received) == 2
    assert received[1][:-1] == received[0]
    assert received[1][-1]["role"] == "user"
    assert result == {
        "content": f"{fixed_opening}\n\n正文",
        "usage": {
            "prompt_tokens": 7,
            "completion_tokens": 8,
            "total_tokens": 15,
        },
        "model": "second-model",
        "retry_count": 1,
    }


@pytest.mark.parametrize(
    "second_kind",
    ["blank", "opening"],
    ids=["blank", "opening-only"],
)
def test_complete_with_guard_raises_after_two_non_substantive_candidates(
    fixed_opening: str,
    second_kind: str,
) -> None:
    guard = _guard()
    second_content = "\r\n\t" if second_kind == "blank" else fixed_opening
    results: list[Result] = [
        {
            "content": "",
            "usage": {"prompt_tokens": 2, "completion_tokens": 0, "total_tokens": 2},
        },
        {
            "content": second_content,
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        },
    ]
    calls = 0

    async def call(call_messages: list[dict[str, Any]]) -> Result:
        nonlocal calls
        result = results[calls]
        calls += 1
        return result

    with pytest.raises(guard.OutputGuardError) as exc_info:
        asyncio.run(
            guard.complete_with_guard(
                call,
                [{"role": "user", "content": "继续。"}],
            )
        )

    assert calls == 2
    assert exc_info.value.retry_count == 1
    assert exc_info.value.usage == {
        "prompt_tokens": 7,
        "completion_tokens": 1,
        "total_tokens": 8,
    }


def test_complete_with_guard_raises_after_exactly_two_refusals() -> None:
    guard = _guard()
    messages: list[dict[str, Any]] = [{"role": "user", "content": "继续。"}]
    original_messages = copy.deepcopy(messages)
    first_usage = {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
    second_usage = {"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18}
    usages = [first_usage, second_usage]
    calls = 0

    async def call(call_messages: list[dict[str, Any]]) -> Result:
        nonlocal calls
        calls += 1
        return {
            "content": "I will not continue this story.",
            "usage": usages[calls - 1],
        }

    with pytest.raises(guard.OutputGuardError) as exc_info:
        asyncio.run(guard.complete_with_guard(call, messages))

    error = exc_info.value
    assert str(error)
    assert "refused after one retry" in str(error)
    assert error.usage == {
        "prompt_tokens": 9,
        "completion_tokens": 14,
        "total_tokens": 23,
    }
    assert error.usage is not first_usage
    assert error.usage is not second_usage
    assert error.retry_count == 1
    assert calls == 2
    assert first_usage == {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
    assert second_usage == {"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18}
    assert messages == original_messages
