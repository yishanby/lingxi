"""Token estimation and truncation utilities."""

from __future__ import annotations

import re


def estimate_tokens(text: str) -> int:
    """Estimate token count for mixed Chinese/English text.

    Heuristic: 1 token ≈ 0.75 Chinese characters or 4 English characters.
    Claude tokenizer uses ~0.7 CJK chars per token; we use 0.75 for slight margin.
    """
    if not text:
        return 0
    # Count Chinese characters (CJK Unified Ideographs + common punctuation)
    cjk = len(re.findall(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]', text))
    non_cjk = len(text) - cjk
    return int(cjk / 0.75 + non_cjk / 4)


def estimate_messages_tokens(messages: list[dict[str, str]]) -> int:
    """Estimate chat-message tokens, including per-message framing overhead."""
    return sum(estimate_tokens(message["content"]) + 4 for message in messages)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximately max_tokens, preserving complete lines."""
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    lines = text.split('\n')
    result: list[str] = []
    total = 0
    for line in lines:
        line_tokens = estimate_tokens(line)
        if total + line_tokens > max_tokens and result:
            break
        result.append(line)
        total += line_tokens

    candidate = '\n'.join(result)
    if estimate_tokens(candidate) <= max_tokens:
        return candidate
    low = 0
    high = len(candidate)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_tokens(candidate[:middle]) <= max_tokens:
            low = middle
        else:
            high = middle - 1
    return candidate[:low]
