"""Token estimation and truncation utilities."""

from __future__ import annotations

import re


def estimate_tokens(text: str) -> int:
    """Estimate token count for mixed Chinese/English text.

    Heuristic: 1 token ≈ 1.5 Chinese characters or 4 English characters.
    """
    if not text:
        return 0
    # Count Chinese characters (CJK Unified Ideographs + common punctuation)
    cjk = len(re.findall(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]', text))
    non_cjk = len(text) - cjk
    return int(cjk / 1.5 + non_cjk / 4)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximately max_tokens, preserving complete lines."""
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
    return '\n'.join(result)
