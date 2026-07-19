from app.services.token_utils import estimate_tokens


def test_estimate_tokens_returns_positive_count() -> None:
    assert estimate_tokens("测试 memory v2") > 0
