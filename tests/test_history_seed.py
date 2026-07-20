from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.services import history_seed, memory
from app.services.md_store import MarkdownMemoryStore


class _Scalars:
    def __init__(self, values):
        self._values = values

    def all(self):
        return self._values


class _Result:
    def __init__(self, values):
        self._values = values

    def scalars(self):
        return _Scalars(self._values)


class _DB:
    def __init__(self, sessions):
        self.sessions = sessions

    async def execute(self, statement):
        return _Result(self.sessions)


def _legacy_messages(marker: str) -> str:
    return json.dumps(
        [
            {"role": "user", "content": "greeting"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": f"{marker}-user"},
            {"role": "assistant", "content": f"{marker}-assistant"},
        ]
    )


@pytest.mark.asyncio
async def test_history_seed_prefers_existing_markdown_over_stale_db(
    tmp_path, monkeypatch
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    monkeypatch.setattr(memory, "md_store", store)
    await store.append_pair(1, "greeting", "hello", char_name="C", user_name="U")
    await store.append_pair(1, "md-user", "md-assistant", char_name="C", user_name="U")
    session = SimpleNamespace(id=1, messages=_legacy_messages("db"))

    seed = await history_seed.get_history_seed(_DB([session]), 7)

    assert seed == [
        {"role": "user", "content": "md-user"},
        {"role": "assistant", "content": "md-assistant"},
    ]


@pytest.mark.asyncio
async def test_history_seed_empty_markdown_suppresses_stale_db_fallback(
    tmp_path, monkeypatch
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    monkeypatch.setattr(memory, "md_store", store)
    await store.write_text(1, "chat.md", "")
    session = SimpleNamespace(id=1, messages=_legacy_messages("stale"))

    seed = await history_seed.get_history_seed(_DB([session]), 7)

    assert seed == []


@pytest.mark.asyncio
async def test_history_seed_uses_v1_db_only_when_chat_markdown_is_absent(
    tmp_path, monkeypatch
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    monkeypatch.setattr(memory, "md_store", store)
    session = SimpleNamespace(id=1, messages=_legacy_messages("v1"))

    seed = await history_seed.get_history_seed(_DB([session]), 7)

    assert seed == [
        {"role": "user", "content": "v1-user"},
        {"role": "assistant", "content": "v1-assistant"},
    ]


@pytest.mark.asyncio
async def test_history_seed_skips_corrupt_markdown_without_stale_db_fallback(
    tmp_path, monkeypatch, caplog
) -> None:
    store = MarkdownMemoryStore(tmp_path)
    monkeypatch.setattr(memory, "md_store", store)
    await store.write_text(1, "chat.md", "corrupt-secret invalid markdown")
    corrupt = SimpleNamespace(id=1, messages=_legacy_messages("stale"))
    fallback = SimpleNamespace(id=2, messages=_legacy_messages("v1"))

    seed = await history_seed.get_history_seed(_DB([corrupt, fallback]), 7)

    assert seed == [
        {"role": "user", "content": "v1-user"},
        {"role": "assistant", "content": "v1-assistant"},
    ]
    assert "corrupt-secret" not in caplog.text
    assert "session 1" in caplog.text.casefold()
