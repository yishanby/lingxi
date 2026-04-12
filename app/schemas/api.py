from __future__ import annotations

import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Character ────────────────────────────────────────────────────────────────

class CharacterCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=256)
    avatar: Optional[str] = None
    description: str = ""
    personality: str = ""
    scenario: str = ""
    first_message: str = ""
    example_dialogues: str = ""
    system_prompt: str = ""
    creator_notes: str = ""
    tags: list[str] = []


class CharacterUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=256)
    avatar: Optional[str] = None
    description: Optional[str] = None
    personality: Optional[str] = None
    scenario: Optional[str] = None
    first_message: Optional[str] = None
    example_dialogues: Optional[str] = None
    system_prompt: Optional[str] = None
    creator_notes: Optional[str] = None
    tags: Optional[list[str]] = None


class CharacterOut(BaseModel):
    id: int
    name: str
    avatar: Optional[str] = None
    description: str
    personality: str
    scenario: str
    first_message: str
    example_dialogues: str
    system_prompt: str
    creator_notes: str
    tags: list[str]
    source: str
    created_at: datetime.datetime
    updated_at: datetime.datetime

    model_config = {"from_attributes": True}


# ── WorldBook ────────────────────────────────────────────────────────────────

class WorldBookEntry(BaseModel):
    keyword: str
    content: str
    position: str = "before_char"  # before_char / after_char
    enabled: bool = True


class WorldBookCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=256)
    entries: list[WorldBookEntry] = []


class WorldBookUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=256)
    entries: Optional[list[WorldBookEntry]] = None


class WorldBookOut(BaseModel):
    id: int
    name: str
    entries: list[WorldBookEntry]
    created_at: datetime.datetime
    updated_at: datetime.datetime

    model_config = {"from_attributes": True}


# ── Session ──────────────────────────────────────────────────────────────────

class MessageItem(BaseModel):
    role: str  # user / assistant / system
    content: str
    timestamp: Optional[datetime.datetime] = None


class SessionCreate(BaseModel):
    character_id: int
    worldbook_ids: list[int] = []
    feishu_chat_id: Optional[str] = None
    user_id: Optional[str] = None


class SessionOut(BaseModel):
    id: int
    character_id: int
    worldbook_ids: list[int]
    feishu_chat_id: Optional[str]
    user_id: Optional[str]
    messages: list[MessageItem]
    status: str
    created_at: datetime.datetime

    model_config = {"from_attributes": True}


class SessionMessageIn(BaseModel):
    content: str = Field(..., min_length=1)
    backend_id: Optional[int] = None  # which backend to use; None = first available


# ── Backend ──────────────────────────────────────────────────────────────────

class BackendCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=256)
    provider: str = Field(..., pattern=r"^(openai|anthropic|custom)$")
    api_key: str = ""
    model: str = ""
    base_url: str = ""
    params: dict[str, Any] = {}


class BackendUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=256)
    provider: Optional[str] = Field(None, pattern=r"^(openai|anthropic|custom)$")
    api_key: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    params: Optional[dict[str, Any]] = None


class BackendOut(BaseModel):
    id: int
    name: str
    provider: str
    model: str
    base_url: str
    params: dict[str, Any]
    created_at: datetime.datetime
    updated_at: datetime.datetime
    # api_key intentionally excluded from output

    model_config = {"from_attributes": True}
