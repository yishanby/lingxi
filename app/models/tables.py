from __future__ import annotations

import datetime
import json
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Character(Base):
    __tablename__ = "characters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    avatar: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # base64 or URL
    description: Mapped[str] = mapped_column(Text, default="")
    personality: Mapped[str] = mapped_column(Text, default="")
    scenario: Mapped[str] = mapped_column(Text, default="")
    first_message: Mapped[str] = mapped_column(Text, default="")
    example_dialogues: Mapped[str] = mapped_column(Text, default="")
    system_prompt: Mapped[str] = mapped_column(Text, default="")
    creator_notes: Mapped[str] = mapped_column(Text, default="")
    tags: Mapped[str] = mapped_column(Text, default="[]")  # JSON array
    linked_worldbook_ids: Mapped[str] = mapped_column(Text, default="[]")  # JSON array of worldbook IDs
    source: Mapped[str] = mapped_column(String(32), default="manual")  # imported / manual
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    sessions: Mapped[list[Session]] = relationship(back_populates="character")

    @property
    def tags_list(self) -> list[str]:
        return json.loads(self.tags) if self.tags else []

    @tags_list.setter
    def tags_list(self, value: list[str]) -> None:
        self.tags = json.dumps(value, ensure_ascii=False)


class WorldBook(Base):
    __tablename__ = "worldbooks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    entries: Mapped[str] = mapped_column(Text, default="[]")  # JSON array of entries
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    character_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("characters.id"), nullable=False
    )
    worldbook_ids: Mapped[str] = mapped_column(Text, default="[]")  # JSON array of ints
    feishu_chat_id: Mapped[Optional[str]] = mapped_column(String(256), nullable=True, index=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    persona_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("personas.id"), nullable=True)
    user_name: Mapped[str] = mapped_column(String(256), default="用户")  # protagonist name
    user_persona: Mapped[str] = mapped_column(Text, default="")  # protagonist description
    messages: Mapped[str] = mapped_column(Text, default="[]")  # JSON array of message dicts
    summary: Mapped[str] = mapped_column(Text, default="")  # rolling summary of older messages
    summary_up_to: Mapped[int] = mapped_column(Integer, default=0)  # messages[0:N] already summarised
    backend_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("backends.id"), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="active")  # active / archived
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now()
    )

    character: Mapped[Character] = relationship(back_populates="sessions")
    persona: Mapped[Optional["Persona"]] = relationship()
    backend: Mapped[Optional["Backend"]] = relationship()


class Backend(Base):
    __tablename__ = "backends"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)  # openai / anthropic / custom
    api_key: Mapped[str] = mapped_column(Text, default="")
    model: Mapped[str] = mapped_column(String(256), default="")
    base_url: Mapped[str] = mapped_column(Text, default="")
    params: Mapped[str] = mapped_column(Text, default="{}")  # JSON dict
    is_default: Mapped[bool] = mapped_column(Integer, default=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class TokenUsage(Base):
    __tablename__ = "token_usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("sessions.id"), nullable=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(256), nullable=True, index=True)
    character_name: Mapped[str] = mapped_column(String(256), default="")
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    model: Mapped[str] = mapped_column(String(256), default="")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now()
    )


class Persona(Base):
    __tablename__ = "personas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    avatar: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    description: Mapped[str] = mapped_column(Text, default="")  # full persona text
    description_position: Mapped[str] = mapped_column(
        String(32), default="in_prompt"  # in_prompt / after_scenario / top_an / bottom_an / none
    )
    is_default: Mapped[bool] = mapped_column(Integer, default=False)  # default persona
    linked_character_ids: Mapped[str] = mapped_column(Text, default="[]")  # JSON array of char IDs
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
