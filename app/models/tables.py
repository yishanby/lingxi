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
    status: Mapped[str] = mapped_column(String(32), default="active")  # active / archived
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now()
    )

    character: Mapped[Character] = relationship(back_populates="sessions")
    persona: Mapped[Optional["Persona"]] = relationship()


class Backend(Base):
    __tablename__ = "backends"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)  # openai / anthropic / custom
    api_key: Mapped[str] = mapped_column(Text, default="")
    model: Mapped[str] = mapped_column(String(256), default="")
    base_url: Mapped[str] = mapped_column(Text, default="")
    params: Mapped[str] = mapped_column(Text, default="{}")  # JSON dict
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class Persona(Base):
    __tablename__ = "personas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    avatar: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    description: Mapped[str] = mapped_column(Text, default="")  # full persona text
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
