"""Session management + chat endpoint."""

from __future__ import annotations

import datetime
import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.tables import Backend, Character, Session, WorldBook
from app.schemas.api import (
    MessageItem,
    SessionCreate,
    SessionMessageIn,
    SessionOut,
    WorldBookEntry,
)
from app.services.llm import LLMError, chat_completion
from app.services.prompt import assemble_prompt

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def _row_to_out(s: Session) -> SessionOut:
    return SessionOut(
        id=s.id,
        character_id=s.character_id,
        worldbook_ids=json.loads(s.worldbook_ids) if s.worldbook_ids else [],
        feishu_chat_id=s.feishu_chat_id,
        user_id=s.user_id,
        messages=[MessageItem(**m) for m in (json.loads(s.messages) if s.messages else [])],
        status=s.status,
        created_at=s.created_at,
    )


@router.post("", response_model=SessionOut, status_code=201)
async def create_session(body: SessionCreate, db: AsyncSession = Depends(get_db)):
    char = await db.get(Character, body.character_id)
    if not char:
        raise HTTPException(404, "Character not found")
    # Prepare initial messages with first_message if available
    initial_messages = []
    if char.first_message:
        first_msg = char.first_message.replace("{{user}}", "User").replace("{{char}}", char.name)
        now = datetime.datetime.utcnow().isoformat()
        initial_messages.append({"role": "assistant", "content": first_msg, "timestamp": now})

    # Merge user-specified worldbook_ids with character's linked worldbooks
    wb_ids = list(body.worldbook_ids)
    char_linked = json.loads(char.linked_worldbook_ids) if char.linked_worldbook_ids else []
    for wid in char_linked:
        if wid not in wb_ids:
            wb_ids.append(wid)

    session = Session(
        character_id=body.character_id,
        worldbook_ids=json.dumps(wb_ids),
        feishu_chat_id=body.feishu_chat_id,
        user_id=body.user_id,
        messages=json.dumps(initial_messages, ensure_ascii=False),
        status="active",
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return _row_to_out(session)


@router.get("", response_model=list[SessionOut])
async def list_sessions(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Session).order_by(Session.created_at.desc()))
    return [_row_to_out(s) for s in result.scalars().all()]


@router.get("/{session_id}", response_model=SessionOut)
async def get_session(session_id: int, db: AsyncSession = Depends(get_db)):
    session = await db.get(Session, session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return _row_to_out(session)


@router.delete("/{session_id}", status_code=204)
async def delete_session(session_id: int, db: AsyncSession = Depends(get_db)):
    session = await db.get(Session, session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    await db.delete(session)
    await db.commit()


@router.post("/{session_id}/message", response_model=SessionOut)
async def send_message(
    session_id: int,
    body: SessionMessageIn,
    db: AsyncSession = Depends(get_db),
):
    """Core chat endpoint: append user message, call LLM, append reply."""
    session = await db.get(Session, session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.status != "active":
        raise HTTPException(400, "Session is archived")

    # Load character
    char = await db.get(Character, session.character_id)
    if not char:
        raise HTTPException(500, "Character associated with session not found")

    # Load worldbook entries
    wb_ids = json.loads(session.worldbook_ids) if session.worldbook_ids else []
    all_entries: list[WorldBookEntry] = []
    for wid in wb_ids:
        wb = await db.get(WorldBook, wid)
        if wb:
            for e in json.loads(wb.entries) if wb.entries else []:
                all_entries.append(WorldBookEntry(**e))

    # Choose backend
    backend = await _resolve_backend(body.backend_id, db)

    # Current messages
    messages = [
        MessageItem(**m) for m in (json.loads(session.messages) if session.messages else [])
    ]

    # Build character dict
    char_dict = {
        "name": char.name,
        "description": char.description,
        "personality": char.personality,
        "scenario": char.scenario,
        "first_message": char.first_message,
        "example_dialogues": char.example_dialogues,
        "system_prompt": char.system_prompt,
    }

    # Assemble prompt
    llm_messages = assemble_prompt(
        character=char_dict,
        worldbook_entries=all_entries,
        chat_history=messages,
        user_message=body.content,
    )

    # Call LLM
    try:
        reply = await chat_completion(
            provider=backend["provider"],
            api_key=backend["api_key"],
            model=backend["model"],
            base_url=backend["base_url"],
            messages=llm_messages,
            params=backend["params"],
        )
    except LLMError as exc:
        raise HTTPException(502, f"LLM error: {exc}")

    # Persist messages
    now = datetime.datetime.utcnow().isoformat()
    messages.append(MessageItem(role="user", content=body.content, timestamp=now))
    messages.append(MessageItem(role="assistant", content=reply, timestamp=now))
    session.messages = json.dumps(
        [m.model_dump(mode="json") for m in messages], ensure_ascii=False
    )
    await db.commit()
    await db.refresh(session)
    return _row_to_out(session)


async def _resolve_backend(
    backend_id: int | None, db: AsyncSession
) -> dict:
    """Return a dict with provider/api_key/model/base_url/params."""
    if backend_id:
        b = await db.get(Backend, backend_id)
        if not b:
            raise HTTPException(404, "Backend not found")
        return {
            "provider": b.provider,
            "api_key": b.api_key,
            "model": b.model,
            "base_url": b.base_url,
            "params": json.loads(b.params) if b.params else {},
        }
    # Fallback: first available backend
    result = await db.execute(select(Backend).limit(1))
    b = result.scalar_one_or_none()
    if b:
        return {
            "provider": b.provider,
            "api_key": b.api_key,
            "model": b.model,
            "base_url": b.base_url,
            "params": json.loads(b.params) if b.params else {},
        }
    # Final fallback: env config
    from app.config import settings

    return {
        "provider": settings.default_llm_provider,
        "api_key": settings.default_llm_api_key,
        "model": settings.default_llm_model,
        "base_url": settings.default_llm_base_url,
        "params": {},
    }
