"""Session management + chat endpoint."""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import re

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.tables import Backend, Character, Persona, Session, WorldBook
from app.schemas.api import (
    MessageItem,
    SessionCreate,
    SessionMessageIn,
    SessionOut,
    WorldBookEntry,
)
from app.services.llm import LLMError, chat_completion, chat_completion_stream
from app.services.memory import (
    append_chat_md,
    append_manual_memory,
    extract_memory,
    extract_memory_full,
    inject_memory_into_prompt,
    load_chat_md,
    load_memory,
    remove_last_chat_md_entries,
    save_memory,
    should_extract_memory,
)
from app.services.prompt import assemble_prompt

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/sessions", tags=["sessions"])

# OOC pattern: content wrapped in (( ))
_OOC_PATTERN = re.compile(r"^\(\((.+)\)\)$", re.DOTALL)

# OOC system instruction
_OOC_INSTRUCTION = (
    "Messages in (( )) are OOC (out of character) instructions. "
    "Adjust your behavior accordingly without breaking character."
)


def _chat_md_to_messages(raw: list[dict]) -> list[MessageItem]:
    """Convert load_chat_md dicts to MessageItem list."""
    return [
        MessageItem(role=m["role"], content=m["content"], timestamp=m.get("timestamp"))
        for m in raw
    ]


async def _load_session_messages(session_id: int) -> list[MessageItem]:
    """Load messages from chat.md for a session."""
    raw = await load_chat_md(session_id)
    return _chat_md_to_messages(raw)


async def _row_to_out(s: Session) -> SessionOut:
    messages = await _load_session_messages(s.id)
    return SessionOut(
        id=s.id,
        character_id=s.character_id,
        worldbook_ids=json.loads(s.worldbook_ids) if s.worldbook_ids else [],
        feishu_chat_id=s.feishu_chat_id,
        user_id=s.user_id,
        persona_id=s.persona_id,
        user_name=s.user_name or "用户",
        user_persona=s.user_persona or "",
        messages=messages,
        status=s.status,
        created_at=s.created_at,
    )


@router.post("", response_model=SessionOut, status_code=201)
async def create_session(body: SessionCreate, db: AsyncSession = Depends(get_db)):
    char = await db.get(Character, body.character_id)
    if not char:
        raise HTTPException(404, "Character not found")

    # Resolve persona: if persona_id is set, use it; otherwise auto-detect from character link, or use inline fields
    user_name = body.user_name or "用户"
    user_persona = body.user_persona or ""
    persona_id = body.persona_id

    if not persona_id:
        # Try to auto-find persona linked to this character
        result = await db.execute(select(Persona))
        all_personas = result.scalars().all()
        for p in all_personas:
            linked = json.loads(p.linked_character_ids) if p.linked_character_ids else []
            if body.character_id in linked:
                persona_id = p.id
                break
        # Fall back to default persona
        if not persona_id:
            for p in all_personas:
                if p.is_default:
                    persona_id = p.id
                    break

    if persona_id:
        persona = await db.get(Persona, persona_id)
        if persona:
            user_name = persona.name
            user_persona = persona.description

    # Prepare initial messages with first_message if available
    initial_messages = []
    if char.first_message:
        first_msg = char.first_message.replace("{{user}}", user_name).replace("{{char}}", char.name)
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
        persona_id=persona_id,
        user_name=user_name,
        user_persona=user_persona,
        status="active",
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)

    # Write first_message to chat.md if present
    if initial_messages:
        try:
            await append_chat_md(
                session.id, "assistant", initial_messages[0]["content"],
                char_name=char.name, user_name=user_name,
            )
        except Exception as exc:
            logger.error(f"Failed to write first message to chat.md: {exc}")

    return await _row_to_out(session)


@router.get("", response_model=list[SessionOut])
async def list_sessions(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Session).order_by(Session.created_at.desc()))
    return [await _row_to_out(s) for s in result.scalars().all()]


@router.get("/{session_id}", response_model=SessionOut)
async def get_session(session_id: int, db: AsyncSession = Depends(get_db)):
    session = await db.get(Session, session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return await _row_to_out(session)


@router.delete("/{session_id}", status_code=204)
async def delete_session(session_id: int, db: AsyncSession = Depends(get_db)):
    session = await db.get(Session, session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    await db.delete(session)
    await db.commit()


@router.patch("/{session_id}/status", response_model=SessionOut)
async def update_session_status(
    session_id: int,
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    """Update session status (active/archived)."""
    session = await db.get(Session, session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    status = body.get("status")
    if status not in ("active", "archived"):
        raise HTTPException(400, "Status must be 'active' or 'archived'")
    session.status = status
    await db.commit()
    await db.refresh(session)
    return await _row_to_out(session)


@router.post("/{session_id}/message", response_model=SessionOut)
async def send_message(
    session_id: int,
    body: SessionMessageIn,
    db: AsyncSession = Depends(get_db),
):
    """Core chat endpoint: append user message, call LLM, append reply.
    Supports /commands, OOC messages, and memory integration.
    """
    session = await db.get(Session, session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.status != "active":
        raise HTTPException(400, "Session is archived")

    # Load character
    char = await db.get(Character, session.character_id)
    if not char:
        raise HTTPException(500, "Character associated with session not found")

    # Current messages (from chat.md)
    messages = await _load_session_messages(session_id)

    user_name = session.user_name or "用户"
    content = body.content.strip()

    # ── /commands ────────────────────────────────────────────────────────
    if content.startswith("/"):
        return await _handle_command(
            content, session, session_id, char, messages, body.backend_id, db,
        )

    # ── OOC detection ───────────────────────────────────────────────────
    ooc_match = _OOC_PATTERN.match(content)
    msg_type = "ooc" if ooc_match else "ic"

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

    # Resolve persona position
    persona_position = "in_prompt"
    if session.persona_id:
        persona = await db.get(Persona, session.persona_id)
        if persona:
            persona_position = persona.description_position or "in_prompt"

    # Load memory context
    memory_context = await load_memory(session_id)

    # Add OOC instruction if needed
    ooc_extra = ""
    if msg_type == "ooc":
        ooc_extra = _OOC_INSTRUCTION

    # Assemble prompt
    full_memory = memory_context
    if ooc_extra:
        full_memory = (full_memory + "\n\n" + ooc_extra) if full_memory else ooc_extra

    llm_messages = assemble_prompt(
        character=char_dict,
        worldbook_entries=all_entries,
        chat_history=messages,
        user_message=content,
        user_name=user_name,
        user_persona=session.user_persona or "",
        persona_position=persona_position,
        memory_context=full_memory,
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

    # Persist to chat.md (append user + assistant)
    now = datetime.datetime.utcnow().isoformat()
    try:
        await append_chat_md(
            session_id, "user", content,
            char_name=char.name, user_name=user_name, msg_type=msg_type,
        )
        await append_chat_md(
            session_id, "assistant", reply,
            char_name=char.name, user_name=user_name,
        )
    except Exception as exc:
        logger.error(f"Failed to append chat.md: {exc}")

    # Reload messages from chat.md for accurate count
    messages = await _load_session_messages(session_id)

    # Check if memory extraction is needed (background task)
    msg_count = len(messages)
    if await should_extract_memory(session_id, msg_count):
        asyncio.create_task(
            extract_memory(
                session_id,
                [m.model_dump(mode="json") for m in messages],
                backend,
            )
        )

    return await _row_to_out(session)


async def _handle_command(
    content: str,
    session: Session,
    session_id: int,
    char: Character,
    messages: list[MessageItem],
    backend_id: int | None,
    db: AsyncSession,
) -> SessionOut:
    """Handle /commands. Returns SessionOut for all commands."""
    user_name = session.user_name or "用户"
    parts = content.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "/memory":
        # Subcommands: /memory, /memory edit <content>, /memory delete <keyword>, /memory clear
        sub_parts = arg.split(maxsplit=1)
        sub_cmd = sub_parts[0].lower() if sub_parts else ""
        sub_arg = sub_parts[1] if len(sub_parts) > 1 else ""

        if sub_cmd == "edit":
            if not sub_arg:
                response = "Usage: /memory edit <new memory content>\n\nThis replaces the ENTIRE memory. Use /memory to see current content first."
            else:
                await save_memory(session_id, sub_arg)
                response = "✅ Memory updated."
        elif sub_cmd == "delete":
            if not sub_arg:
                response = "Usage: /memory delete <keyword>\n\nDeletes all lines containing the keyword."
            else:
                memory = await load_memory(session_id)
                if not memory:
                    response = "(No memories to delete from)"
                else:
                    lines = memory.split("\n")
                    keyword_lower = sub_arg.lower()
                    removed = [l for l in lines if keyword_lower in l.lower()]
                    kept = [l for l in lines if keyword_lower not in l.lower()]
                    await save_memory(session_id, "\n".join(kept))
                    if removed:
                        response = f"🗑️ Deleted {len(removed)} line(s) containing '{sub_arg}':\n" + "\n".join(removed)
                    else:
                        response = f"No lines found containing '{sub_arg}'"
        elif sub_cmd == "extract":
            # Full extraction from conversation — synchronous, returns result
            backend = await _resolve_backend(backend_id, db)
            msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
            try:
                result = await extract_memory_full(
                    session_id, msg_dicts, backend,
                    custom_instruction=sub_arg or None,
                )
                response = f"📋 Extracted memory (preview, not saved yet):\n\n{result}\n\nUse `/memory edit <content>` to save."
            except Exception as exc:
                response = f"❌ Extraction failed: {exc}"
        elif sub_cmd == "clear":
            await save_memory(session_id, "")
            response = "🗑️ Memory cleared."
        else:
            # Plain /memory — show current content
            memory = await load_memory(session_id)
            response = memory if memory else "(No memories recorded yet)"

        now = datetime.datetime.utcnow().isoformat()
        try:
            await append_chat_md(session_id, "user", content, char_name=char.name, user_name=user_name)
            await append_chat_md(session_id, "assistant", response, char_name=char.name, user_name=user_name)
        except Exception as exc:
            logger.error(f"Failed to append chat.md for /memory: {exc}")
        return await _row_to_out(session)

    elif cmd == "/remember":
        if not arg:
            response = "Usage: /remember <text to remember>"
        else:
            await append_manual_memory(session_id, arg)
            response = f"✓ Remembered: {arg}"
        now = datetime.datetime.utcnow().isoformat()
        try:
            await append_chat_md(session_id, "user", content, char_name=char.name, user_name=user_name)
            await append_chat_md(session_id, "assistant", response, char_name=char.name, user_name=user_name)
        except Exception as exc:
            logger.error(f"Failed to append chat.md for /remember: {exc}")
        return await _row_to_out(session)

    elif cmd == "/undo":
        if len(messages) < 2:
            raise HTTPException(400, "No messages to undo")
        # Remove from chat.md
        try:
            await remove_last_chat_md_entries(session_id, count=2)
        except Exception as exc:
            logger.error(f"Failed to undo chat.md entries: {exc}")
        return await _row_to_out(session)

    elif cmd == "/retry":
        if not messages or messages[-1].role != "assistant":
            raise HTTPException(400, "No assistant message to retry")
        # Remove last assistant message
        messages.pop()
        # The last message should now be the user message — re-use its content
        if not messages or messages[-1].role != "user":
            raise HTTPException(400, "No user message found to retry")
        retry_content = messages[-1].content
        # Remove the user message too (will be re-added after LLM call)
        messages.pop()

        # Remove from chat.md
        try:
            await remove_last_chat_md_entries(session_id, count=2)
        except Exception as exc:
            logger.error(f"Failed to remove chat.md entries for retry: {exc}")

        # Load worldbook, backend, assemble prompt, call LLM
        wb_ids = json.loads(session.worldbook_ids) if session.worldbook_ids else []
        all_entries: list[WorldBookEntry] = []
        for wid in wb_ids:
            wb = await db.get(WorldBook, wid)
            if wb:
                for e in json.loads(wb.entries) if wb.entries else []:
                    all_entries.append(WorldBookEntry(**e))

        backend = await _resolve_backend(backend_id, db)
        char_dict = {
            "name": char.name,
            "description": char.description,
            "personality": char.personality,
            "scenario": char.scenario,
            "first_message": char.first_message,
            "example_dialogues": char.example_dialogues,
            "system_prompt": char.system_prompt,
        }

        persona_position = "in_prompt"
        if session.persona_id:
            persona = await db.get(Persona, session.persona_id)
            if persona:
                persona_position = persona.description_position or "in_prompt"

        memory_context = await load_memory(session_id)

        llm_messages = assemble_prompt(
            character=char_dict,
            worldbook_entries=all_entries,
            chat_history=messages,
            user_message=retry_content,
            user_name=user_name,
            user_persona=session.user_persona or "",
            persona_position=persona_position,
            memory_context=memory_context,
        )

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

        now = datetime.datetime.utcnow().isoformat()
        try:
            await append_chat_md(
                session_id, "user", retry_content,
                char_name=char.name, user_name=user_name,
            )
            await append_chat_md(
                session_id, "assistant", reply,
                char_name=char.name, user_name=user_name,
            )
        except Exception as exc:
            logger.error(f"Failed to append chat.md on retry: {exc}")

        return await _row_to_out(session)

    else:
        raise HTTPException(400, f"Unknown command: {cmd}")


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


@router.post("/{session_id}/message/stream")
async def send_message_stream(
    session_id: int,
    body: SessionMessageIn,
    db: AsyncSession = Depends(get_db),
):
    """Streaming chat endpoint: yields SSE chunks as LLM generates."""
    session = await db.get(Session, session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.status != "active":
        raise HTTPException(400, "Session is archived")

    char = await db.get(Character, session.character_id)
    if not char:
        raise HTTPException(500, "Character associated with session not found")

    content = body.content.strip()

    # OOC detection
    ooc_match = _OOC_PATTERN.match(content)
    msg_type = "ooc" if ooc_match else "ic"

    wb_ids = json.loads(session.worldbook_ids) if session.worldbook_ids else []
    all_entries: list[WorldBookEntry] = []
    for wid in wb_ids:
        wb = await db.get(WorldBook, wid)
        if wb:
            for e in json.loads(wb.entries) if wb.entries else []:
                all_entries.append(WorldBookEntry(**e))

    backend = await _resolve_backend(body.backend_id, db)

    messages = await _load_session_messages(session_id)

    char_dict = {
        "name": char.name,
        "description": char.description,
        "personality": char.personality,
        "scenario": char.scenario,
        "first_message": char.first_message,
        "example_dialogues": char.example_dialogues,
        "system_prompt": char.system_prompt,
    }

    persona_position = "in_prompt"
    if session.persona_id:
        persona = await db.get(Persona, session.persona_id)
        if persona:
            persona_position = persona.description_position or "in_prompt"

    # Load memory context
    memory_context = await load_memory(session_id)

    # Add OOC instruction if needed
    if msg_type == "ooc":
        memory_context = (memory_context + "\n\n" + _OOC_INSTRUCTION) if memory_context else _OOC_INSTRUCTION

    llm_messages = assemble_prompt(
        character=char_dict,
        worldbook_entries=all_entries,
        chat_history=messages,
        user_message=content,
        user_name=session.user_name or "用户",
        user_persona=session.user_persona or "",
        persona_position=persona_position,
        memory_context=memory_context,
    )

    # Capture values needed for DB save after streaming
    _session_id = session.id
    _user_content = content
    _current_messages = messages
    _char_name = char.name
    _user_name = session.user_name or "用户"
    _msg_type = msg_type
    _backend = backend

    async def generate():
        full_reply = ""
        try:
            async for chunk in chat_completion_stream(
                provider=backend["provider"],
                api_key=backend["api_key"],
                model=backend["model"],
                base_url=backend["base_url"],
                messages=llm_messages,
                params=backend["params"],
            ):
                full_reply += chunk
                yield f"data: {json.dumps({'delta': chunk}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.error(f"Streaming LLM error: {exc}")
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        # Write to chat.md
        try:
            await append_chat_md(
                _session_id, "user", _user_content,
                char_name=_char_name, user_name=_user_name, msg_type=_msg_type,
            )
            await append_chat_md(
                _session_id, "assistant", full_reply,
                char_name=_char_name, user_name=_user_name,
            )
        except Exception as exc:
            logger.error(f"Failed to append chat.md in stream: {exc}")

        # Reload messages for memory extraction count
        _current_messages = await _load_session_messages(_session_id)

        # Check memory extraction
        msg_count = len(_current_messages)
        if await should_extract_memory(_session_id, msg_count):
            asyncio.create_task(
                extract_memory(
                    _session_id,
                    [m.model_dump(mode="json") for m in _current_messages],
                    _backend,
                )
            )

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
