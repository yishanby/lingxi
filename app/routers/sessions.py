"""Session management + chat endpoint."""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import re

# Delay for background LLM tasks to let main conversation go first
_BG_DELAY_SECONDS = 30

async def _delayed(coro, delay: float = _BG_DELAY_SECONDS):
    """Wait *delay* seconds then run *coro*, giving the main LLM priority."""
    await asyncio.sleep(delay)
    return await coro

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
from app.services.token_tracker import record_usage
from app.services.memory import (
    append_chat_md,
    append_manual_memory,
    extract_memory,
    extract_memory_and_characters,
    extract_memory_full,
    extract_memory_rebuild,
    compact_memory,
    inject_memory_into_prompt,
    is_asset_relevant,
    load_assets,
    load_assets_summary,
    load_chat_md,
    load_mentioned_character_profiles,
    load_memory,
    load_memory_relevant,
    load_summary,
    remove_last_chat_md_entries,
    save_memory,
    should_extract_memory,
    should_update_summary,
    update_assets,
    update_character_profiles,
    update_rolling_summary,
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


async def _run_memory_rebuild(session_id, msg_dicts, backend, session, db, custom_instruction=None):
    """Background task for memory rebuild — notifies feishu chat when done."""
    try:
        result = await extract_memory_rebuild(session_id, msg_dicts, backend, custom_instruction=custom_instruction)
        chat_id = session.feishu_chat_id
        if chat_id:
            from app.services.feishu_ws_worker import send_text
            preview = result[:1500] + ("…" if len(result) > 1500 else "")
            send_text(chat_id, f"✅ 记忆重建完成！内容已保存到预览文件。\n\n{preview}\n\n使用 /memory apply 确认替换当前记忆，或忽略保留原记忆。")
    except Exception as exc:
        logger.exception(f"Memory rebuild failed for session {session_id}")
        chat_id = session.feishu_chat_id
        if chat_id:
            from app.services.feishu_ws_worker import send_text
            send_text(chat_id, f"❌ 记忆重建失败: {exc}")


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


@router.get("")
async def list_sessions(
    lite: bool = False,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Session).order_by(Session.created_at.desc()))
    if lite:
        sessions = []
        for s in result.scalars().all():
            raw = await load_chat_md(s.id)
            msg_count = len(raw)
            last_summary = ""
            if raw:
                last_content = raw[-1].get("content", "")
                last_summary = last_content.replace("\n", " ")[:50]
            sessions.append({
                "id": s.id,
                "character_id": s.character_id,
                "feishu_chat_id": s.feishu_chat_id,
                "status": s.status,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "user_name": s.user_name or "用户",
                "msg_count": msg_count,
                "last_summary": last_summary,
            })
        from fastapi.responses import JSONResponse
        return JSONResponse(content=sessions)
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

    # Choose backend: explicit request > session binding > default
    effective_backend_id = body.backend_id or session.backend_id
    backend = await _resolve_backend(effective_backend_id, db)

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

    # Load memory context (relevant chunks only, within token budget)
    recent_dicts = [{"role": m.role, "content": m.content} for m in messages[-10:]]
    memory_context = await load_memory_relevant(session_id, content, recent_dicts)

    # Load asset context (Layer 0: summary always, Layer 1: full when relevant)
    assets_summary = await load_assets_summary(session_id)
    assets_full = await load_assets(session_id) if is_asset_relevant(content) else ""

    # Load character profiles for mentioned characters
    character_profiles = await load_mentioned_character_profiles(session_id, content, recent_dicts)

    # Load rolling summary
    summary_context = await load_summary(session_id)

    # Update rolling summary if history exceeds context budget
    from app.config import settings
    msg_dicts_for_summary = [{"role": m.role, "content": m.content} for m in messages]
    bg_backend = await _resolve_bg_backend(backend, db)
    if should_update_summary(msg_dicts_for_summary, settings.prompt_history_max_chars):
        asyncio.create_task(
            _delayed(update_rolling_summary(
                session_id, msg_dicts_for_summary, bg_backend,
                settings.prompt_history_max_chars,
            ))
        )

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
        summary_context=summary_context,
        assets_summary=assets_summary,
        assets_full=assets_full,
        character_profiles=character_profiles,
    )

    # Call LLM
    try:
        result = await chat_completion(
            provider=backend["provider"],
            api_key=backend["api_key"],
            model=backend["model"],
            base_url=backend["base_url"],
            messages=llm_messages,
            params=backend["params"],
        )
        reply = result["content"]
        asyncio.create_task(record_usage(
            session_id=session_id, user_id=session.user_id,
            character_name=char.name, model=backend["model"], usage=result["usage"],
        ))
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
        # Unified memory + character extraction (single LLM call)
        asyncio.create_task(
            _delayed(extract_memory_and_characters(
                session_id,
                [m.model_dump(mode="json") for m in messages],
                bg_backend,
            ))
        )
        # Update assets separately
        asyncio.create_task(
            _delayed(update_assets(
                session_id,
                [m.model_dump(mode="json") for m in messages],
                bg_backend,
            ), delay=_BG_DELAY_SECONDS + 5)
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
        elif sub_cmd == "rebuild":
            if sub_arg.lower() == "preview":
                # Show current rebuild progress
                from pathlib import Path
                import aiofiles
                preview_path = Path(f"data/memory/{session_id}/memory_rebuild_preview.md")
                if preview_path.exists():
                    async with aiofiles.open(preview_path, mode="r", encoding="utf-8") as f:
                        preview_content = await f.read()
                    response = f"📋 Rebuild进度预览（{len(preview_content)}字）：\n\n{preview_content}"
                else:
                    response = "❌ 没有正在进行的rebuild，也没有待应用的预览。"
            else:
                # Chunked rebuild from entire chat.md
                backend = await _resolve_backend(backend_id, db)
                msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
                msg_count = len(msg_dicts)
                chunk_count = max(1, sum(len(f"{m['role']}: {m['content']}") for m in msg_dicts) // 60000 + 1)
                response = f"⏳ 开始重建记忆（{msg_count}条消息，预计{chunk_count}个分块），这可能需要几分钟..."
                if sub_arg:
                    response += f"\n📝 自定义指令：{sub_arg}"
                # Run in background so we can reply immediately
                asyncio.create_task(
                    _run_memory_rebuild(session_id, msg_dicts, backend, session, db, custom_instruction=sub_arg or None)
                )
        elif sub_cmd == "apply":
            from pathlib import Path
            import aiofiles
            # Check for rebuild or compact preview
            rebuild_path = Path(f"data/memory/{session_id}/memory_rebuild_preview.md")
            compact_path = Path(f"data/memory/{session_id}/memory_compact_preview.md")
            preview_path = None
            source = ""
            if rebuild_path.exists():
                preview_path = rebuild_path
                source = "重建"
            elif compact_path.exists():
                preview_path = compact_path
                source = "压缩"
            
            if preview_path:
                async with aiofiles.open(preview_path, mode="r", encoding="utf-8") as f:
                    new_memory = await f.read()
                await save_memory(session_id, new_memory)
                preview_path.unlink()
                response = f"✅ {source}记忆已应用！（{len(new_memory)}字）"
            else:
                response = "❌ 没有待应用的预览。请先运行 /memory rebuild 或 /memory compact"
        elif sub_cmd == "compact":
            if sub_arg.lower() == "preview":
                from pathlib import Path
                import aiofiles
                preview_path = Path(f"data/memory/{session_id}/memory_compact_preview.md")
                if preview_path.exists():
                    async with aiofiles.open(preview_path, mode="r", encoding="utf-8") as f:
                        preview_content = await f.read()
                    response = f"📋 Compact预览（{len(preview_content)}字）：\n\n{preview_content}"
                else:
                    response = "❌ 没有compact预览。请先运行 /memory compact"
            else:
                backend = await _resolve_backend(backend_id, db)
                try:
                    compacted, old_len, new_len, backup_name = await compact_memory(session_id, backend)
                    response = (
                        f"📦 记忆压缩预览：{old_len}字 → {new_len}字 ({new_len*100//old_len}%)\n"
                        f"备份: {backup_name}\n\n{compacted}\n\n"
                        f"使用 /memory apply 确认替换，或忽略保留原记忆。"
                    )
                except ValueError as exc:
                    response = f"⚠️ {exc}"
                except Exception as exc:
                    response = f"❌ 压缩失败: {exc}"
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

    elif cmd == "/assets":
        assets = await load_assets(session_id)
        summary = await load_assets_summary(session_id)
        if assets:
            response = f"📊 资产概览：{summary}\n\n{assets}"
        else:
            response = "还没有资产记录。资产会在对话中自动跟踪更新。"
        try:
            await append_chat_md(session_id, "user", content, char_name=char.name, user_name=user_name)
            await append_chat_md(session_id, "assistant", response, char_name=char.name, user_name=user_name)
        except Exception as exc:
            logger.error(f"Failed to append chat.md for /assets: {exc}")
        return await _row_to_out(session)

    elif cmd == "/summary":
        sub_parts = arg.split(maxsplit=1)
        sub_cmd = sub_parts[0].lower() if sub_parts else ""

        if sub_cmd == "update":
            backend = await _resolve_backend(backend_id, db)
            msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
            try:
                await update_rolling_summary(
                    session_id, msg_dicts, backend, settings.prompt_history_max_chars
                )
                summary = await load_summary(session_id)
                response = f"✅ 摘要已更新：\n\n{summary[:500]}" + ("…" if len(summary) > 500 else "")
            except Exception as exc:
                response = f"❌ 更新失败: {exc}"
        elif sub_cmd == "clear":
            await save_summary(session_id, "")
            response = "🗑️ 摘要已清空。"
        else:
            summary = await load_summary(session_id)
            response = summary if summary else "(还没有摘要)"

        try:
            await append_chat_md(session_id, "user", content, char_name=char.name, user_name=user_name)
            await append_chat_md(session_id, "assistant", response, char_name=char.name, user_name=user_name)
        except Exception as exc:
            logger.error(f"Failed to append chat.md for /summary: {exc}")
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

        recent_dicts_retry = [{"role": m.role, "content": m.content} for m in messages[-10:]]
        memory_context = await load_memory_relevant(session_id, retry_content, recent_dicts_retry)

        summary_context = await load_summary(session_id)

        llm_messages = assemble_prompt(
            character=char_dict,
            worldbook_entries=all_entries,
            chat_history=messages,
            user_message=retry_content,
            user_name=user_name,
            user_persona=session.user_persona or "",
            persona_position=persona_position,
            memory_context=memory_context,
            summary_context=summary_context,
        )

        try:
            result = await chat_completion(
                provider=backend["provider"],
                api_key=backend["api_key"],
                model=backend["model"],
                base_url=backend["base_url"],
                messages=llm_messages,
                params=backend["params"],
            )
            reply = result["content"]
            asyncio.create_task(record_usage(
                session_id=session_id, user_id=session.user_id,
                character_name=char.name, model=backend["model"], usage=result["usage"],
            ))
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
    # Fallback: default backend (is_default=1), then first available
    result = await db.execute(select(Backend).where(Backend.is_default == True).limit(1))
    b = result.scalar_one_or_none()
    if b:
        return {
            "provider": b.provider,
            "api_key": b.api_key,
            "model": b.model,
            "base_url": b.base_url,
            "params": json.loads(b.params) if b.params else {},
        }
    # No default marked, fall back to first available
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


async def _resolve_bg_backend(session_backend: dict, db: AsyncSession) -> dict:
    """Resolve backend for background tasks (summary, memory, assets).

    Uses settings.background_backend_id if configured, otherwise falls back
    to the session's own backend."""
    from app.config import settings
    if settings.background_backend_id:
        try:
            return await _resolve_backend(settings.background_backend_id, db)
        except Exception:
            pass  # fall back to session backend
    return session_backend


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

    effective_backend_id_stream = body.backend_id or session.backend_id
    backend = await _resolve_backend(effective_backend_id_stream, db)

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

    # Load memory context (relevant chunks only)
    _recent_dicts = [{"role": m.role, "content": m.content} for m in messages[-10:]]
    memory_context = await load_memory_relevant(session_id, content, _recent_dicts)

    # Load asset context (Layer 0: summary always, Layer 1: full when relevant)
    _assets_summary = await load_assets_summary(session_id)
    _assets_full = await load_assets(session_id) if is_asset_relevant(content) else ""

    # Load character profiles for mentioned characters
    _char_profiles = await load_mentioned_character_profiles(session_id, content, _recent_dicts)

    # Load rolling summary
    summary_context = await load_summary(session_id)

    # Update rolling summary if needed
    from app.config import settings as _settings
    _bg_backend = await _resolve_bg_backend(backend, db)
    _msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
    if should_update_summary(_msg_dicts, _settings.prompt_history_max_chars):
        asyncio.create_task(
            _delayed(update_rolling_summary(
                session_id, _msg_dicts, _bg_backend,
                _settings.prompt_history_max_chars,
            ))
        )

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
        summary_context=summary_context,
        assets_summary=_assets_summary,
        assets_full=_assets_full,
        character_profiles=_char_profiles,
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
        stream_usage = {}
        try:
            async for chunk in chat_completion_stream(
                provider=backend["provider"],
                api_key=backend["api_key"],
                model=backend["model"],
                base_url=backend["base_url"],
                messages=llm_messages,
                params=backend["params"],
            ):
                if isinstance(chunk, dict):
                    stream_usage = chunk.get("usage", {})
                    continue
                full_reply += chunk
                yield f"data: {json.dumps({'delta': chunk}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.error(f"Streaming LLM error: {exc}")
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        # Record token usage
        if stream_usage:
            asyncio.create_task(record_usage(
                session_id=_session_id, user_id=session.user_id,
                character_name=_char_name, model=_backend["model"], usage=stream_usage,
            ))

        # Schedule post-stream work as a separate task so it survives client disconnect
        async def _post_stream_work():
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
            current_msgs = await _load_session_messages(_session_id)

            # Check memory extraction
            msg_count = len(current_msgs)
            should_extract = await should_extract_memory(_session_id, msg_count)
            logger.info(f"Stream post-work: session {_session_id}, msg_count={msg_count}, extract={should_extract}")
            if should_extract:
                # Unified memory + character extraction (single LLM call)
                asyncio.create_task(
                    _delayed(extract_memory_and_characters(
                        _session_id,
                        [m.model_dump(mode="json") for m in current_msgs],
                        _bg_backend,
                    ))
                )
                # Update assets separately
                asyncio.create_task(
                    _delayed(update_assets(
                        _session_id,
                        [m.model_dump(mode="json") for m in current_msgs],
                        _bg_backend,
                    ), delay=_BG_DELAY_SECONDS + 5)
                )

        asyncio.create_task(_post_stream_work())

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
