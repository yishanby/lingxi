"""Character CRUD + import router."""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.tables import Character, WorldBook
from app.schemas.api import CharacterCreate, CharacterOut, CharacterUpdate
from app.services.character_import import (
    parse_character_card,
    parse_raw_card,
    extract_character_book,
    extract_linked_world_name,
)

router = APIRouter(prefix="/api/characters", tags=["characters"])


def _row_to_out(c: Character) -> CharacterOut:
    return CharacterOut(
        id=c.id,
        name=c.name,
        avatar=c.avatar,
        description=c.description,
        personality=c.personality,
        scenario=c.scenario,
        first_message=c.first_message,
        example_dialogues=c.example_dialogues,
        system_prompt=c.system_prompt,
        creator_notes=c.creator_notes,
        tags=json.loads(c.tags) if c.tags else [],
        source=c.source,
        created_at=c.created_at,
        updated_at=c.updated_at,
    )


@router.post("", response_model=CharacterOut, status_code=201)
async def create_character(body: CharacterCreate, db: AsyncSession = Depends(get_db)):
    char = Character(
        name=body.name,
        avatar=body.avatar,
        description=body.description,
        personality=body.personality,
        scenario=body.scenario,
        first_message=body.first_message,
        example_dialogues=body.example_dialogues,
        system_prompt=body.system_prompt,
        creator_notes=body.creator_notes,
        tags=json.dumps(body.tags, ensure_ascii=False),
        source="manual",
    )
    db.add(char)
    await db.commit()
    await db.refresh(char)
    return _row_to_out(char)


@router.get("", response_model=list[CharacterOut])
async def list_characters(
    search: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Character).order_by(Character.updated_at.desc())
    if search:
        stmt = stmt.where(Character.name.ilike(f"%{search}%"))
    result = await db.execute(stmt)
    return [_row_to_out(c) for c in result.scalars().all()]


@router.get("/{char_id}", response_model=CharacterOut)
async def get_character(char_id: int, db: AsyncSession = Depends(get_db)):
    char = await db.get(Character, char_id)
    if not char:
        raise HTTPException(404, "Character not found")
    return _row_to_out(char)


@router.put("/{char_id}", response_model=CharacterOut)
async def update_character(
    char_id: int, body: CharacterUpdate, db: AsyncSession = Depends(get_db)
):
    char = await db.get(Character, char_id)
    if not char:
        raise HTTPException(404, "Character not found")
    update_data = body.model_dump(exclude_unset=True)
    if "tags" in update_data:
        update_data["tags"] = json.dumps(update_data["tags"], ensure_ascii=False)
    for key, val in update_data.items():
        setattr(char, key, val)
    await db.commit()
    await db.refresh(char)
    return _row_to_out(char)


@router.delete("/{char_id}", status_code=204)
async def delete_character(char_id: int, db: AsyncSession = Depends(get_db)):
    char = await db.get(Character, char_id)
    if not char:
        raise HTTPException(404, "Character not found")
    await db.delete(char)
    await db.commit()


@router.post("/import", response_model=CharacterOut, status_code=201)
async def import_character(
    file: UploadFile = File(...), db: AsyncSession = Depends(get_db)
):
    contents = await file.read()
    try:
        card_data = parse_character_card(contents, file.filename or "card.json")
        raw = parse_raw_card(contents, file.filename or "card.json")
    except (ValueError, Exception) as exc:
        raise HTTPException(400, f"Failed to parse character card: {exc}")

    tags = card_data.pop("tags", [])
    char = Character(**card_data, tags=json.dumps(tags, ensure_ascii=False))
    db.add(char)
    await db.flush()  # get char.id before commit

    linked_wb_ids: list[int] = []

    # 1. Handle embedded character_book -> create a worldbook from it
    embedded_entries = extract_character_book(raw)
    if embedded_entries:
        wb = WorldBook(
            name=f"{char.name} (Embedded)",
            entries=json.dumps(embedded_entries, ensure_ascii=False),
        )
        db.add(wb)
        await db.flush()
        linked_wb_ids.append(wb.id)

    # 2. Handle extensions.world -> find matching worldbook by name
    world_name = extract_linked_world_name(raw)
    if world_name:
        result = await db.execute(
            select(WorldBook).where(WorldBook.name.ilike(f"%{world_name}%")).limit(1)
        )
        matched_wb = result.scalar_one_or_none()
        if matched_wb and matched_wb.id not in linked_wb_ids:
            linked_wb_ids.append(matched_wb.id)

    char.linked_worldbook_ids = json.dumps(linked_wb_ids)
    await db.commit()
    await db.refresh(char)
    return _row_to_out(char)
