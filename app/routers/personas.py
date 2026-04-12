"""Persona (user character) management – SillyTavern style."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.tables import Character, Persona
from app.schemas.api import PersonaCreate, PersonaOut, PersonaUpdate

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/personas", tags=["personas"])


def _row_to_out(p: Persona) -> PersonaOut:
    return PersonaOut(
        id=p.id,
        name=p.name,
        avatar=p.avatar,
        description=p.description,
        description_position=p.description_position or "in_prompt",
        is_default=bool(p.is_default),
        linked_character_ids=json.loads(p.linked_character_ids) if p.linked_character_ids else [],
        created_at=p.created_at,
        updated_at=p.updated_at,
    )


@router.post("", response_model=PersonaOut, status_code=201)
async def create_persona(body: PersonaCreate, db: AsyncSession = Depends(get_db)):
    # If setting as default, clear other defaults
    if body.is_default:
        result = await db.execute(select(Persona).where(Persona.is_default == True))
        for p in result.scalars().all():
            p.is_default = False

    persona = Persona(
        name=body.name,
        avatar=body.avatar,
        description=body.description,
        description_position=body.description_position,
        is_default=body.is_default,
        linked_character_ids=json.dumps(body.linked_character_ids),
    )
    db.add(persona)
    await db.commit()
    await db.refresh(persona)
    return _row_to_out(persona)


@router.get("", response_model=list[PersonaOut])
async def list_personas(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Persona).order_by(Persona.created_at.desc()))
    return [_row_to_out(p) for p in result.scalars().all()]


@router.get("/for-character/{character_id}", response_model=PersonaOut | None)
async def get_persona_for_character(character_id: int, db: AsyncSession = Depends(get_db)):
    """Find the persona linked to a specific character, or the default persona."""
    result = await db.execute(select(Persona))
    all_personas = result.scalars().all()

    # First try to find one linked to this character
    for p in all_personas:
        linked = json.loads(p.linked_character_ids) if p.linked_character_ids else []
        if character_id in linked:
            return _row_to_out(p)

    # Fall back to default persona
    for p in all_personas:
        if p.is_default:
            return _row_to_out(p)

    return None


@router.get("/{persona_id}", response_model=PersonaOut)
async def get_persona(persona_id: int, db: AsyncSession = Depends(get_db)):
    persona = await db.get(Persona, persona_id)
    if not persona:
        raise HTTPException(404, "Persona not found")
    return _row_to_out(persona)


@router.patch("/{persona_id}", response_model=PersonaOut)
async def update_persona(
    persona_id: int, body: PersonaUpdate, db: AsyncSession = Depends(get_db)
):
    persona = await db.get(Persona, persona_id)
    if not persona:
        raise HTTPException(404, "Persona not found")

    update_data = body.model_dump(exclude_unset=True)

    # Handle default flag
    if update_data.get("is_default"):
        result = await db.execute(select(Persona).where(Persona.is_default == True))
        for p in result.scalars().all():
            if p.id != persona_id:
                p.is_default = False

    # Handle linked_character_ids
    if "linked_character_ids" in update_data:
        update_data["linked_character_ids"] = json.dumps(update_data["linked_character_ids"])

    for key, val in update_data.items():
        setattr(persona, key, val)

    await db.commit()
    await db.refresh(persona)
    return _row_to_out(persona)


@router.delete("/{persona_id}", status_code=204)
async def delete_persona(persona_id: int, db: AsyncSession = Depends(get_db)):
    persona = await db.get(Persona, persona_id)
    if not persona:
        raise HTTPException(404, "Persona not found")
    await db.delete(persona)
    await db.commit()


@router.post("/from-character/{character_id}", response_model=PersonaOut, status_code=201)
async def convert_character_to_persona(character_id: int, db: AsyncSession = Depends(get_db)):
    """Convert a character card into a persona (like SillyTavern's 'Convert to Persona')."""
    char = await db.get(Character, character_id)
    if not char:
        raise HTTPException(404, "Character not found")

    persona = Persona(
        name=char.name,
        avatar=char.avatar,
        description=char.description,
        description_position="in_prompt",
        is_default=False,
        linked_character_ids="[]",
    )
    db.add(persona)
    await db.commit()
    await db.refresh(persona)
    return _row_to_out(persona)
