"""Persona (user character) management."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.tables import Persona
from app.schemas.api import PersonaCreate, PersonaOut, PersonaUpdate

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/personas", tags=["personas"])


def _row_to_out(p: Persona) -> PersonaOut:
    return PersonaOut(
        id=p.id,
        name=p.name,
        avatar=p.avatar,
        description=p.description,
        created_at=p.created_at,
        updated_at=p.updated_at,
    )


@router.post("", response_model=PersonaOut, status_code=201)
async def create_persona(body: PersonaCreate, db: AsyncSession = Depends(get_db)):
    persona = Persona(
        name=body.name,
        avatar=body.avatar,
        description=body.description,
    )
    db.add(persona)
    await db.commit()
    await db.refresh(persona)
    return _row_to_out(persona)


@router.get("", response_model=list[PersonaOut])
async def list_personas(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Persona).order_by(Persona.created_at.desc()))
    return [_row_to_out(p) for p in result.scalars().all()]


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
