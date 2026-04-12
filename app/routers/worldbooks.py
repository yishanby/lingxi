"""WorldBook CRUD + import router."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.tables import WorldBook
from app.schemas.api import WorldBookCreate, WorldBookEntry, WorldBookOut, WorldBookUpdate

router = APIRouter(prefix="/api/worldbooks", tags=["worldbooks"])


def _row_to_out(wb: WorldBook) -> WorldBookOut:
    entries_raw = json.loads(wb.entries) if wb.entries else []
    entries = [WorldBookEntry(**e) for e in entries_raw]
    return WorldBookOut(
        id=wb.id,
        name=wb.name,
        entries=entries,
        created_at=wb.created_at,
        updated_at=wb.updated_at,
    )


@router.post("", response_model=WorldBookOut, status_code=201)
async def create_worldbook(body: WorldBookCreate, db: AsyncSession = Depends(get_db)):
    wb = WorldBook(
        name=body.name,
        entries=json.dumps([e.model_dump() for e in body.entries], ensure_ascii=False),
    )
    db.add(wb)
    await db.commit()
    await db.refresh(wb)
    return _row_to_out(wb)


@router.get("", response_model=list[WorldBookOut])
async def list_worldbooks(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(WorldBook).order_by(WorldBook.updated_at.desc()))
    return [_row_to_out(wb) for wb in result.scalars().all()]


@router.put("/{wb_id}", response_model=WorldBookOut)
async def update_worldbook(
    wb_id: int, body: WorldBookUpdate, db: AsyncSession = Depends(get_db)
):
    wb = await db.get(WorldBook, wb_id)
    if not wb:
        raise HTTPException(404, "WorldBook not found")
    if body.name is not None:
        wb.name = body.name
    if body.entries is not None:
        wb.entries = json.dumps(
            [e.model_dump() for e in body.entries], ensure_ascii=False
        )
    await db.commit()
    await db.refresh(wb)
    return _row_to_out(wb)


@router.post("/import", response_model=WorldBookOut, status_code=201)
async def import_worldbook(
    file: UploadFile = File(...), db: AsyncSession = Depends(get_db)
):
    """Import a SillyTavern worldbook JSON file.

    ST worldbook format: ``{"entries": {"0": {"key": [...], "content": "...", ...}, ...}}``
    """
    contents = await file.read()
    try:
        raw = json.loads(contents.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(400, f"Invalid JSON: {exc}")

    entries_raw = raw.get("entries", {})
    if isinstance(entries_raw, dict):
        entries_raw = entries_raw.values()

    entries: list[dict] = []
    name = raw.get("name", file.filename or "Imported WorldBook")
    for e in entries_raw:
        keys = e.get("key", e.get("keyword", []))
        if isinstance(keys, list):
            keys = ", ".join(keys)
        entries.append(
            {
                "keyword": keys,
                "content": e.get("content", ""),
                "position": "before_char" if e.get("position", 0) == 0 else "after_char",
                "enabled": not e.get("disable", False) if "disable" in e else e.get("enabled", True),
            }
        )

    wb = WorldBook(
        name=name,
        entries=json.dumps(entries, ensure_ascii=False),
    )
    db.add(wb)
    await db.commit()
    await db.refresh(wb)
    return _row_to_out(wb)
