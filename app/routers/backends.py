"""LLM Backend configuration router."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.tables import Backend
from app.schemas.api import BackendCreate, BackendOut, BackendUpdate

router = APIRouter(prefix="/api/backends", tags=["backends"])


def _row_to_out(b: Backend) -> BackendOut:
    return BackendOut(
        id=b.id,
        name=b.name,
        provider=b.provider,
        model=b.model,
        base_url=b.base_url,
        params=json.loads(b.params) if b.params else {},
        created_at=b.created_at,
        updated_at=b.updated_at,
    )


@router.post("", response_model=BackendOut, status_code=201)
async def create_backend(body: BackendCreate, db: AsyncSession = Depends(get_db)):
    b = Backend(
        name=body.name,
        provider=body.provider,
        api_key=body.api_key,
        model=body.model,
        base_url=body.base_url,
        params=json.dumps(body.params, ensure_ascii=False),
    )
    db.add(b)
    await db.commit()
    await db.refresh(b)
    return _row_to_out(b)


@router.get("", response_model=list[BackendOut])
async def list_backends(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Backend).order_by(Backend.name))
    return [_row_to_out(b) for b in result.scalars().all()]


@router.get("/{backend_id}", response_model=BackendOut)
async def get_backend(backend_id: int, db: AsyncSession = Depends(get_db)):
    b = await db.get(Backend, backend_id)
    if not b:
        raise HTTPException(404, "Backend not found")
    return _row_to_out(b)


@router.put("/{backend_id}", response_model=BackendOut)
async def update_backend(
    backend_id: int, body: BackendUpdate, db: AsyncSession = Depends(get_db)
):
    b = await db.get(Backend, backend_id)
    if not b:
        raise HTTPException(404, "Backend not found")
    update_data = body.model_dump(exclude_unset=True)
    if "params" in update_data:
        update_data["params"] = json.dumps(update_data["params"], ensure_ascii=False)
    for key, val in update_data.items():
        setattr(b, key, val)
    await db.commit()
    await db.refresh(b)
    return _row_to_out(b)


@router.delete("/{backend_id}", status_code=204)
async def delete_backend(backend_id: int, db: AsyncSession = Depends(get_db)):
    b = await db.get(Backend, backend_id)
    if not b:
        raise HTTPException(404, "Backend not found")
    await db.delete(b)
    await db.commit()
