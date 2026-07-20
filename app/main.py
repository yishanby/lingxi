"""SillyTavern-Feishu Bridge – FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import async_session, engine
from app.models.tables import Base, Session
from app.routers import backends, characters, feishu, personas, sessions, stats, worldbooks
from app.services import memory
from app.services.memory_pipeline import MemoryPipeline
from app.services.memory_tasks import MemoryTaskManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
memory_pipeline: MemoryPipeline | None = None
memory_task_manager: MemoryTaskManager | None = None


async def _column_missing(conn, table: str, column: str) -> bool:
    """Check if a column is missing from a SQLite table."""
    import sqlalchemy
    result = await conn.execute(sqlalchemy.text(f"PRAGMA table_info({table})"))
    columns = [row[1] for row in result.fetchall()]
    return column not in columns


async def _resolve_memory_backend(session_id: int) -> dict:
    async with async_session() as db:
        session = await db.get(Session, session_id)
        backend = await sessions._resolve_backend(
            session.backend_id if session is not None else None,
            db,
        )
        return await sessions._resolve_bg_backend(backend, db)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global memory_pipeline, memory_task_manager

    memory_pipeline = None
    memory_task_manager = None
    stop_ws_client = None
    ws_start_attempted = False
    try:
        # Create tables on startup
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

            # Migrate: add summary columns if missing (SQLite)
            if await _column_missing(conn, "sessions", "summary"):
                await conn.execute(
                    __import__("sqlalchemy").text(
                        "ALTER TABLE sessions ADD COLUMN summary TEXT DEFAULT ''"
                    )
                )
            if await _column_missing(conn, "sessions", "summary_up_to"):
                await conn.execute(
                    __import__("sqlalchemy").text(
                        "ALTER TABLE sessions ADD COLUMN summary_up_to INTEGER DEFAULT 0"
                    )
                )

        logger.info("Database tables ready")

        if settings.memory_v2_enabled:
            memory_pipeline = MemoryPipeline(memory.md_store)
            memory_task_manager = MemoryTaskManager(
                memory_pipeline,
                _resolve_memory_backend,
            )
            await memory_task_manager.start()
            await memory_task_manager.scan_pending_sessions()

        from app.services.feishu_ws import set_session_factory, start_ws_client
        from app.services.feishu_ws import stop_ws_client as stop_client

        stop_ws_client = stop_client
        set_session_factory(async_session)
        ws_start_attempted = True
        start_ws_client()
        yield
    finally:
        try:
            if memory_task_manager is not None:
                await memory_task_manager.stop()
        finally:
            try:
                if ws_start_attempted and stop_ws_client is not None:
                    stop_ws_client()
            finally:
                await engine.dispose()


app = FastAPI(
    title="SillyTavern-Feishu Bridge",
    description="Roleplay chat service bridging SillyTavern characters with Feishu bots",
    version="0.1.0",
    lifespan=lifespan,
)

# ── Routers ──────────────────────────────────────────────────────────────────
app.include_router(characters.router)
app.include_router(worldbooks.router)
app.include_router(personas.router)
app.include_router(sessions.router)
app.include_router(backends.router)
app.include_router(stats.router)
app.include_router(feishu.router)

# ── Static files / SPA ──────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True,
    )
