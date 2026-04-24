"""SillyTavern-Feishu Bridge – FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import engine
from app.models.tables import Base
from app.routers import backends, characters, feishu, personas, sessions, stats, worldbooks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


async def _column_missing(conn, table: str, column: str) -> bool:
    """Check if a column is missing from a SQLite table."""
    import sqlalchemy
    result = await conn.execute(sqlalchemy.text(f"PRAGMA table_info({table})"))
    columns = [row[1] for row in result.fetchall()]
    return column not in columns


@asynccontextmanager
async def lifespan(app: FastAPI):
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

    # Start Feishu WebSocket client
    from app.database import async_session
    from app.services.feishu_ws import set_session_factory, start_ws_client
    set_session_factory(async_session)
    start_ws_client()

    yield

    # Cleanup
    from app.services.feishu_ws import stop_ws_client
    stop_ws_client()


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
