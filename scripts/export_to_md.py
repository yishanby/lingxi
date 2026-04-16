"""Export existing SQLite conversation data to markdown files.

Usage:
    python -m scripts.export_to_md
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Fix Windows console encoding for CJK characters
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
sys.stdout.reconfigure(encoding='utf-8')

import aiosqlite

DB_PATH = Path("data/sillytavern_feishu.db")
MEMORY_BASE = Path("data/memory")

MEMORY_TEMPLATE = """# Session Memory

## Characters
- (none yet)

## Relationships
- (none yet)

## Key Events
- (none yet)

## Important Decisions
- (none yet)

## User Preferences (from OOC)
- (none yet)
"""


async def export_session(
    session_id: int,
    character_name: str,
    user_name: str,
    messages_json: str | None,
) -> int:
    """Export one session's messages to chat.md. Returns message count."""
    messages = json.loads(messages_json) if messages_json else []
    if not messages:
        return 0

    session_dir = MEMORY_BASE / str(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)

    chat_path = session_dir / "chat.md"
    memory_path = session_dir / "memory.md"

    lines: list[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        ts = m.get("timestamp")

        if ts:
            try:
                dt = datetime.fromisoformat(ts)
                ts_str = dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, TypeError):
                ts_str = "unknown"
        else:
            ts_str = "unknown"

        if role == "user":
            header = f"## [{ts_str}] {user_name}"
        elif role == "assistant":
            header = f"## [{ts_str}] {character_name}"
        else:
            header = f"## [{ts_str}] System"

        lines.append(f"\n{header}\n{content}\n")

    chat_path.write_text("".join(lines), encoding="utf-8")

    if not memory_path.exists():
        memory_path.write_text(MEMORY_TEMPLATE, encoding="utf-8")

    return len(messages)


async def main() -> None:
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        sys.exit(1)

    MEMORY_BASE.mkdir(parents=True, exist_ok=True)

    total_sessions = 0
    total_messages = 0

    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row

        query = """
            SELECT s.id, s.messages, s.user_name,
                   c.name as character_name
            FROM sessions s
            LEFT JOIN characters c ON s.character_id = c.id
        """
        async with db.execute(query) as cursor:
            rows = await cursor.fetchall()

        for row in rows:
            session_id = row["id"]
            char_name = row["character_name"] or "Assistant"
            user_name = row["user_name"] or "User"
            messages_json = row["messages"]

            count = await export_session(session_id, char_name, user_name, messages_json)
            if count > 0:
                total_sessions += 1
                total_messages += count
                print(f"  Session {session_id} ({char_name}): {count} messages")

    print(f"\nExport complete: {total_sessions} sessions, {total_messages} messages total")
    print(f"Output directory: {MEMORY_BASE.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
