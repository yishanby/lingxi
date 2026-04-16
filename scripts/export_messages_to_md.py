"""Export all session messages from SQLite to chat.md files.

For each session with messages in the DB, writes to data/memory/<session_id>/chat.md.
Skips sessions that already have a non-empty chat.md.

Usage: python scripts/export_messages_to_md.py
"""

import json
import os
import sys
import sqlite3
from pathlib import Path
from datetime import datetime

# Project root
ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "sillytavern_feishu.db"
MEMORY_BASE = ROOT / "data" / "memory"


def main():
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get all sessions
    cursor.execute("SELECT id, character_id, messages, user_name FROM sessions")
    sessions = cursor.fetchall()

    # Get character names
    cursor.execute("SELECT id, name FROM characters")
    char_map = {row["id"]: row["name"] for row in cursor.fetchall()}

    exported = 0
    skipped = 0

    for session in sessions:
        session_id = session["id"]
        messages_json = session["messages"]
        if not messages_json:
            continue

        try:
            messages = json.loads(messages_json)
        except json.JSONDecodeError:
            print(f"  Session {session_id}: invalid JSON, skipping")
            continue

        if not messages:
            continue

        # Check if chat.md already exists and has content
        session_dir = MEMORY_BASE / str(session_id)
        chat_md_path = session_dir / "chat.md"

        if chat_md_path.exists() and chat_md_path.stat().st_size > 0:
            print(f"  Session {session_id}: chat.md already exists ({chat_md_path.stat().st_size} bytes), skipping")
            skipped += 1
            continue

        # Create directory
        session_dir.mkdir(parents=True, exist_ok=True)

        char_name = char_map.get(session["character_id"], f"Character#{session['character_id']}")
        user_name = session["user_name"] or "用户"

        # Write chat.md
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "")

            # Parse timestamp
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    ts_str = dt.strftime("%Y-%m-%d %H:%M")
                except (ValueError, TypeError):
                    ts_str = timestamp[:16] if len(timestamp) >= 16 else timestamp
            else:
                ts_str = "unknown"

            if role == "user":
                header = f"## [{ts_str}] {user_name}"
            elif role == "assistant":
                header = f"## [{ts_str}] {char_name}"
            else:
                header = f"## [{ts_str}] System"

            lines.append(f"\n{header}\n{content}\n")

        with open(chat_md_path, "w", encoding="utf-8") as f:
            f.write("".join(lines))

        print(f"  Session {session_id}: exported {len(messages)} messages -> {chat_md_path}")
        exported += 1

    conn.close()
    print(f"\nDone! Exported: {exported}, Skipped (already exist): {skipped}, Total sessions: {len(sessions)}")


if __name__ == "__main__":
    main()
