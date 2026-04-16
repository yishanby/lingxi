"""Fix: archive old sessions, keep only the most recent one active per chat_id."""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "sillytavern_feishu.db"

conn = sqlite3.connect(str(DB_PATH))
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# Group active sessions by feishu_chat_id
cur.execute("SELECT id, feishu_chat_id FROM sessions WHERE status='active' ORDER BY id DESC")
rows = cur.fetchall()

chat_seen = set()
archived = []
for row in rows:
    chat_id = row["feishu_chat_id"]
    if chat_id in chat_seen:
        # Archive this older one
        cur.execute("UPDATE sessions SET status='archived' WHERE id=?", (row["id"],))
        archived.append(row["id"])
    else:
        chat_seen.add(chat_id)

conn.commit()
conn.close()
print(f"Archived {len(archived)} old sessions: {archived}")
print(f"Kept latest active for {len(chat_seen)} chat(s)")
