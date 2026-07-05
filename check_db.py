import sqlite3, sys
sys.stdout.reconfigure(encoding='utf-8')
conn = sqlite3.connect('data/sillytavern_feishu.db')
conn.execute("UPDATE sessions SET backend_id=6 WHERE id != 9")
conn.commit()
for r in conn.execute("""
    SELECT s.id, c.name, s.status, b.name FROM sessions s
    LEFT JOIN characters c ON s.character_id=c.id
    LEFT JOIN backends b ON s.backend_id=b.id
    WHERE s.status='active' OR s.status IS NULL
    ORDER BY s.id
""").fetchall():
    print(r)
