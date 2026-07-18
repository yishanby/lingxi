import sqlite3
conn = sqlite3.connect('data/sillytavern_feishu.db')
cur = conn.cursor()
cur.execute("PRAGMA table_info(characters)")
print("Cols:", cur.fetchall())
cur.execute("SELECT id, name, substr(system_prompt, 1, 100) FROM characters")
for r in cur.fetchall():
    print(r)
