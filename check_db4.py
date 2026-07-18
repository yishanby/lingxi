import sqlite3, sys
sys.stdout.reconfigure(encoding='utf-8')
conn = sqlite3.connect('data/sillytavern_feishu.db')
cur = conn.cursor()
cur.execute("SELECT id, name FROM characters")
for r in cur.fetchall():
    print(r)
print("---")
print("Total:", cur.execute("SELECT count(*) FROM characters").fetchone()[0])
