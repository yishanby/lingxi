import sqlite3
conn = sqlite3.connect('data/sillytavern_feishu.db')
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("Tables:", cur.fetchall())
