import sqlite3
import sys
sys.stdout.reconfigure(encoding='utf-8')

conn = sqlite3.connect('data/sillytavern_feishu.db')
cur = conn.cursor()

# Check how many characters have example_dialogues
cur.execute("SELECT COUNT(*) FROM characters WHERE example_dialogues IS NOT NULL AND example_dialogues != ''")
has_examples = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM characters")
total = cur.fetchone()[0]
print(f"Characters with example_dialogues: {has_examples}/{total}")

# Show those that have it
cur.execute("SELECT name, LENGTH(example_dialogues) FROM characters WHERE example_dialogues IS NOT NULL AND example_dialogues != ''")
for row in cur.fetchall():
    print(f"  {row[0]}: {row[1]} chars")

conn.close()
