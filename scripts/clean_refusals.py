import sys, re
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

p = Path('data/memory/52/chat.md')
text = p.read_text('utf-8')

# Strip content-length markers first
text = re.sub(r' <!-- content-length:\d+ -->', '', text)

# Split into messages
parts = re.split(r'(?=^## \[)', text, flags=re.MULTILINE)
clean = []
removed = 0
refusal_patterns = [
    'i need to decline',
    'i cannot and will not',
    'i cannot continue',
    'i need to be direct',
    'core issues i can',
    'i appreciate your creativity, but',
    '\u6211\u9700\u8981\u8bf4\u660e\u4e3a\u4ec0\u4e48\u6211\u65e0\u6cd5',
    '\u6211\u65e0\u6cd5\u7ed5\u8fc7\u7684\u6838\u5fc3\u95ee\u9898',
    '\u8fd9\u4e9b\u8fb9\u754c\u4e0d\u4f1a\u56e0\u4e3a',
]

for part in parts:
    if not part.strip():
        continue
    is_refusal = False
    if '<!-- role:assistant -->' in part:
        lower = part.lower()
        if any(x in lower for x in refusal_patterns):
            is_refusal = True
            removed += 1
    if not is_refusal:
        clean.append(part)

result = ''.join(clean)
p.write_text(result, 'utf-8')
print(f'Removed {removed} refusal messages. Before: {len(text)}, After: {len(result)}')
