import sys
sys.stdout.reconfigure(encoding='utf-8')
with open('app/services/prompt.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
for i, l in enumerate(lines[118:125], start=119):
    print(f"{i}: {repr(l)}")
