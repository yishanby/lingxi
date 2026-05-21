"""Rebuild all character profiles for session 9 using RAG."""
import asyncio
import sys
import os
import re
from pathlib import Path

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')

from app.services.rag import rebuild_character_from_history
from app.services.memory import save_character_profile, list_character_names

CHARS_DIR = Path('data/memory/9/characters')


async def main():
    names = await list_character_names(9)

    # Get unique base names
    base_names = set()
    for n in names:
        base = re.split(r'[(\uff08]', n)[0].strip()
        if base:
            base_names.add(base)
    base_names = sorted(base_names)
    print(f'Rebuilding {len(base_names)} unique characters...', flush=True)

    done = 0
    failed = []
    summary = []

    for name in base_names:
        try:
            profile = await asyncio.wait_for(
                rebuild_character_from_history(9, name), timeout=120
            )
            await save_character_profile(9, name, profile)
            done += 1
            # Extract info line
            info_line = ''
            for l in profile.split('\n'):
                if l.startswith('- ') and ('身份' in l or '外貌' in l):
                    info_line = l[2:].strip()
                    break
            summary.append({'name': name, 'length': len(profile), 'info': info_line})
            print(f'  [{done}/{len(base_names)}] {name} ({len(profile)} chars)', flush=True)
        except asyncio.TimeoutError:
            failed.append(name)
            print(f'  [TIMEOUT] {name}', flush=True)
        except Exception as e:
            failed.append(name)
            print(f'  [ERROR] {name}: {e}', flush=True)

    # Delete old duplicate versions
    deleted = 0
    for f in CHARS_DIR.glob('*.md'):
        stem = f.stem
        if '(' in stem or '\uff08' in stem:
            f.unlink()
            deleted += 1
    print(f'\nDeleted {deleted} old duplicate files', flush=True)

    # Write summary index
    lines = ['# Session 9 - 角色档案索引\n']
    lines.append(f'共 {len(summary)} 个角色（RAG重建）\n')
    for i, s in enumerate(summary, 1):
        info = s['info'][:50] if s['info'] else ''
        lines.append(f'{i}. **{s["name"]}** — {info} ({s["length"]}字)')

    if failed:
        lines.append(f'\n## 失败 ({len(failed)})')
        for f in failed:
            lines.append(f'- {f}')

    summary_path = CHARS_DIR.parent / 'characters_index.md'
    summary_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nSummary → {summary_path}', flush=True)
    print(f'Done! {done} rebuilt, {len(failed)} failed, {deleted} old files deleted.', flush=True)


asyncio.run(main())
