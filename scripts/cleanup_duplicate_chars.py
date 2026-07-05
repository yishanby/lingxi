"""Clean up duplicate character profile files with parenthetical suffixes."""
import os
import re
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

MEMORY_BASE = Path("data/memory")


def normalize_name(name: str) -> str:
    """Remove parenthetical suffixes like （升级档案） or (upgraded)."""
    name = re.sub(r'[（(][^）)]*[）)]', '', name)
    return name.strip()


def main():
    total_deleted = 0
    total_merged = 0

    for session_dir in sorted(MEMORY_BASE.iterdir()):
        if not session_dir.is_dir():
            continue
        chars_dir = session_dir / "characters"
        if not chars_dir.exists():
            continue

        # Group files by base name
        groups: dict[str, list[Path]] = {}
        for f in chars_dir.glob("*.md"):
            base = normalize_name(f.stem)
            if not base:
                continue
            groups.setdefault(base, []).append(f)

        for base_name, files in groups.items():
            if len(files) <= 1:
                # Single file — just check if it needs renaming
                f = files[0]
                if f.stem != base_name:
                    target = chars_dir / f"{base_name}.md"
                    if not target.exists():
                        print(f"  [RENAME] {f.name} -> {base_name}.md")
                        shutil.move(str(f), str(target))
                        total_merged += 1
                continue

            # Multiple files for same base name
            session_name = session_dir.name
            print(f"\nSession {session_name} - '{base_name}' has {len(files)} variants:")
            for f in files:
                stat = f.stat()
                print(f"    {f.name}  size={stat.st_size}  mtime={stat.st_mtime:.0f}")

            # Pick best: prefer the clean base name file if it exists, else newest+longest
            base_file = chars_dir / f"{base_name}.md"
            if base_file.exists() and base_file in files:
                best = base_file
            else:
                # Sort by (size desc, mtime desc) — prefer longest content, then most recent
                files_sorted = sorted(files, key=lambda f: (f.stat().st_size, f.stat().st_mtime), reverse=True)
                best = files_sorted[0]

            print(f"  [KEEP] {best.name}")

            # If best is not the base name, copy its content to base name
            if best.name != f"{base_name}.md":
                target = chars_dir / f"{base_name}.md"
                content = best.read_text(encoding="utf-8")
                target.write_text(content, encoding="utf-8")
                print(f"  [SAVE] -> {base_name}.md ({len(content)} chars)")
                total_merged += 1

            # Delete all non-base variants
            for f in files:
                if f.name != f"{base_name}.md":
                    f.unlink()
                    print(f"  [DELETE] {f.name}")
                    total_deleted += 1

    print(f"\nDone! Merged: {total_merged}, Deleted: {total_deleted}")


if __name__ == "__main__":
    main()
