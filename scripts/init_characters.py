"""Initial character extraction from session 9 chat history."""
import asyncio
import json
import re
from pathlib import Path

async def main():
    from app.services.memory import (
        load_chat_md, load_memory, save_character_profile,
        CHARACTER_EXTRACTION_PROMPT, _characters_dir,
    )
    from app.services.llm import chat_completion
    from app.database import async_session
    from app.models.tables import Backend
    from sqlalchemy import select

    session_id = 9

    msgs = await load_chat_md(session_id)
    print(f"Loaded {len(msgs)} messages")

    async with async_session() as db:
        b = (await db.execute(select(Backend).where(Backend.id == 1))).scalar_one()
        params = json.loads(b.params) if isinstance(b.params, str) else (b.params or {})
        backend = {
            "provider": b.provider, "api_key": b.api_key,
            "model": b.model, "base_url": b.base_url, "params": params,
        }

    memory = await load_memory(session_id)

    # Use memory + last 60 messages for rich context
    conversation_text = "\n".join(
        f"{m.get('role', 'user')}: {m.get('content', '')}"
        for m in msgs[-60:]
    )

    user_prompt = (
        f"## Existing Character Profiles\n(none yet — this is the initial extraction)\n\n"
        f"## Memory Context (for reference)\n{memory[:8000]}\n\n"
        f"## Recent Conversation\n{conversation_text}\n\n"
        "Extract ALL characters that appear in the conversation. "
        "Output complete profiles for each one. This is the initial extraction, "
        "so include everyone mentioned."
    )

    result = await chat_completion(
        provider=backend["provider"],
        api_key=backend["api_key"],
        model=backend["model"],
        base_url=backend["base_url"],
        messages=[
            {"role": "system", "content": CHARACTER_EXTRACTION_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        params=params,
    )

    print(f"LLM output length: {len(result)}")

    # Save raw output for inspection
    raw_path = Path(f"data/memory/{session_id}/characters_raw.txt")
    raw_path.write_text(result, encoding="utf-8")

    # Parse and save
    blocks = re.split(r"\n---\n", result.strip())
    saved = 0
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        match = re.match(r"^#\s+(.+)", block)
        if match:
            char_name = match.group(1).strip()
            await save_character_profile(session_id, char_name, block)
            print(f"  Saved: {char_name}", flush=True)
            saved += 1

    print(f"\nTotal saved: {saved}")

    # List files
    d = Path(f"data/memory/{session_id}/characters")
    if d.exists():
        for f in d.glob("*.md"):
            print(f"  {f.name}")

if __name__ == "__main__":
    asyncio.run(main())
