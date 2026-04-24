"""Full character extraction from entire chat history, processing in chunks."""
import asyncio
import json
import re
from pathlib import Path

CHUNK_SIZE = 60  # messages per chunk

async def main():
    from app.services.memory import (
        load_chat_md, load_memory, save_character_profile,
        load_character_profile, list_character_names,
        CHARACTER_EXTRACTION_PROMPT, _characters_dir,
    )
    from app.services.llm import chat_completion
    from app.database import async_session
    from app.models.tables import Backend
    from sqlalchemy import select

    session_id = 9

    msgs = await load_chat_md(session_id)
    total = len(msgs)
    print(f"Loaded {total} messages, processing in chunks of {CHUNK_SIZE}")

    async with async_session() as db:
        b = (await db.execute(select(Backend).where(Backend.id == 1))).scalar_one()
        params = json.loads(b.params) if isinstance(b.params, str) else (b.params or {})
        backend = {
            "provider": b.provider, "api_key": b.api_key,
            "model": b.model, "base_url": b.base_url, "params": params,
        }

    START_CHUNK = 17  # resume from this chunk (0-indexed)

    chunk_count = (total + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"Total chunks: {chunk_count}, starting from {START_CHUNK + 1}")

    for i in range(START_CHUNK, chunk_count):
        start = i * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, total)
        chunk = msgs[start:end]

        # Load existing profiles for context
        names = await list_character_names(session_id)
        existing = ""
        for name in names:
            p = await load_character_profile(session_id, name)
            if p:
                existing += f"\n\n---\n{p}"

        conversation_text = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}"
            for m in chunk
        )

        # Truncate if too long
        if len(conversation_text) > 60000:
            conversation_text = conversation_text[:60000]
        if len(existing) > 10000:
            existing = existing[:10000]

        user_prompt = (
            f"## Existing Character Profiles\n{existing or '(none yet)'}\n\n"
            f"## Conversation (messages {start+1}-{end} of {total})\n{conversation_text}\n\n"
            "Extract ALL new characters and UPDATE existing ones with any new information. "
            "Output complete updated profiles for characters with changes. "
            "If no changes, output NO_CHANGE."
        )

        print(f"  Chunk {i+1}/{chunk_count} (msgs {start+1}-{end})...", end=" ", flush=True)

        try:
            result = (await chat_completion(
                provider=backend["provider"],
                api_key=backend["api_key"],
                model=backend["model"],
                base_url=backend["base_url"],
                messages=[
                    {"role": "system", "content": CHARACTER_EXTRACTION_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                params=params,
            ))["content"]

            if "NO_CHANGE" in result.strip():
                print("no changes")
                continue

            # Parse and save
            blocks = re.split(r"\n---\n", result.strip())
            saved = []
            for block in blocks:
                block = block.strip()
                if not block:
                    continue
                match = re.match(r"^#\s+(.+)", block)
                if match:
                    char_name = match.group(1).strip()
                    await save_character_profile(session_id, char_name, block)
                    saved.append(char_name)

            print(f"updated: {', '.join(saved) if saved else 'parse failed'}")

        except Exception as exc:
            print(f"error: {exc}")

    # Final summary
    names = await list_character_names(session_id)
    print(f"\nDone! Total characters: {len(names)}")
    for name in sorted(names):
        p = await load_character_profile(session_id, name)
        print(f"  {name}: {len(p)} chars")

if __name__ == "__main__":
    asyncio.run(main())
