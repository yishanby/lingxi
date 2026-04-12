"""SillyTavern character card import (JSON and PNG with embedded tEXt chunk)."""

from __future__ import annotations

import base64
import io
import json
import struct
import zlib
from typing import Any


def _read_png_text_chunks(data: bytes) -> dict[str, str]:
    """Extract tEXt chunks from raw PNG bytes.

    PNG spec: each chunk is [4-byte length][4-byte type][payload][4-byte CRC].
    tEXt payload is: keyword\\x00text
    """
    chunks: dict[str, str] = {}
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return chunks
    pos = 8
    while pos < len(data):
        if pos + 8 > len(data):
            break
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        chunk_type = data[pos + 4 : pos + 8]
        chunk_data = data[pos + 8 : pos + 8 + length]
        pos += 12 + length  # 4 len + 4 type + payload + 4 crc
        if chunk_type == b"tEXt":
            sep = chunk_data.index(b"\x00")
            keyword = chunk_data[:sep].decode("latin-1")
            text = chunk_data[sep + 1 :].decode("latin-1")
            chunks[keyword] = text
    return chunks


def parse_character_card_json(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalise a SillyTavern JSON character card into our internal field set.

    Handles both V1 (flat) and V2 (`spec: "chara_card_v2"`, nested under `data`).
    """
    # V2 cards nest everything under "data"
    card = raw.get("data", raw) if raw.get("spec") == "chara_card_v2" else raw

    tags_raw = card.get("tags", [])
    if isinstance(tags_raw, str):
        tags_raw = [t.strip() for t in tags_raw.split(",") if t.strip()]

    return {
        "name": card.get("name", card.get("char_name", "Unknown")),
        "avatar": card.get("avatar", None),
        "description": card.get("description", ""),
        "personality": card.get("personality", ""),
        "scenario": card.get("scenario", ""),
        "first_message": card.get("first_mes", card.get("first_message", "")),
        "example_dialogues": card.get("mes_example", card.get("example_dialogues", "")),
        "system_prompt": card.get("system_prompt", ""),
        "creator_notes": card.get("creator_notes", card.get("creatorcomment", "")),
        "tags": tags_raw,
        "source": "imported",
    }


def parse_character_card(file_bytes: bytes, filename: str) -> dict[str, Any]:
    """Parse a SillyTavern character card from either JSON or PNG file bytes."""
    lower = filename.lower()

    if lower.endswith(".json"):
        raw = json.loads(file_bytes.decode("utf-8"))
        return parse_character_card_json(raw)

    if lower.endswith(".png"):
        chunks = _read_png_text_chunks(file_bytes)
        chara_b64 = chunks.get("chara")
        if not chara_b64:
            raise ValueError("PNG file does not contain a 'chara' tEXt chunk")
        decoded = base64.b64decode(chara_b64)
        raw = json.loads(decoded.decode("utf-8"))
        card = parse_character_card_json(raw)
        # Store the entire PNG as a data-URI avatar
        png_b64 = base64.b64encode(file_bytes).decode("ascii")
        card["avatar"] = f"data:image/png;base64,{png_b64}"
        return card

    raise ValueError(f"Unsupported file type: {filename}. Use .json or .png")
