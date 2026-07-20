"""RAG (Retrieval-Augmented Generation) for chat history.

Lightweight vector search using embeddings stored as JSON files.
No external vector database needed — uses numpy-free cosine similarity.
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any

import aiofiles
import httpx

from app.config import settings
from app.services.md_store import ChatRecord, MarkdownMemoryStore

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────

MEMORY_BASE = Path("data/memory")
CHUNK_SIZE = 5  # messages per chunk
EMBEDDING_DIM = 1536  # will auto-detect on first call
TOP_K_DEFAULT = 10


def _index_dir(session_id: int) -> Path:
    d = MEMORY_BASE / str(session_id) / "rag"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Embedding ─────────────────────────────────────────────────────────────


# ── Local model singleton ─────────────────────────────────────────────────

_local_model = None


def _get_local_model():
    """Lazy-load sentence-transformers model."""
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        model_name = settings.rag_embedding_model or "all-MiniLM-L6-v2"
        logger.info(f"Loading local embedding model: {model_name}")
        _local_model = SentenceTransformer(model_name)
        dim = getattr(_local_model, 'get_embedding_dimension', _local_model.get_sentence_embedding_dimension)()
        logger.info(f"Embedding model loaded, dim={dim}")
    return _local_model


async def get_embeddings(
    texts: list[str],
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str = "",
) -> list[list[float]]:
    """Get embeddings using local sentence-transformers or OpenAI-compatible API.
    
    If base_url is provided and not empty, uses the API.
    Otherwise uses local sentence-transformers model.
    """
    # Try API if configured
    if base_url and base_url.strip():
        url = base_url.rstrip("/") + "/embeddings"
        key = api_key or settings.default_llm_api_key or "no-key"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {"model": model or "text-embedding-3-small", "input": texts}

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        return [item["embedding"] for item in data["data"]]

    # Local model
    import asyncio
    loop = asyncio.get_event_loop()
    model_instance = _get_local_model()
    embeddings = await loop.run_in_executor(
        None, lambda: model_instance.encode(texts, normalize_embeddings=True).tolist()
    )
    return embeddings


async def get_embedding(text: str, **kwargs) -> list[float]:
    """Get embedding for a single text."""
    results = await get_embeddings([text], **kwargs)
    return results[0]


# ── Vector math (no numpy needed) ─────────────────────────────────────────


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: list[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    na, nb = _norm(a), _norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return _dot(a, b) / (na * nb)


# ── Chunking ──────────────────────────────────────────────────────────────


def chunk_messages(messages: list[dict[str, str]], chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Split messages into chunks of N messages each, returning text for each chunk."""
    chunks = []
    for i in range(0, len(messages), chunk_size):
        group = messages[i:i + chunk_size]
        text = "\n".join(
            f"{'用户' if m.get('role') == 'user' else '角色'}: {m.get('content', '')}"
            for m in group
        )
        chunks.append(text)
    return chunks


def _chunk_records(
    records: list[ChatRecord],
    chunk_size: int = CHUNK_SIZE,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for index in range(0, len(records), chunk_size):
        group = records[index : index + chunk_size]
        text = "\n".join(
            f"{'用户' if record.role == 'user' else '角色'}: {record.content}"
            for record in group
        )
        chunks.append(
            {
                "text": text,
                "start_message": group[0].number,
                "end_message": group[-1].number,
            }
        )
    return chunks


def parse_chat_md(chat_md_text: str) -> list[dict[str, str]]:
    """Parse chat.md into list of {role, content} dicts."""
    messages = []
    current_role = None
    current_lines: list[str] = []

    for line in chat_md_text.split("\n"):
        # Match headers like: ## [2024-01-01 12:00] 角色名 <!-- role:assistant -->
        header_match = re.match(r"^## \[.*?\].*<!-- role:(user|assistant) -->", line)
        if header_match:
            if current_role and current_lines:
                messages.append({"role": current_role, "content": "\n".join(current_lines).strip()})
            current_role = header_match.group(1)
            current_lines = []
        else:
            current_lines.append(line)

    if current_role and current_lines:
        messages.append({"role": current_role, "content": "\n".join(current_lines).strip()})

    return messages


# ── Index management ──────────────────────────────────────────────────────


async def load_index(session_id: int) -> dict[str, Any]:
    """Load the RAG index for a session. Returns {chunks: [...], embeddings: [...]}."""
    path = _index_dir(session_id) / "index.json"
    if not path.exists():
        return {"chunks": [], "embeddings": [], "indexed_messages": 0}
    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        return json.loads(await f.read())


async def save_index(session_id: int, index: dict[str, Any]) -> None:
    """Save the RAG index."""
    await MarkdownMemoryStore(MEMORY_BASE).write_text(
        session_id,
        "rag/index.json",
        json.dumps(index, ensure_ascii=False),
    )


async def invalidate_after(session_id: int, message_number: int) -> None:
    """Remove range-aware chunks that extend past a rollback boundary."""
    if type(message_number) is not int or message_number < 0:
        raise ValueError("message_number must be a nonnegative integer")

    index = await load_index(session_id)
    retained_chunks: list[Any] = []
    retained_embeddings: list[Any] = []
    retained_range_ends: list[int] = []
    for chunk, embedding in zip(
        index.get("chunks", []),
        index.get("embeddings", []),
    ):
        end_message = chunk.get("end_message", 0) if isinstance(chunk, dict) else 0
        if (
            type(end_message) is int
            and end_message != 0
            and end_message > message_number
        ):
            continue
        retained_chunks.append(chunk)
        retained_embeddings.append(embedding)
        if type(end_message) is int and end_message > 0:
            retained_range_ends.append(end_message)

    index["chunks"] = retained_chunks
    index["embeddings"] = retained_embeddings
    index["indexed_messages"] = min(
        index.get("indexed_messages", 0),
        max(retained_range_ends, default=0),
    )
    await save_index(session_id, index)


async def build_index(
    session_id: int,
    *,
    force_rebuild: bool = False,
    embedding_base_url: str | None = None,
    embedding_api_key: str | None = None,
    embedding_model: str = "text-embedding-3-small",
) -> dict[str, Any]:
    """Build or incrementally update the RAG index from chat.md.
    
    Only embeds new chunks since last index build.
    """
    chat_path = MEMORY_BASE / str(session_id) / "chat.md"
    if not chat_path.exists():
        logger.warning(f"No chat.md for session {session_id}")
        return {"chunks": [], "embeddings": [], "indexed_messages": 0}

    records = await MarkdownMemoryStore(MEMORY_BASE).load_chat(session_id)
    total_messages = len(records)

    if force_rebuild:
        index = {"chunks": [], "embeddings": [], "indexed_messages": 0}
    else:
        index = await load_index(session_id)

    already_indexed = index.get("indexed_messages", 0)
    if already_indexed >= total_messages:
        logger.info(f"RAG index up to date for session {session_id} ({total_messages} msgs)")
        return index

    # Only process new messages
    new_records = records[already_indexed:]
    new_chunks = _chunk_records(new_records)

    if not new_chunks:
        return index

    # Get embeddings in batches
    batch_size = 20
    new_embeddings = []
    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i:i + batch_size]
        embs = await get_embeddings(
            [chunk["text"] for chunk in batch],
            base_url=embedding_base_url,
            api_key=embedding_api_key,
            model=embedding_model,
        )
        new_embeddings.extend(embs)
        logger.info(f"Embedded batch {i//batch_size + 1} ({len(batch)} chunks) for session {session_id}")

    # Append to index
    index["chunks"].extend(new_chunks)
    index["embeddings"].extend(new_embeddings)
    index["indexed_messages"] = total_messages

    await save_index(session_id, index)
    logger.info(f"RAG index updated: {already_indexed} → {total_messages} messages, {len(index['chunks'])} chunks")
    return index


# ── Search ────────────────────────────────────────────────────────────────


async def search(
    session_id: int,
    query: str,
    *,
    top_k: int = TOP_K_DEFAULT,
    embedding_base_url: str | None = None,
    embedding_api_key: str | None = None,
    embedding_model: str = "text-embedding-3-small",
) -> list[dict[str, Any]]:
    """Search the RAG index for chunks most relevant to query.
    
    Returns list of {text, score, index} sorted by relevance.
    """
    index = await load_index(session_id)
    if not index["chunks"]:
        return []

    # Get query embedding
    query_emb = await get_embedding(
        query,
        base_url=embedding_base_url,
        api_key=embedding_api_key,
        model=embedding_model,
    )

    # Compute similarities
    results = []
    for i, (chunk, emb) in enumerate(zip(index["chunks"], index["embeddings"])):
        score = cosine_similarity(query_emb, emb)
        chunk_text = chunk.get("text", "") if isinstance(chunk, dict) else chunk
        results.append({"text": chunk_text, "score": score, "index": i})

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


async def search_character(
    session_id: int,
    character_name: str,
    *,
    top_k: int = 15,
    **kwargs,
) -> list[dict[str, Any]]:
    """Search for chunks related to a specific character.
    
    Primary: keyword filter (chunks containing the name).
    Secondary: semantic search as supplement.
    """
    # 1. Load index and keyword-filter chunks containing the character name
    index = await load_index(session_id)
    if not index.get("chunks"):
        return []
    
    keyword_chunks = []
    for i, chunk in enumerate(index["chunks"]):
        chunk_text = chunk if isinstance(chunk, str) else chunk.get("text", "")
        if character_name in chunk_text:
            keyword_chunks.append({"text": chunk_text, "index": i, "score": 1.0, "method": "keyword"})
    
    # 2. Also do semantic search for supplementary results
    queries = [
        character_name,
        f"{character_name}的关系",
        f"{character_name}的事件",
    ]
    semantic_indices = set()
    semantic_results = []
    for q in queries:
        results = await search(session_id, q, top_k=top_k, **kwargs)
        for r in results:
            if r["index"] not in semantic_indices:
                semantic_indices.add(r["index"])
                # Only include if character name actually appears in text
                if character_name in r["text"]:
                    r["score"] = min(r["score"] * 1.2, 1.0)
                    semantic_results.append(r)
    
    # 3. Merge: keyword results first, then semantic supplements
    seen = set(c["index"] for c in keyword_chunks)
    for r in semantic_results:
        if r["index"] not in seen:
            keyword_chunks.append(r)
            seen.add(r["index"])
    
    # Sort by score descending, then by index descending (most recent first) for same score
    keyword_chunks.sort(key=lambda x: (x["score"], x["index"]), reverse=True)
    return keyword_chunks[:top_k]


# ── High-level: rebuild character profile from RAG ────────────────────────


async def rebuild_character_from_history(
    session_id: int,
    character_name: str,
    *,
    top_k: int = 20,
    embedding_base_url: str | None = None,
    embedding_api_key: str | None = None,
    embedding_model: str = "text-embedding-3-small",
    llm_provider: str | None = None,
    llm_api_key: str | None = None,
    llm_model: str | None = None,
    llm_base_url: str | None = None,
) -> str:
    """Search history for character info and rebuild their profile using LLM.
    
    Returns the new character profile markdown.
    """
    from app.services.llm import chat_completion
    from app.services.memory import load_character_profile

    # Ensure index is built
    await build_index(
        session_id,
        embedding_base_url=embedding_base_url,
        embedding_api_key=embedding_api_key,
        embedding_model=embedding_model,
    )

    # Search for relevant chunks
    results = await search_character(
        session_id,
        character_name,
        top_k=top_k,
        embedding_base_url=embedding_base_url,
        embedding_api_key=embedding_api_key,
        embedding_model=embedding_model,
    )

    if not results:
        return f"# {character_name}\n\n(未找到相关历史记录)"

    # Load existing profile for reference
    existing = await load_character_profile(session_id, character_name)

    # Build context from search results
    retrieved_text = "\n\n---\n\n".join(r["text"] for r in results)

    system_prompt = (
        "你是一个角色档案提取系统。根据提供的对话片段，提取并整理该角色的完整档案。\n"
        "输出格式：\n"
        f"# {character_name}\n\n"
        "## 基本信息\n（外貌、年龄、身份、职业等）\n\n"
        "## 性格特点\n（性格、说话方式、习惯等）\n\n"
        "## 人物关系\n（与其他角色的关系）\n\n"
        "## 当前状态\n（最新的状态、处境、情绪等）\n\n"
        "## 关键事件\n（重要的经历和转折点）\n\n"
        "要求：\n"
        "- 只写有证据支持的内容，不要猜测\n"
        "- 用中文\n"
        "- 如果现有档案中有信息但检索到的片段中没有矛盾，保留现有信息\n"
        "- 如果检索到的片段与现有档案矛盾，以检索到的内容为准（更新）"
    )

    user_prompt = (
        f"## 现有档案\n{existing or '(空)'}\n\n"
        f"## 检索到的相关对话片段\n{retrieved_text}\n\n"
        f"请输出{character_name}的完整更新档案。"
    )

    result = await chat_completion(
        provider=llm_provider or settings.default_llm_provider,
        api_key=llm_api_key or settings.default_llm_api_key,
        model=llm_model or settings.default_llm_model,
        base_url=llm_base_url or settings.default_llm_base_url,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return result.get("content", f"# {character_name}\n\n(提取失败)")
