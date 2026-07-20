from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    database_url: str = "sqlite+aiosqlite:///./data/sillytavern_feishu.db"

    # Feishu
    feishu_app_id: str = ""
    feishu_app_secret: str = ""
    feishu_verification_token: str = ""
    feishu_encrypt_key: str = ""

    # Default LLM (bootstrap convenience; backends table is the real source)
    default_llm_provider: str = "openai"
    default_llm_api_key: str = ""
    default_llm_model: str = "gpt-4o"
    default_llm_base_url: str = "https://api.openai.com/v1"

    # Prompt context limit (max chars of chat history to include in prompt)
    prompt_history_max_chars: int = 120000

    # Summary / context compression
    summary_recent_window: int = 10       # keep last N messages verbatim (legacy, unused by budget system)
    summary_max_chars: int = 3000         # max chars for the rolling summary
    summary_trigger_threshold: int = 20   # unsummarised msgs before triggering

    # Token budget system
    total_token_budget: int = 40000
    reply_token_reserve: int = 4096
    layer0_budget: int = 5000             # fixed layer (system prompt, character, worldbook)
    layer1_budget: int = 5000             # recall layer (memory, assets, character profiles)
    layer2_budget: int = 30000            # conversation layer (summary + recent messages)
    min_recent_messages: int = 4          # at least 2 rounds (4 messages)
    summary_max_tokens: int = 3000
    compressed_history_max_tokens: int = 5000

    # RAG settings
    rag_embedding_model: str = "BAAI/bge-small-zh-v1.5"  # Chinese embedding model
    rag_embedding_base_url: str = ""  # empty = use local model; set for API
    rag_embedding_api_key: str = ""   # only needed if using API
    rag_chunk_size: int = 5            # messages per chunk
    rag_auto_index: bool = True        # legacy compatibility; V2 owns indexing

    # Background tasks backend (summary, memory extraction, assets)
    # If set, background LLM tasks use this backend instead of the session's backend
    background_backend_id: int | None = None

    # Checkpointed Markdown memory pipeline
    memory_v2_enabled: bool = True
    story_state_interval_messages: int = 2
    memory_extract_interval_messages: int = 10
    episode_size_messages: int = 20
    rag_index_interval_messages: int = 10
    assets_interval_messages: int = 10
    memory_backup_count: int = 3
    stream_guard_chars: int = 512

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
