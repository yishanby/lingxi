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

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
