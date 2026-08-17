from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    sidecar_token: str
    corpus_path: Path
    model_name: str = "all-MiniLM-L6-v2"
    host: str = "127.0.0.1"
    port: int = 8310
    cache_dir: str = "cache"
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_api_key: str = ""
    llm_model: str = "meta-llama/llama-3.3-70b-instruct"
    llm_max_tokens: int = 512

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
