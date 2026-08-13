from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    sidecar_token: str
    corpus_path: Path
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    host: str = "127.0.0.1"
    port: int = 8310
    cache_dir: str = "cache"

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
