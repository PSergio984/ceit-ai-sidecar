from pathlib import Path
from typing import Literal

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
    # Query rewriting (LLM) and re-ranking (blend|llm) in the /search flow.
    # Both degrade safely: no API key / provider failure -> original behavior.
    query_rewrite: bool = True
    rerank_mode: Literal["blend", "llm", "none"] = "blend"
    # Below this top-cosine, the semantic channel is treated as "no relevant
    # match" (off-corpus queries return nothing instead of nearest-neighbour
    # noise). Tunable; see the golden-set cosine separation in README.
    min_semantic_similarity: float = 0.25
    # Embed the corpus in bounded batches so a large corpus (1000+ docs)
    # never holds every text and vector in memory at once — small cloud
    # instances OOM'd rebuilding the full production corpus in one shot.
    embed_batch_size: int = 64
    # Durable thumbs-up/down feedback log (JSONL).
    feedback_path: Path = Path("var/feedback.jsonl")

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
