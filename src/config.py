# src/config.py
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Groq ──────────────────────────────────────────────────
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"

    # ── Qdrant ────────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str | None = None          # ← ADD THIS
    qdrant_use_https: bool = False              # ← ADD THIS
    primary_collection: str = "bge_chunks"
    secondary_collection: str = "minilm_chunks"

    # ── Redis ─────────────────────────────────────────────────
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str | None = None          # ← ADD THIS
    embedding_cache_ttl: int = 86400     # 24 hours
    query_cache_ttl: int = 3600          # 1 hour

    # ── Embeddings ────────────────────────────────────────────
    primary_embedding_model: str = "BAAI/bge-base-en-v1.5"
    secondary_embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 32

    # ── Chunking ──────────────────────────────────────────────
    default_chunk_size: int = 512
    default_chunk_overlap: int = 102     # ~20% of 512

    # ── Retrieval ─────────────────────────────────────────────
    top_k_retrieval: int = 20
    top_k_rerank: int = 5
    rrf_k_constant: int = 60
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)

    # ── Evaluation ────────────────────────────────────────────
    eval_k_values: list[int] = [1, 3, 5, 10]

    # ── App ───────────────────────────────────────────────────
    log_level: str = "INFO"
    api_url: str = "http://localhost:8000"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Cached settings loader.
    Use get_settings() everywhere — never instantiate Settings() directly.
    This ensures .env is read once and the same object is shared across the app.
    """
    return Settings()


# Module-level singleton for convenience imports
# Usage: from src.config import settings
settings = get_settings()
