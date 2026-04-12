# src/retrieval/cache.py
import hashlib
import pickle
from typing import Any

import numpy as np
import redis
from loguru import logger

from src.config import settings
from src.utils.correlation_id import get_correlation_id


def _make_redis_client() -> redis.Redis:
    return redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=0,
        decode_responses=False,   # we store binary (pickled numpy arrays)
        socket_connect_timeout=3,
        socket_timeout=3,
    )


class EmbeddingCache:
    """
    Redis-backed cache for embedding vectors.

    Key:   sha256(text + model_name)
    Value: pickled numpy array
    TTL:   settings.embedding_cache_ttl (default 24h)

    On Redis failure, logs a warning and returns None (graceful degradation).
    The pipeline works without cache — it just re-computes embeddings.
    """

    def __init__(self) -> None:
        self._client = _make_redis_client()
        self._ttl = settings.embedding_cache_ttl
        self._prefix = "emb:"

    def _key(self, text: str, model_name: str) -> str:
        payload = f"{text}::{model_name}".encode("utf-8")
        return self._prefix + hashlib.sha256(payload).hexdigest()

    def get(self, text: str, model_name: str) -> list[float] | None:
        """Return cached embedding or None on miss/error."""
        cid = get_correlation_id()
        key = self._key(text, model_name)
        try:
            raw = self._client.get(key)
            if raw is None:
                return None
            vector: np.ndarray = pickle.loads(raw)
            logger.debug(f"[{cid}] Embedding cache HIT: {key[:20]}...")
            return vector.tolist()
        except Exception as e:
            logger.warning(f"[{cid}] Embedding cache GET error: {e}")
            return None

    def set(self, text: str, model_name: str, vector: list[float]) -> None:
        """Cache an embedding vector."""
        cid = get_correlation_id()
        key = self._key(text, model_name)
        try:
            raw = pickle.dumps(np.array(vector, dtype=np.float32))
            self._client.setex(key, self._ttl, raw)
            logger.debug(f"[{cid}] Embedding cache SET: {key[:20]}...")
        except Exception as e:
            logger.warning(f"[{cid}] Embedding cache SET error: {e}")

    def get_many(
        self, texts: list[str], model_name: str
    ) -> tuple[list[list[float] | None], list[int]]:
        """
        Batch cache lookup.

        Returns:
            results: list aligned with texts — None for misses
            miss_indices: indices of texts that need embedding
        """
        results: list[list[float] | None] = []
        miss_indices: list[int] = []

        for idx, text in enumerate(texts):
            cached = self.get(text, model_name)
            results.append(cached)
            if cached is None:
                miss_indices.append(idx)

        return results, miss_indices

    def set_many(
        self, texts: list[str], model_name: str, vectors: list[list[float]]
    ) -> None:
        """Batch cache write."""
        for text, vector in zip(texts, vectors):
            self.set(text, model_name, vector)


class QueryCache:
    """
    Redis-backed cache for full query results.

    Key:   sha256(query + top_k + model_name)
    Value: pickled list of result dicts
    TTL:   settings.query_cache_ttl (default 1h)
    """

    def __init__(self) -> None:
        self._client = _make_redis_client()
        self._ttl = settings.query_cache_ttl
        self._prefix = "qry:"

    def _key(self, query: str, top_k: int, model_name: str) -> str:
        payload = f"{query}::{top_k}::{model_name}".encode("utf-8")
        return self._prefix + hashlib.sha256(payload).hexdigest()

    def get(self, query: str, top_k: int, model_name: str) -> list[dict] | None:
        """Return cached results or None."""
        cid = get_correlation_id()
        key = self._key(query, top_k, model_name)
        try:
            raw = self._client.get(key)
            if raw is None:
                return None
            logger.debug(f"[{cid}] Query cache HIT: {query[:40]}...")
            return pickle.loads(raw)
        except Exception as e:
            logger.warning(f"[{cid}] Query cache GET error: {e}")
            return None

    def set(
        self, query: str, top_k: int, model_name: str, results: list[dict]
    ) -> None:
        """Cache query results."""
        cid = get_correlation_id()
        key = self._key(query, top_k, model_name)
        try:
            self._client.setex(key, self._ttl, pickle.dumps(results))
            logger.debug(f"[{cid}] Query cache SET: {query[:40]}...")
        except Exception as e:
            logger.warning(f"[{cid}] Query cache SET error: {e}")

    def invalidate_all(self) -> int:
        """Flush all query cache keys. Returns count deleted."""
        try:
            keys = self._client.keys(f"{self._prefix}*")
            if keys:
                return self._client.delete(*keys)
            return 0
        except Exception as e:
            logger.warning(f"Query cache flush error: {e}")
            return 0


# Module-level singletons
embedding_cache = EmbeddingCache()
query_cache = QueryCache()
