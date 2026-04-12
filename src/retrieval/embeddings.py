# src/retrieval/embeddings.py
import asyncio
from functools import lru_cache
from typing import Any

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.utils.correlation_id import get_correlation_id
from src.utils.timer import timed


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> SentenceTransformer:
    """
    Load and cache a SentenceTransformer model.
    Called at most once per model name per process lifetime.
    """
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    logger.info(f"Model loaded: {model_name} — dim={model.get_sentence_embedding_dimension()}")
    return model


class EmbeddingEngine:
    """
    Dual-model embedding engine.

    Wraps two SentenceTransformer models (primary + secondary).
    Provides batch embedding with progress logging.
    Models are loaded lazily and cached for the process lifetime.

    Usage:
        engine = EmbeddingEngine()
        vectors = engine.embed_texts(["text1", "text2"])          # primary model
        vectors = engine.embed_texts(["text1"], model="secondary") # secondary model
    """

    def __init__(self) -> None:
        self._primary_name = settings.primary_embedding_model
        self._secondary_name = settings.secondary_embedding_model
        self._batch_size = settings.embedding_batch_size

    @property
    def primary_dim(self) -> int:
        return _load_model(self._primary_name).get_sentence_embedding_dimension()

    @property
    def secondary_dim(self) -> int:
        return _load_model(self._secondary_name).get_sentence_embedding_dimension()

    @timed("embed_texts")
    def embed_texts(
        self,
        texts: list[str],
        model: str = "primary",
        show_progress: bool = False,
    ) -> list[list[float]]:
        """
        Embed a list of texts using the specified model.

        Args:
            texts: List of strings to embed
            model: "primary" (bge-base) or "secondary" (minilm)
            show_progress: Show tqdm progress bar for large batches

        Returns:
            List of embedding vectors as Python lists of floats
        """
        cid = get_correlation_id()
        if not texts:
            return []

        model_name = self._primary_name if model == "primary" else self._secondary_name
        st_model = _load_model(model_name)

        logger.debug(
            f"[{cid}] Embedding {len(texts)} texts "
            f"with {model} model ({model_name})"
        )

        # sentence-transformers handles batching internally
        embeddings: np.ndarray = st_model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,   # cosine similarity needs normalized vectors
            convert_to_numpy=True,
        )

        return embeddings.tolist()

    def embed_query(self, query: str, model: str = "primary") -> list[float]:
        """
        Embed a single query string.
        Identical to embed_texts but clearer API for query-time use.
        """
        results = self.embed_texts([query], model=model)
        return results[0]

    async def embed_texts_async(
        self,
        texts: list[str],
        model: str = "primary",
    ) -> list[list[float]]:
        """
        Async wrapper — runs CPU-bound embedding in a thread pool
        so it doesn't block the FastAPI event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.embed_texts(texts, model=model),
        )

    async def embed_query_async(self, query: str, model: str = "primary") -> list[float]:
        """Async single-query embedding."""
        results = await self.embed_texts_async([query], model=model)
        return results[0]

    def compare_models(self, texts: list[str]) -> dict[str, Any]:
        """
        Embed the same texts with both models and return comparison stats.
        Used in notebooks/02_embedding_comparison.ipynb.
        """
        import time

        results: dict[str, Any] = {}

        for model_label, model_name in [
            ("primary", self._primary_name),
            ("secondary", self._secondary_name),
        ]:
            start = time.perf_counter()
            vectors = self.embed_texts(texts, model=model_label)
            elapsed = (time.perf_counter() - start) * 1000

            results[model_label] = {
                "model_name": model_name,
                "num_texts": len(texts),
                "embedding_dim": len(vectors[0]) if vectors else 0,
                "total_ms": round(elapsed, 2),
                "ms_per_text": round(elapsed / max(len(texts), 1), 2),
            }

        return results


# Module-level singleton
embedding_engine = EmbeddingEngine()
