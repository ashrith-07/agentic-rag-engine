# src/retrieval/hybrid_retriever.py
from loguru import logger

from src.config import settings
from src.retrieval.vector_store import vector_store
from src.retrieval.bm25_index import bm25_index
from src.retrieval.cache import query_cache
from src.utils.correlation_id import get_correlation_id
from src.utils.timer import timed


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int | None = None,
) -> list[dict]:
    """
    Reciprocal Rank Fusion (RRF) over multiple ranked result lists.

    RRF score for a chunk = Σ 1 / (k + rank_i)
    where rank_i is its 1-based position in list i.

    k=60 is the standard constant (from the original RRF paper).
    It dampens the effect of very high ranks without eliminating them.

    Args:
        result_lists: Each list is a ranked list of result dicts
                      (must have "chunk_id" key)
        k: RRF constant (default: settings.rrf_k_constant = 60)

    Returns:
        Merged list sorted by RRF score descending, deduplicated
    """
    k = k or settings.rrf_k_constant
    rrf_scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}

    for result_list in result_lists:
        for rank, result in enumerate(result_list, start=1):
            chunk_id = result["chunk_id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            # Keep the result dict for later (prefer higher-scored entry)
            if chunk_id not in chunk_data or result["score"] > chunk_data[chunk_id]["score"]:
                chunk_data[chunk_id] = result

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

    merged = []
    for cid in sorted_ids:
        entry = dict(chunk_data[cid])
        entry["rrf_score"] = round(rrf_scores[cid], 6)
        merged.append(entry)

    return merged


class HybridRetriever:
    """
    Hybrid retriever: dense (Qdrant) + sparse (BM25) fused via RRF.

    This is the primary retrieval interface for the RAG pipeline.
    Falls back to dense-only if BM25 index is not built.

    Usage:
        retriever = HybridRetriever()
        results = retriever.search("What is RRF fusion?", top_k=10)
    """

    def __init__(self) -> None:
        self._vector_store = vector_store
        self._bm25 = bm25_index

    @timed("hybrid_search")
    def search(
        self,
        query: str,
        top_k: int | None = None,
        model: str = "primary",
        filters: dict | None = None,
        use_cache: bool = True,
    ) -> list[dict]:
        """
        Hybrid search: RRF fusion of dense + BM25 results.

        Args:
            query: User query string
            top_k: Final number of results to return
            model: Embedding model ("primary" or "secondary")
            filters: Qdrant metadata filters
            use_cache: Check/write Redis query cache

        Returns:
            Merged ranked results sorted by RRF score
        """
        cid = get_correlation_id()
        k = top_k or settings.top_k_retrieval
        model_name = (
            settings.primary_embedding_model
            if model == "primary"
            else settings.secondary_embedding_model
        )

        # Cache check
        if use_cache:
            cached = query_cache.get(query, k, model_name)
            if cached is not None:
                logger.info(f"[{cid}] Hybrid search cache HIT: '{query[:50]}'")
                return cached[:k]

        # Dense retrieval — always runs
        dense_results = self._vector_store.search(
            query=query,
            top_k=k * 2,          # fetch more, RRF will trim
            model=model,
            filters=filters,
        )
        logger.debug(f"[{cid}] Dense: {len(dense_results)} results")

        # Sparse retrieval — only if index is built
        if self._bm25.is_built:
            sparse_results = self._bm25.search(query=query, top_k=k * 2)
            logger.debug(f"[{cid}] Sparse (BM25): {len(sparse_results)} results")
            result_lists = [dense_results, sparse_results]
        else:
            logger.warning(
                f"[{cid}] BM25 index not built — falling back to dense-only"
            )
            result_lists = [dense_results]

        # RRF fusion
        fused = reciprocal_rank_fusion(result_lists)[:k]

        logger.info(
            f"[{cid}] Hybrid search: {len(fused)} results after RRF fusion "
            f"(dense={len(dense_results)}, "
            f"sparse={len(sparse_results) if self._bm25.is_built else 'N/A'})"
        )

        # Cache result
        if use_cache:
            query_cache.set(query, k, model_name, fused)

        return fused

    def search_dense_only(
        self,
        query: str,
        top_k: int | None = None,
        model: str = "primary",
        filters: dict | None = None,
    ) -> list[dict]:
        """
        Dense-only search — used by SIMPLE query router path.
        Skips BM25 and RRF for lower latency.
        """
        k = top_k or settings.top_k_retrieval
        return self._vector_store.search(
            query=query,
            top_k=k,
            model=model,
            filters=filters,
        )


# Module-level singleton
hybrid_retriever = HybridRetriever()
