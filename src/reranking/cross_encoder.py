# src/reranking/cross_encoder.py
from functools import lru_cache

from loguru import logger
from sentence_transformers import CrossEncoder

from src.config import settings
from src.utils.correlation_id import get_correlation_id
from src.utils.timer import timed

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _load_cross_encoder() -> CrossEncoder:
    """Load cross-encoder model once, cache for process lifetime."""
    logger.info(f"Loading cross-encoder: {_MODEL_NAME}")
    model = CrossEncoder(_MODEL_NAME, max_length=512)
    logger.info("Cross-encoder loaded")
    return model


class CrossEncoderReranker:
    """
    Cross-encoder re-ranker using ms-marco-MiniLM-L-6-v2.

    A cross-encoder sees both query and document together in a single
    forward pass — far more accurate than bi-encoder cosine similarity,
    but O(n) with candidate count. This is why we run it only on the
    top-20 candidates from bi-encoder retrieval, not the full corpus.

    This two-stage pattern (bi-encoder → cross-encoder) is how every
    production search system works: Google, Bing, Cohere Rerank, etc.

    Usage:
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(query, candidates, top_k=5)
    """

    def __init__(self) -> None:
        self._model_name = _MODEL_NAME

    @timed("cross_encoder_rerank")
    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Re-rank candidates using cross-encoder relevance scoring.

        Args:
            query: User query string
            candidates: List of result dicts (must have "text" key)
                        Typically the top-20 from hybrid retrieval
            top_k: How many to return after re-ranking
                   (default: settings.top_k_rerank = 5)

        Returns:
            Re-ranked list of result dicts, sorted by cross-encoder
            score descending. Each dict gains a "ce_score" field.
        """
        cid = get_correlation_id()
        k = top_k or settings.top_k_rerank

        if not candidates:
            return []

        model = _load_cross_encoder()

        # Build (query, passage) pairs for the cross-encoder
        pairs = [(query, c.get("text", "")) for c in candidates]

        # Score all pairs in one batched forward pass
        scores = model.predict(pairs, show_progress_bar=False)

        # Attach scores and sort
        scored = []
        for candidate, score in zip(candidates, scores):
            entry = dict(candidate)
            entry["ce_score"] = round(float(score), 4)
            scored.append(entry)

        scored.sort(key=lambda x: x["ce_score"], reverse=True)
        reranked = scored[:k]

        logger.debug(
            f"[{cid}] Cross-encoder: {len(candidates)} → {len(reranked)} "
            f"(top score={reranked[0]['ce_score']:.4f}, "
            f"bottom={reranked[-1]['ce_score']:.4f})"
        )

        return reranked

    def score_pair(self, query: str, text: str) -> float:
        """Score a single (query, text) pair. Useful for debugging."""
        model = _load_cross_encoder()
        score = model.predict([(query, text)])
        return round(float(score[0]), 4)


# Module-level singleton
cross_encoder_reranker = CrossEncoderReranker()
