# src/reranking/diversity_reranker.py
import numpy as np
from loguru import logger

from src.config import settings
from src.retrieval.embeddings import embedding_engine
from src.utils.correlation_id import get_correlation_id
from src.utils.timer import timed


class DiversityReranker:
    """
    Maximal Marginal Relevance (MMR) diversity re-ranker.

    MMR prevents the top-K results from being near-identical passages.
    Without diversity enforcement, a document with the same fact repeated
    three times will dominate the top results — the LLM gets redundant
    context instead of breadth.

    MMR score = λ × sim(chunk, query) − (1−λ) × max_sim(chunk, selected)

    λ=1.0 → pure relevance (same as original ranking)
    λ=0.0 → pure diversity (maximally different results)
    λ=0.7 → default: strong relevance bias with diversity enforcement

    Reference: Carbonell & Goldstein, 1998 — "The use of MMR,
    diversity-based reranking for reordering documents and producing
    summaries"

    Usage:
        reranker = DiversityReranker()
        diverse = reranker.rerank(query, candidates, top_k=5)
    """

    def __init__(self, lambda_param: float | None = None) -> None:
        self.lambda_param = lambda_param or settings.mmr_lambda

    @timed("mmr_rerank")
    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int | None = None,
        lambda_param: float | None = None,
    ) -> list[dict]:
        """
        Re-rank candidates using MMR for diversity.

        Args:
            query: User query string
            candidates: Ranked result dicts (must have "text" key)
            top_k: Number of results to return
            lambda_param: Override λ for this call (0.0–1.0)

        Returns:
            Diversity-balanced list of top_k result dicts.
            Each dict gains a "mmr_score" field.
        """
        cid = get_correlation_id()
        k = top_k or settings.top_k_rerank
        lam = lambda_param if lambda_param is not None else self.lambda_param

        if not candidates:
            return []

        if len(candidates) <= k:
            # Not enough candidates to apply MMR meaningfully
            return candidates[:k]

        # Embed query and all candidate texts
        all_texts = [query] + [c.get("text", "") for c in candidates]
        all_vectors = embedding_engine.embed_texts(all_texts, model="primary")

        query_vec = np.array(all_vectors[0])
        candidate_vecs = np.array(all_vectors[1:])   # shape: (n_candidates, dim)

        # Compute query–candidate similarities (already normalized)
        query_sims = candidate_vecs @ query_vec       # shape: (n_candidates,)

        selected_indices: list[int] = []
        selected_vecs: list[np.ndarray] = []

        remaining = list(range(len(candidates)))

        for _ in range(k):
            if not remaining:
                break

            if not selected_vecs:
                # First selection: pick highest query similarity
                best_idx = max(remaining, key=lambda i: query_sims[i])
            else:
                # MMR: balance relevance vs redundancy
                selected_matrix = np.array(selected_vecs)  # (n_selected, dim)

                mmr_scores = {}
                for i in remaining:
                    relevance = lam * float(query_sims[i])
                    # Max similarity to any already-selected chunk
                    redundancy = (1 - lam) * float(
                        np.max(candidate_vecs[i] @ selected_matrix.T)
                    )
                    mmr_scores[i] = relevance - redundancy

                best_idx = max(mmr_scores, key=lambda i: mmr_scores[i])

            selected_indices.append(best_idx)
            selected_vecs.append(candidate_vecs[best_idx])
            remaining.remove(best_idx)

        # Build output with MMR scores
        result = []
        for rank, idx in enumerate(selected_indices):
            entry = dict(candidates[idx])
            # MMR score = normalized position score for display
            entry["mmr_score"] = round(float(query_sims[idx]), 4)
            entry["mmr_rank"] = rank + 1
            result.append(entry)

        logger.debug(
            f"[{cid}] MMR rerank (λ={lam}): "
            f"{len(candidates)} → {len(result)} diverse results"
        )

        return result


# Module-level singleton
diversity_reranker = DiversityReranker()
