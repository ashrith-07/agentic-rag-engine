# src/reranking/ab_comparator.py
"""
A/B comparator: measures the impact of re-ranking on retrieval quality.

Runs two pipelines side by side for every query:
  Pipeline A (baseline): hybrid retrieval only
  Pipeline B (reranked): hybrid retrieval → cross-encoder → MMR

Returns a structured comparison dict showing:
  - NDCG@5 before and after
  - Latency overhead added by re-ranking
  - Side-by-side result lists

Used in notebooks/04_reranking_analysis.ipynb and the Streamlit dashboard.
"""
import time

from loguru import logger

from src.config import settings
from src.evaluation.retrieval_metrics import compute_metrics
from src.reranking.cross_encoder import cross_encoder_reranker
from src.reranking.diversity_reranker import diversity_reranker
from src.retrieval.hybrid_retriever import hybrid_retriever
from src.utils.correlation_id import get_correlation_id
from src.utils.timer import timed


@timed("ab_compare")
def compare(
    query: str,
    relevant_ids: list[str] | None = None,
    top_k_retrieve: int | None = None,
    top_k_rerank: int | None = None,
    lambda_param: float | None = None,
) -> dict:
    """
    Run A/B comparison for a single query.

    Pipeline A: hybrid retrieval → top_k_rerank results
    Pipeline B: hybrid retrieval → cross-encoder → MMR → top_k_rerank results

    Args:
        query: User query string
        relevant_ids: Ground truth chunk IDs (for NDCG computation)
                      If None, NDCG is not computed
        top_k_retrieve: Candidates from hybrid retrieval (default: 20)
        top_k_rerank: Final results after re-ranking (default: 5)
        lambda_param: MMR λ override

    Returns:
        Comparison dict with results, latencies, and optional metrics
    """
    cid = get_correlation_id()
    k_retrieve = top_k_retrieve or settings.top_k_retrieval
    k_rerank = top_k_rerank or settings.top_k_rerank

    # ── Pipeline A: baseline (hybrid only) ───────────────────────────────────
    t0 = time.perf_counter()
    baseline_results = hybrid_retriever.search(
        query=query,
        top_k=k_retrieve,
        use_cache=False,
    )
    baseline_latency_ms = (time.perf_counter() - t0) * 1000

    # Trim to final k for fair comparison
    baseline_top_k = baseline_results[:k_rerank]

    # ── Pipeline B: reranked ──────────────────────────────────────────────────
    t1 = time.perf_counter()

    # Stage 1: cross-encoder on the full candidate set
    ce_results = cross_encoder_reranker.rerank(
        query=query,
        candidates=baseline_results,
        top_k=k_retrieve,   # keep all for MMR input
    )

    # Stage 2: MMR diversity on cross-encoder output
    reranked_results = diversity_reranker.rerank(
        query=query,
        candidates=ce_results,
        top_k=k_rerank,
        lambda_param=lambda_param,
    )

    rerank_latency_ms = (time.perf_counter() - t1) * 1000
    total_reranked_latency_ms = baseline_latency_ms + rerank_latency_ms

    # ── Metrics (only if ground truth provided) ───────────────────────────────
    baseline_metrics: dict = {}
    reranked_metrics: dict = {}

    if relevant_ids:
        baseline_ids = [r["chunk_id"] for r in baseline_top_k]
        reranked_ids = [r["chunk_id"] for r in reranked_results]

        b_metrics = compute_metrics(baseline_ids, relevant_ids, k_values=[k_rerank])
        r_metrics = compute_metrics(reranked_ids, relevant_ids, k_values=[k_rerank])

        baseline_metrics = {
            f"ndcg_at_{k_rerank}": b_metrics.ndcg_at_k.get(k_rerank, 0.0),
            f"precision_at_{k_rerank}": b_metrics.precision_at_k.get(k_rerank, 0.0),
            "mrr": b_metrics.mrr,
        }
        reranked_metrics = {
            f"ndcg_at_{k_rerank}": r_metrics.ndcg_at_k.get(k_rerank, 0.0),
            f"precision_at_{k_rerank}": r_metrics.precision_at_k.get(k_rerank, 0.0),
            "mrr": r_metrics.mrr,
        }

    # ── Build comparison dict ─────────────────────────────────────────────────
    comparison = {
        "query": query,
        "pipeline_a_baseline": {
            "results": _format_results(baseline_top_k),
            "latency_ms": round(baseline_latency_ms, 2),
            "metrics": baseline_metrics,
        },
        "pipeline_b_reranked": {
            "results": _format_results(reranked_results),
            "latency_ms": round(total_reranked_latency_ms, 2),
            "metrics": reranked_metrics,
        },
        "delta": {
            "latency_overhead_ms": round(rerank_latency_ms, 2),
            "latency_overhead_pct": round(
                rerank_latency_ms / max(baseline_latency_ms, 1) * 100, 1
            ),
        },
    }

    # Add NDCG delta if we have ground truth
    if relevant_ids and baseline_metrics and reranked_metrics:
        ndcg_key = f"ndcg_at_{k_rerank}"
        delta_ndcg = reranked_metrics[ndcg_key] - baseline_metrics[ndcg_key]
        comparison["delta"]["ndcg_improvement"] = round(delta_ndcg, 4)
        comparison["delta"]["mrr_improvement"] = round(
            reranked_metrics["mrr"] - baseline_metrics["mrr"], 4
        )

    logger.info(
        f"[{cid}] A/B compare: "
        f"baseline={baseline_latency_ms:.0f}ms, "
        f"rerank overhead={rerank_latency_ms:.0f}ms"
    )

    return comparison


def run_ab_benchmark(
    queries: list[str],
    relevant_ids_list: list[list[str]] | None = None,
) -> dict:
    """
    Run A/B comparison across multiple queries and aggregate results.

    Args:
        queries: List of query strings
        relevant_ids_list: Ground truth per query (optional)

    Returns:
        Aggregated comparison with per-query details and averages
    """
    if relevant_ids_list is None:
        relevant_ids_list = [None] * len(queries)  # type: ignore

    comparisons = []
    for query, relevant_ids in zip(queries, relevant_ids_list):
        result = compare(query=query, relevant_ids=relevant_ids)
        comparisons.append(result)

    # Aggregate latencies
    baseline_latencies = [c["pipeline_a_baseline"]["latency_ms"] for c in comparisons]
    reranked_latencies = [c["pipeline_b_reranked"]["latency_ms"] for c in comparisons]
    overhead_ms = [c["delta"]["latency_overhead_ms"] for c in comparisons]

    summary = {
        "num_queries": len(comparisons),
        "avg_baseline_latency_ms": round(
            sum(baseline_latencies) / len(baseline_latencies), 2
        ),
        "avg_reranked_latency_ms": round(
            sum(reranked_latencies) / len(reranked_latencies), 2
        ),
        "avg_overhead_ms": round(sum(overhead_ms) / len(overhead_ms), 2),
        "per_query": comparisons,
    }

    # Add NDCG aggregates if available
    ndcg_deltas = [
        c["delta"].get("ndcg_improvement")
        for c in comparisons
        if c["delta"].get("ndcg_improvement") is not None
    ]
    if ndcg_deltas:
        summary["avg_ndcg_improvement"] = round(
            sum(ndcg_deltas) / len(ndcg_deltas), 4
        )

    return summary


def _format_results(results: list[dict]) -> list[dict]:
    """Trim result dicts to essential fields for clean output."""
    return [
        {
            "chunk_id": r.get("chunk_id", ""),
            "text_preview": r.get("text", "")[:120] + "...",
            "score": r.get("rrf_score") or r.get("score", 0.0),
            "ce_score": r.get("ce_score"),
            "mmr_score": r.get("mmr_score"),
        }
        for r in results
    ]
