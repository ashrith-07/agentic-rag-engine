# src/evaluation/retrieval_metrics.py
import math
from dataclasses import dataclass, field


@dataclass
class RetrievalMetrics:
    """
    Computed retrieval metrics for a single query.
    All values are floats in [0.0, 1.0] except where noted.
    """
    query_id: str
    precision_at_k: dict[int, float] = field(default_factory=dict)  # k → score
    recall_at_k: dict[int, float] = field(default_factory=dict)
    f1_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0          # Mean Reciprocal Rank
    ndcg_at_k: dict[int, float] = field(default_factory=dict)
    hit_rate: float = 0.0     # 1.0 if any relevant doc in top-K

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "f1_at_k": self.f1_at_k,
            "mrr": round(self.mrr, 4),
            "ndcg_at_k": self.ndcg_at_k,
            "hit_rate": round(self.hit_rate, 4),
        }


def compute_metrics(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k_values: list[int] | None = None,
) -> RetrievalMetrics:
    """
    Compute all retrieval metrics for a single query.

    Args:
        retrieved_ids: Ordered list of chunk IDs returned by retrieval
                       (index 0 = highest ranked)
        relevant_ids:  Ground truth set of relevant chunk IDs
        k_values:      List of K values to evaluate at (default [1,3,5,10])

    Returns:
        RetrievalMetrics dataclass
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    relevant_set = set(relevant_ids)
    metrics = RetrievalMetrics(query_id="")

    # ── Precision@K, Recall@K, F1@K ──────────────────────────────────────────
    for k in k_values:
        top_k = retrieved_ids[:k]
        hits = sum(1 for doc_id in top_k if doc_id in relevant_set)

        precision = hits / k if k > 0 else 0.0
        recall = hits / len(relevant_set) if relevant_set else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics.precision_at_k[k] = round(precision, 4)
        metrics.recall_at_k[k] = round(recall, 4)
        metrics.f1_at_k[k] = round(f1, 4)

    # ── MRR ──────────────────────────────────────────────────────────────────
    # Reciprocal rank of the first relevant result
    mrr = 0.0
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            mrr = 1.0 / rank
            break
    metrics.mrr = round(mrr, 4)

    # ── NDCG@K ───────────────────────────────────────────────────────────────
    for k in k_values:
        metrics.ndcg_at_k[k] = round(_ndcg_at_k(retrieved_ids, relevant_set, k), 4)

    # ── Hit Rate ─────────────────────────────────────────────────────────────
    max_k = max(k_values)
    top_max = set(retrieved_ids[:max_k])
    metrics.hit_rate = 1.0 if top_max & relevant_set else 0.0

    return metrics


def _ndcg_at_k(retrieved_ids: list[str], relevant_set: set[str], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at K.

    DCG@K  = Σ rel_i / log2(i + 1)   for i in 1..K
    NDCG@K = DCG@K / IDCG@K

    where rel_i = 1 if retrieved_ids[i-1] is relevant, else 0.
    IDCG is the DCG of a perfect ranking.
    """
    top_k = retrieved_ids[:k]

    # Actual DCG
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(top_k, start=1)
        if doc_id in relevant_set
    )

    # Ideal DCG: all relevant docs ranked first
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


def aggregate_metrics(
    all_metrics: list[RetrievalMetrics],
    k_values: list[int] | None = None,
) -> dict:
    """
    Aggregate per-query metrics into macro-averages over a test set.

    Args:
        all_metrics: List of RetrievalMetrics (one per test query)
        k_values: K values to aggregate (default [1,3,5,10])

    Returns:
        Dict with averaged metrics + per-query detail list
    """
    if not all_metrics:
        return {}

    if k_values is None:
        k_values = [1, 3, 5, 10]

    n = len(all_metrics)

    def avg(values: list[float]) -> float:
        return round(sum(values) / n, 4) if values else 0.0

    aggregated: dict = {
        "num_queries": n,
        "mrr": avg([m.mrr for m in all_metrics]),
        "hit_rate": avg([m.hit_rate for m in all_metrics]),
        "precision_at_k": {},
        "recall_at_k": {},
        "f1_at_k": {},
        "ndcg_at_k": {},
    }

    for k in k_values:
        aggregated["precision_at_k"][k] = avg(
            [m.precision_at_k.get(k, 0.0) for m in all_metrics]
        )
        aggregated["recall_at_k"][k] = avg(
            [m.recall_at_k.get(k, 0.0) for m in all_metrics]
        )
        aggregated["f1_at_k"][k] = avg(
            [m.f1_at_k.get(k, 0.0) for m in all_metrics]
        )
        aggregated["ndcg_at_k"][k] = avg(
            [m.ndcg_at_k.get(k, 0.0) for m in all_metrics]
        )

    aggregated["per_query"] = [m.to_dict() for m in all_metrics]
    return aggregated
