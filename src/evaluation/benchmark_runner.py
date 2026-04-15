# src/evaluation/benchmark_runner.py
"""
Full evaluation benchmark runner.

Runs the complete evaluation suite:
  1. Load test dataset from data/evaluation/test_dataset.json
  2. Run retrieval for each query
  3. Compute Precision@K, Recall@K, F1@K, MRR, NDCG@K, Hit Rate
  4. Optionally run RAGAS (requires LLM answers)
  5. Save results to data/evaluation/benchmark_report.json

Run via:
    make eval
    # or:
    python -m src.evaluation.benchmark_runner
"""
import json
import time
from pathlib import Path

from loguru import logger

from src.config import settings
from src.evaluation.retrieval_metrics import aggregate_metrics, compute_metrics
from src.evaluation.test_dataset_generator import load_dataset
from src.retrieval.hybrid_retriever import hybrid_retriever
from src.utils.correlation_id import set_correlation_id

_REPORT_PATH = Path("data/evaluation/benchmark_report.json")


def run_retrieval_benchmark(
    dataset: list[dict] | None = None,
    top_k: int = 10,
    report_path: Path = _REPORT_PATH,
) -> dict:
    """
    Run retrieval evaluation over the full test dataset.

    For each query:
      1. Run hybrid retrieval
      2. Compare retrieved chunk IDs to ground truth relevant IDs
      3. Compute all 6 retrieval metrics

    Args:
        dataset: Test dataset (loaded from disk if None)
        top_k: Retrieval depth for evaluation
        report_path: Where to save the JSON report

    Returns:
        Aggregated benchmark results dict
    """
    set_correlation_id()

    if dataset is None:
        dataset = load_dataset()

    logger.info(f"Running benchmark: {len(dataset)} queries, top_k={top_k}")
    start_time = time.perf_counter()

    all_metrics = []
    retrieval_latencies: list[float] = []

    for idx, entry in enumerate(dataset):
        query = entry["query"]
        relevant_ids = entry.get("relevant_chunk_ids", [])
        query_id = entry.get("query_id", str(idx))

        if not relevant_ids:
            logger.debug(f"Skipping query {idx} — no relevant_chunk_ids")
            continue

        # Retrieve
        t0 = time.perf_counter()
        try:
            results = hybrid_retriever.search(query=query, top_k=top_k, use_cache=False)
        except Exception as e:
            logger.warning(f"Query {idx} retrieval failed: {e}")
            continue
        latency_ms = (time.perf_counter() - t0) * 1000
        retrieval_latencies.append(latency_ms)

        # Extract retrieved IDs in ranked order
        retrieved_ids = [r["chunk_id"] for r in results]

        # Compute metrics
        metrics = compute_metrics(
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
            k_values=settings.eval_k_values,
        )
        metrics.query_id = query_id
        all_metrics.append(metrics)

        if (idx + 1) % 10 == 0:
            logger.info(f"Progress: {idx + 1}/{len(dataset)} queries evaluated")

    # Aggregate
    aggregated = aggregate_metrics(all_metrics, k_values=settings.eval_k_values)

    total_time = (time.perf_counter() - start_time) * 1000
    avg_latency = (
        sum(retrieval_latencies) / len(retrieval_latencies)
        if retrieval_latencies else 0.0
    )

    report = {
        "benchmark_config": {
            "top_k": top_k,
            "k_values": settings.eval_k_values,
            "total_queries": len(dataset),
            "evaluated_queries": len(all_metrics),
            "total_time_ms": round(total_time, 2),
            "avg_retrieval_latency_ms": round(avg_latency, 2),
        },
        "retrieval_metrics": aggregated,
    }

    # Save report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Benchmark report saved: {report_path}")
    _print_summary(report)

    return report


def _print_summary(report: dict) -> None:
    """Print a clean summary table to stdout."""
    cfg = report["benchmark_config"]
    metrics = report["retrieval_metrics"]

    print("\n" + "═" * 55)
    print("  RETRIEVAL BENCHMARK RESULTS")
    print("═" * 55)
    print(f"  Queries evaluated : {cfg['evaluated_queries']} / {cfg['total_queries']}")
    print(f"  Avg latency       : {cfg['avg_retrieval_latency_ms']:.1f}ms per query")
    print(f"  Total time        : {cfg['total_time_ms']:.0f}ms")
    print("─" * 55)

    k_vals = sorted(metrics.get("precision_at_k", {}).keys())
    print(f"  {'Metric':<20} " + "  ".join(f"@{k}" for k in k_vals))
    print("─" * 55)

    for metric_key, label in [
        ("precision_at_k", "Precision"),
        ("recall_at_k", "Recall"),
        ("f1_at_k", "F1"),
        ("ndcg_at_k", "NDCG"),
    ]:
        row = f"  {label:<20} "
        row += "  ".join(
            f"{metrics[metric_key].get(k, 0.0):.3f}  " for k in k_vals
        )
        print(row)

    print("─" * 55)
    print(f"  {'MRR':<20} {metrics.get('mrr', 0.0):.4f}")
    print(f"  {'Hit Rate':<20} {metrics.get('hit_rate', 0.0):.4f}")
    print("═" * 55 + "\n")


if __name__ == "__main__":
    run_retrieval_benchmark()
