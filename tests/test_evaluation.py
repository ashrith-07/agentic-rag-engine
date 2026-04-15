
from src.evaluation.retrieval_metrics import (
    aggregate_metrics,
    compute_metrics,
)


class TestComputeMetrics:
    def test_perfect_retrieval(self):
        m = compute_metrics(["A", "B", "C"], ["A", "B", "C"], k_values=[1, 3, 5])
        assert m.precision_at_k[3] == 1.0
        assert m.recall_at_k[3] == 1.0
        assert m.f1_at_k[3] == 1.0
        assert m.mrr == 1.0
        assert m.ndcg_at_k[3] == 1.0
        assert m.hit_rate == 1.0

    def test_zero_retrieval(self):
        m = compute_metrics(["X", "Y", "Z"], ["A", "B"], k_values=[1, 3, 5])
        assert m.mrr == 0.0
        assert m.hit_rate == 0.0
        assert m.precision_at_k[3] == 0.0
        assert m.recall_at_k[3] == 0.0
        assert m.ndcg_at_k[3] == 0.0

    def test_partial_retrieval(self):
        # Only first result is relevant
        m = compute_metrics(["A", "X", "Y"], ["A", "B", "C"], k_values=[1, 3])
        assert m.precision_at_k[1] == 1.0
        assert m.precision_at_k[3] == round(1 / 3, 4)
        assert m.recall_at_k[3] == round(1 / 3, 4)
        assert m.mrr == 1.0

    def test_mrr_second_rank(self):
        # Relevant doc is at rank 2
        m = compute_metrics(["X", "A", "Y"], ["A"], k_values=[3])
        assert m.mrr == round(1 / 2, 4)

    def test_mrr_third_rank(self):
        m = compute_metrics(["X", "Y", "A"], ["A"], k_values=[3])
        assert m.mrr == round(1 / 3, 4)

    def test_hit_rate_true(self):
        m = compute_metrics(["X", "Y", "A"], ["A"], k_values=[3])
        assert m.hit_rate == 1.0

    def test_hit_rate_false(self):
        m = compute_metrics(["X", "Y", "Z"], ["A"], k_values=[3])
        assert m.hit_rate == 0.0

    def test_ndcg_perfect_is_1(self):
        m = compute_metrics(["A", "B"], ["A", "B"], k_values=[2])
        assert m.ndcg_at_k[2] == 1.0

    def test_ndcg_worst_is_0(self):
        m = compute_metrics(["X", "Y"], ["A", "B"], k_values=[2])
        assert m.ndcg_at_k[2] == 0.0

    def test_f1_harmonic_mean(self):
        m = compute_metrics(["A", "X", "Y"], ["A", "B", "C"], k_values=[3])
        p = m.precision_at_k[3]
        r = m.recall_at_k[3]
        expected_f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        assert abs(m.f1_at_k[3] - round(expected_f1, 4)) < 1e-4

    def test_empty_relevant_set(self):
        m = compute_metrics(["A", "B"], [], k_values=[3])
        assert m.precision_at_k[3] == 0.0
        assert m.recall_at_k[3] == 0.0

    def test_empty_retrieved(self):
        m = compute_metrics([], ["A", "B"], k_values=[3])
        assert m.mrr == 0.0
        assert m.hit_rate == 0.0


class TestAggregateMetrics:
    def test_aggregation_averages_correctly(self):
        m1 = compute_metrics(["A", "B"], ["A", "B"], k_values=[2])
        m2 = compute_metrics(["X", "Y"], ["A", "B"], k_values=[2])
        agg = aggregate_metrics([m1, m2], k_values=[2])

        assert agg["num_queries"] == 2
        assert agg["mrr"] == round((1.0 + 0.0) / 2, 4)
        assert agg["hit_rate"] == round((1.0 + 0.0) / 2, 4)

    def test_aggregation_empty_returns_empty(self):
        assert aggregate_metrics([]) == {}

    def test_aggregation_contains_per_query(self):
        m = compute_metrics(["A"], ["A"], k_values=[1])
        agg = aggregate_metrics([m], k_values=[1])
        assert "per_query" in agg
        assert len(agg["per_query"]) == 1

    def test_all_k_values_present(self):
        m = compute_metrics(["A", "B", "C"], ["A"], k_values=[1, 3, 5])
        agg = aggregate_metrics([m], k_values=[1, 3, 5])
        for k in [1, 3, 5]:
            assert k in agg["precision_at_k"]
            assert k in agg["ndcg_at_k"]
