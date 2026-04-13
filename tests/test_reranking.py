import pytest

from src.reranking.cross_encoder import CrossEncoderReranker
from src.reranking.diversity_reranker import DiversityReranker
from src.utils.correlation_id import set_correlation_id


@pytest.fixture(autouse=True)
def set_cid():
    set_correlation_id()


@pytest.fixture
def query() -> str:
    return "What are the chunking strategies for RAG pipelines?"


@pytest.fixture
def candidates() -> list[dict]:
    return [
        {
            "chunk_id": f"chunk-{i:04d}",
            "score": 0.9 - i * 0.1,
            "text": text,
            "metadata": {},
        }
        for i, text in enumerate(
            [
                "Fixed-size chunking splits documents into equal token chunks with overlap.",
                "Semantic chunking respects sentence boundaries using NLP models.",
                "Hierarchical chunking preserves document structure with parent-child links.",
                "Structure-aware chunking keeps tables and code blocks atomic.",
                "Retrieval-augmented generation improves LLM factual accuracy.",
                "Vector databases store embedding representations of text chunks.",
            ]
        )
    ]


class TestCrossEncoderReranker:
    def test_returns_top_k_results(self, query, candidates):
        reranker = CrossEncoderReranker()
        results = reranker.rerank(query, candidates, top_k=3)
        assert len(results) == 3

    def test_scores_are_sorted_descending(self, query, candidates):
        reranker = CrossEncoderReranker()
        results = reranker.rerank(query, candidates, top_k=5)
        scores = [r["ce_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_ce_score_field_present(self, query, candidates):
        reranker = CrossEncoderReranker()
        results = reranker.rerank(query, candidates, top_k=3)
        assert all("ce_score" in r for r in results)

    def test_empty_candidates_returns_empty(self, query):
        reranker = CrossEncoderReranker()
        assert reranker.rerank(query, [], top_k=5) == []

    def test_fewer_candidates_than_top_k(self, query):
        reranker = CrossEncoderReranker()
        small = [{"chunk_id": "x", "score": 0.5, "text": "hello world", "metadata": {}}]
        results = reranker.rerank(query, small, top_k=10)
        assert len(results) == 1

    def test_score_pair(self, query):
        reranker = CrossEncoderReranker()
        score = reranker.score_pair(query, "chunking strategies for documents")
        assert isinstance(score, float)


class TestDiversityReranker:
    def test_returns_top_k_results(self, query, candidates):
        reranker = DiversityReranker()
        results = reranker.rerank(query, candidates, top_k=3)
        assert len(results) == 3

    def test_mmr_score_field_present(self, query, candidates):
        reranker = DiversityReranker()
        results = reranker.rerank(query, candidates, top_k=3)
        assert all("mmr_score" in r for r in results)

    def test_empty_candidates_returns_empty(self, query):
        reranker = DiversityReranker()
        assert reranker.rerank(query, [], top_k=5) == []

    def test_lambda_1_pure_relevance(self, query, candidates):
        reranker = DiversityReranker()
        results = reranker.rerank(query, candidates, top_k=3, lambda_param=1.0)
        assert len(results) == 3

    def test_lambda_0_pure_diversity(self, query, candidates):
        reranker = DiversityReranker()
        results = reranker.rerank(query, candidates, top_k=3, lambda_param=0.0)
        assert len(results) == 3

    def test_no_duplicate_chunk_ids(self, query, candidates):
        reranker = DiversityReranker()
        results = reranker.rerank(query, candidates, top_k=4)
        ids = [r["chunk_id"] for r in results]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs in MMR output"
