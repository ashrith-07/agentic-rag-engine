import pytest

from src.retrieval.bm25_index import BM25Index, _tokenize
from src.retrieval.hybrid_retriever import reciprocal_rank_fusion
from src.utils.correlation_id import set_correlation_id


@pytest.fixture(autouse=True)
def set_cid():
    set_correlation_id()


class TestTokenize:
    def test_lowercases(self):
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_removes_punctuation(self):
        tokens = _tokenize("hello, world! foo.")
        assert "," not in tokens
        assert "!" not in tokens

    def test_removes_short_tokens(self):
        tokens = _tokenize("a b cd efg")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "cd" in tokens

    def test_empty_string(self):
        assert _tokenize("") == []


class TestBM25Index:
    def test_build_and_search(self, sample_chunks):
        index = BM25Index()
        index.build(sample_chunks)
        assert index.is_built
        assert index.doc_count == len(sample_chunks)

        results = index.search("chunking strategy", top_k=3)
        assert isinstance(results, list)

    def test_search_returns_sorted_by_score(self, sample_chunks):
        index = BM25Index()
        index.build(sample_chunks)
        results = index.search("RAG retrieval generation", top_k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_excludes_zero_scores(self, sample_chunks):
        index = BM25Index()
        index.build(sample_chunks)
        results = index.search("xyzzy foobar nonsense", top_k=5)
        assert all(r["score"] > 0 for r in results)

    def test_empty_query_returns_empty(self, sample_chunks):
        index = BM25Index()
        index.build(sample_chunks)
        results = index.search("", top_k=5)
        assert results == []

    def test_not_built_raises(self):
        index = BM25Index()
        with pytest.raises(RuntimeError, match="not built"):
            index.search("test", top_k=5)

    def test_save_and_load(self, sample_chunks, tmp_path):
        index_path = tmp_path / "bm25.pkl"
        meta_path = tmp_path / "bm25_meta.json"

        index = BM25Index()
        index.build(sample_chunks)
        index.save(index_path, meta_path)

        loaded = BM25Index.load(index_path, meta_path)
        assert loaded.is_built
        assert loaded.doc_count == index.doc_count

        original_results = index.search("chunking", top_k=3)
        loaded_results = loaded.search("chunking", top_k=3)
        assert len(original_results) == len(loaded_results)

    def test_add_chunks_increases_doc_count(self, sample_chunks):
        index = BM25Index()
        index.build(sample_chunks[:1])
        initial_count = index.doc_count
        index.add_chunks(sample_chunks[1:])
        assert index.doc_count == len(sample_chunks)
        assert index.doc_count > initial_count


class TestRRF:
    def test_chunk_in_both_lists_ranks_higher(self):
        list_a = [
            {"chunk_id": "A", "score": 0.9},
            {"chunk_id": "B", "score": 0.8},
            {"chunk_id": "C", "score": 0.7},
        ]
        list_b = [
            {"chunk_id": "B", "score": 5.0},
            {"chunk_id": "D", "score": 4.0},
            {"chunk_id": "A", "score": 3.0},
        ]
        fused = reciprocal_rank_fusion([list_a, list_b])
        assert fused[0]["chunk_id"] == "B"

    def test_output_contains_all_unique_chunks(self):
        list_a = [{"chunk_id": "A", "score": 1.0}, {"chunk_id": "B", "score": 0.5}]
        list_b = [{"chunk_id": "C", "score": 2.0}, {"chunk_id": "A", "score": 1.5}]
        fused = reciprocal_rank_fusion([list_a, list_b])
        ids = {r["chunk_id"] for r in fused}
        assert ids == {"A", "B", "C"}

    def test_rrf_scores_are_positive(self):
        list_a = [{"chunk_id": "X", "score": 0.5}]
        fused = reciprocal_rank_fusion([list_a])
        assert all(r["rrf_score"] > 0 for r in fused)

    def test_single_list_preserves_order(self):
        result_list = [
            {"chunk_id": "A", "score": 0.9},
            {"chunk_id": "B", "score": 0.8},
            {"chunk_id": "C", "score": 0.7},
        ]
        fused = reciprocal_rank_fusion([result_list])
        ids = [r["chunk_id"] for r in fused]
        assert ids == ["A", "B", "C"]

    def test_empty_lists(self):
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[]]) == []

    def test_rrf_score_field_present(self):
        list_a = [{"chunk_id": "A", "score": 1.0}]
        fused = reciprocal_rank_fusion([list_a])
        assert "rrf_score" in fused[0]
