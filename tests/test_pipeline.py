"""
End-to-end integration test for the RAGPipeline.

These tests use the real pipeline (no mocks) but operate on a
synthetic in-memory document rather than a real PDF file.
They verify the full ingest → query flow works correctly.
"""

import pytest

from src.ingestion.metadata import ParsedDocument
from src.pipeline import RAGPipeline
from src.retrieval.bm25_index import bm25_index
from src.retrieval.vector_store import vector_store
from src.utils.correlation_id import set_correlation_id


@pytest.fixture(scope="module")
def pipeline():
    set_correlation_id()
    return RAGPipeline()


@pytest.fixture(scope="module", autouse=True)
def ingest_sample(pipeline, sample_parsed_doc):
    """
    Ingest a synthetic document before running pipeline tests.

    Uses the sample_parsed_doc fixture (no real PDF needed).
    Clears Qdrant collections first to ensure clean state.
    """
    set_correlation_id()

    try:
        vector_store.delete_collection("bge_chunks")
        vector_store.delete_collection("minilm_chunks")
    except Exception:
        pass
    vector_store.ensure_collections()

    from src.ingestion.chunker import chunk_document

    chunks = chunk_document(sample_parsed_doc, strategy="fixed")

    vector_store.upsert_chunks(chunks, model="primary")
    vector_store.upsert_chunks(chunks, model="secondary")
    bm25_index.build(chunks)

    yield chunks

    try:
        vector_store.delete_collection("bge_chunks")
        vector_store.delete_collection("minilm_chunks")
    except Exception:
        pass


class TestQueryResult:
    def test_query_returns_result(self, pipeline):
        result = pipeline.query_sync("What is chunking?")
        assert result is not None
        assert result.answer
        assert result.correlation_id

    def test_query_has_valid_query_type(self, pipeline):
        result = pipeline.query_sync("What chunking strategies are available?")
        assert result.query_type in {
            "SIMPLE",
            "ANALYTICAL",
            "COMPARATIVE",
            "MULTI_HOP",
            "OUT_OF_SCOPE",
        }

    def test_out_of_scope_query_returns_refusal(self, pipeline):
        result = pipeline.query_sync("What is the population of Mars?")
        assert result.query_type == "OUT_OF_SCOPE"
        assert "cannot find" in result.answer.lower()
        assert len(result.retrieval_results) == 0

    def test_query_has_stage_trace(self, pipeline):
        result = pipeline.query_sync("What is fixed-size chunking?")
        assert result.trace is not None
        assert result.trace.total_ms > 0
        assert "routing_ms" in result.trace.stages
        assert "llm_ms" in result.trace.stages

    def test_query_has_token_usage(self, pipeline):
        result = pipeline.query_sync("Explain semantic chunking.")
        assert result.token_usage is not None
        assert result.token_usage.total_input_tokens > 0
        assert result.token_usage.total_cost_usd > 0

    def test_query_has_hallucination_report(self, pipeline):
        result = pipeline.query_sync("What metrics are used for evaluation?")
        assert result.hallucination_report is not None
        assert 0.0 <= result.hallucination_report.confidence_score <= 1.0

    def test_to_dict_is_serializable(self, pipeline):
        import json

        result = pipeline.query_sync("What is RAG?")
        d = result.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_cited_answer_has_citations(self, pipeline):
        """
        With a grounded answer, citations should be extractable.
        Note: this may return 0 citations if the LLM doesn't include
        chunk ID tags — we just verify the structure is present.
        """
        result = pipeline.query_sync("What is fixed-size chunking?")
        assert result.cited_answer is not None
        assert isinstance(result.cited_answer.citations, list)
        assert isinstance(result.cited_answer.cited_chunk_ids, list)
