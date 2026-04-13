"""
Shared pytest fixtures.
All tests import from here — keeps test files clean.
"""

import pytest
from src.ingestion.metadata import ChunkMetadata, ParsedDocument


@pytest.fixture(scope="module")
def sample_markdown() -> str:
    return """# Introduction to RAG Systems

Retrieval-Augmented Generation (RAG) combines retrieval and generation.
It retrieves relevant documents and uses them to generate accurate answers.

## Chunking Strategies

There are four main chunking strategies used in RAG pipelines.

### Fixed-size Chunking

Fixed-size chunking splits text into chunks of a fixed token count.
The default chunk size is 512 tokens with 20% overlap.
This approach works well for uniform technical documents.

### Semantic Chunking

Semantic chunking respects sentence and paragraph boundaries.
It uses spaCy for sentence segmentation.
This approach works best for narrative text and research papers.

## Vector Databases

| Database | Type | Persistence |
|----------|------|-------------|
| Qdrant   | Vector | Yes |
| FAISS    | Vector | No  |
| Redis    | Cache  | Yes |

## Code Example

```python
def embed_text(text: str) -> list[float]:
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    return model.encode(text).tolist()
```

## Evaluation Metrics

Precision@K measures the fraction of top-K results that are relevant.
Recall@K measures coverage of all relevant documents in top-K results.
MRR is the Mean Reciprocal Rank of the first relevant result.
NDCG is the Normalized Discounted Cumulative Gain at K.
"""


@pytest.fixture(scope="module")
def sample_parsed_doc(sample_markdown) -> ParsedDocument:
    return ParsedDocument(
        doc_id="abc123def456" * 4,
        source_file="test_document.pdf",
        total_pages=3,
        raw_markdown=sample_markdown,
        pages=[
            {
                "page_number": 1,
                "text": "Introduction to RAG Systems",
                "has_table": False,
                "has_code": False,
                "char_count": 100,
            },
            {
                "page_number": 2,
                "text": "Chunking Strategies",
                "has_table": True,
                "has_code": True,
                "char_count": 200,
            },
            {
                "page_number": 3,
                "text": "Evaluation Metrics",
                "has_table": False,
                "has_code": False,
                "char_count": 150,
            },
        ],
        file_size_bytes=4096,
    )


@pytest.fixture
def sample_chunk():
    from src.ingestion.chunker import Chunk

    meta = ChunkMetadata(
        doc_id="abc123def456" * 4,
        source_file="test_document.pdf",
        page_number=1,
        section_title="Introduction",
        section_depth=1,
        chunk_index=0,
        total_chunks=5,
        strategy_used="fixed",
        token_count=150,
        has_table=False,
        has_code=False,
    )
    return Chunk(
        text=(
            "Retrieval-Augmented Generation combines retrieval and generation. "
            "It retrieves relevant documents and uses them to generate accurate answers."
        ),
        metadata=meta,
    )


@pytest.fixture
def sample_chunks(sample_parsed_doc):
    """Five chunks with varied content for retrieval tests."""
    from src.ingestion.chunker import chunk_document
    from src.utils.correlation_id import set_correlation_id

    set_correlation_id()
    return chunk_document(sample_parsed_doc, strategy="fixed", chunk_size=120, overlap=12)


@pytest.fixture
def sample_retrieval_results() -> list[dict]:
    """Mock retrieval results for LLM layer tests."""
    return [
        {
            "chunk_id": "chunk-id-0001-abcd-efgh-ijkl",
            "score": 0.92,
            "text": "Fixed-size chunking splits text into chunks of 512 tokens with 20% overlap.",
            "metadata": {
                "chunk_id": "chunk-id-0001-abcd-efgh-ijkl",
                "doc_id": "abc123",
                "source_file": "test_document.pdf",
                "page_number": 2,
                "section_title": "Fixed-size Chunking",
                "strategy_used": "fixed",
                "token_count": 18,
            },
        },
        {
            "chunk_id": "chunk-id-0002-abcd-efgh-ijkl",
            "score": 0.85,
            "text": "Semantic chunking respects sentence and paragraph boundaries using spaCy.",
            "metadata": {
                "chunk_id": "chunk-id-0002-abcd-efgh-ijkl",
                "doc_id": "abc123",
                "source_file": "test_document.pdf",
                "page_number": 2,
                "section_title": "Semantic Chunking",
                "strategy_used": "semantic",
                "token_count": 14,
            },
        },
        {
            "chunk_id": "chunk-id-0003-abcd-efgh-ijkl",
            "score": 0.78,
            "text": "Precision@K measures the fraction of top-K results that are relevant.",
            "metadata": {
                "chunk_id": "chunk-id-0003-abcd-efgh-ijkl",
                "doc_id": "abc123",
                "source_file": "test_document.pdf",
                "page_number": 3,
                "section_title": "Evaluation Metrics",
                "strategy_used": "fixed",
                "token_count": 13,
            },
        },
    ]
