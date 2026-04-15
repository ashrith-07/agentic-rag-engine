# Architecture

## System Overview

Agentic RAG Engine is a production-grade Retrieval-Augmented Generation
pipeline. It processes PDF documents, stores them in a vector database, and
answers natural language questions with source citations and hallucination
detection.

## Data Flow
PDF
└─► Parser (pymupdf4llm)
└─► DocTypeDetector → AdaptiveChunker
└─► ChunkMetadata attached to every chunk
├─► EmbeddingEngine (bge-base + minilm)
│    ├─► Qdrant (bge_chunks collection)
│    └─► Qdrant (minilm_chunks collection)
└─► BM25Index (sparse, persisted to disk)
Query
└─► CorrelationID assigned
└─► Redis cache check → hit? return immediately
└─► QueryRouter (Groq llama-3.3-70b)
├─► SIMPLE      → dense-only search
├─► ANALYTICAL  → hybrid RRF search
├─► COMPARATIVE → hybrid RRF search
├─► MULTI_HOP   → hybrid RRF search
└─► OUT_OF_SCOPE → immediate refusal
└─► CrossEncoderReranker (ms-marco)
└─► DiversityReranker (MMR λ=0.7)
└─► ContextFormatter
└─► Groq (answer generation)
└─► CitationEngine
└─► HallucinationDetector (Groq)
└─► QueryResult
├─► Redis cache write
└─► StageTrace log

## Component Responsibilities

### Ingestion Layer (`src/ingestion/`)

**`parser.py`** — Converts PDF binary to structured markdown using
pymupdf4llm. Preserves tables as pipe-delimited markdown and code blocks as
fenced sections. Computes a SHA-256 doc_id from the binary for deduplication.

**`doc_type_detector.py`** — Analyses the first 3 pages using three
heuristics: table density, header density, and average sentence length.
Returns a strategy recommendation with confidence score.

**`chunker.py`** — Four strategy classes plus `AdaptiveChunker`:
- `FixedChunker`: tiktoken sliding window
- `SemanticChunker`: spaCy sentence segmentation
- `HierarchicalChunker`: markdown heading tree
- `StructureChunker`: atomic table/code block preservation

### Retrieval Layer (`src/retrieval/`)

**`embeddings.py`** — Wraps two SentenceTransformer models. Both are loaded
once via `lru_cache` and shared across all requests. Async wrapper prevents
blocking the FastAPI event loop during inference.

**`vector_store.py`** — Qdrant client wrapper. Manages two collections
(one per embedding model). Batch upsert (100 vectors/request) with embedding
cache integration. Metadata filtering on doc_id, page_number, strategy.

**`bm25_index.py`** — BM25Okapi sparse index. Complements dense retrieval
for exact keyword matches. Persisted to disk and loaded on startup.

**`hybrid_retriever.py`** — Reciprocal Rank Fusion of dense + BM25 results.
k=60 constant (standard). Falls back to dense-only if BM25 is not built.

**`query_router.py`** — Single Groq call classifies the query into 5 types.
Routes SIMPLE queries to the fast path (dense-only, no reranking), cutting
average latency by ~60% for factual lookups.

**`cache.py`** — Redis-backed caches for embeddings (24h TTL) and query
results (1h TTL). Graceful degradation — pipeline works without Redis.

### Re-ranking Layer (`src/reranking/`)

**`cross_encoder.py`** — ms-marco-MiniLM-L-6-v2 cross-encoder. Jointly
encodes query + passage for precise relevance scoring. Applied to the top-20
candidates from retrieval, not the full corpus.

**`diversity_reranker.py`** — Maximal Marginal Relevance (MMR). Prevents
redundant passages dominating the top results. λ=0.7 balances relevance and
diversity.

**`ab_comparator.py`** — Runs both pipelines (baseline and reranked) for
every query. Measures NDCG delta and latency overhead.

### LLM Layer (`src/llm/`)

**`groq_client.py`** — Async Groq SDK wrapper. Tenacity retry (3× exponential
backoff) on rate limits and server errors. Token usage tracked per call.

**`prompt_templates.py`** — All prompts in one place. System prompts for
answer generation, query routing, and hallucination auditing. Context
formatter with token budget management.

**`citation_engine.py`** — Regex extraction of `[CHUNK_xxx]` tags from
generated answers. Resolves short IDs back to full chunk metadata.

**`hallucination_detector.py`** — Second Groq call with adversarial framing.
Returns per-claim breakdown and 0.0–1.0 confidence score.

### Pipeline Orchestrator (`src/pipeline.py`)

`RAGPipeline` is the single entry point. `.ingest()` and `.query()` are the
only public methods. All stage timings flow through `StageTrace`. All token
costs accumulate in `TokenUsageTracker`. Correlation ID flows through every
log line.

## Design Decisions

### Why Qdrant over FAISS?
FAISS is an in-process library with no persistence, no metadata filtering,
and no REST API. Qdrant is a production server that survives restarts,
supports concurrent clients, and scales horizontally. FAISS is appropriate
for research prototypes; Qdrant is appropriate for production systems.

### Why RRF over weighted score averaging?
Dense cosine scores and BM25 scores are on incompatible scales. RRF operates
on ranks, which are always comparable across retrieval methods regardless of
score distribution. No tuning required, robust to corpus changes.

### Why two-stage retrieval?
Cross-encoders are O(n) with corpus size — infeasible at query time for large
corpora. Bi-encoder retrieval narrows candidates to 20 in ~90ms; cross-encoder
re-ranks those 20 in ~150ms. Total: ~240ms for precision comparable to full
cross-encoder scan at a fraction of the cost.

### Why Groq?
Groq's LPU hardware runs llama-3.3-70b at ~800 tokens/second versus ~50
tokens/second for comparable API providers. For a pipeline making 3 LLM
calls per query (router + answer + hallucination), this reduces LLM latency
from ~3s to ~600ms. Cost is approximately 14× cheaper than GPT-4o.

### Why correlation IDs?
A single query spans 4 services (API, Qdrant, Redis, Groq). Without a
request ID in every log line, correlating what happened for a specific query
is impossible. Correlation IDs are standard practice in distributed systems.

### Why hallucination detection as a separate call?
Generation and verification require different prompting strategies. A single
"be careful" instruction is weak. A separate call with adversarial framing
("find what is NOT supported") activates verification behaviour rather than
generation behaviour.

## Infrastructure
docker-compose.yml
├── qdrant      (qdrant/qdrant:v1.9.2)   :6333 :6334
├── redis       (redis:7.2-alpine)        :6379
├── app         (multi-stage Dockerfile)  :8000  FastAPI
└── dashboard   (same image, diff CMD)    :8501  Streamlit

Multi-stage Dockerfile: builder stage installs dependencies, runtime stage
copies only the installed packages. Runtime image has no build tools.

## Observability

Every query produces a structured trace:

```json
{
  "correlation_id": "uuid4",
  "query_type": "ANALYTICAL",
  "stages": {
    "routing_ms": 560,
    "retrieval_ms": 85,
    "reranking_ms": 308,
    "llm_ms": 422,
    "hallucination_ms": 470
  },
  "token_usage": {
    "total_input_tokens": 4200,
    "total_output_tokens": 380,
    "total_cost_usd": 0.000047
  },
  "hallucination_score": 0.97
}
```
