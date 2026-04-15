# ARCHITECTURE.md — Agentic RAG Engine

---

## 1. Executive Summary

The Agentic RAG Engine is a production-grade Retrieval-Augmented Generation pipeline built for high-accuracy document question-answering. It combines LLM-based agentic query routing, two-stage hybrid retrieval (dense + sparse with RRF fusion), cross-encoder re-ranking, Maximal Marginal Relevance diversity selection, and a dedicated hallucination detection pass. The system is deployed as a single Docker container on Hugging Face Spaces, with Qdrant Cloud for vector storage and Upstash Redis for semantic caching. Three distinct LLM invocations (router → generator → auditor) are orchestrated per non-cached query, with each stage designed to fail safely and independently.

---

## 2. System Goals & Non-Goals

### Goals

- Accurately answer natural language questions over ingested PDF documents with source citations.
- Route queries to appropriate retrieval strategies based on query complexity.
- Detect and surface hallucinations before responses reach the user.
- Operate reliably within a resource-constrained single-container deployment.
- Maintain deterministic, reproducible evaluation metrics across ingestion and retrieval passes.

### Non-Goals

- Real-time document streaming or incremental ingestion.
- Multi-tenant data isolation (all documents share one Qdrant collection).
- Support for modalities beyond text (images, tables are treated as text via pymupdf4llm output).
- Sub-100ms p99 latency — the pipeline prioritizes accuracy over speed.

---

## 3. High-Level Architecture

### Agentic vs. Traditional RAG

Traditional RAG applies a fixed retrieve-then-generate pattern regardless of query type. This fails for multi-hop questions (require chained retrieval), comparative questions (require parallel retrieval across concepts), and analytical questions (require broader top-K). It also has no mechanism to reject out-of-scope queries before incurring retrieval cost.

This system adds an **agentic routing layer**: an LLM classifies each query into one of five types (`SIMPLE`, `ANALYTICAL`, `COMPARATIVE`, `MULTI_HOP`, `OUT_OF_SCOPE`) before any retrieval occurs. This classification gate serves two purposes: it adapts retrieval parameters per query type, and it acts as a security guardrail by routing adversarial inputs to a fail-closed response path without hitting the vector store or generator.

### Architecture Layers

```
┌──────────────────────────────────────────────────────────────┐
│                     Hugging Face Space                        │
│  ┌───────────────┐          ┌──────────────────────────────┐ │
│  │  Streamlit UI │◄────────►│  FastAPI Backend (:8000)     │ │
│  │  (:7860)      │  HTTP    │  RAGPipeline Orchestrator    │ │
│  └───────────────┘          └──────────────────────────────┘ │
│           supervised by supervisord                           │
└──────────────────────────────────────────────────────────────┘
         │                          │                  │
         ▼                          ▼                  ▼
  Groq Cloud LLM          Qdrant Cloud Vector DB   Upstash Redis
  (llama-3.3-70b)         (dense + sparse index)   (semantic cache)
```

---

## 4. Component Breakdown

---

### 4.1 Query Router (LLM-Based Classifier)

**Responsibility:** Classify incoming user queries into one of five categories to determine downstream pipeline behavior.

**Inputs:** Raw user query string.  
**Outputs:** `QueryType` enum — `SIMPLE | ANALYTICAL | COMPARATIVE | MULTI_HOP | OUT_OF_SCOPE`.

**Behavior per classification:**
- `SIMPLE` → standard hybrid retrieval, top-K = 5.
- `ANALYTICAL` → broader retrieval, top-K = 10–20.
- `COMPARATIVE` → parallel retrieval over multiple concept anchors.
- `MULTI_HOP` → sequential retrieval with intermediate context injection.
- `OUT_OF_SCOPE` → immediate pipeline short-circuit; no retrieval, no LLM generation.

**Why chosen:** Rule-based classifiers cannot handle semantic ambiguity in query phrasing. An LLM router generalizes robustly to paraphrased, mixed-intent, and adversarial inputs.

**Failure modes:**
- Misclassification of complex queries as `SIMPLE` → lower recall; mitigated by conservative top-K defaults.
- LLM latency spike → adds ~300–500ms to query path; not retried (classification errors are recoverable).
- Prompt injection → hardened prompt template causes any injection attempt to produce `OUT_OF_SCOPE`; pipeline fails closed.

---

### 4.2 Hybrid Retriever (Dense + BM25 + RRF)

**Responsibility:** Retrieve candidate chunks combining semantic similarity (dense vectors) and lexical overlap (sparse BM25 index).

**Inputs:** Query embedding (768-dim from bge-base-en-v1.5) + raw query string for BM25.  
**Outputs:** Merged candidate list (up to top-K = 20) with RRF-fused scores.

**Dense retrieval:** Cosine similarity search over Qdrant's HNSW index.  
**Sparse retrieval:** BM25 keyword scoring over the same corpus.  
**Fusion:** Reciprocal Rank Fusion — final score = Σ 1/(k + rankᵢ) across retriever ranks. Default k = 60.

**Why RRF over score normalization:** Dense and sparse retrievers produce scores on incompatible scales. Normalization requires assumptions about score distributions that break under domain shift. RRF uses only rank positions, making it distribution-agnostic and stable across document types.

**Failure modes:**
- Qdrant connection failure → exception propagates to API layer; 503 returned to UI.
- BM25 index stale after ingestion → mitigated by rebuild-on-ingest.
- Poor recall for highly specific queries → mitigated by re-ranking stage.

---

### 4.3 Cross-Encoder Re-Ranker

**Responsibility:** Score each retrieved candidate against the query jointly (query + passage as single input), replacing bi-encoder's approximate similarity with exact relevance scoring.

**Inputs:** Query string + list of up to 20 candidate chunks.  
**Outputs:** Ranked list with cross-encoder relevance scores.

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`.

**Why cross-encoder over bi-encoder for re-ranking:** Bi-encoders embed query and passage independently, sacrificing fine-grained token-level interaction. Cross-encoders attend over the full (query, passage) pair, producing higher-quality relevance scores at the cost of O(N) inference calls. Re-ranking only 20 candidates keeps latency acceptable (~150–300ms on CPU).

**Failure modes:**
- CPU inference latency spikes under load → top-K passed through without re-ranking as fallback (degraded quality, not failure).
- Model load failure at startup → container health check fails; supervisord does not restart (model is a hard dependency).

---

### 4.4 MMR Diversity Layer

**Responsibility:** Select a final subset from re-ranked candidates that maximizes both relevance and result diversity, preventing redundant chunks from occupying LLM context.

**Inputs:** Re-ranked candidate list with embeddings.  
**Outputs:** Top-N diverse chunks (typically N = 5–7).

**Algorithm:** Maximal Marginal Relevance — iteratively selects the candidate that maximizes λ·sim(query, chunk) − (1−λ)·max_sim(chunk, selected). λ = 0.7 (relevance-weighted).

**Why this matters:** Without MMR, the top-5 chunks from a dense index often cover the same passage from slightly different angles — especially with 20% overlap chunking. The LLM context fills with near-duplicate content, crowding out complementary information.

**Failure modes:** Degenerate case where all candidates are highly similar → MMR selects top-N by relevance score. No hard failure path.

---

### 4.5 LLM Answer Generator

**Responsibility:** Generate a grounded, cited answer from the retrieved context chunks.

**Inputs:** User query + ordered list of context chunks with source metadata.  
**Outputs:** Answer text with inline citations referencing chunk sources.

**Model:** Groq `llama-3.3-70b-versatile` via Groq Cloud API (~800 tok/s throughput).

**Prompt design:** Strict grounding instruction — model is instructed to cite only from provided context and flag uncertainty explicitly. No parametric knowledge generation permitted.

**Failure modes:**
- Groq API rate limit → exponential backoff with 3 retries; timeout after 30s.
- Context window overflow → chunks truncated to fit; longest chunks pruned first.
- Model generates uncited claims → caught by hallucination detector in the next stage.

---

### 4.6 Hallucination Detector

**Responsibility:** Audit the generated answer against retrieved context to identify claims unsupported by source material.

**Inputs:** Generated answer + original context chunks.  
**Outputs:** Hallucination verdict (pass/flag) + per-claim grounding assessment.

**Mechanism:** A second LLM call with an adversarially-framed prompt — the model is instructed to act as an auditor looking for unsupported claims, rather than a generator trying to be helpful. This framing increases detection sensitivity. Results are logged to JSON and surfaced in the UI.

**Why a separate LLM call:** Self-evaluation in the same generation pass is unreliable — the model is biased toward confirming its own output. A fresh invocation with an adversarial system prompt breaks this bias.

**Failure modes:**
- False positive → user sees a flagged answer that is actually grounded; acceptable cost.
- False negative (missed hallucination) → no downstream catch; mitigated by strict generator prompt.
- Groq API failure → answer passed through with a UI warning; fails open by design (detection failure is surfaced, not silenced).

---

### 4.7 Chunking System (AdaptiveChunker)

**Responsibility:** Split ingested PDF content into semantically coherent, retrieval-optimized chunks with deterministic IDs.

**Inputs:** Raw text from pymupdf4llm + document metadata.  
**Outputs:** List of chunks with content, metadata, and MD5-based deterministic IDs.

**Strategies (in priority order):**

| Strategy | Trigger | Chunk Size |
|---|---|---|
| Structure-Aware | Tables/code blocks detected | Atomic (variable) |
| Hierarchical | High header density | 512 tokens + parent prefix |
| Semantic | Mixed structure | 400–600 tokens (soft limit) |
| Fixed | Fallback | 512 tokens, 102-token overlap |

**Auto-selection:** Heuristics applied to the first three pages only (O(1) with respect to document length).

**Deterministic IDs:** `MD5(doc_id + "::" + strategy + "::" + chunk_index)`. Prevents evaluation metric corruption caused by ID mismatch between static ingestion and dynamic evaluation dataset generation.

**Failure modes:**
- spaCy model unavailable → semantic chunker falls back to fixed.
- Malformed PDF → pymupdf4llm partial parse; ingested with warning, not rejected.

---

### 4.8 Vector Store (Qdrant)

**Responsibility:** Persist and serve dense vector embeddings with HNSW approximate nearest-neighbor search; also holds BM25 sparse index.

**Inputs:** Chunk embeddings (768-dim float32) + payload metadata.  
**Outputs:** Top-K results by cosine similarity with payloads.

**Why Qdrant over FAISS:** FAISS requires the index to fit in process memory and is rebuilt from scratch on restart. Qdrant is a persistent, network-accessible service with native support for both dense and sparse vectors, payload filtering, and horizontal scaling. It also supports gRPC transport, reducing serialization overhead.

**Connection handling:** `QDRANT_HOST` env var is stripped of `https://` prefix in `vector_store.py` to prevent SSL confusion in the gRPC client.

**Failure modes:**
- Network partition → retrieval fails; API returns 503.
- Collection not initialized → caught at startup; ingestion required before querying.

---

### 4.9 Cache Layer (Redis via Upstash)

**Responsibility:** Short-circuit repeated or semantically equivalent queries without re-running retrieval and generation.

**Two cache levels:**
1. **Embedding cache:** Stores computed embeddings keyed by query hash. Avoids re-embedding identical queries.
2. **Semantic response cache:** Stores full pipeline responses keyed by query embedding similarity. Cache hit if cosine similarity > threshold (0.92).

**TTL:** 24 hours for embeddings; 6 hours for responses (response freshness degrades faster).

**Connection handling:** `REDIS_HOST` strips `https://` prefix in `cache.py` for Upstash serverless compatibility.

**Failure modes:**
- Redis unavailable → cache miss treated as cold query; pipeline proceeds normally. No hard dependency.
- Cache poisoning → TTL expiry is the only mitigation; no cache invalidation API exposed.

---

### 4.10 API Layer (FastAPI)

**Responsibility:** Expose the RAGPipeline as HTTP endpoints consumed by the Streamlit UI.

**Endpoints (representative):**
- `POST /ingest` — document ingestion.
- `POST /query` — full pipeline query execution.
- `GET /health` — container liveness probe.
- `POST /evaluate` — trigger benchmark run.

**Design:** Bound to `0.0.0.0:8000` inside the container. Not exposed to the public internet — Hugging Face Spaces only routes `:7860` externally. The UI makes all backend calls via `localhost:8000`.

**Correlation IDs:** Every request is assigned a UUID at the API boundary, threaded through all pipeline stages for latency attribution and log correlation.

**Failure modes:**
- Unhandled exception → 500 with structured error body; no stack trace exposed to client.
- Startup ordering — FastAPI must be healthy before Streamlit accepts traffic; supervisord `startsecs` handles this.

---

### 4.11 UI Layer (Streamlit)

**Responsibility:** Provide an interactive dashboard for document upload, query submission, result inspection, and evaluation benchmarking.

**Key surfaces:** Document upload + ingestion trigger; query input with query-type display; answer view with source citations; hallucination flag indicators; evaluation results table (Precision@K, MRR, NDCG, Hit Rate).

**Design constraint:** Streamlit is single-threaded; all API calls to FastAPI are synchronous HTTP. Long-running ingestion or benchmark runs block the UI thread. This is an acceptable trade-off for the current deployment scale.

**Failure modes:**
- Backend unreachable → user-facing error message with retry suggestion.
- Session state loss → query history not persisted across page refreshes (stateless).

---

## 5. Data Flow — Request Lifecycle

```
User Query
    │
    ▼
[1] Assign Correlation ID
    │
    ▼
[2] Redis Cache Lookup (embedding similarity)
    │── HIT ──► Return cached response → END
    │
    MISS
    │
    ▼
[3] QueryRouter (LLM Call #1)
    │── OUT_OF_SCOPE ──► Return rejection → END
    │
    ▼
[4] Embed Query (bge-base-en-v1.5)
    │
    ▼
[5] Hybrid Retrieval
    ├── Dense: Qdrant HNSW search (top-20)
    └── Sparse: BM25 keyword search (top-20)
    │
    ▼
[6] RRF Fusion → merged top-20 candidates
    │
    ▼
[7] Cross-Encoder Re-Ranking (ms-marco)
    │
    ▼
[8] MMR Diversity Selection → top 5–7 chunks
    │
    ▼
[9] LLM Answer Generation (LLM Call #2, Groq)
    │
    ▼
[10] Hallucination Detection (LLM Call #3, Groq)
    │── FLAGGED ──► Return answer + hallucination warning
    │
    PASS
    │
    ▼
[11] Cache Write (Redis)
    │
    ▼
[12] Return response + citations + correlation ID → UI
```

---

## 6. Deployment Architecture

### Topology

Single Docker container managed by `supervisord` on Hugging Face Spaces (Docker SDK).

```
supervisord
├── streamlit (app.py, :7860)   — public-facing
└── uvicorn   (FastAPI, :8000)  — internal only
```

### Why single container over microservices

At the current scale (single-user demo, HF Spaces resource limits), the operational overhead of a service mesh outweighs the isolation benefits. The two processes share a Python environment and communicate over localhost, eliminating network serialization between UI and API.

The trade-off is that a crash in one process does not automatically stop the other — supervisord keeps both alive independently. A Streamlit crash with a healthy backend means users see a broken UI but the API continues processing inflight requests.

### Security implications

FastAPI is bound internally. All public traffic enters only through Streamlit on `:7860`. This prevents direct API access, prompt injection via raw HTTP, and endpoint enumeration by external actors.

---

## 7. Scalability Considerations

### Current bottlenecks

| Bottleneck | Root Cause | Mitigation |
|---|---|---|
| LLM latency | 3 sequential Groq calls per query | Groq LPU reduces generation time; caching absorbs repeat queries |
| Cross-encoder re-ranking | O(N) CPU inference over top-20 | Candidate cap at 20; async batch inference not yet implemented |
| Streamlit single-thread | Blocking HTTP calls to backend | Not mitigated; acceptable at demo scale |
| Qdrant cold start | Collection hydration on first query | Pre-warm via health check at startup |

### Horizontal scaling limits

The FastAPI backend is stateless and horizontally scalable. Qdrant and Redis are external services that already support clustering. The hard constraint is Streamlit — scaling the frontend requires a load balancer and session affinity. At production scale, the Streamlit UI would be replaced with a React frontend consuming the FastAPI directly.

---

## 8. Reliability & Fault Tolerance

| Failure Scenario | Behavior |
|---|---|
| Redis unavailable | Cache miss; pipeline proceeds cold |
| Groq rate limit | Exponential backoff, 3 retries, then 503 |
| Qdrant unreachable | 503 to client; no silent data loss |
| Hallucination detector failure | Answer returned with UI warning flag |
| Cross-encoder crash | Top-K returned in RRF order (graceful degradation) |
| OUT_OF_SCOPE query | Immediate rejection; no retrieval or generation cost |

---

## 9. Observability

- **Correlation IDs:** UUID assigned per request at the API boundary; threaded through all pipeline stages. Enables end-to-end log tracing for any specific query.
- **Latency tracking:** Per-stage timing logged (router, retrieval, re-ranking, generation, detection). Surfaced in API response metadata.
- **Token cost tracking:** Groq token usage logged per call (prompt + completion tokens). Aggregated per session for cost attribution.
- **Evaluation output:** Benchmark results (MRR, NDCG, Precision@K, Hit Rate) written to `data/evaluation/` as JSON. Loaded directly into UI dashboard.
- **Hallucination logs:** Each detection pass result written to JSON. Surfaced in UI for user inspection.

---

## 10. Security

| Threat | Mitigation |
|---|---|
| Prompt injection | QueryRouter prompt hardened; any injection triggers OUT_OF_SCOPE routing |
| Direct API access | FastAPI bound to internal port; not publicly exposed |
| Secret leakage | All credentials injected via env vars; no secrets in codebase |
| Qdrant SSL confusion | `https://` stripped from host before gRPC client init |
| Redis SSL confusion | `https://` stripped from host before connection |
| Malicious PDF | pymupdf4llm sandboxed; no code execution from document content |

**Fail-closed routing:** The query router defaults to `OUT_OF_SCOPE` on any parse failure, ambiguous output, or prompt injection signal. The pipeline never proceeds to retrieval on an uncertain classification.

---

## 11. Trade-offs

### Latency vs. Accuracy

Three sequential LLM calls per query add ~1–2 seconds of irreducible latency (routing + generation + hallucination detection). This is the deliberate cost of agentic accuracy. A single-call pipeline would be faster but would sacrifice query-type adaptation and hallucination auditing. The Redis semantic cache absorbs the latency cost for repeated query patterns.

### Cost vs. Quality

Cross-encoder re-ranking and hallucination detection each add inference cost. At Groq's LPU throughput rates, the three LLM calls cost approximately the same wall-clock time as a single call on a standard GPU endpoint. The quality gains from re-ranking (NDCG +0.14 over dense-only) justify the cost at current usage volumes.

### Simplicity vs. Flexibility

Single-container deployment trades operational simplicity for scaling flexibility. The architecture is appropriate for a demo and early-stage production workload. Migrating to microservices would require extracting the FastAPI backend, containerizing independently, and introducing a message queue for async ingestion jobs — work deferred until usage patterns justify it.

---
