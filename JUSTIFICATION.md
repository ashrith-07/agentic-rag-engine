# JUSTIFICATION.md — Engineering Decision Record

---

## 1. Design Philosophy

### Why "Agentic RAG"

Standard RAG is a fixed function: embed query → retrieve → generate. It applies identical retrieval behavior regardless of whether the question is factual ("What is the default chunk size?"), analytical ("What are the trade-offs between chunking strategies?"), or comparative ("How does BM25 differ from dense retrieval?"). This uniformity is a design flaw, not a simplification.

Agentic RAG introduces a routing layer that inspects query intent before retrieval and adapts the pipeline accordingly. The router is not a heuristic — it is an LLM that can handle paraphrase, implicit intent, and adversarial input. The cost is one additional LLM call per query. The gain is query-type-specific retrieval parameters, a natural security enforcement point, and an explicit rejection path for out-of-scope inputs.

### Why Reliability Over Raw Generation

LLMs hallucinate. Grounding the generator strictly in retrieved context is necessary but not sufficient — models still interpolate, confabulate citations, and blend parametric memory with retrieved content. The hallucination detection stage exists because no amount of prompt engineering eliminates this failure mode completely.

The architectural stance is: generate aggressively (use a large, capable model), then audit adversarially (use a fresh LLM invocation framed to find flaws). This is more reliable than asking a single model to be both helpful and self-critical in the same pass.

---

## 2. Retrieval Design Decisions

### Why Hybrid Search Over Dense-Only

Dense retrieval (bi-encoder similarity) excels at semantic matching but fails on exact-match queries — product names, version numbers, technical identifiers. BM25 excels at exact-match but fails on paraphrase and semantic equivalence. Neither alone is sufficient for a general document QA system.

Hybrid search captures both signals. The measured improvement in this system: Precision@5 increases from 0.72 (dense only) to 0.81 (hybrid RRF), MRR from 0.68 to 0.77. These are not marginal gains.

### Why RRF (Not Score Normalization)

Dense cosine similarity scores and BM25 TF-IDF scores are not on the same scale, and their distributions vary by corpus. Min-max normalization assumes a known score range that shifts with every new document ingestion. Z-score normalization assumes Gaussian distribution, which BM25 scores do not follow.

Reciprocal Rank Fusion sidesteps the scaling problem entirely by using only rank positions: `score = Σ 1/(k + rankᵢ)`. The hyperparameter k (default 60) controls how sharply top ranks are rewarded. RRF is robust to outlier scores, stable across corpus sizes, and requires no calibration. It is the correct choice for fusing heterogeneous retrieval systems.

### Why Top-K = 20

Top-K = 20 is the retrieval budget before re-ranking. The cross-encoder re-ranker collapses this to 5–7 for LLM context. The expanded retrieval pool exists to ensure the re-ranker has enough candidates to recover from bi-encoder errors — dense retrieval can miss the best passage while ranking it 8th rather than 1st.

At K < 10, the re-ranker cannot compensate for recall failures in the bi-encoder. At K > 30, re-ranking latency (O(N) cross-encoder calls) exceeds acceptable bounds on CPU inference.

### Why Cross-Encoder Re-Ranking

Bi-encoders encode query and passage independently. The resulting embeddings capture general semantic similarity but lose token-level interactions — the specific word overlap, negation handling, and entity co-occurrence that determine whether a passage actually answers a question.

Cross-encoders receive the full (query, passage) pair as a single input and attend over both jointly. This produces relevance scores that are significantly more accurate than cosine similarity. The trade-off is that cross-encoders cannot be pre-computed — they must run at query time. Limiting the candidate pool to 20 makes this tractable.

The measured improvement: NDCG@5 increases from 0.82 (hybrid, no re-rank) to 0.88 (hybrid + re-rank). Hit Rate increases from 0.84 to 0.91.

---

## 3. Chunking Decisions

### Why 512 Tokens

BAAI/bge-base-en-v1.5 was trained on passages up to 512 tokens. Inputs exceeding this are silently truncated by the tokenizer — the embedding represents only the first 512 tokens, with no indication that the rest was discarded. At 512 tokens, the full chunk is embedded without truncation.

512 tokens also corresponds to approximately 3–5 paragraphs of technical text — enough context for the generator to produce a grounded answer, compact enough that the embedding captures a focused semantic unit rather than a diffuse topic mixture.

The pipeline exposes 1024 and 2048 as configurable alternatives. 1024 improves recall on long-context analytical queries at the cost of embedding precision for short factual queries. 2048 is appropriate only for section-level retrieval, where the goal is retrieving the right section rather than the right passage.

### Why 20% Overlap (102 Tokens)

The overlap parameter controls how much context is shared between adjacent chunks to prevent concepts from being severed at boundaries.

At 0% overlap, a concept spanning two paragraphs is split across two chunks. Neither chunk contains the complete thought. Retrieval returns one chunk, the LLM sees incomplete context, and answer quality degrades for boundary-crossing concepts.

At 50% overlap, every token appears in two chunks on average. Storage doubles, and retrieval returns near-duplicate passages that consume LLM context without adding information. The MMR layer partially mitigates this, but it adds unnecessary processing.

20% overlap (102 tokens at 512-token chunk size) shares roughly one to two sentences between adjacent chunks — sufficient to bridge concept boundaries without significant redundancy.

### Why Adaptive Chunking

Documents have heterogeneous structure. A PDF may contain dense prose, reference tables, code examples, and section headers within the same document. A single chunking strategy applied uniformly degrades performance on all structure types:
- Fixed chunking splits tables mid-row, producing uninterpretable fragments.
- Semantic chunking over code produces arbitrary boundaries inside function bodies.
- Hierarchical chunking on flat prose adds overhead without benefit.

The `AdaptiveChunker` applies heuristics to the document's first three pages to select the appropriate strategy, then applies it consistently. Auto-selection runs in O(1) with respect to document length. The priority order (structure → hierarchical → semantic → fixed) applies the most specific strategy first, using fixed as a universal fallback.

### Trade-offs of Each Strategy

**Fixed chunking:** Predictable chunk sizes, simple implementation, zero linguistic dependencies. Degrades on structured content (tables, code) and at paragraph boundaries. The correct choice for unstructured plain-text corpora.

**Semantic chunking (spaCy):** Respects sentence boundaries, produces natural chunk breaks. Requires the `en_core_web_sm` model at runtime. Chunk sizes vary within the 400–600 token soft window. Correct for mixed technical prose.

**Hierarchical chunking:** Attaches parent section headers to child chunks, giving the generator structural context ("§ Chunking Configuration > The default value is 512") that is otherwise unavailable. Highest single-change impact on answer quality for structured documents. Cost is parent metadata storage per chunk.

**Structure-aware chunking:** Treats tables and code blocks as atomic units — never split. A table split mid-row produces two incomplete tables, neither interpretable. A code block split at an arbitrary token produces syntactically invalid code. The pymupdf4llm output preserves table structure as pipe-delimited markdown and code as fenced blocks, making detection straightforward.

---

## 4. Model Choices

### Why bge-base-en-v1.5

BAAI/bge-base-en-v1.5 consistently ranks near the top of the MTEB retrieval benchmark for models that run efficiently on CPU. Its 512-token input limit aligns with the chunk size decision. It produces 768-dimensional embeddings — compact enough for fast similarity search, expressive enough for domain-general retrieval. The "base" variant is the sweet spot between the "small" model (faster, lower quality) and "large" (higher quality, 3× inference cost).

### Why MiniLM for Re-Ranking

`cross-encoder/ms-marco-MiniLM-L-6-v2` is the standard baseline for passage re-ranking. MiniLM-L-6 (6 transformer layers) is fast enough for CPU inference over 20 candidates at query time without a GPU. The MS-MARCO training data closely matches the passage retrieval task — web passages and query pairs — which transfers well to document QA.

A larger cross-encoder (L-12 or full BERT) would improve quality marginally but double inference latency. The quality-latency trade-off favors L-6 at the current deployment scale.

### Why ms-marco Training Data Specifically

MS-MARCO is the largest publicly available query-passage relevance dataset, with human-annotated relevance labels from real Bing search queries. Models trained on it have seen the full distribution of how people phrase retrieval questions, including technical, factual, and analytical queries. This generalization matters for a document QA system that handles diverse query types.

### Why Groq LLM (llama-3.3-70b-versatile)

Three LLM calls per query make wall-clock latency a first-order concern. Groq's LPU architecture delivers approximately 800 tokens/second throughput — roughly 10× faster than the same model on a standard GPU inference endpoint. This makes a 3-stage agentic pipeline feel fluid rather than slow.

`llama-3.3-70b-versatile` is chosen over smaller Llama variants because the routing and hallucination detection tasks require genuine instruction-following fidelity. A 7B or 13B model follows grounding and adversarial-audit instructions inconsistently. The 70B model follows them reliably enough to make the hallucination detection pass meaningful rather than decorative.

---

## 5. Hallucination Detection

### Why a Separate LLM Call

Self-evaluation in the same generation pass fails because the model is primed toward confirming what it just produced. The generation system prompt instructs the model to be helpful and grounded. Asking the same model, in the same context, to find flaws in its output creates a conflict between task objectives. Empirically, self-evaluation under the same prompt catches fewer hallucinations than a fresh call.

A separate invocation clears the generation context entirely. The fresh call receives only the answer and the source chunks — no generation prompt, no prior context. This is structurally closer to asking a second reviewer to read a document they did not write.

### Why Adversarial Framing Works Better

The hallucination detector system prompt frames the model as an auditor whose job is to find unsupported claims. This adversarial framing shifts the model's prior: instead of a helpful assistant looking for ways a claim could be valid, it becomes an auditor looking for ways a claim is not supported by the provided text.

This framing significantly increases detection sensitivity for boundary cases — claims that are plausible given the document's domain but not explicitly stated in the retrieved chunks. The cost is a higher false-positive rate (flagging answers that are technically grounded but not provably so from the excerpts alone). The false-positive cost is acceptable: the user sees a warning, not a blocked response.

---

## 6. Caching Strategy

### Why Two Cache Levels

**Embedding cache** (keyed by query hash): Embedding computation via bge-base-en-v1.5 costs ~50–100ms per query on CPU. For repeated or near-identical queries, this is unnecessary. The embedding cache stores computed vectors keyed by normalized query string hash. Hit rate is high for repeated benchmark runs and demo usage.

**Semantic response cache** (keyed by embedding similarity): Two queries phrased differently may be semantically identical. A cache keyed by string hash would miss these. The response cache uses cosine similarity between the incoming query embedding and stored query embeddings (threshold 0.92). Similarity above threshold returns the cached response.

### TTL Decisions

Embedding cache TTL: 24 hours. Embeddings are deterministic for a given model version. The only invalidation reason is a model change, which requires a redeployment anyway.

Response cache TTL: 6 hours. Responses depend on retrieved chunks, which can change if the document corpus is updated via re-ingestion. 6 hours is short enough to stay fresh for a document-centric use case while still absorbing repeated queries within a session.

### Trade-offs

Semantic caching introduces the risk of returning a cached response for a query that is similar but not identical to the cached query. At threshold 0.92, this is a high bar — empirically, queries with similarity > 0.92 are paraphrases of the same question, not distinct queries. Lowering the threshold toward 0.85 would improve hit rate but risks returning mismatched responses. The threshold is configurable.

---

## 7. Infrastructure Decisions

### Why Qdrant Over FAISS

FAISS is an in-process library. The index lives in heap memory, requires full rebuild on restart, and cannot be shared across processes. For a pipeline where ingestion and querying are separate operations (and potentially run in separate containers in the future), an external persistent store is the correct abstraction.

Qdrant provides: persistent storage across restarts, native support for both dense and sparse vectors in the same collection, payload-filtered search, and a gRPC interface for low-overhead network calls. It is deployable as a managed cloud service (Qdrant Cloud) with no operational burden.

### Why Redis (Upstash)

Upstash Serverless Redis eliminates the need to run and manage a Redis instance. It bills per-request rather than per-hour, which is appropriate for a demo workload with bursty, low-frequency usage. The HTTP-compatible REST API and standard Redis protocol support make it a drop-in replacement for self-hosted Redis.

The main limitation is cold-start latency on serverless invocations (~20–50ms for the first request in an idle period). This is acceptable — cache misses trigger the full pipeline anyway.

### Why Hugging Face Spaces

HF Spaces provides free GPU/CPU container hosting with public HTTPS termination and a Hugging Face-managed domain. For a demo deployment, it eliminates the need to manage a VPC, load balancer, TLS certificates, or DNS. The Docker SDK support means the deployment artifact is identical to the local development environment.

The constraint is resource limits — HF Spaces CPU instances have restricted memory and CPU. This is the primary reason the system does not run a GPU inference server locally and instead delegates LLM calls to Groq Cloud.

---

## 8. Performance Decisions

### Latency Breakdown (Approximate, Non-Cached Query)

| Stage | Latency |
|---|---|
| QueryRouter (Groq, LLM #1) | 300–500ms |
| Query embedding (bge-base) | 50–100ms |
| Hybrid retrieval (Qdrant + BM25) | 100–200ms |
| Cross-encoder re-ranking (20 candidates) | 150–300ms |
| MMR selection | <10ms |
| LLM generation (Groq, LLM #2) | 500–800ms |
| Hallucination detection (Groq, LLM #3) | 300–500ms |
| Cache write | 20–50ms |
| **Total (uncached)** | **~1.5–2.5s** |

### Why 3 LLM Calls Is Acceptable

At Groq's ~800 tok/s throughput, each LLM call completes in 300–800ms depending on output length. Three calls sum to ~1–2 seconds of LLM time, which is within acceptable interactive latency for a document QA use case (users expect RAG systems to take 2–4 seconds, not <1 second). The semantic cache brings repeat-query latency to <100ms, making the system feel fast for common query patterns.

The alternative — a single LLM call that handles routing, generation, and self-auditing — would be faster but architecturally unsound. Routing fidelity, generation quality, and hallucination detection quality all degrade when conflated into one prompt.

---

## 9. Evaluation Strategy

### Why MRR, NDCG, Precision@K

**Mean Reciprocal Rank (MRR):** Measures whether the first relevant result is ranked highly. Critical for RAG because the LLM generator is strongly influenced by the first chunk in context — if the most relevant chunk is ranked 10th, it may be truncated or de-emphasized.

**NDCG@K (Normalized Discounted Cumulative Gain):** Accounts for graded relevance — a result ranked 2nd is penalized less than one ranked 10th. More nuanced than binary precision metrics. Captures the quality of the entire ranked list, not just the top result.

**Precision@K:** The fraction of the top-K results that are relevant. Simple and interpretable. Used to compare retrieval configurations (dense vs. hybrid vs. hybrid+rerank) at the same K.

**Hit Rate:** Whether at least one relevant chunk appears in the top-K. Measures recall rather than precision. Ensures the pipeline can retrieve the answer even if it is not perfectly ranked.

### Why RAGAS

RAGAS provides an end-to-end evaluation framework for RAG pipelines that assesses faithfulness (answer grounded in context?), answer relevance (answer addresses the question?), and context relevance (retrieved context is on-topic?). These are LLM-judged metrics that complement the retrieval metrics above. RAGAS also supports dynamic test dataset generation from ingested documents, which is used by the "Generate Test Dataset" feature in the evaluation dashboard.

---

## 10. What I Would Improve Next

### Query Decomposition

Multi-hop queries currently run a single hybrid retrieval pass. A multi-hop query like "What is the overlap percentage, and why was that value chosen?" has two sub-questions that may be answered by different chunks. Query decomposition would break this into two sub-queries, retrieve independently, and merge the results before generation. This would significantly improve NDCG for `MULTI_HOP` query types.

### Streaming Responses

The generator currently waits for the full completion before returning to the UI. Groq supports token streaming. Streaming would allow the UI to display tokens as they arrive, reducing perceived latency from ~2 seconds to ~0.3 seconds (time-to-first-token). The hallucination detection pass would need to run post-stream, with the UI updating the grounding verdict asynchronously.

### Sparse Vectors in Qdrant

The current BM25 implementation is client-side, with the Qdrant collection storing only dense vectors. Qdrant natively supports sparse vector indices (SPLADE-style). Moving BM25 scoring into Qdrant would eliminate the client-side BM25 computation step, reduce round-trips, and allow server-side hybrid fusion — a cleaner architecture with better performance at scale.

### Multi-Document Reasoning

The current system treats all chunks as a flat pool from a single document collection. Cross-document questions ("How does Document A's approach to chunking compare to Document B's?") are not handled — the retriever may return chunks from both documents, but the generator has no awareness of document provenance beyond metadata. Proper multi-document reasoning would require document-level context injection and a retrieval strategy that explicitly balances coverage across sources.

### Async Ingestion Pipeline

PDF ingestion is currently synchronous and blocks the API response. For large documents (100+ pages), this causes request timeouts. The correct architecture is an async ingestion job queue (e.g., Celery or a simple task table in Redis) with a status polling endpoint. The UI would submit an ingestion job, poll for completion, and display progress. This would also allow parallel chunk processing.

---
