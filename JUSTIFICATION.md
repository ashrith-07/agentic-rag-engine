# Chunking Strategy Justification

## Parameter Choices and Design Rationale

### Chunk Size: 512 Tokens (Default)

The 512-token default was selected based on the context window requirements
of transformer-based bi-encoder retrieval models. BAAI/bge-base-en-v1.5, the
primary embedding model in this pipeline, was trained on passages up to 512
tokens. Exceeding this limit causes the model to truncate input silently,
degrading embedding quality without any visible error.

Empirically, 512 tokens corresponds to roughly 3–5 paragraphs of technical
text — enough context for the LLM to generate a grounded answer, but small
enough that the embedding captures a focused semantic unit rather than a
diffuse mixture of topics.

The pipeline exposes 1024 and 2048 as configurable alternatives via
`config.py`. 1024 tokens improves recall for long-context analytical queries
at the cost of embedding precision for short factual queries. 2048 tokens
is appropriate only for document-level retrieval tasks where the goal is
section-level matching rather than passage-level matching.

### Overlap: 20% (102 Tokens)

A 20% overlap (102 tokens at the 512-token default) was chosen to preserve
semantic continuity at chunk boundaries without incurring excessive storage
cost.

At 0% overlap, concepts that span paragraph boundaries are split across
chunks — neither chunk contains the full context, and retrieval misses the
relationship entirely. This is particularly damaging for hierarchical
technical documents where conclusions often reference definitions from the
preceding paragraph.

At 50% overlap, every token appears in two chunks on average. This doubles
the vector storage requirement and increases retrieval noise — the same
passage is returned twice with slightly different surrounding context,
forcing the LLM to reconcile near-duplicate content.

20% overlap corresponds to approximately one to two sentences of shared
context — enough to bridge concept boundaries without redundancy.

### Semantic Chunking: Soft Limit 400–600 Tokens

The semantic chunker targets a soft window of 400–600 tokens rather than a
hard cutoff. Hard cutoffs on sentence-boundary chunkers produce erratic
chunk sizes — a single long sentence can push a chunk 30% over the limit.
The soft window allows the chunker to complete the current sentence before
splitting, producing more natural chunk boundaries at the cost of slight
size variation.

spaCy's `en_core_web_sm` was selected for sentence segmentation over
rule-based splitters (splitting on `.`) because it correctly handles
abbreviations, decimal numbers, and quoted speech — all common in technical
documents.

### Hierarchical Chunking: Parent Context Prefix

The hierarchical strategy attaches a `parent_chunk_id` to every child chunk.
At retrieval time, the parent section header is prepended to the child chunk
text before passing to the LLM. This is the single highest-impact
improvement to answer quality in the pipeline.

Without the parent prefix, a child chunk reading "The default value is 512"
is uninterpretable without knowing which parameter is being discussed. With
the parent prefix ("## Chunking Configuration > The default value is 512"),
the LLM has full context.

### Structure-Aware Chunking: Atomic Blocks

Tables and code blocks are treated as atomic units — never split mid-block.
A table split across two chunks produces two incomplete tables, neither of
which is interpretable. A code block split at an arbitrary token boundary
produces syntactically invalid code.

The structure chunker detects these blocks via the markdown output of
pymupdf4llm (which preserves table structure as pipe-delimited markdown and
code blocks as fenced `` ``` `` sections) and wraps each in its own chunk
with a surrounding context sentence to anchor it semantically.

### Auto-Selection Heuristics

The `AdaptiveChunker` applies heuristics to the first three pages of a
document rather than the full document. This makes auto-selection O(1) with
respect to document length. The three-page sample is sufficient because
document structure (header density, table density) is established in the
opening pages of well-structured technical documents.

The priority order (structure → hierarchical → semantic → fixed) was chosen
to apply the most specific strategy first. Structure-aware chunking is more
specific than hierarchical (it handles mixed content), which is more specific
than semantic (it handles structured text), which is more specific than fixed
(the universal fallback).

## Security & Observability Rationale

### Deterministic Chunk Identification (MD5)
Earlier iterations generated completely randomized `uuid.uuid4()` identities per chunk during ingestion. When evaluation datasets (such as RAGAS dynamic generation) ran, the chunk IDs mismatched against the generated UUID references in the vector DB, yielding artificial `0.000` Mean Reciprocal Rank (MRR) and Hit Rate metrics.

Moving the chunk generation to `hashlib.md5(f"{doc_id}::{strategy}::{chunk_index}".encode()).hexdigest()` solved the indexing alignment. A chunk dynamically sliced at evaluation identical to a chunk statically inserted via UI ingestion will now share the identical deterministic hash and guarantee metrics reliability seamlessly in production.

### Guardrails on Prompt Injection (`QueryRouter`)
Due to strict Hugging Face Spaces environment hosting and the agentic autonomy of the QA process, prompt injection forms real risk. The `QueryRouter` prompt (`src/llm/prompt_templates.py`) was hardened under an explicit `OUT_OF_SCOPE` definition. Any attempt to modify persona, spoof system rules, or ignore contextual instructions triggers immediate refusal. This bypasses the entire heavy hybrid retrieval+Rerank latency path, instantly failing closed at Streamlit UI.

### Qdrant Cloud and Upstash Redis Parsing
When deployed as a Docker container to Hugging Face Spaces via `supervisord`, raw `https://` environmental variables native to typical cloud dashboards broke TCP handshakes.
*   **Qdrant:** `QDRANT_HOST` variables parsing is safely modified in `src/retrieval/vector_store.py` to prevent SSL confusion.
*   **Redis Cache:** Strips `https` prefix directly in `src/retrieval/cache.py` eliminating `.ping()` connection drops.

Both allow the system to operate on cloud infrastructure purely over the `docker` buildx layer securely rather than spinning up heavy DBs inside a severely un-provisioned virtual machine chunk.
