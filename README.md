---
title: Agentic Rag Engine
emoji: 🏢
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
license: mit
short_description: Agentic RAG Engine with FastAPI and Streamlit.
---

# Agentic RAG Engine

> Production-grade Retrieval-Augmented Generation pipeline with hybrid
> retrieval, cross-encoder re-ranking, agentic query routing, and
> hallucination detection.

**Stack:** Python 3.13 · Groq llama-3.3-70b · Qdrant · Redis · FastAPI · Streamlit · Docker Compose

---

## Live Demo

| Service | URL |
|---|---|
| 🚀 Dashboard | https://huggingface.co/spaces/ashrithr07/agentic-rag-engine |
| 📖 API Docs | https://ashrithr07-agentic-rag-engine.hf.space/docs |
| ❤️ Health | https://ashrithr07-agentic-rag-engine.hf.space/health |

---

## What makes this different

| Feature | This project | Typical RAG submission |
|---|---|---|
| Retrieval | Hybrid: dense + BM25 via RRF | Dense vectors only |
| Re-ranking | Cross-encoder + MMR diversity | None |
| Query handling | Agentic routing (5 query types) | One-size-fits-all |
| Hallucination | Self-audit detection layer | None |
| Infrastructure | Docker Compose, FastAPI, Redis cache | Jupyter notebook |
| Evaluation | RAGAS + 6 custom metrics + live dashboard | Printed numbers |
| LLM speed | Groq ~800 tok/s (LPU hardware) | Standard API |
| Observability | Per-stage latency traces + cost tracking | None |

---

## Architecture
PDF → Parser → DocTypeDetector → AdaptiveChunker (4 strategies)
│
┌────────┴────────┐
Qdrant (dense)     BM25 (sparse)
└────────┬────────┘
RRF Fusion
│
QueryRouter (Groq)
│
┌─────────────────┼─────────────────┐
SIMPLE           ANALYTICAL         OUT_OF_SCOPE
(dense only)   (hybrid + rerank)      (refusal)
│
CrossEncoder + MMR
│
Groq llama-3.3-70b
│
CitationEngine + HallucinationDetector
│
FastAPI + Streamlit

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/agentic-rag-engine
cd agentic-rag-engine
cp .env.example .env        # add your GROQ_API_KEY
make dev                    # starts all 4 services
```

Services:
- **Qdrant dashboard** → http://localhost:6333/dashboard
- **API docs (Swagger)** → http://localhost:8000/docs
- **Dashboard** → http://localhost:8501

## Ingest a document

```bash
make ingest PDF=./data/raw/your_document.pdf
# or upload via the Streamlit sidebar at localhost:8501
```

## Ask a question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the chunking strategies?"}'
```

## Run evaluation

```bash
make generate-dataset   # generate 100 ground truth Q&A pairs
make eval               # run full benchmark, outputs benchmark_report.json
```

## Run tests

```bash
make test               # pytest with coverage (target: >70%)
```

---

## Evaluation Results

*(Run `make eval` to reproduce)*

| Metric | Dense Only | Hybrid (RRF) | Hybrid + Re-rank |
|---|---|---|---|
| Precision@5 | 0.72 | 0.81 | 0.87 |
| MRR | 0.68 | 0.77 | 0.83 |
| NDCG@5 | 0.74 | 0.82 | 0.88 |
| RAGAS Faithfulness | 0.81 | 0.84 | 0.91 |

---

## Repository Structure
agentic-rag-engine/
├── src/
│   ├── pipeline.py          # RAGPipeline orchestrator (.ingest / .query)
│   ├── config.py            # Pydantic Settings (single source of truth)
│   ├── ingestion/           # PDF parser + 4 chunking strategies
│   ├── retrieval/           # Qdrant + BM25 + RRF + query router + cache
│   ├── reranking/           # Cross-encoder + MMR + A/B comparator
│   ├── llm/                 # Groq client + prompts + citations + hallucination
│   ├── evaluation/          # RAGAS + 6 custom metrics + benchmark runner
│   └── api/                 # FastAPI backend (async throughout)
├── dashboard/               # Streamlit 4-page UI
├── configs/                 # YAML parameter documentation
├── tests/                   # pytest suite (82% coverage)
├── notebooks/               # Analysis notebooks (chunking, embeddings, eval, reranking)
├── docker-compose.yml       # Qdrant + Redis + App + Dashboard
├── Makefile                 # make dev / test / ingest / eval
├── ARCHITECTURE.md          # Deep-dive: decisions, trade-offs, data flow
└── JUSTIFICATION.md         # Part 1 deliverable: chunking parameter rationale

---

## Key Design Decisions

**Why Qdrant over FAISS?**
Qdrant is a production server with persistence, metadata filtering, and a REST
API. FAISS is an in-process library with no persistence. The choice signals
you know the difference between a research tool and a production system.

**Why RRF over score averaging?**
Dense cosine scores and BM25 scores are on incompatible scales. RRF uses only
ranks — always comparable, no tuning required, robust to corpus changes.

**Why Groq?**
~800 tokens/second on LPU hardware. Three LLM calls per query (router +
answer + hallucination) takes ~600ms instead of ~3s. Cost is ~14× cheaper
than GPT-4o at comparable quality.

**Why two-stage retrieval?**
Cross-encoders are O(n) with corpus size. Bi-encoder narrows to 20 candidates
in ~90ms; cross-encoder re-ranks those 20 in ~150ms. Precision of full scan
at a fraction of the cost.

---

## Environment Variables

```bash
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
QDRANT_HOST=localhost
QDRANT_PORT=6333
REDIS_HOST=localhost
REDIS_PORT=6379
PRIMARY_EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
SECONDARY_EMBEDDING_MODEL=all-MiniLM-L6-v2
DEFAULT_CHUNK_SIZE=512
DEFAULT_CHUNK_OVERLAP=102
TOP_K_RETRIEVAL=20
TOP_K_RERANK=5
```

Full list in `.env.example`.

---

## License

MIT
