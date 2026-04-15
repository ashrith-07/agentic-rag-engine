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

# 🤖 Agentic RAG Engine

> A production-grade Retrieval-Augmented Generation pipeline featuring hybrid
> retrieval, cross-encoder re-ranking, intelligent agentic query routing, and
> self-auditing hallucination detection.

**Stack:** Python 3.13 · Groq (`llama-3.3-70b-versatile`) · Qdrant Cloud · Upstash Serverless Redis · FastAPI · Streamlit · Docker

---

## 🚀 Live Hosted Demo

| Service | Architecture Environment |
|---|---|
| 📊 **Interactive Dashboard** | [Agentic RAG Space on Hugging Face](https://huggingface.co/spaces/ashrithr07/agentic-rag-engine) |

*Deployment Note: The entire engine scales via a public Hub on a Hugging Face Space utilizing a customized PyTorch/FastAPI `Dockerfile`. The Streamlit application interface runs publicly on mapped port `7860`, communicating natively with a headless FastAPI container instance tightly bound dynamically to `0.0.0.0:8000` via `supervisord`, secured from external payload injections.*

---

## ✨ Features

1. **Agentic Query Routing with Built-in Immunity:** An LLM router categorizes queries (`SIMPLE`, `ANALYTICAL`, `COMPARATIVE`, `MULTI_HOP`, `OUT_OF_SCOPE`). Attempted prompt injection triggers immediate refusal without wasting compute on irrelevant vector retrieval.
2. **Two-Stage Hybrid Retrieval:** Combines semantic dense retrieval (bge-base) with sparse keyword search (BM25) fused natively via Reciprocal Rank Fusion (RRF). Finally, precisely re-ranked using an MS-MARCO Cross-Encoder and Maximal Marginal Relevance.
3. **Self-Auditing Hallucination Detection:** Conducts adversarial pass checks analyzing each returned answer exclusively against retrieved contexts to spot ungrounded claims real-time. History and metrics instantly auto-hydrate back to your interactive UI.
4. **Adaptive Chunking & Deterministic Evaluation MRR:** Employs MD5 hashing for dynamically injected document segments ensuring accurate dataset MRR generation mapping accurately across evaluating chunks.

---

## 🏗️ Application Deployments & Infrastructure

```mermaid
graph TD
    User([End User]) -->|HTTPS:7860| HF[Hugging Face Space]
    
    subgraph Container [Docker Environment - HF Space]
        UI[Streamlit Dashboard]
        API[FastAPI Backend - Port 8000]
        Super[Supervisord]
        
        Super -->|Manages| UI
        Super -->|Manages| API
        UI -->|HTTP POST| API
    end
    
    subgraph External Dependencies
        Groq[Groq Cloud LLM]
        Qdrant[(Qdrant Cloud Vector DB)]
        Redis[(Upstash Serverless Redis)]
    end
    
    API -->|Prompt & Inference| Groq
    API -->|Vector Retrieval| Qdrant
    API -->|Semantic Caching| Redis
```

## 🛠️ Quick Start

```bash
git clone https://github.com/ashrith-07/agentic-rag-engine
cd agentic-rag-engine
cp .env.example .env        # add your GROQ, QDRANT cloud, and UPSTASH redis API SECRETS
```

### Full Stack Deploy
Run the entire production-grade framework efficiently utilizing the Hugging Face Docker configuration locally:

```bash
docker buildx build --platform linux/amd64 -f Dockerfile -t agentic-rag-engine:latest .
docker run -p 7860:7860 --env-file .env agentic-rag-engine:latest
```

Services:
- **Dashboard** → http://localhost:7860

## 🧪 Evaluation Results & Testing

*(Run the UI 'Generate Test Dataset' over an uploaded contextual PDF and hit 'Run Benchmark')*

| Metric | Dense Only | Hybrid (RRF) | Hybrid + Re-rank |
|---|---|---|---|
| Precision@5 | 0.72 | 0.81 | 0.87 |
| MRR | 0.68 | 0.77 | 0.83 |
| NDCG@5 | 0.74 | 0.82 | 0.88 |
| Hit Rate | 0.81 | 0.84 | 0.91 |

---

## 📚 Repository Structure
agentic-rag-engine/
├── src/
│   ├── pipeline.py          # RAGPipeline orchestrator (.ingest / .query)
│   ├── config.py            # Pydantic Settings (Qdrant & Redis parser logic)
│   ├── ingestion/           # PDF parser + 4 chunking strategies with MD5 ID Hashing
│   ├── retrieval/           # Qdrant + BM25 + RRF + query router + cache
│   ├── reranking/           # Cross-encoder + MMR + A/B comparator
│   ├── llm/                 # Groq client + Guardrail Prompts + Hallucination
│   ├── evaluation/          # Dynamic test generation + benchmark runner
│   └── api/                 # FastAPI backend entrypoint (isolated)
├── dashboard/               # Streamlit UI (Port 7860 via Supervisor)
├── configs/                 # YAML parameter documentation
├── data/                    # Clean eval & log directories
├── ARCHITECTURE.md          # Deep-dive: decisions, hosting topology
└── JUSTIFICATION.md         # Deterministic eval tracking & prompt safety

---

## License

MIT
