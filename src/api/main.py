import sys
import time
from contextlib import asynccontextmanager

import redis
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from qdrant_client import QdrantClient
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from src.api.models import HealthResponse
from src.api.routes import eval as eval_router
from src.api.routes import ingest, query
from src.config import settings
from src.retrieval.bm25_index import _INDEX_PATH, bm25_index
from src.retrieval.embeddings import embedding_engine
from src.utils.correlation_id import set_correlation_id

# ── Logging setup ─────────────────────────────────────────────────────────────

logger.remove()
logger.add(
    sys.stdout,
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "{message}"
    ),
    level=settings.log_level,
    colorize=True,
)
logger.add(
    "data/logs/app.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown logic.
    """
    cid = set_correlation_id()
    logger.info(f"[{cid}] Starting Agentic RAG Engine API...")

    # Pre-warm embedding models
    logger.info("Pre-warming embedding models...")
    try:
        embedding_engine.embed_query("warmup", model="primary")
        embedding_engine.embed_query("warmup", model="secondary")
        logger.info("✓ Embedding models warmed up")
    except Exception as e:
        logger.warning(f"Model warmup failed: {e}")

    # Load BM25 index from disk
    if _INDEX_PATH.exists():
        try:
            from src.retrieval.bm25_index import BM25Index

            loaded = BM25Index.load()
            bm25_index._bm25 = loaded._bm25
            bm25_index._chunks = loaded._chunks
            bm25_index._corpus_tokens = loaded._corpus_tokens
            logger.info(f"✓ BM25 index loaded: {bm25_index.doc_count} docs")
        except Exception as e:
            logger.warning(f"BM25 index load failed: {e}")
    else:
        logger.warning("No BM25 index found on disk — ingest a document first")

    logger.info("✓ Agentic RAG Engine API ready")
    logger.info("  Docs → http://localhost:8000/docs")

    yield

    logger.info("Shutting down Agentic RAG Engine API...")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Agentic RAG Engine",
    description=(
        "Production-grade RAG pipeline with hybrid retrieval, "
        "cross-encoder re-ranking, agentic query routing, "
        "and hallucination detection."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow dashboard and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Raise upload body limit to 210 MB (UI shows 200 MB max)
_MAX_UPLOAD_BYTES = 210 * 1024 * 1024  # 210 MB


class LimitUploadSizeMiddleware(BaseHTTPMiddleware):
    """Reject requests whose Content-Length exceeds the configured limit."""

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > _MAX_UPLOAD_BYTES:
            return Response(
                content=f"File too large. Maximum allowed size is 200 MB.",
                status_code=413,
            )
        return await call_next(request)


app.add_middleware(LimitUploadSizeMiddleware)


# ── Request logging middleware ────────────────────────────────────────────────


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with correlation ID and latency."""
    cid = set_correlation_id()
    start = time.perf_counter()

    response = await call_next(request)

    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"[{cid}] {request.method} {request.url.path} "
        f"→ {response.status_code} ({duration_ms:.0f}ms)"
    )
    return response


# ── Routes ────────────────────────────────────────────────────────────────────

app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(eval_router.router)


# ── Health check ──────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    """
    Check health of all downstream services.
    Returns status of Qdrant, Redis, BM25 index, and LLM model.
    """
    # Qdrant
    try:
        from src.pipeline import vector_store
        vector_store._client.get_collections()
        qdrant_status = "healthy"
    except Exception as e:
        qdrant_status = f"unhealthy: {e}"

    # Redis
    try:
        from src.retrieval.cache import embedding_cache
        embedding_cache._client.ping()
        redis_status = "healthy"
    except Exception as e:
        redis_status = f"unhealthy: {e}"

    overall = (
        "healthy"
        if qdrant_status == "healthy" and redis_status == "healthy"
        else "degraded"
    )

    return HealthResponse(
        status=overall,
        qdrant=qdrant_status,
        redis=redis_status,
        bm25_docs=bm25_index.doc_count,
        model=settings.groq_model,
    )


@app.get("/", tags=["health"])
async def root() -> dict:
    return {
        "name": "Agentic RAG Engine",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }
