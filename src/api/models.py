from typing import Any

from pydantic import BaseModel, Field

# ── Ingest ────────────────────────────────────────────────────────────────────


class IngestResponse(BaseModel):
    doc_id: str
    source_file: str
    total_chunks: int
    strategy_used: str
    message: str = "Ingestion complete"


# ── Query ─────────────────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    run_ab_comparison: bool = False


class CitationOut(BaseModel):
    chunk_id: str
    chunk_id_short: str
    source_file: str
    page_number: int | None
    section_title: str | None


class HallucinationOut(BaseModel):
    supported_claims: list[str]
    hallucinated_claims: list[str]
    unsupported_inferences: list[str]
    confidence_score: float
    is_reliable: bool
    error: str | None = None


class StageTraceOut(BaseModel):
    correlation_id: str
    stages: dict[str, float]
    total_ms: float


class TokenUsageOut(BaseModel):
    model: str
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float


class QueryResponse(BaseModel):
    correlation_id: str
    query: str
    query_type: str
    answer: str
    citations: list[CitationOut]
    hallucination: HallucinationOut
    trace: StageTraceOut
    token_usage: TokenUsageOut
    chunks_retrieved: int
    ab_comparison: dict[str, Any] | None = None


# ── Evaluation ────────────────────────────────────────────────────────────────


class EvalResponse(BaseModel):
    benchmark_config: dict[str, Any]
    retrieval_metrics: dict[str, Any]
    report_path: str


# ── Health ────────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    redis: str
    bm25_docs: int
    model: str
