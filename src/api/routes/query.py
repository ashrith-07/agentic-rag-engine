from fastapi import APIRouter, HTTPException
from loguru import logger

from src.api.models import (
    CitationOut,
    HallucinationOut,
    QueryRequest,
    QueryResponse,
    StageTraceOut,
    TokenUsageOut,
)
from src.pipeline import pipeline
from src.utils.correlation_id import get_correlation_id

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Run a full RAG query through the pipeline.

    Returns the answer with:
    - Inline citations referencing source chunks
    - Hallucination detection report with confidence score
    - Full stage latency trace (routing, retrieval, reranking, LLM, audit)
    - Token usage and cost breakdown
    - Optional A/B comparison (before/after re-ranking)
    """
    try:
        result = await pipeline.query(
            question=request.question,
            run_ab_comparison=request.run_ab_comparison,
        )
    except Exception as e:
        cid = get_correlation_id()
        logger.error(f"[{cid}] Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

    return QueryResponse(
        correlation_id=result.correlation_id,
        query=result.query,
        query_type=result.query_type,
        answer=result.answer,
        citations=[
            CitationOut(
                chunk_id=c.chunk_id,
                chunk_id_short=c.chunk_id_short,
                source_file=c.source_file,
                page_number=c.page_number,
                section_title=c.section_title,
            )
            for c in result.cited_answer.citations
        ],
        hallucination=HallucinationOut(
            supported_claims=result.hallucination_report.supported_claims,
            hallucinated_claims=result.hallucination_report.hallucinated_claims,
            unsupported_inferences=result.hallucination_report.unsupported_inferences,
            confidence_score=result.hallucination_report.confidence_score,
            is_reliable=result.hallucination_report.is_reliable,
            error=result.hallucination_report.error,
        ),
        trace=StageTraceOut(
            correlation_id=result.trace.correlation_id,
            stages=result.trace.stages,
            total_ms=result.trace.total_ms,
        ),
        token_usage=TokenUsageOut(
            model=result.token_usage.model,
            total_input_tokens=result.token_usage.total_input_tokens,
            total_output_tokens=result.token_usage.total_output_tokens,
            total_cost_usd=result.token_usage.total_cost_usd,
        ),
        chunks_retrieved=len(result.retrieval_results),
        ab_comparison=result.ab_comparison,
    )
