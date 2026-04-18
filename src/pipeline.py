"""
RAGPipeline — the main orchestrator.

Clean public interface:
    pipeline = RAGPipeline()
    pipeline.ingest("path/to/doc.pdf")
    result = await pipeline.query("What is RRF fusion?")

Everything else (chunking, embedding, retrieval, reranking, LLM,
evaluation) is an implementation detail wired together here.
"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from src.config import settings
from src.ingestion.chunker import Chunk, chunk_document
from src.ingestion.parser import parse_pdf
from src.llm.citation_engine import CitedAnswer, citation_engine
from src.llm.groq_client import groq_client
from src.llm.hallucination_detector import HallucinationReport, hallucination_detector
from src.llm.prompt_templates import build_answer_messages, format_context
from src.reranking.ab_comparator import compare as ab_compare
from src.reranking.cross_encoder import cross_encoder_reranker
from src.reranking.diversity_reranker import diversity_reranker
from src.retrieval.bm25_index import bm25_index
from src.retrieval.hybrid_retriever import hybrid_retriever
from src.retrieval.query_router import OUT_OF_SCOPE_RESPONSE, query_router
from src.retrieval.vector_store import vector_store
from src.utils.correlation_id import set_correlation_id
from src.utils.cost_estimator import TokenUsageTracker
from src.utils.timer import StageTrace, stage_timer


@dataclass
class IngestResult:
    """Result of a document ingestion."""

    doc_id: str
    source_file: str
    total_chunks: int
    strategy_used: str
    chunks: list[Chunk] = field(repr=False)


@dataclass
class QueryResult:
    """Complete result of a pipeline query."""

    correlation_id: str
    query: str
    query_type: str
    answer: str
    cited_answer: CitedAnswer
    hallucination_report: HallucinationReport
    retrieval_results: list[dict]
    trace: StageTrace
    token_usage: TokenUsageTracker
    ab_comparison: dict | None = None

    def to_dict(self) -> dict:
        return {
            "correlation_id": self.correlation_id,
            "query": self.query,
            "query_type": self.query_type,
            "answer": self.answer,
            "citations": [c.__dict__ for c in self.cited_answer.citations],
            "hallucination": self.hallucination_report.to_dict(),
            "trace": self.trace.to_dict(),
            "token_usage": self.token_usage.to_dict(),
            "chunks_retrieved": len(self.retrieval_results),
        }

    def pretty(self) -> str:
        """Human-readable summary for CLI/notebook use."""
        lines = [
            f"\n{'═' * 60}",
            f"  Query      : {self.query}",
            f"  Type       : {self.query_type}",
            f"  Confidence : {self.hallucination_report.confidence_score:.2f}",
            f"  Cost       : ${self.token_usage.total_cost_usd:.6f}",
            f"  Latency    : {self.trace.total_ms:.0f}ms",
            f"{'─' * 60}",
            "  Answer:\n",
            f"  {self.answer}",
            f"{'─' * 60}",
        ]

        if self.cited_answer.citations:
            lines.append("  Sources:")
            for c in self.cited_answer.citations:
                page = f", page {c.page_number}" if c.page_number else ""
                lines.append(f"    [{c.chunk_id_short}] {c.source_file}{page}")

        if self.hallucination_report.hallucinated_claims:
            lines.append("\n  ⚠ Hallucinated claims:")
            for claim in self.hallucination_report.hallucinated_claims:
                lines.append(f"    • {claim}")

        lines.append(f"{'═' * 60}")
        return "\n".join(lines)


class RAGPipeline:
    """
    Production RAG pipeline orchestrator.
    """

    def __init__(self) -> None:
        self._ingested_chunks: list[Chunk] = []
        # vector_store.ensure_collections() # Lazy-loaded or handled in app startup
        logger.info("RAGPipeline initialized")

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest(
        self,
        pdf_path: str | Path,
        strategy: str = "auto",
        chunk_size: int | None = None,
        clear_existing: bool = False,
    ) -> IngestResult:
        """
        Ingest a PDF document into the pipeline.
        """
        cid = set_correlation_id()
        logger.info(f"[{cid}] Ingesting: {pdf_path}")

        try:
            if clear_existing:
                logger.info(f"[{cid}] Clearing existing documents from vector store and BM25 index")
                vector_store.delete_collection(settings.primary_collection)
                vector_store.delete_collection(settings.secondary_collection)
                vector_store.ensure_collections()
                
                bm25_index.build([])
                bm25_index.save()
                self._ingested_chunks.clear()
            else:
                vector_store.ensure_collections()
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}. Are HF Secret variables QDRANT_HOST and QDRANT_API_KEY set?")
            raise Exception("Failed to connect to Qdrant Cloud. Check HF Space Secrets.") from e

        # Parse
        doc = parse_pdf(pdf_path)

        # Chunk
        chunks = chunk_document(doc, strategy=strategy, chunk_size=chunk_size)
        strategy_used = chunks[0].metadata.strategy_used if chunks else strategy

        # Embed + index (both models)
        vector_store.upsert_chunks(chunks, model="primary")
        vector_store.upsert_chunks(chunks, model="secondary")

        # BM25 — add to existing index and persist
        bm25_index.add_chunks(chunks)
        bm25_index.save()

        # Track for session
        self._ingested_chunks.extend(chunks)

        result = IngestResult(
            doc_id=doc.doc_id,
            source_file=doc.source_file,
            total_chunks=len(chunks),
            strategy_used=strategy_used,
            chunks=chunks,
        )

        logger.info(
            f"[{cid}] Ingestion complete: {doc.source_file} → "
            f"{len(chunks)} chunks (strategy={strategy_used})"
        )

        return result

    # ── Query ─────────────────────────────────────────────────────────────────

    async def query(
        self,
        question: str,
        run_ab_comparison: bool = False,
    ) -> QueryResult:
        """
        Run a full RAG query through the pipeline.
        """
        cid = set_correlation_id()
        trace = StageTrace(correlation_id=cid)
        tracker = TokenUsageTracker(model=settings.groq_model)

        logger.info(f"[{cid}] Query: '{question}'")

        # ── Step 1: Route ─────────────────────────────────────────────────────
        with stage_timer(trace, "routing_ms"):
            routing = await query_router.route(question, tracker=tracker)

        query_type = routing["query_type"]

        # Short-circuit for out-of-scope queries
        if routing["is_out_of_scope"]:
            return self._out_of_scope_result(question, cid, trace, tracker)

        # ── Step 2: Retrieve ──────────────────────────────────────────────────
        with stage_timer(trace, "retrieval_ms"):
            if routing["use_hybrid"]:
                candidates = hybrid_retriever.search(
                    query=question,
                    top_k=settings.top_k_retrieval,
                )
            else:
                candidates = hybrid_retriever.search_dense_only(
                    query=question,
                    top_k=settings.top_k_retrieval,
                )

        # ── Step 3: Re-rank ───────────────────────────────────────────────────
        with stage_timer(trace, "reranking_ms"):
            if routing["use_reranking"] and candidates:
                reranked = cross_encoder_reranker.rerank(
                    query=question,
                    candidates=candidates,
                    top_k=settings.top_k_retrieval,
                )
                final_results = diversity_reranker.rerank(
                    query=question,
                    candidates=reranked,
                    top_k=settings.top_k_rerank,
                )
            else:
                final_results = candidates[: settings.top_k_rerank]

        # ── Step 4: Generate answer ───────────────────────────────────────────
        with stage_timer(trace, "llm_ms"):
            context = format_context(final_results)
            messages = build_answer_messages(question, context)
            answer = await groq_client.complete(
                messages=messages,
                tracker=tracker,
                stage="answer",
                temperature=0.1,
                max_tokens=1024,
            )

        # ── Step 5: Extract citations ─────────────────────────────────────────
        cited_answer = citation_engine.extract(answer, final_results)

        # ── Step 6: Hallucination check ───────────────────────────────────────
        with stage_timer(trace, "hallucination_ms"):
            hallucination_report = await hallucination_detector.check(
                answer=answer,
                retrieval_results=final_results,
                tracker=tracker,
            )

        # ── Step 7: A/B comparison (optional) ────────────────────────────────
        ab_comparison = None
        if run_ab_comparison:
            with stage_timer(trace, "ab_comparison_ms"):
                ab_comparison = ab_compare(query=question)

        logger.info(
            f"[{cid}] Query complete: "
            f"type={query_type}, "
            f"chunks={len(final_results)}, "
            f"confidence={hallucination_report.confidence_score:.2f}, "
            f"cost=${tracker.total_cost_usd:.6f}, "
            f"total={trace.total_ms:.0f}ms"
        )

        return QueryResult(
            correlation_id=cid,
            query=question,
            query_type=query_type,
            answer=answer,
            cited_answer=cited_answer,
            hallucination_report=hallucination_report,
            retrieval_results=final_results,
            trace=trace,
            token_usage=tracker,
            ab_comparison=ab_comparison,
        )

    def query_sync(self, question: str, run_ab_comparison: bool = False) -> QueryResult:
        """Synchronous wrapper for scripts and notebooks."""
        return asyncio.run(self.query(question, run_ab_comparison))

    def _out_of_scope_result(
        self,
        question: str,
        cid: str,
        trace: StageTrace,
        tracker: TokenUsageTracker,
    ) -> QueryResult:
        """Build a QueryResult for out-of-scope queries."""
        from src.llm.citation_engine import CitedAnswer
        from src.llm.hallucination_detector import HallucinationReport

        logger.info(f"[{cid}] Query classified OUT_OF_SCOPE — returning refusal")
        return QueryResult(
            correlation_id=cid,
            query=question,
            query_type="OUT_OF_SCOPE",
            answer=OUT_OF_SCOPE_RESPONSE,
            cited_answer=CitedAnswer(
                answer_text=OUT_OF_SCOPE_RESPONSE,
                citations=[],
                cited_chunk_ids=[],
            ),
            hallucination_report=HallucinationReport(confidence_score=1.0),
            retrieval_results=[],
            trace=trace,
            token_usage=tracker,
        )

    # ── CLI entry point ───────────────────────────────────────────────────────

    @staticmethod
    def main() -> None:
        """
        CLI entry point — used by `make ingest` and `make query`.
        """
        import typer

        app = typer.Typer()
        pipeline = RAGPipeline()

        @app.command()
        def ingest(pdf: str = typer.Option(..., help="Path to PDF file")) -> None:
            result = pipeline.ingest(pdf)
            print(f"✓ Ingested: {result.source_file}")
            print(f"  chunks={result.total_chunks}, strategy={result.strategy_used}")
            print(f"  doc_id={result.doc_id[:16]}...")

        @app.command()
        def query(question: str = typer.Option(..., help="Question to ask")) -> None:
            result = pipeline.query_sync(question)
            print(result.pretty())

        app()


# Module-level singleton
pipeline = RAGPipeline()


if __name__ == "__main__":
    RAGPipeline.main()
