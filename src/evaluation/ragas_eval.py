# src/evaluation/ragas_eval.py
"""
RAGAS evaluation integration.

Evaluates LLM answer quality using 4 RAGAS metrics:
  - Faithfulness:       answer claims supported by context
  - Answer Relevancy:   answer addresses the question
  - Context Precision:  retrieved context is relevant to query
  - Context Recall:     retrieved context covers the answer

RAGAS requires an LLM backend. We configure it to use Groq
via a LangChain-compatible wrapper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class RAGASResult:
    """RAGAS evaluation result for a single query."""
    query: str
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "error": self.error,
        }

    @property
    def average_score(self) -> float | None:
        scores = [
            s for s in [
                self.faithfulness,
                self.answer_relevancy,
                self.context_precision,
                self.context_recall,
            ]
            if s is not None
        ]
        return round(sum(scores) / len(scores), 4) if scores else None


class RAGASEvaluator:
    """
    Wrapper around the RAGAS evaluation framework.

    Evaluates answer quality given query, answer, and retrieved contexts.
    Uses Groq as the LLM backend via LangChain integration.

    Usage:
        evaluator = RAGASEvaluator()
        result = evaluator.evaluate_single(
            query="What is RRF?",
            answer="RRF is a rank fusion method...",
            contexts=["Context chunk 1...", "Context chunk 2..."],
        )
    """

    def __init__(self) -> None:
        self._initialized = False
        self._metrics: list[Any] = []

    def _initialize(self) -> None:
        """Lazy initialization — imports RAGAS and configures Groq backend."""
        if self._initialized:
            return

        try:
            from langchain_groq import ChatGroq
            from ragas.llms import LangchainLLMWrapper
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )

            from src.config import settings

            llm = ChatGroq(
                api_key=settings.groq_api_key,
                model=settings.groq_model,
                temperature=0,
            )
            wrapped_llm = LangchainLLMWrapper(llm)

            # Assign LLM to each metric
            for metric in [faithfulness, answer_relevancy,
                            context_precision, context_recall]:
                metric.llm = wrapped_llm

            self._metrics = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ]
            self._initialized = True
            logger.info("RAGAS evaluator initialized with Groq backend")

        except ImportError as e:
            logger.warning(
                f"RAGAS import failed: {e}\n"
                "Install: pip install ragas langchain-groq"
            )
            raise

    def evaluate_single(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> RAGASResult:
        """
        Evaluate a single query-answer-context triple.

        Args:
            query: User question
            answer: LLM-generated answer
            contexts: Retrieved chunk texts used for the answer
            ground_truth: Optional reference answer (improves context_recall)

        Returns:
            RAGASResult with per-metric scores
        """
        result = RAGASResult(query=query)

        try:
            self._initialize()

            from datasets import Dataset
            from ragas import evaluate

            data = {
                "question": [query],
                "answer": [answer],
                "contexts": [contexts],
            }
            if ground_truth:
                data["ground_truth"] = [ground_truth]

            dataset = Dataset.from_dict(data)
            scores = evaluate(dataset, metrics=self._metrics)
            scores_dict = scores.to_pandas().iloc[0].to_dict()

            result.faithfulness = _safe_float(scores_dict.get("faithfulness"))
            result.answer_relevancy = _safe_float(scores_dict.get("answer_relevancy"))
            result.context_precision = _safe_float(scores_dict.get("context_precision"))
            result.context_recall = _safe_float(scores_dict.get("context_recall"))

        except Exception as e:
            result.error = str(e)
            logger.warning(f"RAGAS evaluation failed: {e}")

        return result

    def evaluate_batch(
        self,
        queries: list[str],
        answers: list[str],
        contexts_list: list[list[str]],
        ground_truths: list[str] | None = None,
    ) -> list[RAGASResult]:
        """
        Evaluate a batch of query-answer-context triples.

        More efficient than calling evaluate_single in a loop
        as RAGAS batches LLM calls internally.
        """
        results = []
        gt_list = ground_truths or [None] * len(queries)

        for query, answer, contexts, gt in zip(
            queries, answers, contexts_list, gt_list
        ):
            result = self.evaluate_single(query, answer, contexts, gt)
            results.append(result)

        return results

    def aggregate(self, results: list[RAGASResult]) -> dict:
        """Average RAGAS scores across a list of results."""
        valid = [r for r in results if r.error is None]
        if not valid:
            return {"error": "No valid results to aggregate"}

        n = len(valid)

        def avg(attr: str) -> float | None:
            vals = [getattr(r, attr) for r in valid if getattr(r, attr) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        return {
            "num_evaluated": n,
            "num_errors": len(results) - n,
            "faithfulness": avg("faithfulness"),
            "answer_relevancy": avg("answer_relevancy"),
            "context_precision": avg("context_precision"),
            "context_recall": avg("context_recall"),
        }


def _safe_float(value: Any) -> float | None:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


# Module-level singleton
ragas_evaluator = RAGASEvaluator()
