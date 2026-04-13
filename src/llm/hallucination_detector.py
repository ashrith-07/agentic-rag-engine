import json
from dataclasses import dataclass, field

from loguru import logger

from src.llm.groq_client import groq_client
from src.llm.prompt_templates import build_hallucination_messages, format_context
from src.utils.correlation_id import get_correlation_id
from src.utils.cost_estimator import TokenUsageTracker
from src.utils.timer import timed


@dataclass
class HallucinationReport:
    """
    Result of the hallucination self-audit call.

    confidence_score: 1.0 = fully supported, 0.0 = entirely hallucinated
    """

    supported_claims: list[str] = field(default_factory=list)
    hallucinated_claims: list[str] = field(default_factory=list)
    unsupported_inferences: list[str] = field(default_factory=list)
    confidence_score: float = 1.0
    error: str | None = None

    @property
    def is_reliable(self) -> bool:
        """True if answer has no hallucinated claims and score >= 0.7."""
        return len(self.hallucinated_claims) == 0 and self.confidence_score >= 0.7

    def to_dict(self) -> dict:
        return {
            "supported_claims": self.supported_claims,
            "hallucinated_claims": self.hallucinated_claims,
            "unsupported_inferences": self.unsupported_inferences,
            "confidence_score": round(self.confidence_score, 3),
            "is_reliable": self.is_reliable,
            "error": self.error,
        }


class HallucinationDetector:
    """
    Self-audit layer: verifies every answer against its source chunks.
    """

    @timed("hallucination_check")
    async def check(
        self,
        answer: str,
        retrieval_results: list[dict],
        tracker: TokenUsageTracker | None = None,
    ) -> HallucinationReport:
        """
        Run hallucination detection on an answer.
        """
        cid = get_correlation_id()

        # Skip detection for out-of-scope refusals
        if "cannot find sufficient information" in answer.lower():
            return HallucinationReport(confidence_score=1.0)

        context = format_context(retrieval_results, max_tokens=6000)
        messages = build_hallucination_messages(answer, context)

        try:
            raw = await groq_client.complete(
                messages=messages,
                tracker=tracker,
                stage="hallucination",
                temperature=0.0,
                max_tokens=512,
                json_mode=True,
            )

            # Strip markdown fences if present
            clean = raw.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            clean = clean.strip()

            parsed = json.loads(clean)

            report = HallucinationReport(
                supported_claims=parsed.get("supported_claims", []),
                hallucinated_claims=parsed.get("hallucinated_claims", []),
                unsupported_inferences=parsed.get("unsupported_inferences", []),
                confidence_score=float(parsed.get("confidence_score", 1.0)),
            )

            logger.info(
                f"[{cid}] Hallucination check: "
                f"score={report.confidence_score:.2f}, "
                f"supported={len(report.supported_claims)}, "
                f"hallucinated={len(report.hallucinated_claims)}, "
                f"inferences={len(report.unsupported_inferences)}"
            )

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"[{cid}] Hallucination detector error: {e}")
            report = HallucinationReport(
                confidence_score=0.5,
                error=str(e),
            )

        return report


# Module-level singleton
hallucination_detector = HallucinationDetector()
