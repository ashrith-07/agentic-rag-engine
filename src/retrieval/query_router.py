import json

from loguru import logger

from src.llm.groq_client import groq_client
from src.llm.prompt_templates import build_router_messages
from src.utils.correlation_id import get_correlation_id
from src.utils.cost_estimator import TokenUsageTracker
from src.utils.timer import timed

# Valid query types — matches ROUTER_SYSTEM_PROMPT
QUERY_TYPES = frozenset(
    {
        "SIMPLE",
        "ANALYTICAL",
        "COMPARATIVE",
        "MULTI_HOP",
        "OUT_OF_SCOPE",
    }
)

OUT_OF_SCOPE_RESPONSE = "I cannot find sufficient information in the provided documents."


class QueryRouter:
    """
    Groq-powered agentic query router.

    Classifies each incoming query into one of 5 types before retrieval.
    This enables the pipeline to choose the optimal retrieval strategy
    per query — simple queries take a fast path, complex queries get
    the full hybrid + rerank treatment.
    """

    @timed("query_routing")
    async def route(
        self,
        query: str,
        tracker: TokenUsageTracker | None = None,
    ) -> dict:
        """
        Classify a query and return routing decision.
        """
        cid = get_correlation_id()
        messages = build_router_messages(query)

        try:
            raw = await groq_client.complete(
                messages=messages,
                tracker=tracker,
                stage="router",
                temperature=0.0,
                max_tokens=128,
                json_mode=True,
            )

            parsed = json.loads(raw)
            query_type = parsed.get("query_type", "ANALYTICAL").upper()

            # Validate — default to ANALYTICAL if unrecognised
            if query_type not in QUERY_TYPES:
                logger.warning(
                    f"[{cid}] Router returned unknown type '{query_type}' "
                    f"— defaulting to ANALYTICAL"
                )
                query_type = "ANALYTICAL"

            reasoning = parsed.get("reasoning", "")

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"[{cid}] Router error: {e} — defaulting to ANALYTICAL")
            query_type = "ANALYTICAL"
            reasoning = f"Router failed: {e}"

        result = {
            "query_type": query_type,
            "reasoning": reasoning,
            "use_reranking": query_type not in ("SIMPLE", "OUT_OF_SCOPE"),
            "use_hybrid": query_type not in ("OUT_OF_SCOPE",),
            "is_out_of_scope": query_type == "OUT_OF_SCOPE",
        }

        logger.info(
            f"[{cid}] Query routed: type={query_type}, "
            f"rerank={result['use_reranking']}, "
            f"reasoning='{reasoning}'"
        )

        return result


# Module-level singleton
query_router = QueryRouter()
