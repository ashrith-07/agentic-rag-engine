import asyncio
from typing import Any

from groq import AsyncGroq, Groq
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import settings
from src.utils.correlation_id import get_correlation_id
from src.utils.cost_estimator import TokenUsageTracker
from src.utils.timer import timed


def _is_retryable(exc: BaseException) -> bool:
    """Retry on rate limits and transient server errors only."""
    from groq import InternalServerError, RateLimitError

    return isinstance(exc, (RateLimitError, InternalServerError))


class GroqClient:
    """
    Async-first Groq SDK wrapper with retry logic and cost tracking.

    Every call is tracked via TokenUsageTracker so per-request
    cost appears in the query trace.

    Usage:
        client = GroqClient()
        response = await client.complete(
            messages=[{"role": "user", "content": "Hello"}],
            tracker=tracker,
            stage="router",
        )
    """

    def __init__(self) -> None:
        self._sync_client = Groq(api_key=settings.groq_api_key)
        self._async_client = AsyncGroq(api_key=settings.groq_api_key)
        self._model = settings.groq_model

    @timed("groq_complete")
    async def complete(
        self,
        messages: list[dict],
        tracker: TokenUsageTracker | None = None,
        stage: str = "llm",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> str:
        """
        Async chat completion with automatic retry.

        Args:
            messages: List of {role, content} dicts
            tracker: TokenUsageTracker to record this call's cost
            stage: Label for cost tracking (e.g. "router", "answer")
            temperature: 0.0 for deterministic outputs (router, detector)
            max_tokens: Output token budget
            json_mode: Set response_format to JSON (for structured outputs)

        Returns:
            Response content string
        """
        cid = get_correlation_id()

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = await self._complete_with_retry(**kwargs)

        content = response.choices[0].message.content or ""
        usage = response.usage

        if tracker and usage:
            tracker.add(
                stage=stage,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
            )

        logger.debug(
            f"[{cid}] Groq {stage}: "
            f"in={usage.prompt_tokens if usage else '?'} "
            f"out={usage.completion_tokens if usage else '?'} tokens"
        )

        return content

    async def _complete_with_retry(self, **kwargs: Any) -> Any:
        """Inner call with tenacity retry (3x exponential backoff)."""

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        async def _call() -> Any:
            return await self._async_client.chat.completions.create(**kwargs)

        return await _call()

    def complete_sync(
        self,
        messages: list[dict],
        tracker: TokenUsageTracker | None = None,
        stage: str = "llm",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> str:
        """
        Synchronous completion — used in scripts and notebooks.
        Wraps the async version in a new event loop.
        """
        return asyncio.run(
            self.complete(
                messages=messages,
                tracker=tracker,
                stage=stage,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )
        )


# Module-level singleton
groq_client = GroqClient()
