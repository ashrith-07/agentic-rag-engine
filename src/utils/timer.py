# src/utils/timer.py
import functools
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from src.utils.correlation_id import get_correlation_id


@dataclass
class StageTrace:
    """Holds timing results for all pipeline stages in a single request."""
    correlation_id: str = ""
    stages: dict[str, float] = field(default_factory=dict)  # stage_name → ms

    def record(self, stage: str, duration_ms: float) -> None:
        self.stages[stage] = round(duration_ms, 2)
        logger.debug(
            f"[{self.correlation_id}] stage={stage} duration={duration_ms:.1f}ms"
        )

    @property
    def total_ms(self) -> float:
        return round(sum(self.stages.values()), 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "correlation_id": self.correlation_id,
            "stages": self.stages,
            "total_ms": self.total_ms,
        }


@contextmanager
def stage_timer(trace: StageTrace, stage_name: str):
    """
    Context manager to time a named pipeline stage.

    Usage:
        with stage_timer(trace, "retrieval"):
            results = retriever.search(query)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        trace.record(stage_name, duration_ms)


def timed(stage_name: str | None = None) -> Callable:
    """
    Decorator to time a function and log its duration.
    Logs correlation ID if available.

    Usage:
        @timed("embedding")
        def embed_text(text: str) -> list[float]: ...

        @timed()
        def my_function(): ...
    """
    def decorator(func: Callable) -> Callable:
        name = stage_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            cid = get_correlation_id()
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.perf_counter() - start) * 1000
                logger.debug(f"[{cid}] fn={name} duration={duration_ms:.1f}ms")

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            cid = get_correlation_id()
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.perf_counter() - start) * 1000
                logger.debug(f"[{cid}] fn={name} duration={duration_ms:.1f}ms")

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator
