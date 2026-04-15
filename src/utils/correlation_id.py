# src/utils/correlation_id.py
import uuid
from contextvars import ContextVar

# ContextVar is safe across async tasks — each request gets its own value
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="no-request")


def set_correlation_id(cid: str | None = None) -> str:
    """
    Set a new correlation ID for the current async context.
    If none provided, a new UUID4 is generated.
    Returns the ID that was set.
    """
    cid = cid or str(uuid.uuid4())
    _correlation_id.set(cid)
    return cid


def get_correlation_id() -> str:
    """Get the correlation ID for the current async context."""
    return _correlation_id.get()


def reset_correlation_id() -> None:
    """Reset to default (useful in tests)."""
    _correlation_id.set("no-request")
