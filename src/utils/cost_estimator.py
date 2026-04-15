# src/utils/cost_estimator.py
from dataclasses import dataclass, field

# Groq pricing (USD per million tokens) — pay-as-you-go rates
# Source: https://groq.com/pricing  (update if Groq changes pricing)
_PRICING: dict[str, dict[str, float]] = {
    # Llama 3.3 70B — primary model
    "llama-3.3-70b-versatile": {
        "input_per_mtok": 0.59,
        "output_per_mtok": 0.79,
    },
    # Llama 3.1 8B — fast/cheap fallback
    "llama-3.1-8b-instant": {
        "input_per_mtok": 0.05,
        "output_per_mtok": 0.08,
    },
    # Mixtral 8x7B — alternative mid-tier
    "mixtral-8x7b-32768": {
        "input_per_mtok": 0.24,
        "output_per_mtok": 0.24,
    },
    # Gemma 2 9B — lightweight option
    "gemma2-9b-it": {
        "input_per_mtok": 0.20,
        "output_per_mtok": 0.20,
    },
}

# Fallback if model not in table (use 70B Versatile rates)
_DEFAULT_PRICING = {"input_per_mtok": 0.59, "output_per_mtok": 0.79}


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """
    Estimate the USD cost of a single Groq API call.

    Args:
        model: Groq model ID (e.g. "llama-3.3-70b-versatile")
        input_tokens: Number of prompt/input tokens
        output_tokens: Number of completion/output tokens

    Returns:
        Cost in USD (float, e.g. 0.000340)
    """
    pricing = _PRICING.get(model, _DEFAULT_PRICING)
    input_cost = (input_tokens / 1_000_000) * pricing["input_per_mtok"]
    output_cost = (output_tokens / 1_000_000) * pricing["output_per_mtok"]
    return round(input_cost + output_cost, 6)


@dataclass
class TokenUsageTracker:
    """
    Accumulates token usage and cost across multiple Groq API calls
    within a single pipeline request.

    Usage:
        tracker = TokenUsageTracker(model="llama-3.3-70b-versatile")
        tracker.add("router", input_tokens=180, output_tokens=20)
        tracker.add("answer", input_tokens=3420, output_tokens=287)
        print(tracker.to_dict())
    """
    model: str
    _calls: list[dict] = field(default_factory=list, repr=False)

    def add(self, stage: str, input_tokens: int, output_tokens: int) -> None:
        cost = estimate_cost(self.model, input_tokens, output_tokens)
        self._calls.append({
            "stage": stage,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
        })

    @property
    def total_input_tokens(self) -> int:
        return sum(c["input_tokens"] for c in self._calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c["output_tokens"] for c in self._calls)

    @property
    def total_cost_usd(self) -> float:
        return round(sum(c["cost_usd"] for c in self._calls), 6)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "calls": self._calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": self.total_cost_usd,
        }
