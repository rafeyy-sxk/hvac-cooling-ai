"""Per-call latency, token and cost tracking for both assistant loops.

Every LLM request the loops make is recorded as an ``LLMCall``: wall-clock latency, the token counts
the API reported, and an estimated cost from the price table below.

Prices are ESTIMATES, in US dollars per million tokens, read on 2026-09-26:
  claude-sonnet-5      $2.00 in / $10.00 out  (Anthropic list price; cache writes ~1.25x input,
                                               cache reads ~0.1x input)
  openai/gpt-oss-120b  $0.15 in / $0.60 out   (Groq model page, console.groq.com)
They change; override or extend them without editing code by pointing HVAC_AI_PRICES at a JSON file
shaped like {"model-id": {"input_per_mtok": 1.0, "output_per_mtok": 2.0}}. A model with no price
gets cost None rather than a made-up number.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PRICES_AS_OF = "2026-09-26"
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10

DEFAULT_PRICES: dict[str, dict[str, float]] = {
    "claude-sonnet-5": {"input_per_mtok": 2.00, "output_per_mtok": 10.00},
    "openai/gpt-oss-120b": {"input_per_mtok": 0.15, "output_per_mtok": 0.60},
}


def load_prices() -> dict[str, dict[str, float]]:
    """Default price table, overlaid with the JSON file named by HVAC_AI_PRICES (if set)."""
    prices = {k: dict(v) for k, v in DEFAULT_PRICES.items()}
    path = os.environ.get("HVAC_AI_PRICES")
    if path:
        extra = json.loads(Path(path).read_text())
        for model, row in extra.items():
            prices[model] = {
                "input_per_mtok": float(row["input_per_mtok"]),
                "output_per_mtok": float(row["output_per_mtok"]),
            }
    return prices


@dataclass(frozen=True)
class LLMCall:
    provider: str
    model: str
    latency_s: float
    input_tokens: int
    output_tokens: int
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float | None = None
    usage_reported: bool = True


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
    prices: dict[str, dict[str, float]] | None = None,
) -> float | None:
    row = (prices if prices is not None else load_prices()).get(model)
    if row is None:
        return None
    effective_input = (
        input_tokens + cache_write_tokens * CACHE_WRITE_MULTIPLIER + cache_read_tokens * CACHE_READ_MULTIPLIER
    )
    return (effective_input * row["input_per_mtok"] + output_tokens * row["output_per_mtok"]) / 1e6


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def from_anthropic(model: str, response: Any, latency_s: float) -> LLMCall:
    """Anthropic Messages API: response.usage.{input_tokens, output_tokens, cache_*_input_tokens}."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return LLMCall("anthropic", model, latency_s, 0, 0, usage_reported=False)
    tin = _int(getattr(usage, "input_tokens", 0))
    tout = _int(getattr(usage, "output_tokens", 0))
    cw = _int(getattr(usage, "cache_creation_input_tokens", 0))
    cr = _int(getattr(usage, "cache_read_input_tokens", 0))
    return LLMCall("anthropic", model, latency_s, tin, tout, cw, cr, estimate_cost(model, tin, tout, cw, cr))


def from_openai(model: str, data: dict[str, Any], latency_s: float) -> LLMCall:
    """OpenAI-compatible chat API (Groq): data["usage"]["prompt_tokens" / "completion_tokens"]."""
    usage = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(usage, dict):
        return LLMCall("groq", model, latency_s, 0, 0, usage_reported=False)
    tin = _int(usage.get("prompt_tokens"))
    tout = _int(usage.get("completion_tokens"))
    # prompt_tokens already includes any cached prompt tokens; all are priced at the input rate.
    return LLMCall("groq", model, latency_s, tin, tout, cost_usd=estimate_cost(model, tin, tout))


@dataclass
class UsageSummary:
    calls: list[LLMCall] = field(default_factory=list)

    @property
    def total_latency_s(self) -> float:
        return sum(c.latency_s for c in self.calls)

    @property
    def input_tokens(self) -> int:
        return sum(c.input_tokens + c.cache_write_tokens + c.cache_read_tokens for c in self.calls)

    @property
    def output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def cost_usd(self) -> float | None:
        """Total estimated cost, or None if any call had no price or no usage report."""
        if any(c.cost_usd is None or not c.usage_reported for c in self.calls):
            return None
        return sum(c.cost_usd or 0.0 for c in self.calls)

    def lines(self) -> list[str]:
        out = []
        for i, c in enumerate(self.calls, 1):
            cost = "n/a" if c.cost_usd is None else f"${c.cost_usd:.5f}"
            note = "" if c.usage_reported else "  (no usage reported)"
            out.append(
                f"call {i}  {c.model}  {c.latency_s:.2f} s  in {c.input_tokens} tok  "
                f"out {c.output_tokens} tok  {cost}{note}"
            )
        total = "n/a" if self.cost_usd is None else f"${self.cost_usd:.5f}"
        out.append(
            f"total  {len(self.calls)} calls  {self.total_latency_s:.2f} s  in {self.input_tokens} tok  "
            f"out {self.output_tokens} tok  est. {total} (prices are estimates as of {PRICES_AS_OF})"
        )
        return out
