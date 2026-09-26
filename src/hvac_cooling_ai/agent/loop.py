"""Tool-use loop on the Anthropic Messages API (manual loop, no beta helpers).

Needs ANTHROPIC_API_KEY. The client is injectable so the loop logic itself is
tested with a scripted fake client and no network.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from hvac_cooling_ai.agent.guard import WITHHELD, check_answer
from hvac_cooling_ai.agent.tools import TOOL_SPECS, ToolRecord, ToolSession
from hvac_cooling_ai.agent.usage import LLMCall, UsageSummary, from_anthropic
from hvac_cooling_ai.envelope import ENVELOPE

MODEL = "claude-sonnet-5"
MAX_TOKENS = 16000
MAX_TURNS = 10

SYSTEM_PROMPT = f"""You help HVAC contractors size an indirect evaporative (M-cycle style) cooler.

You have tools backed by a physics model and a fast ML surrogate of it. Every number you state about
air properties or cooler performance must come from a tool result in this conversation. Each tool
result has an id such as R1; cite the id in square brackets right after the figure it supports,
for example "supply air about 27.1 C [R2]". An answer without citations is discarded.

The model is validated only inside this envelope: {json.dumps(ENVELOPE.to_dict())}
(temperatures in C, humidity ratio in g/kg). If a tool refuses a request as outside the envelope,
say so plainly and do not estimate the answer another way.

Convert Fahrenheit to Celsius before calling tools. Sizing answers are in channel pairs of
200 x 200 mm plates; say that the modelled core is a bench-scale design, not a catalogue product.
Keep answers short and practical: the verdict first, then the supporting figures."""


@dataclass
class AgentResult:
    answer: str
    grounded: bool
    guard_reason: str
    tool_calls: list[ToolRecord] = field(default_factory=list)
    stop_reason: str = ""
    llm_calls: list[LLMCall] = field(default_factory=list)

    @property
    def usage(self) -> UsageSummary:
        """Latency, tokens and estimated cost across every LLM call this answer took."""
        return UsageSummary(self.llm_calls)


def _text_of(content: list[Any]) -> str:
    return "\n".join(b.text for b in content if getattr(b, "type", None) == "text").strip()


def _make_client():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("Set ANTHROPIC_API_KEY to use 'ask'. The tools work without it: see 'simulate'.")
    import anthropic

    return anthropic.Anthropic()


def ask(question: str, client: Any | None = None, model: str = MODEL) -> AgentResult:
    client = client or _make_client()
    session = ToolSession()
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    retried_citation = False
    calls: list[LLMCall] = []

    for _ in range(MAX_TURNS):
        started = time.perf_counter()
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOL_SPECS,
            messages=messages,
        )
        calls.append(from_anthropic(model, response, time.perf_counter() - started))
        stop = response.stop_reason
        if stop == "refusal":
            return AgentResult(
                "The model declined this request.",
                False,
                "model refusal",
                session.records,
                stop,
                llm_calls=calls,
            )
        if stop == "max_tokens":
            return AgentResult(
                WITHHELD, False, "response hit max_tokens", session.records, stop, llm_calls=calls
            )
        messages.append({"role": "assistant", "content": response.content})
        if stop == "pause_turn":
            continue
        if stop == "tool_use":
            results = []
            for block in response.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                record = session.call(block.name, dict(block.input))
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(
                            {"result_id": record.result_id, "status": record.status, **record.output}
                        ),
                        "is_error": record.status == "error",
                    }
                )
            messages.append({"role": "user", "content": results})
            continue

        verdict = check_answer(_text_of(response.content), session)
        if verdict.ok:
            return AgentResult(verdict.text, True, verdict.reason, session.records, stop, llm_calls=calls)
        if retried_citation:
            return AgentResult(WITHHELD, False, verdict.reason, session.records, stop, llm_calls=calls)
        retried_citation = True
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Your answer was rejected: {verdict.reason}. Call the tools you need and cite each "
                    "supporting result id in square brackets, or say you cannot answer."
                ),
            }
        )

    return AgentResult(WITHHELD, False, "too many turns", session.records, "max_turns", llm_calls=calls)
