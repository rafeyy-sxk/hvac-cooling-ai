"""Prompt and workflow regression suite for the assistant.

``eval_cases.json`` holds contractor questions, what must happen for each (the outcome, the tools that
must be called, argument checks such as a Fahrenheit input converted to Celsius) and a script of what a
fake model says. The scripted replay clients below speak the Anthropic and the Groq/OpenAI wire
formats, including usage fields, so the full loop (tools, citation guard, usage tracking) is tested in
CI without a network. With an API key the same questions go to the live model and the same
expectations apply (see tests/test_prompt_regression.py).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from hvac_cooling_ai.agent.loop import AgentResult

CASES_PATH = Path(__file__).with_name("eval_cases.json")
OUTCOMES = ("grounded", "refused", "withheld")
# Token counts the scripted replies report, so cost tracking is exercised end to end.
SCRIPTED_INPUT_TOKENS = 1200
SCRIPTED_OUTPUT_TOKENS = 80


def load_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    cases = json.loads(Path(path).read_text())["cases"]
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case ids in eval cases")
    for c in cases:
        if c["expect"]["outcome"] not in OUTCOMES:
            raise ValueError(f"case {c['id']}: outcome must be one of {OUTCOMES}")
    return cases


def outcome_of(result: AgentResult) -> str:
    if result.guard_reason == "outside validated envelope":
        return "refused"
    return "grounded" if result.grounded else "withheld"


def check_case(case: dict[str, Any], result: AgentResult) -> list[str]:
    """Every way ``result`` misses the case's expectations ([] means it passed)."""
    expect = case["expect"]
    failures: list[str] = []
    got = outcome_of(result)
    if got != expect["outcome"]:
        failures.append(f"outcome {got!r}, expected {expect['outcome']!r} ({result.guard_reason})")
    called = [r.name for r in result.tool_calls]
    for tool in expect.get("tools", []):
        if tool not in called:
            failures.append(f"expected a call to {tool}, got {called}")
    any_of = expect.get("tools_any_of")
    if any_of and not set(any_of) & set(called):
        failures.append(f"expected a call to one of {any_of}, got {called}")
    for check in expect.get("args", []):
        tools = [check["tool"]] if isinstance(check["tool"], str) else check["tool"]
        values = [r.args.get(check["arg"]) for r in result.tool_calls if r.name in tools]
        if not values:
            continue  # a tools/tools_any_of rule decides whether the call was required
        if not any(isinstance(v, int | float) and abs(v - check["approx"]) <= check["tol"] for v in values):
            want = f"{check['approx']} +/- {check['tol']}"
            failures.append(f"{'/'.join(tools)}.{check['arg']} was {values}, expected {want}")
    return failures


# ---- scripted replay clients ----


class ScriptedAnthropic:
    """Replays a case script as Anthropic Messages API responses (the ``client`` for loop.ask)."""

    def __init__(self, script: list[dict[str, Any]]):
        self.turns = list(script)
        self.requests: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append(kwargs)
        turn = self.turns.pop(0)
        usage = SimpleNamespace(input_tokens=SCRIPTED_INPUT_TOKENS, output_tokens=SCRIPTED_OUTPUT_TOKENS)
        if "tool_calls" in turn:
            blocks = [
                SimpleNamespace(
                    type="tool_use", id=f"tu{len(self.requests)}_{i}", name=c["name"], input=c["args"]
                )
                for i, c in enumerate(turn["tool_calls"])
            ]
            return SimpleNamespace(stop_reason="tool_use", content=blocks, usage=usage)
        return SimpleNamespace(
            stop_reason="end_turn", content=[SimpleNamespace(type="text", text=turn["text"])], usage=usage
        )


class ScriptedGroq:
    """Replays a case script as OpenAI-compatible chat responses (the ``post`` for groq_loop.ask)."""

    def __init__(self, script: list[dict[str, Any]]):
        self.turns = list(script)
        self.payloads: list[dict[str, Any]] = []

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(payload)
        turn = self.turns.pop(0)
        usage = {
            "prompt_tokens": SCRIPTED_INPUT_TOKENS,
            "completion_tokens": SCRIPTED_OUTPUT_TOKENS,
            "total_tokens": SCRIPTED_INPUT_TOKENS + SCRIPTED_OUTPUT_TOKENS,
        }
        if "tool_calls" in turn:
            calls = [
                {
                    "id": f"c{len(self.payloads)}_{i}",
                    "type": "function",
                    "function": {"name": c["name"], "arguments": json.dumps(c["args"])},
                }
                for i, c in enumerate(turn["tool_calls"])
            ]
            message = {"role": "assistant", "content": None, "tool_calls": calls}
            finish = "tool_calls"
        else:
            message = {"role": "assistant", "content": turn["text"]}
            finish = "stop"
        return {"choices": [{"message": message, "finish_reason": finish}], "usage": usage}
