"""Grounding guard and the tool-use loop, driven by a scripted fake client.

The fake stands in for the Messages API so the loop logic is tested offline.
The live test at the bottom runs only when ANTHROPIC_API_KEY is set.
"""

import json
import os
from types import SimpleNamespace

import pytest

from hvac_cooling_ai.agent import loop
from hvac_cooling_ai.agent.guard import WITHHELD, check_answer
from hvac_cooling_ai.agent.tools import ToolSession


def text(t):
    return SimpleNamespace(type="text", text=t)


def tool_use(tid, name, args):
    return SimpleNamespace(type="tool_use", id=tid, name=name, input=args)


def reply(stop, *blocks):
    return SimpleNamespace(stop_reason=stop, content=list(blocks))


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(json.loads(json.dumps(kwargs, default=lambda o: o.__dict__)))
        return self.replies.pop(0)


# ---- guard ----


def test_guard_accepts_cited_ok_result():
    s = ToolSession()
    s.call("simulate_cooler", {"dry_bulb_c": 40, "relative_humidity_pct": 20})
    assert check_answer("Supply is 25 C [R1].", s).ok


def test_guard_reads_lenticular_bracket_citations_as_square():
    s = ToolSession()
    s.call("simulate_cooler", {"dry_bulb_c": 40, "relative_humidity_pct": 20})
    verdict = check_answer("Supply is 25 C 【R1】.", s)
    assert verdict.ok and "[R1]" in verdict.text


def test_guard_rejects_missing_unknown_and_failed_citations():
    s = ToolSession()
    s.call("simulate_cooler", {"dry_bulb_c": 40})  # error: no humidity
    s.call("psychrometric_lookup", {"dry_bulb_c": 40, "relative_humidity_pct": 20})
    assert not check_answer("It will be fine.", s).ok
    assert "unknown" in check_answer("See [R9].", s).reason
    assert "failed or refused" in check_answer("See [R1] and [R2].", s).reason


def test_guard_replaces_answer_when_only_out_of_envelope_results():
    s = ToolSession()
    s.call("simulate_cooler", {"dry_bulb_c": 55, "relative_humidity_pct": 10})
    s.call("psychrometric_lookup", {"dry_bulb_c": 55, "relative_humidity_pct": 10})
    verdict = check_answer("Probably about 30 C [R2].", s)
    assert verdict.ok and verdict.reason == "outside validated envelope"
    assert "can't answer" in verdict.text and "55C" in verdict.text and "[R1]" in verdict.text


# ---- loop ----


def test_loop_runs_tool_and_returns_grounded_answer():
    client = FakeClient(
        [
            reply(
                "tool_use",
                text("Checking."),
                tool_use("tu1", "simulate_cooler", {"dry_bulb_c": 44, "relative_humidity_pct": 30}),
            ),
            reply("end_turn", text("No: supply air would be about 31.5 C [R1], above your 26 C target.")),
        ]
    )
    result = loop.ask("Will it hold 26 C supply at 44 C, 30% RH?", client=client)
    assert result.grounded and "[R1]" in result.answer
    assert [r.name for r in result.tool_calls] == ["simulate_cooler"]
    second = client.requests[1]
    assert second["model"] == "claude-sonnet-5"
    tool_result = second["messages"][-1]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == "tu1"
    assert json.loads(tool_result["content"])["result_id"] == "R1"
    assert tool_result["is_error"] is False


def test_loop_retries_once_then_withholds_uncited_answer():
    client = FakeClient(
        [
            reply("end_turn", text("It should be fine.")),
            reply("end_turn", text("Trust me, it is fine.")),
        ]
    )
    result = loop.ask("Will it hold 26 C?", client=client)
    assert not result.grounded and result.answer == WITHHELD
    assert "rejected" in client.requests[1]["messages"][-1]["content"]


def test_loop_refuses_outside_envelope_even_if_model_guesses():
    client = FakeClient(
        [
            reply(
                "tool_use",
                tool_use("tu1", "simulate_cooler", {"dry_bulb_c": 52, "relative_humidity_pct": 10}),
            ),
            reply("end_turn", text("It would probably reach 28 C.")),
        ]
    )
    result = loop.ask("52 C afternoon?", client=client)
    assert "can't answer" in result.answer and "52C" in result.answer
    assert result.tool_calls[0].status == "refused"


def test_loop_handles_model_refusal_and_truncation():
    assert loop.ask("x", client=FakeClient([reply("refusal")])).grounded is False
    assert loop.ask("x", client=FakeClient([reply("max_tokens", text("partial"))])).answer == WITHHELD


def test_tool_input_error_is_flagged_to_model():
    client = FakeClient(
        [
            reply("tool_use", tool_use("tu1", "simulate_cooler", {"dry_bulb_c": 44})),
            reply("end_turn", text("I need the humidity to answer.")),
            reply("end_turn", text("Please give the humidity.")),
        ]
    )
    result = loop.ask("44 C?", client=client)
    assert client.requests[1]["messages"][-1]["content"][0]["is_error"] is True
    assert result.answer == WITHHELD


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="needs ANTHROPIC_API_KEY")
def test_live_agent_cites_tools():
    result = loop.ask("Will this cooler hold 26 C supply air on a 44 C, 30% RH afternoon?")
    assert result.tool_calls, "expected at least one tool call"
    assert result.grounded
