"""The Groq loop, driven by a scripted fake HTTP call. Live test runs only with GROQ_API_KEY."""

import json
import os

import pytest

from hvac_cooling_ai.agent import groq_loop
from hvac_cooling_ai.agent.guard import WITHHELD


def choice(content="", calls=None, finish="stop"):
    return {"choices": [{"message": {"content": content, "tool_calls": calls}, "finish_reason": finish}]}


def call(cid, name, args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class FakePost:
    def __init__(self, replies):
        self.replies = list(replies)
        self.payloads = []

    def __call__(self, payload):
        self.payloads.append(json.loads(json.dumps(payload)))
        return self.replies.pop(0)


def test_groq_loop_runs_tool_and_returns_grounded_answer():
    post = FakePost(
        [
            choice(
                calls=[call("c1", "simulate_cooler", {"dry_bulb_c": 44, "relative_humidity_pct": 30})],
                finish="tool_calls",
            ),
            choice("No: supply air would be about 31.5 C [R1], above your 26 C target."),
        ]
    )
    result = groq_loop.ask("Will it hold 26 C at 44 C, 30% RH?", post=post, model="m")
    assert result.grounded and "[R1]" in result.answer
    second = post.payloads[1]
    tool_msg = second["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "c1"
    assert json.loads(tool_msg["content"])["result_id"] == "R1"
    assert {t["function"]["name"] for t in second["tools"]} >= {"simulate_cooler", "units_needed"}


def test_groq_loop_withholds_answer_that_never_cites():
    post = FakePost([choice("It will be fine."), choice("Trust me, it is fine.")])
    result = groq_loop.ask("Will it hold?", post=post, model="m")
    assert not result.grounded and result.answer == WITHHELD


def test_groq_loop_stops_on_truncation():
    result = groq_loop.ask("q", post=FakePost([choice("partial", finish="length")]), model="m")
    assert result.answer == WITHHELD and result.guard_reason == "response hit max_tokens"


def test_groq_loop_needs_a_key_without_injected_post(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        groq_loop.ask("q")


@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="live Groq test needs GROQ_API_KEY")
def test_groq_live_answer_is_grounded():
    result = groq_loop.ask("Will this cooler hold 26 C supply air on a 40 C, 20% RH afternoon?")
    assert result.grounded, result.guard_reason
