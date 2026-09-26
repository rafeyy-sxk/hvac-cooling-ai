"""Prompt/workflow regression suite: every case in agent/eval_cases.json, on both loops.

Offline, a scripted fake model replays each case, so a change to the tools, the citation guard, the
envelope or the loops that alters an outcome fails here. With ANTHROPIC_API_KEY or GROQ_API_KEY set,
the same questions go to the live model and must meet the same expectations.
"""

import os

import pytest

from hvac_cooling_ai.agent import evals, groq_loop, loop
from hvac_cooling_ai.agent.guard import GuardVerdict
from hvac_cooling_ai.agent.usage import estimate_cost

CASES = evals.load_cases()
LIVE_CASES = [c for c in CASES if c.get("live", True)]


def run_scripted(case, provider):
    if provider == "claude":
        return loop.ask(case["question"], client=evals.ScriptedAnthropic(case["script"]))
    return groq_loop.ask(
        case["question"], post=evals.ScriptedGroq(case["script"]), model=groq_loop.DEFAULT_MODEL
    )


def test_cases_cover_every_outcome():
    assert {c["expect"]["outcome"] for c in CASES} == set(evals.OUTCOMES)
    assert len(CASES) >= 8


@pytest.mark.parametrize("provider", ["claude", "groq"])
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_scripted_case(case, provider):
    result = run_scripted(case, provider)
    assert evals.check_case(case, result) == []
    # usage: one record per model request, with the scripted token counts and a priced cost
    turns = len(case["script"])
    assert len(result.llm_calls) == turns
    assert result.usage.input_tokens == turns * evals.SCRIPTED_INPUT_TOKENS
    assert result.usage.output_tokens == turns * evals.SCRIPTED_OUTPUT_TOKENS
    model = loop.MODEL if provider == "claude" else groq_loop.DEFAULT_MODEL
    per_call = estimate_cost(model, evals.SCRIPTED_INPUT_TOKENS, evals.SCRIPTED_OUTPUT_TOKENS)
    assert result.usage.cost_usd == pytest.approx(turns * per_call)
    assert all(c.latency_s >= 0 for c in result.llm_calls)


def test_checker_reports_wrong_outcome_missing_tool_and_bad_argument():
    case = next(c for c in CASES if c["id"] == "fahrenheit-converted")
    # the model forgot to convert 104 F: passes 104 as Celsius, gets refused
    bad = {
        **case,
        "script": [
            {
                "tool_calls": [
                    {"name": "simulate_cooler", "args": {"dry_bulb_c": 104, "relative_humidity_pct": 20}}
                ]
            },
            {"text": "About 26 C."},
        ],
    }
    failures = evals.check_case(case, run_scripted(bad, "claude"))
    assert any("outcome 'refused'" in f for f in failures)
    assert any("dry_bulb_c was [104]" in f for f in failures)
    sizing = next(c for c in CASES if c["id"] == "sizing-12kw")
    assert any("units_needed" in f for f in evals.check_case(sizing, run_scripted(case, "claude")))


def test_suite_catches_a_disabled_guard(monkeypatch):
    """Planted defect: a guard that accepts every answer. The suite must go red."""
    monkeypatch.setattr(loop, "check_answer", lambda text, session: GuardVerdict(ok=True, text=text))
    failing = [c["id"] for c in CASES if evals.check_case(c, run_scripted(c, "claude"))]
    assert {"uncited-answer-withheld", "humid-day-refused", "too-hot-refused"} <= set(failing)


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="live cases need ANTHROPIC_API_KEY")
@pytest.mark.parametrize("case", LIVE_CASES, ids=[c["id"] for c in LIVE_CASES])
def test_live_claude_case(case):
    result = loop.ask(case["question"])
    assert evals.check_case(case, result) == [], result.answer


@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="live cases need GROQ_API_KEY")
@pytest.mark.parametrize("case", LIVE_CASES, ids=[c["id"] for c in LIVE_CASES])
def test_live_groq_case(case):
    result = groq_loop.ask(case["question"])
    assert evals.check_case(case, result) == [], result.answer
