"""Latency, token and cost tracking: both API usage formats, the price table, and ask --show-usage."""

import json
from types import SimpleNamespace

import pytest

from hvac_cooling_ai.agent import evals, groq_loop, loop, usage
from hvac_cooling_ai.cli import main


def test_anthropic_usage_fields_and_cost():
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=1000,
            output_tokens=200,
            cache_creation_input_tokens=400,
            cache_read_input_tokens=2000,
        )
    )
    call = usage.from_anthropic("claude-sonnet-5", response, 1.5)
    assert (call.input_tokens, call.output_tokens, call.cache_write_tokens, call.cache_read_tokens) == (
        1000,
        200,
        400,
        2000,
    )
    # $2/M input, $10/M output; cache writes 1.25x input, reads 0.1x input
    expected = ((1000 + 400 * 1.25 + 2000 * 0.1) * 2.00 + 200 * 10.00) / 1e6
    assert call.cost_usd == pytest.approx(expected)
    assert usage.UsageSummary([call]).input_tokens == 3400


def test_openai_usage_fields_and_cost():
    data = {"usage": {"prompt_tokens": 3000, "completion_tokens": 500, "total_tokens": 3500}}
    call = usage.from_openai("openai/gpt-oss-120b", data, 0.8)
    assert (call.provider, call.input_tokens, call.output_tokens) == ("groq", 3000, 500)
    assert call.cost_usd == pytest.approx((3000 * 0.15 + 500 * 0.60) / 1e6)


def test_missing_usage_and_unknown_model_give_no_cost():
    no_usage = usage.from_openai("openai/gpt-oss-120b", {"choices": []}, 0.1)
    assert not no_usage.usage_reported and no_usage.cost_usd is None
    unknown = usage.from_anthropic(
        "some-future-model", SimpleNamespace(usage=SimpleNamespace(input_tokens=5)), 0.1
    )
    assert unknown.cost_usd is None and unknown.input_tokens == 5
    summary = usage.UsageSummary([no_usage, unknown])
    assert summary.cost_usd is None
    assert "n/a" in summary.lines()[-1]


def test_prices_can_be_overridden_from_a_file(tmp_path, monkeypatch):
    prices = tmp_path / "prices.json"
    prices.write_text(json.dumps({"claude-sonnet-5": {"input_per_mtok": 1.0, "output_per_mtok": 1.0}}))
    monkeypatch.setenv("HVAC_AI_PRICES", str(prices))
    assert usage.estimate_cost("claude-sonnet-5", 1_000_000, 1_000_000) == pytest.approx(2.0)
    assert usage.estimate_cost("openai/gpt-oss-120b", 1_000_000, 0) == pytest.approx(0.15)


def test_claude_loop_records_each_request():
    case = next(c for c in evals.load_cases() if c["id"] == "bad-args-then-recovery")
    result = loop.ask(case["question"], client=evals.ScriptedAnthropic(case["script"]))
    assert [c.model for c in result.llm_calls] == ["claude-sonnet-5"] * 3
    assert result.usage.total_latency_s == pytest.approx(sum(c.latency_s for c in result.llm_calls))


def test_ask_show_usage_prints_per_call_and_total(monkeypatch, capsys):
    case = next(c for c in evals.load_cases() if c["id"] == "dry-afternoon-supply")
    monkeypatch.setattr(groq_loop, "_http_post", evals.ScriptedGroq(case["script"]))
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    assert main(["ask", "--provider", "groq", "--show-usage", case["question"]]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert out[-3].startswith("call 1  openai/gpt-oss-120b")
    assert out[-1].startswith("total  2 calls") and "in 2400 tok  out 160 tok" in out[-1]
    assert "est. $0.00046" in out[-1]
