"""Tool contracts, called directly (no API key needed)."""

import json

import pytest

from hvac_cooling_ai.agent import tools
from hvac_cooling_ai.agent.tools import TOOL_SPECS, ToolSession
from hvac_cooling_ai.cli import main as cli_main


def test_specs_match_functions():
    names = {spec["name"] for spec in TOOL_SPECS}
    assert names == {"psychrometric_lookup", "simulate_cooler", "predict_cooler_fast", "units_needed"}
    for spec in TOOL_SPECS:
        schema = spec["input_schema"]
        assert schema["type"] == "object"
        assert set(schema["required"]) <= set(schema["properties"])
        json.dumps(spec)  # serialisable for the API


def test_psychrometric_lookup_contract():
    rec = ToolSession().call("psychrometric_lookup", {"dry_bulb_c": 44, "relative_humidity_pct": 30})
    assert rec.status == "ok" and rec.result_id == "R1"
    out = rec.output
    assert out["dew_point_c"] < out["wet_bulb_c"] < out["dry_bulb_c"]
    assert out["relative_humidity_pct"] == pytest.approx(30.0)


def test_simulate_and_predict_agree():
    s = ToolSession()
    args = {"dry_bulb_c": 40, "relative_humidity_pct": 20}
    phys = s.call("simulate_cooler", args)
    fast = s.call("predict_cooler_fast", args)
    assert [phys.result_id, fast.result_id] == ["R1", "R2"]
    assert phys.status == fast.status == "ok"
    assert abs(phys.output["supply_temp_c"] - fast.output["supply_temp_c"]) < 1.0
    assert phys.output["inlet_dew_point_c"] < phys.output["supply_temp_c"] < 40


def test_units_needed_feasible_and_infeasible():
    s = ToolSession()
    ok = s.call(
        "units_needed",
        {
            "sensible_load_kw": 12,
            "dry_bulb_c": 40,
            "relative_humidity_pct": 15,
            "channel_velocity_m_s": 0.5,
            "working_air_fraction": 0.5,
        },
    )
    assert ok.status == "ok" and ok.output["feasible"]
    per_pair = ok.output["room_capacity_per_channel_pair_w"]
    assert ok.output["channel_pairs_needed"] * per_pair >= 12_000
    assert (ok.output["channel_pairs_needed"] - 1) * per_pair < 12_000
    humid = s.call("units_needed", {"sensible_load_kw": 12, "dry_bulb_c": 44, "relative_humidity_pct": 30})
    assert humid.status == "ok" and humid.output["feasible"] is False


@pytest.mark.parametrize("tool", ["simulate_cooler", "predict_cooler_fast", "units_needed"])
def test_cooler_tools_refuse_outside_envelope(tool):
    args = {"dry_bulb_c": 55, "relative_humidity_pct": 10, "sensible_load_kw": 5}
    rec = ToolSession().call(tool, args)
    assert rec.status == "refused"
    assert any("55C" in reason for reason in rec.output["reasons"])


@pytest.mark.parametrize(
    "args",
    [
        {"dry_bulb_c": 40},  # no humidity
        {"dry_bulb_c": 40, "relative_humidity_pct": 20, "humidity_ratio_g_kg": 8},  # both
        {"dry_bulb_c": "hot", "relative_humidity_pct": 20},
        {"dry_bulb_c": 40, "relative_humidity_pct": 150},
    ],
)
def test_bad_input_is_an_error_not_a_crash(args):
    rec = ToolSession().call("simulate_cooler", args)
    assert rec.status == "error" and "error" in rec.output


def test_unknown_tool():
    assert ToolSession().call("delete_everything", {}).status == "error"


def test_units_needed_requires_load():
    rec = ToolSession().call("units_needed", {"dry_bulb_c": 40, "relative_humidity_pct": 20})
    assert rec.status == "error"


def test_cli_simulate_and_refusal(capsys):
    assert cli_main(["simulate", "--t", "40", "--rh", "20"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ok"
    assert cli_main(["predict", "--t", "60", "--rh", "10"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "refused"


def test_cli_ask_without_key(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert cli_main(["ask", "will it hold 26 C?"]) == 1
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_default_constants_documented():
    assert tools.FAN_POWER_PER_PAIR_W == 0.5
