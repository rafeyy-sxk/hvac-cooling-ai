"""The Lambda handler, driven with function-URL (payload format 2.0) events. No AWS needed."""

import base64
import importlib
import json
import re
from pathlib import Path

from hvac_cooling_ai.lambda_handler import MAX_BODY_BYTES, handler

INFRA = Path(__file__).parent.parent / "infra"


def event(method="POST", body=None, b64=False):
    raw = body if isinstance(body, str) or body is None else json.dumps(body)
    if b64 and raw is not None:
        raw = base64.b64encode(raw.encode()).decode()
    return {
        "version": "2.0",
        "requestContext": {"http": {"method": method, "path": "/"}},
        "body": raw,
        "isBase64Encoded": b64,
    }


def call(ev):
    resp = handler(ev, None)
    assert resp["headers"]["content-type"] == "application/json"
    return resp["statusCode"], json.loads(resp["body"])


def test_get_lists_tools_and_envelope():
    status, body = call(event("GET"))
    assert status == 200
    assert {t["name"] for t in body["tools"]} == {
        "psychrometric_lookup",
        "simulate_cooler",
        "predict_cooler_fast",
        "units_needed",
    }
    assert body["validated_envelope"]["t_in_c"] == [25.0, 50.0]


def test_post_runs_the_physics_tool():
    status, body = call(
        event(body={"tool": "simulate_cooler", "args": {"dry_bulb_c": 40, "relative_humidity_pct": 20}})
    )
    assert status == 200 and body["status"] == "ok" and body["result_id"] == "R1"
    assert body["inlet_dew_point_c"] < body["supply_temp_c"] < 40


def test_base64_body_is_decoded():
    status, body = call(
        event(
            body={"tool": "predict_cooler_fast", "args": {"dry_bulb_c": 45, "humidity_ratio_g_kg": 8}},
            b64=True,
        )
    )
    assert status == 200 and body["tool"] == "predict_cooler_fast"


def test_outside_envelope_is_422_with_reasons():
    status, body = call(
        event(body={"tool": "simulate_cooler", "args": {"dry_bulb_c": 52, "relative_humidity_pct": 10}})
    )
    assert status == 422 and body["status"] == "refused"
    assert "52C" in body["reasons"][0]


def test_bad_requests_are_4xx_not_crashes():
    assert call(event(body="not json"))[0] == 400
    assert call(event(body=[1, 2]))[0] == 400
    assert call(event(body={"tool": "simulate_cooler", "args": "x"}))[0] == 400
    assert call(event(body={"tool": "rm_rf", "args": {}}))[0] == 400
    assert call(event(body={"tool": "simulate_cooler", "args": {"dry_bulb_c": 40}}))[0] == 400
    bad_b64 = event()
    bad_b64.update(body="%%%", isBase64Encoded=True)
    assert call(bad_b64)[0] == 400
    assert call(event(body="x" * (MAX_BODY_BYTES + 1)))[0] == 413
    assert call(event("DELETE"))[0] == 405


def test_terraform_handler_points_at_this_function():
    main_tf = (INFRA / "main.tf").read_text()
    handler_path = re.search(r'handler\s*=\s*"([\w.]+)"', main_tf).group(1)
    module, func = handler_path.rsplit(".", 1)
    assert getattr(importlib.import_module(module), func) is handler
    assert 'runtime                        = "python3.12"' in main_tf
