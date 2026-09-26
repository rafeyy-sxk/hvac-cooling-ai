"""AWS Lambda entry point: the four cooler tools behind a Lambda function URL.

Only the tools are exposed (no LLM, so no API keys live in the function). Same validation, envelope
refusal and result shape as the assistant sees.

    GET  /            -> tool names, input schemas and the validated envelope
    POST / {"tool": "simulate_cooler", "args": {"dry_bulb_c": 40, "relative_humidity_pct": 20}}
         200 ok | 422 refused (outside the envelope) | 400 bad request or bad arguments

The event shape is the Lambda function URL / API Gateway HTTP API payload format 2.0.
Infrastructure is in infra/ (Terraform).
"""

from __future__ import annotations

import base64
import binascii
import json
from typing import Any

from hvac_cooling_ai.agent.tools import TOOL_SPECS, ToolSession
from hvac_cooling_ai.envelope import ENVELOPE

MAX_BODY_BYTES = 10_000
STATUS_BY_RESULT = {"ok": 200, "refused": 422, "error": 400}


def _response(status: int, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json"},
        "body": json.dumps(payload),
    }


def _method(event: dict[str, Any]) -> str:
    http = (event.get("requestContext") or {}).get("http") or {}
    return str(http.get("method") or event.get("httpMethod") or "").upper()


def _body(event: dict[str, Any]) -> bytes:
    raw = event.get("body") or ""
    if event.get("isBase64Encoded"):
        return base64.b64decode(raw, validate=True)
    return raw.encode()


def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    method = _method(event)
    if method == "GET":
        return _response(
            200,
            {
                "tools": [{"name": t["name"], "input_schema": t["input_schema"]} for t in TOOL_SPECS],
                "validated_envelope": ENVELOPE.to_dict(),
            },
        )
    if method != "POST":
        return _response(405, {"error": "use GET to list tools or POST to call one"})

    try:
        body = _body(event)
    except (binascii.Error, ValueError):
        return _response(400, {"error": "body is not valid base64"})
    if len(body) > MAX_BODY_BYTES:
        return _response(413, {"error": f"body is larger than {MAX_BODY_BYTES} bytes"})
    try:
        request = json.loads(body or b"{}")
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _response(400, {"error": "body must be JSON"})
    if not isinstance(request, dict) or not isinstance(request.get("tool"), str):
        return _response(400, {"error": 'body must be {"tool": "<name>", "args": {...}}'})
    args = request.get("args", {})
    if not isinstance(args, dict):
        return _response(400, {"error": "'args' must be an object"})

    record = ToolSession().call(request["tool"], args)
    return _response(
        STATUS_BY_RESULT[record.status],
        {"result_id": record.result_id, "tool": record.name, "status": record.status, **record.output},
    )
