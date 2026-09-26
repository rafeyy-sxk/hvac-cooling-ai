"""The same tool-use loop on Groq's OpenAI-compatible chat API.

Needs GROQ_API_KEY. Uses the same tools, system prompt and citation guard as the
Claude loop, so a Groq answer is held to the same rules. The HTTP call is
injectable so the loop is tested offline with a scripted fake.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from hvac_cooling_ai.agent.guard import WITHHELD, check_answer
from hvac_cooling_ai.agent.loop import MAX_TURNS, SYSTEM_PROMPT, AgentResult
from hvac_cooling_ai.agent.tools import TOOL_SPECS, ToolSession

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"
MAX_TOKENS = 1500
TIMEOUT_S = 60
RATE_LIMIT_RETRIES = 4

Post = Callable[[dict[str, Any]], dict[str, Any]]

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]},
    }
    for t in TOOL_SPECS
]


def _http_post(payload: dict[str, Any]) -> dict[str, Any]:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("Set GROQ_API_KEY to use 'ask --provider groq'.")
    req = urllib.request.Request(
        GROQ_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            # Groq sits behind Cloudflare, which rejects the default Python urllib agent (error 1010).
            "User-Agent": "hvac-cooling-ai/0.1",
        },
    )
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            # The free tier allows a few thousand tokens a minute; wait the time Groq asks for.
            wait = re.search(r"try again in ([0-9.]+)s", detail)
            if exc.code == 429 and wait and attempt < RATE_LIMIT_RETRIES:
                time.sleep(float(wait.group(1)) + 1)
                continue
            raise RuntimeError(f"Groq API returned {exc.code}: {detail}") from exc
    raise RuntimeError("Groq API kept rate limiting the request.")


def ask(question: str, post: Post | None = None, model: str | None = None) -> AgentResult:
    post = post or _http_post
    model = model or os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
    session = ToolSession()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    retried_citation = False

    for _ in range(MAX_TURNS):
        data = post(
            {
                "model": model,
                "messages": messages,
                "tools": OPENAI_TOOLS,
                "max_tokens": MAX_TOKENS,
                "temperature": 0,
            }
        )
        choice = data["choices"][0]
        msg = choice["message"]
        finish = choice.get("finish_reason", "")
        if finish == "length":
            return AgentResult(WITHHELD, False, "response hit max_tokens", session.records, finish)
        calls = msg.get("tool_calls") or []
        messages.append(
            {
                "role": "assistant",
                "content": msg.get("content") or "",
                **({"tool_calls": calls} if calls else {}),
            }
        )
        if calls:
            for call in calls:
                fn = call["function"]
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                record = session.call(fn["name"], args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(
                            {"result_id": record.result_id, "status": record.status, **record.output}
                        ),
                    }
                )
            continue

        verdict = check_answer((msg.get("content") or "").strip(), session)
        if verdict.ok:
            return AgentResult(verdict.text, True, verdict.reason, session.records, finish)
        if retried_citation:
            return AgentResult(WITHHELD, False, verdict.reason, session.records, finish)
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

    return AgentResult(WITHHELD, False, "too many turns", session.records, "max_turns")
