"""Grounding guard applied to every final answer before the user sees it.

Rules
  1. The answer must cite at least one tool result id, like [R2].
  2. Every cited id must exist and must be a successful ("ok") result.
  3. If a cooler-performance tool was refused for being outside the envelope
     and no cooler-performance tool succeeded, the answer is replaced by a
     refusal that states why. The model is not allowed to guess past the
     validated envelope.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from hvac_cooling_ai.agent.tools import COOLER_TOOLS, ToolSession

CITATION = re.compile(r"\[(R\d+)\]")


@dataclass(frozen=True)
class GuardVerdict:
    ok: bool
    text: str
    reason: str = ""


def envelope_refusal_text(session: ToolSession) -> str | None:
    """A refusal message if the question can only be answered outside the envelope."""
    cooler = [r for r in session.records if r.name in COOLER_TOOLS]
    refused = [r for r in cooler if r.status == "refused"]
    if refused and not any(r.status == "ok" for r in cooler):
        reasons = sorted({reason for r in refused for reason in r.output["reasons"]})
        return (
            "I can't answer that reliably: the conditions are outside the range this cooler model was "
            "validated for (" + "; ".join(reasons) + "). "
            f"[{refused[-1].result_id}] refused the request rather than extrapolate."
        )
    return None


def check_answer(text: str, session: ToolSession) -> GuardVerdict:
    refusal = envelope_refusal_text(session)
    if refusal is not None:
        return GuardVerdict(ok=True, text=refusal, reason="outside validated envelope")
    cited = set(CITATION.findall(text))
    if not cited:
        return GuardVerdict(ok=False, text=text, reason="answer cites no tool result")
    missing = sorted(c for c in cited if session.by_id(c) is None)
    if missing:
        return GuardVerdict(ok=False, text=text, reason=f"answer cites unknown results {missing}")
    not_ok = sorted(c for c in cited if c not in session.ok_ids())
    if not_ok:
        return GuardVerdict(ok=False, text=text, reason=f"answer cites failed or refused results {not_ok}")
    return GuardVerdict(ok=True, text=text)


WITHHELD = (
    "Answer withheld: the assistant did not ground its answer in a successful tool result, "
    "so it cannot be trusted. Try rephrasing with the outdoor temperature and humidity."
)
