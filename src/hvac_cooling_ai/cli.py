"""Command line: python -m hvac_cooling_ai {simulate,predict,size,psychro,ask,validate,envelope,drift}."""

from __future__ import annotations

import argparse
import json
import sys

from hvac_cooling_ai.agent.tools import ToolSession


def _add_conditions(p: argparse.ArgumentParser) -> None:
    p.add_argument("--t", type=float, required=True, help="inlet dry-bulb, deg C")
    hum = p.add_mutually_exclusive_group(required=True)
    hum.add_argument("--rh", type=float, help="relative humidity, percent")
    hum.add_argument("--w", type=float, help="humidity ratio, g/kg")


def _add_design(p: argparse.ArgumentParser) -> None:
    p.add_argument("--velocity", type=float, default=None, help="dry-channel velocity, m/s (default 1.0)")
    p.add_argument("--r", type=float, default=None, help="working-air fraction (default 0.333)")


def _conditions(ns: argparse.Namespace) -> dict:
    args = {"dry_bulb_c": ns.t, "relative_humidity_pct": ns.rh, "humidity_ratio_g_kg": ns.w}
    if getattr(ns, "velocity", None) is not None:
        args["channel_velocity_m_s"] = ns.velocity
    if getattr(ns, "r", None) is not None:
        args["working_air_fraction"] = ns.r
    return args


def _run_tool(name: str, args: dict) -> int:
    record = ToolSession().call(name, args)
    print(json.dumps({"status": record.status, **record.output}, indent=2))
    return 0 if record.status == "ok" else 2


def _drift(ns: argparse.Namespace) -> int:
    from hvac_cooling_ai.surrogate import drift

    if ns.inputs:
        report = drift.check_inputs(drift.load_requests(ns.inputs))
    else:
        report = drift.check_model(
            n=ns.n,
            seed=drift.DRIFT_SEED if ns.seed is None else ns.seed,
            tolerance=drift.DEFAULT_TOLERANCE if ns.tolerance is None else ns.tolerance,
        )
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hvac_cooling_ai", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("simulate", help="physics model for one condition")
    _add_conditions(p)
    _add_design(p)
    p.add_argument("--pairs", type=int, default=1, help="channel pairs stacked (default 1)")

    p = sub.add_parser("predict", help="fast surrogate for one condition")
    _add_conditions(p)
    _add_design(p)

    p = sub.add_parser("size", help="channel pairs needed for a room sensible load")
    _add_conditions(p)
    _add_design(p)
    p.add_argument("--load-kw", type=float, required=True)
    p.add_argument("--room", type=float, default=26.0, help="room temperature to hold, deg C")

    p = sub.add_parser("psychro", help="moist-air properties")
    _add_conditions(p)

    p = sub.add_parser("ask", help="ask the AI assistant (Claude, or Groq with --provider groq)")
    p.add_argument("question")
    p.add_argument("--provider", choices=["claude", "groq"], default="claude")
    p.add_argument("--show-tools", action="store_true", help="print the tool calls and results")
    p.add_argument(
        "--show-usage", action="store_true", help="print latency, tokens and estimated cost per LLM call"
    )

    sub.add_parser("validate", help="compare against the source report's numbers")
    sub.add_parser("envelope", help="print the validated operating envelope")

    p = sub.add_parser(
        "drift",
        help="surrogate vs physics on fresh points, or input drift for a file of requests (--inputs)",
    )
    p.add_argument("--n", type=int, default=100, help="fresh operating points to check (default 100)")
    p.add_argument(
        "--seed", type=int, default=None, help="sampling seed (default: a seed unused in training)"
    )
    p.add_argument(
        "--tolerance", type=float, default=None, help="limits = recorded test error x this (default 3)"
    )
    p.add_argument("--inputs", default=None, help="CSV or JSON of real requests: check input drift instead")

    ns = parser.parse_args(argv)

    if ns.command == "simulate":
        return _run_tool("simulate_cooler", {**_conditions(ns), "channel_pairs": ns.pairs})
    if ns.command == "predict":
        return _run_tool("predict_cooler_fast", _conditions(ns))
    if ns.command == "size":
        return _run_tool(
            "units_needed", {**_conditions(ns), "sensible_load_kw": ns.load_kw, "room_temp_c": ns.room}
        )
    if ns.command == "psychro":
        return _run_tool("psychrometric_lookup", _conditions(ns))
    if ns.command == "envelope":
        from hvac_cooling_ai.envelope import ENVELOPE

        print(json.dumps(ENVELOPE.to_dict(), indent=2))
        return 0
    if ns.command == "validate":
        from hvac_cooling_ai import validation

        validation.main()
        return 0
    if ns.command == "drift":
        return _drift(ns)
    if ns.command == "ask":
        if ns.provider == "groq":
            from hvac_cooling_ai.agent.groq_loop import ask
        else:
            from hvac_cooling_ai.agent.loop import ask

        try:
            result = ask(ns.question)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if ns.show_tools:
            for rec in result.tool_calls:
                print(f"[{rec.result_id}] {rec.name}({json.dumps(rec.args)}) -> {rec.status}")
                print("    " + json.dumps(rec.output))
            print()
        print(result.answer)
        if ns.show_usage:
            print()
            for line in result.usage.lines():
                print(line)
        return 0 if result.grounded else 3
    return 1


if __name__ == "__main__":
    sys.exit(main())
