"""Tools the assistant can call. Plain Python: usable and tested with no API key.

Every call goes through ``ToolSession.call``, which validates input, runs the
tool, stamps the output with a result id (R1, R2, ...) and records it. The
assistant must cite those ids; ``guard.py`` checks that it did.

Scale note: the physics describes one repeating channel pair (one dry + one wet
channel, 200 x 200 mm plates, 4 mm gaps). Capacity scales linearly with the
number of pairs stacked, so sizing answers are given in channel pairs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from hvac_cooling_ai.envelope import ENVELOPE
from hvac_cooling_ai.physics import iec, psychro
from hvac_cooling_ai.surrogate import model as surrogate

DEFAULT_VELOCITY = 1.0  # m/s in the dry channel
DEFAULT_WORKING_AIR_FRACTION = 1.0 / 3.0
FAN_POWER_PER_PAIR_W = 0.5  # the source report's electrical input for one channel
M3_S_TO_CFM = 2118.88
COOLER_TOOLS = frozenset({"simulate_cooler", "predict_cooler_fast", "units_needed"})


class ToolInputError(ValueError):
    """Bad or missing tool arguments (reported back to the model as an error)."""


def _num(args: dict, key: str, default: float | None = None) -> float:
    if key not in args or args[key] is None:
        if default is None:
            raise ToolInputError(f"missing required argument '{key}'")
        return default
    try:
        value = float(args[key])
    except (TypeError, ValueError) as exc:
        raise ToolInputError(f"argument '{key}' must be a number, got {args[key]!r}") from exc
    if not math.isfinite(value):
        raise ToolInputError(f"argument '{key}' must be finite")
    return value


def _humidity_g_kg(args: dict) -> float:
    """Accept exactly one of relative_humidity_pct / humidity_ratio_g_kg."""
    has_rh = args.get("relative_humidity_pct") is not None
    has_w = args.get("humidity_ratio_g_kg") is not None
    if has_rh == has_w:
        raise ToolInputError("give exactly one of 'relative_humidity_pct' or 'humidity_ratio_g_kg'")
    t = _num(args, "dry_bulb_c")
    if has_rh:
        rh = _num(args, "relative_humidity_pct")
        if not 0.0 < rh <= 100.0:
            raise ToolInputError("relative_humidity_pct must be in (0, 100]")
        return float(psychro.humidity_ratio(t, rh / 100.0)) * 1000.0
    w = _num(args, "humidity_ratio_g_kg")
    if w <= 0.0:
        raise ToolInputError("humidity_ratio_g_kg must be positive")
    return w


def psychrometric_lookup(args: dict) -> dict:
    t = _num(args, "dry_bulb_c")
    if not -20.0 <= t <= 70.0:
        raise ToolInputError("dry_bulb_c must be between -20 and 70 C for this lookup")
    w_g = _humidity_g_kg(args)
    w = w_g / 1000.0
    rh = float(psychro.relative_humidity(t, w))
    if rh > 1.0 + 1e-9:
        raise ToolInputError(f"that humidity is above saturation at {t:g} C (RH would be {rh:.0%})")
    return {
        "dry_bulb_c": t,
        "relative_humidity_pct": rh * 100.0,
        "humidity_ratio_g_kg": w_g,
        "wet_bulb_c": float(psychro.wet_bulb(t, w)),
        "dew_point_c": float(psychro.dew_point(w)),
        "enthalpy_kj_kg": float(psychro.enthalpy(t, w)) / 1000.0,
        "pressure_pa": psychro.P_ATM,
    }


def operating_point(args: dict) -> tuple[float, float, float, float]:
    """(dry-bulb C, humidity g/kg, velocity m/s, working-air fraction) from tool-style arguments."""
    t = _num(args, "dry_bulb_c")
    w_g = _humidity_g_kg(args)
    v = _num(args, "channel_velocity_m_s", DEFAULT_VELOCITY)
    r = _num(args, "working_air_fraction", DEFAULT_WORKING_AIR_FRACTION)
    return t, w_g, v, r


class EnvelopeRefusalError(Exception):
    def __init__(self, reasons: list[str]):
        super().__init__("; ".join(reasons))
        self.reasons = reasons


def _check_envelope(t: float, w_g: float, v: float, r: float) -> None:
    problems = ENVELOPE.violations(t, w_g, v, r)
    if problems:
        raise EnvelopeRefusalError(problems)


def _per_pair_flows(v: float, r: float) -> tuple[float, float]:
    geo = iec.Geometry()
    m_primary = iec.RHO_AIR * v * geo.channel_gap_m * geo.wet_length_m
    return (1.0 - r) * m_primary, r * m_primary


def simulate_cooler(args: dict) -> dict:
    t, w_g, v, r = operating_point(args)
    _check_envelope(t, w_g, v, r)
    pairs = int(_num(args, "channel_pairs", 1))
    if pairs < 1:
        raise ToolInputError("channel_pairs must be at least 1")
    res = iec.simulate(t, w_g / 1000.0, v, r, fan_power_w=FAN_POWER_PER_PAIR_W)
    return {
        "method": "physics (cell-by-cell epsilon-NTU, 20x20 grid)",
        "inputs": {
            "dry_bulb_c": t,
            "humidity_ratio_g_kg": w_g,
            "channel_velocity_m_s": v,
            "working_air_fraction": r,
            "channel_pairs": pairs,
        },
        "supply_temp_c": res["t_supply_c"],
        "inlet_wet_bulb_c": res["t_wet_bulb_in_c"],
        "inlet_dew_point_c": res["t_dew_point_in_c"],
        "wet_bulb_effectiveness": res["eps_wb"],
        "dew_point_effectiveness": res["eps_dp"],
        "supply_flow_kg_s": res["supply_flow_kg_s"] * pairs,
        "cooling_vs_inlet_w": res["cooling_capacity_w"] * pairs,
        "water_evaporated_l_h": res["water_evaporated_kg_h"] * pairs,
        "fan_power_w_assumed": FAN_POWER_PER_PAIR_W * pairs,
        "cop": res["cop"],
    }


def predict_cooler_fast(args: dict) -> dict:
    t, w_g, v, r = operating_point(args)
    _check_envelope(t, w_g, v, r)
    pred = surrogate.predict(t, w_g, v, r)
    return {
        "method": "surrogate (gradient-boosted model trained on the physics)",
        "inputs": {
            "dry_bulb_c": t,
            "humidity_ratio_g_kg": w_g,
            "channel_velocity_m_s": v,
            "working_air_fraction": r,
        },
        "supply_temp_c": pred["t_supply_c"],
        "inlet_wet_bulb_c": pred["t_wet_bulb_in_c"],
        "inlet_dew_point_c": pred["t_dew_point_in_c"],
        "wet_bulb_effectiveness": pred["eps_wb"],
        "water_g_per_kg_supply": pred["water_g_per_kg_supply"],
        "held_out_supply_temp_mae_k": pred["test_supply_temp_mae_k"],
        "held_out_supply_temp_max_error_k": pred["test_supply_temp_max_k"],
    }


def units_needed(args: dict) -> dict:
    load_kw = _num(args, "sensible_load_kw")
    if not 0.0 < load_kw <= 1000.0:
        raise ToolInputError("sensible_load_kw must be in (0, 1000]")
    room = _num(args, "room_temp_c", 26.0)
    if not 18.0 <= room <= 32.0:
        raise ToolInputError("room_temp_c must be between 18 and 32 C")
    t, w_g, v, r = operating_point(args)
    _check_envelope(t, w_g, v, r)
    res = iec.simulate(t, w_g / 1000.0, v, r, fan_power_w=FAN_POWER_PER_PAIR_W)
    m_supply, _ = _per_pair_flows(v, r)
    cp = float(psychro.moist_cp(w_g / 1000.0))
    per_pair_room_w = m_supply * cp * (room - res["t_supply_c"])
    base = {
        "method": "physics, room sensible capacity = supply flow x cp x (room temp - supply temp)",
        "inputs": {
            "sensible_load_kw": load_kw,
            "room_temp_c": room,
            "dry_bulb_c": t,
            "humidity_ratio_g_kg": w_g,
            "channel_velocity_m_s": v,
            "working_air_fraction": r,
        },
        "supply_temp_c": res["t_supply_c"],
        "room_capacity_per_channel_pair_w": per_pair_room_w,
    }
    if per_pair_room_w <= 0.0:
        return {
            **base,
            "feasible": False,
            "reason": "supply air is not colder than the room, so this cooler cannot remove room load",
        }
    pairs = math.ceil(load_kw * 1000.0 / per_pair_room_w)
    geo = iec.Geometry()
    supply_m3_s = pairs * m_supply / iec.RHO_AIR
    return {
        **base,
        "feasible": True,
        "channel_pairs_needed": pairs,
        "plate_size_mm": [geo.dry_length_m * 1000, geo.wet_length_m * 1000],
        "stack_height_m_approx": pairs * 2 * geo.channel_gap_m,
        "supply_airflow_cfm": supply_m3_s * M3_S_TO_CFM,
        "water_evaporated_l_h": res["water_evaporated_kg_h"] * pairs,
        "fan_power_w_assumed": FAN_POWER_PER_PAIR_W * pairs,
    }


_HUMIDITY_PROPS = {
    "relative_humidity_pct": {"type": "number", "description": "Relative humidity in percent (0-100)."},
    "humidity_ratio_g_kg": {"type": "number", "description": "Humidity ratio, g water per kg dry air."},
}
_OPERATING_PROPS = {
    "dry_bulb_c": {"type": "number", "description": "Outdoor / inlet air dry-bulb temperature, deg C."},
    **_HUMIDITY_PROPS,
    "channel_velocity_m_s": {
        "type": "number",
        "description": f"Dry-channel air velocity, m/s. Default {DEFAULT_VELOCITY}.",
    },
    "working_air_fraction": {
        "type": "number",
        "description": "Share of product air diverted to the wet side. Default 0.333.",
    },
}

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "psychrometric_lookup",
        "description": "Moist-air properties (wet bulb, dew point, humidity ratio, enthalpy) at sea-level "
        "pressure. Give dry_bulb_c and exactly one of relative_humidity_pct or "
        "humidity_ratio_g_kg.",
        "input_schema": {
            "type": "object",
            "properties": {"dry_bulb_c": _OPERATING_PROPS["dry_bulb_c"], **_HUMIDITY_PROPS},
            "required": ["dry_bulb_c"],
        },
    },
    {
        "name": "simulate_cooler",
        "description": "Run the physics model of the crossflow regenerative indirect evaporative cooler "
        "for an outdoor condition. Returns supply air temperature and performance. Refuses "
        "conditions outside the validated envelope.",
        "input_schema": {
            "type": "object",
            "properties": {
                **_OPERATING_PROPS,
                "channel_pairs": {
                    "type": "integer",
                    "description": "Number of stacked channel pairs. Default 1.",
                },
            },
            "required": ["dry_bulb_c"],
        },
    },
    {
        "name": "predict_cooler_fast",
        "description": "Fast ML surrogate of the same cooler (milliseconds). Returns supply temperature "
        "plus the surrogate's measured held-out error. Refuses outside the envelope.",
        "input_schema": {"type": "object", "properties": _OPERATING_PROPS, "required": ["dry_bulb_c"]},
    },
    {
        "name": "units_needed",
        "description": "Size the cooler for a room sensible load: how many channel pairs (200x200 mm "
        "plates) are needed, with airflow, water use and fan power. Refuses outside the "
        "envelope and reports infeasible when supply air is not colder than the room.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sensible_load_kw": {"type": "number", "description": "Room sensible cooling load, kW."},
                "room_temp_c": {
                    "type": "number",
                    "description": "Room temperature to hold, deg C. Default 26.",
                },
                **_OPERATING_PROPS,
            },
            "required": ["sensible_load_kw", "dry_bulb_c"],
        },
    },
]

_FUNCTIONS = {
    "psychrometric_lookup": psychrometric_lookup,
    "simulate_cooler": simulate_cooler,
    "predict_cooler_fast": predict_cooler_fast,
    "units_needed": units_needed,
}


@dataclass
class ToolRecord:
    result_id: str
    name: str
    args: dict
    status: str  # "ok" | "refused" | "error"
    output: dict


@dataclass
class ToolSession:
    """Runs tools, numbers their results and keeps the audit trail."""

    records: list[ToolRecord] = field(default_factory=list)

    def call(self, name: str, args: dict) -> ToolRecord:
        result_id = f"R{len(self.records) + 1}"
        fn = _FUNCTIONS.get(name)
        if fn is None:
            record = ToolRecord(result_id, name, args, "error", {"error": f"unknown tool '{name}'"})
        else:
            try:
                record = ToolRecord(result_id, name, args, "ok", fn(dict(args)))
            except EnvelopeRefusalError as exc:
                record = ToolRecord(
                    result_id,
                    name,
                    args,
                    "refused",
                    {
                        "refused": True,
                        "reasons": exc.reasons,
                        "validated_envelope": ENVELOPE.to_dict(),
                    },
                )
            except ToolInputError as exc:
                record = ToolRecord(result_id, name, args, "error", {"error": str(exc)})
        self.records.append(record)
        return record

    def ok_ids(self) -> set[str]:
        return {r.result_id for r in self.records if r.status == "ok"}

    def by_id(self, result_id: str) -> ToolRecord | None:
        return next((r for r in self.records if r.result_id == result_id), None)
