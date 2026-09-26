"""Load the trained surrogate and predict intensive performance quickly."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from hvac_cooling_ai.envelope import ENVELOPE
from hvac_cooling_ai.physics import psychro

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "surrogate.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"


class OutsideEnvelopeError(ValueError):
    """Raised when a prediction is requested outside the trained envelope."""


@lru_cache(maxsize=1)
def load() -> dict:
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"No surrogate at {ARTIFACT_PATH}. Train one with: python -m hvac_cooling_ai.surrogate.train"
        )
    return joblib.load(ARTIFACT_PATH)


def predict(t_in_c: float, w_in_g_kg: float, velocity_m_s: float, working_air_fraction: float) -> dict:
    """Surrogate prediction for one operating point (refuses outside the envelope)."""
    problems = ENVELOPE.violations(t_in_c, w_in_g_kg, velocity_m_s, working_air_fraction)
    if problems:
        raise OutsideEnvelopeError("; ".join(problems))
    bundle = load()
    x = np.array([[t_in_c, w_in_g_kg, velocity_m_s, working_air_fraction]])
    eps_wb, water_g_per_kg = bundle["model"].predict(x)[0]
    t_wb = float(psychro.wet_bulb(t_in_c, w_in_g_kg / 1000.0))
    t_dp = float(psychro.dew_point(w_in_g_kg / 1000.0))
    t_supply = t_in_c - float(eps_wb) * (t_in_c - t_wb)
    # Physical guard: a regenerative cooler cannot deliver air below the inlet dew point.
    t_supply = max(t_supply, t_dp)
    return {
        "t_supply_c": t_supply,
        "eps_wb": (t_in_c - t_supply) / (t_in_c - t_wb),
        "eps_dp": (t_in_c - t_supply) / (t_in_c - t_dp),
        "t_wet_bulb_in_c": t_wb,
        "t_dew_point_in_c": t_dp,
        "water_g_per_kg_supply": max(float(water_g_per_kg), 0.0),
        "test_supply_temp_mae_k": bundle["metrics"]["test"]["supply_temp_mae_k"],
        "test_supply_temp_max_k": bundle["metrics"]["test"]["supply_temp_max_k"],
    }
