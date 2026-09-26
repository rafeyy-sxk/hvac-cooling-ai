"""Generate a training set by sweeping the physics model over the envelope."""

from __future__ import annotations

import numpy as np

from hvac_cooling_ai.envelope import ENVELOPE, Envelope
from hvac_cooling_ai.physics import iec, psychro

FEATURES = ("t_in_c", "w_in_g_kg", "velocity_m_s", "working_air_fraction")
# Targets are intensive (independent of how many channel pairs are stacked):
#   eps_wb: wet-bulb effectiveness -> supply temperature
#   water_g_per_kg_supply: evaporated water per kg of delivered supply air
TARGETS = ("eps_wb", "water_g_per_kg_supply")


def sample_envelope(n: int, seed: int, env: Envelope = ENVELOPE) -> np.ndarray:
    """Uniform random points inside the envelope (rejection on max RH)."""
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    count = 0
    while count < n:
        m = 2 * (n - count) + 16
        t = rng.uniform(*env.t_in_c, m)
        w = rng.uniform(*env.w_in_g_kg, m)
        v = rng.uniform(*env.velocity_m_s, m)
        r = rng.uniform(*env.working_air_fraction, m)
        ok = psychro.relative_humidity(t, w / 1000.0) <= env.rh_max
        block = np.column_stack([t, w, v, r])[ok]
        rows.append(block)
        count += len(block)
    return np.vstack(rows)[:n]


def label(x: np.ndarray, chunk: int = 1000, grid: iec.GridSpec | None = None) -> np.ndarray:
    """Run the physics model on feature rows; returns the TARGETS columns."""
    out = []
    for start in range(0, len(x), chunk):
        part = x[start : start + chunk]
        res = iec.simulate_batch(part[:, 0], part[:, 1] / 1000.0, part[:, 2], part[:, 3], grid=grid)
        if not np.all(res["converged"]):
            raise RuntimeError("physics model did not converge for part of the batch")
        water_g_per_kg = (res["water_evaporated_kg_h"] / 3600.0) / res["supply_flow_kg_s"] * 1000.0
        out.append(np.column_stack([res["eps_wb"], water_g_per_kg]))
    return np.vstack(out)


def region_groups(
    x: np.ndarray, env: Envelope = ENVELOPE, bins: tuple[int, ...] = (5, 5, 4, 3)
) -> np.ndarray:
    """Assign each point to a block of the envelope (a 4-D grid of regions).

    Splitting by block keeps whole regions out of training, so the test score
    measures interpolation into unseen regions rather than near-duplicates.
    """
    bounds = (env.t_in_c, env.w_in_g_kg, env.velocity_m_s, env.working_air_fraction)
    idx = np.zeros(len(x), dtype=int)
    for col, ((lo, hi), nb) in enumerate(zip(bounds, bins, strict=True)):
        b = np.clip(((x[:, col] - lo) / (hi - lo) * nb).astype(int), 0, nb - 1)
        idx = idx * nb + b
    return idx
