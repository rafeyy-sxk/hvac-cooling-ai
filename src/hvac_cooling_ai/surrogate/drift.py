"""Drift checks for the surrogate: is it still faithful to the physics, and does real traffic look like
what it was trained on?

Two checks, both runnable in CI and from the CLI (``python -m hvac_cooling_ai drift``):

Model drift
    Sample fresh operating points inside the envelope (a seed never used in training), run the physics
    model and the saved surrogate on each, and compare. The pass/fail limits are the held-out test
    errors recorded in ``metrics.json`` times a tolerance factor, so the bar moves with the model that
    was actually shipped. This catches a stale artifact after the physics changes, a corrupted or
    swapped model file, or a library upgrade that changes predictions.

Input drift
    Given a CSV or JSON file of real incoming requests, report the share that falls outside the
    training envelope (the tools refuse those) and a population stability index (PSI) per feature
    against the exact training inputs, regenerated from the recorded seed.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from hvac_cooling_ai.envelope import ENVELOPE
from hvac_cooling_ai.surrogate import dataset
from hvac_cooling_ai.surrogate import model as surrogate
from hvac_cooling_ai.surrogate.train import SEED, supply_temp_error

DRIFT_SEED = SEED + 7001  # never used for training or for the test in tests/test_surrogate.py
DEFAULT_TOLERANCE = 3.0  # limits = recorded held-out error x this factor
PSI_BINS = 10
PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25  # the usual "significant shift" line for PSI
MIN_ROWS_FOR_PSI = 50  # below this a per-feature histogram is mostly noise
MAX_OUTSIDE_SHARE = 0.05

# Request field -> training feature. Humidity is converted to g/kg first.
FEATURE_FIELDS = {
    "t_in_c": "dry_bulb_c",
    "w_in_g_kg": "humidity",
    "velocity_m_s": "channel_velocity_m_s",
    "working_air_fraction": "working_air_fraction",
}


class Predictor(Protocol):
    def predict(self, x: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class ModelThresholds:
    supply_temp_mae_k: float
    supply_temp_p95_k: float
    supply_temp_max_k: float
    water_g_per_kg_mae: float

    @classmethod
    def from_metrics(cls, metrics: dict, tolerance: float = DEFAULT_TOLERANCE) -> ModelThresholds:
        test = metrics["test"]
        return cls(
            supply_temp_mae_k=test["supply_temp_mae_k"] * tolerance,
            supply_temp_p95_k=test["supply_temp_p95_k"] * tolerance,
            supply_temp_max_k=test["supply_temp_max_k"] * tolerance,
            water_g_per_kg_mae=test["water_g_per_kg_mae"] * tolerance,
        )


def load_metrics(path: Path = surrogate.METRICS_PATH) -> dict:
    return json.loads(Path(path).read_text())


def check_model(
    n: int = 100,
    seed: int = DRIFT_SEED,
    tolerance: float = DEFAULT_TOLERANCE,
    predictor: Predictor | None = None,
    metrics: dict | None = None,
) -> dict:
    """Surrogate vs physics on ``n`` fresh points. ``predictor`` defaults to the saved model."""
    if n < 1:
        raise ValueError("n must be at least 1")
    metrics = metrics or load_metrics()
    limits = ModelThresholds.from_metrics(metrics, tolerance)
    predictor = predictor or surrogate.load()["model"]

    x = dataset.sample_envelope(n, seed=seed)
    y = dataset.label(x)
    pred = np.asarray(predictor.predict(x))
    err_t = np.abs(supply_temp_error(x, y, pred))
    err_w = np.abs(pred[:, 1] - y[:, 1])

    observed = {
        "supply_temp_mae_k": float(err_t.mean()),
        "supply_temp_p95_k": float(np.percentile(err_t, 95)),
        "supply_temp_max_k": float(err_t.max()),
        "water_g_per_kg_mae": float(err_w.mean()),
    }
    limit_values = {
        "supply_temp_mae_k": limits.supply_temp_mae_k,
        "supply_temp_p95_k": limits.supply_temp_p95_k,
        "supply_temp_max_k": limits.supply_temp_max_k,
        "water_g_per_kg_mae": limits.water_g_per_kg_mae,
    }
    failures = [
        f"{name} {observed[name]:.4f} exceeds limit {limit_values[name]:.4f}"
        for name in observed
        if observed[name] > limit_values[name]
    ]
    return {
        "check": "model",
        "n": int(n),
        "seed": int(seed),
        "tolerance": tolerance,
        "recorded_test": {k: metrics["test"][k] for k in limit_values},
        "observed": observed,
        "limits": limit_values,
        "passed": not failures,
        "failures": failures,
    }


# ---- input drift ----


def load_requests(path: str | Path) -> list[dict[str, Any]]:
    """Read requests from .csv (header row) or .json (a list, or {"requests": [...]})."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as fh:
            return [{k: v for k, v in row.items() if v not in ("", None)} for row in csv.DictReader(fh)]
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("requests")
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of request objects or {{'requests': [...]}}")
    return data


def _features_of(rows: Iterable[dict[str, Any]]) -> tuple[np.ndarray, int, list[str]]:
    """Turn request dicts into feature rows. Returns (features, n_invalid, sample errors)."""
    from hvac_cooling_ai.agent.tools import ToolInputError, operating_point

    feats: list[tuple[float, float, float, float]] = []
    invalid = 0
    errors: list[str] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            invalid += 1
            errors.append(f"row {i}: not an object")
            continue
        try:
            feats.append(operating_point(row))
        except ToolInputError as exc:
            invalid += 1
            if len(errors) < 5:
                errors.append(f"row {i}: {exc}")
    return np.array(feats, dtype=float).reshape(-1, 4), invalid, errors


def psi(reference: np.ndarray, current: np.ndarray, bins: int = PSI_BINS) -> float:
    """Population stability index of ``current`` against ``reference``.

    Bin edges are reference quantiles, widened to cover both samples; empty bins get a small floor so
    the log term stays finite. 0 means identical histograms.
    """
    edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, bins + 1)))
    edges[0] = min(edges[0], current.min())
    edges[-1] = max(edges[-1], current.max())
    ref_share = np.histogram(reference, edges)[0] / len(reference)
    cur_share = np.histogram(current, edges)[0] / len(current)
    ref_share = np.clip(ref_share, 1e-4, None)
    cur_share = np.clip(cur_share, 1e-4, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def _psi_band(value: float) -> str:
    if value >= PSI_SIGNIFICANT:
        return "significant"
    if value >= PSI_MODERATE:
        return "moderate"
    return "stable"


def training_inputs() -> np.ndarray:
    """The exact inputs the shipped surrogate was trained and tested on (same seed and count)."""
    metrics = load_metrics()
    return dataset.sample_envelope(int(metrics["n_samples"]), seed=int(metrics["seed"]))


def check_inputs(
    rows: list[dict[str, Any]],
    reference: np.ndarray | None = None,
    max_outside_share: float = MAX_OUTSIDE_SHARE,
    psi_limit: float = PSI_SIGNIFICANT,
) -> dict:
    """Share of requests outside the envelope, plus a PSI per supplied feature."""
    x, n_invalid, errors = _features_of(rows)
    n_valid = len(x)
    outside = [bool(ENVELOPE.violations(*row)) for row in x]
    n_outside = int(sum(outside))
    share = n_outside / n_valid if n_valid else 0.0

    reference = training_inputs() if reference is None else reference
    # Only score features the requests actually supplied; defaults are a design choice, not traffic.
    supplied = {
        "t_in_c": True,
        "w_in_g_kg": True,
        "velocity_m_s": any(isinstance(r, dict) and "channel_velocity_m_s" in r for r in rows),
        "working_air_fraction": any(isinstance(r, dict) and "working_air_fraction" in r for r in rows),
    }
    features: dict[str, dict] = {}
    for col, name in enumerate(dataset.FEATURES):
        if not supplied[name]:
            features[name] = {"psi": None, "band": "not supplied (tool default used)"}
        elif n_valid < MIN_ROWS_FOR_PSI:
            features[name] = {"psi": None, "band": f"too few rows (< {MIN_ROWS_FOR_PSI})"}
        else:
            value = psi(reference[:, col], x[:, col])
            features[name] = {
                "psi": value,
                "band": _psi_band(value),
                "request_mean": float(x[:, col].mean()),
                "training_mean": float(reference[:, col].mean()),
            }

    failures = []
    if n_valid == 0:
        failures.append("no valid requests to score")
    if share > max_outside_share:
        failures.append(f"{share:.1%} of requests are outside the envelope (limit {max_outside_share:.0%})")
    for name, info in features.items():
        if info["psi"] is not None and info["psi"] > psi_limit:
            failures.append(f"{name} PSI {info['psi']:.3f} is above {psi_limit:g}")
    return {
        "check": "inputs",
        "n_rows": n_valid + n_invalid,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "invalid_examples": errors,
        "n_outside_envelope": n_outside,
        "outside_envelope_share": share,
        "features": features,
        "limits": {"max_outside_share": max_outside_share, "psi": psi_limit},
        "passed": not failures,
        "failures": failures,
    }
