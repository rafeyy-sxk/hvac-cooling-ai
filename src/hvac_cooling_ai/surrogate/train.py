"""Train the fast surrogate and write the artifact + metrics.

    python -m hvac_cooling_ai.surrogate.train

Protocol
  1. Sample N points uniformly in the envelope, label them with the physics model.
  2. Cut the envelope into 4-D blocks (dataset.region_groups). Hold out 20% of
     the blocks as the TEST set. Nothing from a test block is ever trained on.
  3. From the remaining blocks hold out another 20% of blocks as VALIDATION and
     pick the model family on it (MLP vs gradient boosting).
  4. Refit the chosen family on train + validation, score once on TEST.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hvac_cooling_ai.envelope import ENVELOPE
from hvac_cooling_ai.physics import psychro
from hvac_cooling_ai.surrogate import dataset
from hvac_cooling_ai.surrogate.model import ARTIFACT_DIR, ARTIFACT_PATH, METRICS_PATH

SEED = 20260926


def _candidates() -> dict:
    return {
        "mlp": make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 64),
                activation="tanh",
                max_iter=3000,
                learning_rate_init=3e-3,
                tol=1e-7,
                n_iter_no_change=50,
                random_state=SEED,
            ),
        ),
        "gbr": MultiOutputRegressor(
            HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, random_state=SEED)
        ),
    }


def supply_temp_error(x: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Error in supply temperature (K) implied by an error in eps_wb."""
    t_in = x[:, 0]
    t_wb = psychro.wet_bulb(t_in, x[:, 1] / 1000.0)
    return (y_pred[:, 0] - y_true[:, 0]) * (t_in - t_wb)


def _score(x: np.ndarray, y: np.ndarray, pred: np.ndarray) -> dict:
    err_t = np.abs(supply_temp_error(x, y, pred))
    err_w = np.abs(pred[:, 1] - y[:, 1])
    return {
        "n": int(len(x)),
        "supply_temp_mae_k": float(err_t.mean()),
        "supply_temp_p95_k": float(np.percentile(err_t, 95)),
        "supply_temp_max_k": float(err_t.max()),
        "eps_wb_mae": float(np.abs(pred[:, 0] - y[:, 0]).mean()),
        "water_g_per_kg_mae": float(err_w.mean()),
        "water_g_per_kg_max": float(err_w.max()),
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n", type=int, default=6000, help="number of physics samples")
    args = parser.parse_args(argv)

    t0 = time.time()
    x = dataset.sample_envelope(args.n, seed=SEED)
    y = dataset.label(x)
    label_seconds = time.time() - t0
    groups = dataset.region_groups(x)

    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    trainval_idx, test_idx = next(outer.split(x, y, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED + 1)
    tr_rel, va_rel = next(inner.split(x[trainval_idx], y[trainval_idx], groups[trainval_idx]))
    tr_idx, va_idx = trainval_idx[tr_rel], trainval_idx[va_rel]
    assert not set(groups[test_idx]) & set(groups[trainval_idx]), "test blocks leaked into training"

    validation = {}
    for name, est in _candidates().items():
        est.fit(x[tr_idx], y[tr_idx])
        validation[name] = _score(x[va_idx], y[va_idx], est.predict(x[va_idx]))
    chosen = min(validation, key=lambda k: validation[k]["supply_temp_mae_k"])

    model = _candidates()[chosen]
    model.fit(x[trainval_idx], y[trainval_idx])
    t1 = time.time()
    test_pred = model.predict(x[test_idx])
    predict_seconds = time.time() - t1
    test = _score(x[test_idx], y[test_idx], test_pred)

    metrics = {
        "seed": SEED,
        "n_samples": int(args.n),
        "n_region_blocks_total": int(len(set(groups))),
        "n_region_blocks_test": int(len(set(groups[test_idx]))),
        "physics_label_seconds": round(label_seconds, 2),
        "surrogate_predict_seconds_for_test_set": round(predict_seconds, 4),
        "validation": validation,
        "chosen_model": chosen,
        "test": test,
        "features": list(dataset.FEATURES),
        "targets": list(dataset.TARGETS),
        "envelope": ENVELOPE.to_dict(),
        "sklearn_version": sklearn.__version__,
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metrics": metrics}, ARTIFACT_PATH, compress=3)
    Path(METRICS_PATH).write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({"chosen_model": chosen, "validation": validation, "test": test}, indent=2))
    return metrics


if __name__ == "__main__":
    main()
