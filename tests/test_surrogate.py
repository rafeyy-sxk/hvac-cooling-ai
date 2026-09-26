"""The saved surrogate against fresh physics runs it has never seen."""

import numpy as np
import pytest

from hvac_cooling_ai.physics import psychro
from hvac_cooling_ai.surrogate import dataset
from hvac_cooling_ai.surrogate import model as surrogate
from hvac_cooling_ai.surrogate.train import SEED

# Bounds set with margin above the measured held-out error (MAE 0.047 K, max 0.36 K).
MAE_BOUND_K = 0.15
MAX_BOUND_K = 1.0


def test_artifact_metadata():
    bundle = surrogate.load()
    metrics = bundle["metrics"]
    assert metrics["features"] == list(dataset.FEATURES)
    assert metrics["test"]["supply_temp_mae_k"] < MAE_BOUND_K
    assert metrics["n_region_blocks_test"] > 0


def test_fresh_points_within_error_bound():
    x = dataset.sample_envelope(60, seed=SEED + 999)  # different seed from training
    y = dataset.label(x)
    pred = surrogate.load()["model"].predict(x)
    t_wb = psychro.wet_bulb(x[:, 0], x[:, 1] / 1000.0)
    err = np.abs((pred[:, 0] - y[:, 0]) * (x[:, 0] - t_wb))
    assert err.mean() < MAE_BOUND_K
    assert err.max() < MAX_BOUND_K


def test_region_groups_split_blocks_cleanly():
    x = dataset.sample_envelope(500, seed=1)
    g = dataset.region_groups(x)
    assert g.min() >= 0 and g.max() < 5 * 5 * 4 * 3
    # the same point always lands in the same block
    np.testing.assert_array_equal(g, dataset.region_groups(x.copy()))


def test_predict_refuses_outside_envelope():
    with pytest.raises(surrogate.OutsideEnvelopeError, match="outside the validated range"):
        surrogate.predict(55.0, 10.0, 1.0, 0.33)
    with pytest.raises(surrogate.OutsideEnvelopeError, match="relative humidity"):
        surrogate.predict(25.0, 19.0, 1.0, 0.33)


def test_predict_respects_dew_point_floor():
    out = surrogate.predict(35.0, 12.0, 0.5, 0.6)
    assert out["t_dew_point_in_c"] <= out["t_supply_c"] < 35.0
