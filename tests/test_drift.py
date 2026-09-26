"""Drift checks: the shipped surrogate passes, and planted drift of every kind trips the check."""

import csv
import json

import numpy as np
import pytest

from hvac_cooling_ai.cli import main
from hvac_cooling_ai.surrogate import dataset, drift
from hvac_cooling_ai.surrogate import model as surrogate

N = 40  # small, like CI


class Biased:
    """The saved model with a constant error added to its wet-bulb effectiveness."""

    def __init__(self, eps_offset: float):
        self.inner = surrogate.load()["model"]
        self.eps_offset = eps_offset

    def predict(self, x):
        pred = np.array(self.inner.predict(x), dtype=float)
        pred[:, 0] += self.eps_offset
        return pred


def _rows(x: np.ndarray, design: bool = True) -> list[dict]:
    rows = []
    for t, w, v, r in x:
        row = {"dry_bulb_c": float(t), "humidity_ratio_g_kg": float(w)}
        if design:
            row |= {"channel_velocity_m_s": float(v), "working_air_fraction": float(r)}
        rows.append(row)
    return rows


# ---- model drift ----


def test_limits_come_from_recorded_test_metrics():
    metrics = drift.load_metrics()
    limits = drift.ModelThresholds.from_metrics(metrics, tolerance=3.0)
    assert limits.supply_temp_mae_k == pytest.approx(3.0 * metrics["test"]["supply_temp_mae_k"])
    assert limits.supply_temp_max_k == pytest.approx(3.0 * metrics["test"]["supply_temp_max_k"])


def test_shipped_surrogate_passes_on_fresh_points():
    report = drift.check_model(n=N)
    assert report["passed"], report["failures"]
    assert report["observed"]["supply_temp_mae_k"] < report["limits"]["supply_temp_mae_k"]


def test_biased_model_trips_the_check():
    # 0.02 of wet-bulb effectiveness is about 0.2 K of supply temperature on a typical point
    report = drift.check_model(n=N, predictor=Biased(0.02))
    assert not report["passed"]
    assert any(f.startswith("supply_temp_mae_k") for f in report["failures"])


def test_changed_physics_trips_the_check(monkeypatch):
    """A surrogate left stale after the physics changes (here: effectiveness 3% lower) is caught."""
    real_label = dataset.label

    def changed_physics(x, *args, **kwargs):
        y = real_label(x, *args, **kwargs)
        y[:, 0] *= 0.97
        return y

    monkeypatch.setattr(drift.dataset, "label", changed_physics)
    assert not drift.check_model(n=N)["passed"]


def test_cli_drift_exit_codes(capsys):
    assert main(["drift", "--n", "20"]) == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True
    assert main(["drift", "--n", "20", "--tolerance", "0.05"]) == 4


# ---- input drift ----


def test_psi_is_zero_for_identical_samples_and_grows_with_shift():
    ref = drift.training_inputs()[:, 0]
    assert drift.psi(ref, ref) == pytest.approx(0.0, abs=1e-9)
    assert drift.psi(ref, ref + 2.0) < drift.psi(ref, ref + 6.0)


def test_training_like_requests_pass():
    x = dataset.sample_envelope(500, seed=12345)
    report = drift.check_inputs(_rows(x))
    assert report["passed"], report["failures"]
    assert report["n_outside_envelope"] == 0
    assert all(f["psi"] < drift.PSI_MODERATE for f in report["features"].values())


def test_hotter_requests_trip_the_check():
    x = dataset.sample_envelope(500, seed=12345)
    x[:, 0] += 8.0  # a heat wave: 8 C hotter than anything in the training mix
    report = drift.check_inputs(_rows(x))
    assert not report["passed"]
    assert report["outside_envelope_share"] > drift.MAX_OUTSIDE_SHARE
    assert report["features"]["t_in_c"]["band"] == "significant"
    assert report["features"]["w_in_g_kg"]["band"] == "stable"


def test_concentrated_humidity_trips_psi_even_inside_envelope():
    rng = np.random.default_rng(3)
    x = dataset.sample_envelope(500, seed=12345)
    x[:, 1] = rng.uniform(5.0, 7.0, len(x))  # every request is dry: inside the envelope, but shifted
    report = drift.check_inputs(_rows(x))
    assert report["n_outside_envelope"] == 0
    assert report["features"]["w_in_g_kg"]["band"] == "significant"
    assert not report["passed"]


def test_defaults_are_not_scored_and_bad_rows_are_counted():
    x = dataset.sample_envelope(200, seed=7)
    rows = _rows(x, design=False) + [{"dry_bulb_c": 40}, {"dry_bulb_c": "hot", "relative_humidity_pct": 20}]
    report = drift.check_inputs(rows)
    assert report["n_invalid"] == 2 and report["n_valid"] == 200
    assert report["features"]["velocity_m_s"]["psi"] is None
    assert "not supplied" in report["features"]["working_air_fraction"]["band"]


def test_few_rows_get_no_psi():
    report = drift.check_inputs([{"dry_bulb_c": 40, "relative_humidity_pct": 20}] * 10)
    assert report["features"]["t_in_c"]["psi"] is None and report["passed"]


def test_load_csv_and_json_and_cli(tmp_path, capsys):
    x = dataset.sample_envelope(300, seed=99)
    x[:, 0] += 8.0
    rows = _rows(x)
    csv_path = tmp_path / "requests.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = tmp_path / "requests.json"
    json_path.write_text(json.dumps({"requests": rows}))

    assert len(drift.load_requests(csv_path)) == 300
    assert drift.load_requests(json_path) == rows
    assert main(["drift", "--inputs", str(csv_path)]) == 4
    assert json.loads(capsys.readouterr().out)["check"] == "inputs"
