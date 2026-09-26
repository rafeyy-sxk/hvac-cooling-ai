"""Our psychrometrics against PsychroLib, an independent ASHRAE implementation."""

import numpy as np
import psychrolib
import pytest

from hvac_cooling_ai.physics import psychro

psychrolib.SetUnitSystem(psychrolib.SI)
P = 101325.0


@pytest.mark.parametrize("t", [0.5, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
def test_saturation_pressure_matches_psychrolib(t):
    assert float(psychro.sat_pressure(t)) == pytest.approx(psychrolib.GetSatVapPres(t), rel=1e-6)


@pytest.mark.parametrize(("t", "rh"), [(25.0, 0.2), (32.85, 0.4), (44.0, 0.3), (47.85, 0.2), (30.0, 0.9)])
def test_humidity_wet_bulb_dew_point_match_psychrolib(t, rh):
    w = float(psychro.humidity_ratio(t, rh))
    assert w == pytest.approx(psychrolib.GetHumRatioFromRelHum(t, rh, P), rel=1e-6)
    assert float(psychro.wet_bulb(t, w)) == pytest.approx(
        psychrolib.GetTWetBulbFromHumRatio(t, w, P), abs=0.01
    )
    assert float(psychro.dew_point(w)) == pytest.approx(
        psychrolib.GetTDewPointFromHumRatio(t, w, P), abs=0.01
    )
    assert float(psychro.enthalpy(t, w)) == pytest.approx(psychrolib.GetMoistAirEnthalpy(t, w), rel=1e-4)


def test_round_trips_and_ordering():
    t = np.array([28.0, 35.0, 45.0])
    w = psychro.humidity_ratio(t, np.array([0.3, 0.5, 0.2]))
    np.testing.assert_allclose(psychro.relative_humidity(t, w), [0.3, 0.5, 0.2], rtol=1e-10)
    np.testing.assert_allclose(psychro.temperature_from_enthalpy(psychro.enthalpy(t, w), w), t, rtol=1e-12)
    twb = psychro.wet_bulb(t, w)
    tdp = psychro.dew_point(w)
    assert np.all(tdp < twb) and np.all(twb < t)
