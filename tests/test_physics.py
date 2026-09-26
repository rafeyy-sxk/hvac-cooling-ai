"""Physical sanity of the heat-exchanger model."""

import numpy as np
import pytest

from hvac_cooling_ai.physics import iec, psychro
from hvac_cooling_ai.surrogate import dataset


@pytest.fixture(scope="module")
def batch():
    x = dataset.sample_envelope(200, seed=7)
    res = iec.simulate_batch(x[:, 0], x[:, 1] / 1000.0, x[:, 2], x[:, 3])
    return x, res


def test_converges_everywhere(batch):
    _, res = batch
    assert res["converged"].all()


def test_outlet_below_inlet_and_above_dew_point(batch):
    x, res = batch
    assert np.all(res["t_supply_c"] < x[:, 0])
    assert np.all(res["t_supply_c"] > res["t_dew_point_in_c"])
    assert np.all(res["eps_dp"] < 1.0)


def test_energy_balance_closes(batch):
    """Heat lost by the product air = enthalpy gained by the working air.

    The wet side is re-evaluated from its outlet temperature and humidity with the
    psychrometric enthalpy function, so errors in the humidity update or in the
    saturation clamp would show up here.
    """
    x, res = batch
    m_primary = res["supply_flow_kg_s"] + res["working_flow_kg_s"]
    cp = psychro.moist_cp(x[:, 1] / 1000.0)
    q_product = m_primary * cp * (x[:, 0] - res["t_supply_c"])
    i_in = psychro.enthalpy(res["t_supply_c"], x[:, 1] / 1000.0)
    i_out = psychro.enthalpy(res["t_wet_out_c"], res["w_wet_out"])
    q_wet = res["working_flow_kg_s"] * (i_out - i_in)
    np.testing.assert_allclose(q_wet, q_product, rtol=1e-6)
    np.testing.assert_allclose(res["heat_removed_primary_w"], q_product, rtol=1e-6)


def test_working_air_gains_moisture(batch):
    x, res = batch
    assert np.all(res["w_wet_out"] > x[:, 1] / 1000.0)
    assert np.all(res["water_evaporated_kg_h"] > 0)
    # Each wet channel leaves at or below saturation. (Averaging channels at different
    # temperatures can read slightly above 100%: that is fog from mixing, not a model error.)
    assert np.all(res["rh_wet_channel_outlet_max"] <= 1.0 + 1e-9)


def test_bigger_exchanger_cools_more_but_never_below_dew_point():
    temps = []
    for length in (0.2, 0.5, 1.0, 4.0):
        geo = iec.Geometry(dry_length_m=length, wet_length_m=length)
        temps.append(iec.simulate(40.0, 0.0129, 0.5, 0.4, geometry=geo)["t_supply_c"])
    assert temps == sorted(temps, reverse=True)
    assert temps[-1] > float(psychro.dew_point(0.0129))


def test_grid_refinement_changes_little():
    coarse = iec.simulate(40.0, 0.0129, 1.0, 1 / 3, grid=iec.GridSpec(nx=20, ny=20))["t_supply_c"]
    fine = iec.simulate(40.0, 0.0129, 1.0, 1 / 3, grid=iec.GridSpec(nx=60, ny=60))["t_supply_c"]
    assert abs(coarse - fine) < 0.1


def test_stack_scales_capacity_linearly():
    one = iec.simulate(40.0, 0.0129, 1.0, 1 / 3)
    ten = iec.simulate(40.0, 0.0129, 1.0, 1 / 3, geometry=iec.Geometry(n_channel_pairs=10))
    assert ten["t_supply_c"] == pytest.approx(one["t_supply_c"])
    assert ten["cooling_capacity_w"] == pytest.approx(10 * one["cooling_capacity_w"])


def test_flow_is_laminar_across_envelope(batch):
    _, res = batch
    assert np.all(res["re_dry"] < iec.RE_LAMINAR_LIMIT)
    assert np.all(res["re_wet"] < iec.RE_LAMINAR_LIMIT)
