"""The comparison against the source report is reproducible and self-consistent."""

import math

import pytest

from hvac_cooling_ai import validation


@pytest.fixture(scope="module")
def result():
    return validation.run()


def test_back_calculated_dew_point_is_consistent(result):
    op = result["operating_point"]
    # Both reported inlet temperatures imply the same inlet dew point (~18 C).
    assert op["dew_point_c_from_306"] == pytest.approx(op["dew_point_c_from_321"], abs=0.1)
    assert 0.5 < op["dry_channel_velocity_m_s"] < 3.0


def test_our_model_stays_above_the_dew_point_floor(result):
    for row in result["rows"]:
        assert row["ours"]["t_supply_k"] > row["inlet_dew_point_k"]
        assert row["large_exchanger_ceiling"]["t_supply_k"] > row["inlet_dew_point_k"]
        assert row["ours"]["eps_dp"] < 1.0


def test_reported_eer_is_cop_times_pi(result):
    # Documents a finding: the report's EER equals COP x 3.1416, not COP x 3.412.
    for ratio in result["report_eer_over_cop"]:
        assert ratio == pytest.approx(math.pi, abs=0.001)


def test_table_renders(result):
    table = validation.format_table(result)
    assert table.count("\n") >= 9 and "Dew-point effectiveness" in table
