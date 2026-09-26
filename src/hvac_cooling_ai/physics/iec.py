"""Crossflow regenerative (M-cycle style) indirect evaporative cooler.

Cell-by-cell epsilon-NTU model on a 2-D grid, following the structure of the
source project's model: dry (product) channel with sensible cooling only, wet
(working) channel with combined heat and mass transfer, Lewis factor = 1, a
fully wetted wall, constant air properties, adiabatic outer walls. Each grid
cell is solved as a small crossflow exchanger; the outlet state of one cell is
the inlet state of the next along each flow path.

Flow arrangement (our assumption, see README "Assumptions"):
  * product air enters the dry channel along x;
  * at the dry outlet a fraction ``r`` (the working-air fraction) is turned
    back into the wet channel, which runs along y (crossflow);
  * the remaining ``1 - r`` is delivered as supply air.
The wet-channel inlet therefore depends on the dry-channel outlet, so the grid
march is repeated until that coupling converges (fixed-point iteration).

Wet-side transfer uses the enthalpy-potential (Merkel / Maclaine-cross)
formulation: with Le = 1, q'' = (h_w / cp) * (i_sat(T_wall) - i_air). i_sat is
linearised around a local guess per cell, which turns the wet stream into an
equivalent sensible stream with capacity m_w * d(i_sat)/dT, so the classic
crossflow epsilon-NTU relation can be used. The working-air humidity moves on
the straight line toward the saturated wall state (the Le = 1 result). If that
overshoots saturation, the state is moved back to saturation at constant
enthalpy, so energy is conserved exactly.

Everything is vectorised over a batch of operating points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from hvac_cooling_ai.physics import psychro

# Constant air properties at ~300 K (report assumption: constant thermophysical properties)
RHO_AIR = 1.16  # kg/m3
K_AIR = 0.0263  # W/(m K)
MU_AIR = 1.85e-5  # Pa s
PR_AIR = 0.71
NU_LAMINAR_PARALLEL_PLATES = 7.54  # fully developed, both walls isothermal
RE_LAMINAR_LIMIT = 2300.0
BTU_H_PER_W = 3.412142  # 1 W = 3.412142 Btu/h, so EER = COP * 3.412


@dataclass(frozen=True)
class Geometry:
    """Plate heat and mass exchanger geometry for one repeating channel pair.

    Defaults are OUR assumptions; the source report does not state them.
    ``wet_length_m`` = 0.20 m follows the report's wicking test, which found
    the chosen mesh liner stays fully wetted up to about 200 mm of height.
    """

    dry_length_m: float = 0.20  # product-air flow length (x)
    wet_length_m: float = 0.20  # working-air flow length (y), also dry-channel depth
    channel_gap_m: float = 0.004  # both channels
    n_channel_pairs: int = 1  # the report solved "one channel"
    plate_thickness_m: float = 0.0002
    plate_conductivity: float = 200.0  # aluminium, W/(m K)
    liner_thickness_m: float = 0.0002
    liner_conductivity: float = 0.6  # water-saturated fabric ~ water, W/(m K)

    @property
    def hydraulic_diameter_m(self) -> float:
        return 2.0 * self.channel_gap_m  # infinite parallel plates

    @property
    def area_per_pair_m2(self) -> float:
        # Repeating unit: a dry channel exchanges through both of its walls.
        return 2.0 * self.dry_length_m * self.wet_length_m

    @property
    def wall_resistance(self) -> float:
        return (
            self.plate_thickness_m / self.plate_conductivity
            + self.liner_thickness_m / self.liner_conductivity
        )


@dataclass(frozen=True)
class GridSpec:
    nx: int = 20
    ny: int = 20
    max_iterations: int = 300
    tolerance_k: float = 1e-7


def _nusselt(re: np.ndarray) -> np.ndarray:
    """Laminar parallel-plate value; Gnielinski above the transition."""
    f = (0.79 * np.log(np.maximum(re, 3000.0)) - 1.64) ** -2
    gnielinski = (
        (f / 8.0) * (re - 1000.0) * PR_AIR / (1.0 + 12.7 * np.sqrt(f / 8.0) * (PR_AIR ** (2.0 / 3.0) - 1.0))
    )
    turbulent = np.maximum(gnielinski, NU_LAMINAR_PARALLEL_PLATES)
    return np.where(re < RE_LAMINAR_LIMIT, NU_LAMINAR_PARALLEL_PLATES, turbulent)


def _pressure_drop(re: np.ndarray, velocity: np.ndarray, length: float, dh: float) -> np.ndarray:
    """Friction pressure drop, Pa: laminar parallel plates f = 96/Re, Blasius above transition."""
    f = np.where(re < RE_LAMINAR_LIMIT, 96.0 / np.maximum(re, 1e-9), 0.316 * np.maximum(re, 1.0) ** -0.25)
    return f * (length / dh) * 0.5 * RHO_AIR * velocity**2


def _crossflow_effectiveness(ntu: np.ndarray, cr: np.ndarray) -> np.ndarray:
    """Both-fluids-unmixed crossflow effectiveness (Incropera eq. 11.32)."""
    cr = np.maximum(cr, 1e-12)
    return 1.0 - np.exp((1.0 / cr) * ntu**0.22 * (np.exp(-cr * ntu**0.78) - 1.0))


def _saturate_at_constant_enthalpy(i_air: np.ndarray, t_guess: np.ndarray, p: float) -> np.ndarray:
    """Temperature of saturated air whose enthalpy equals i_air (Newton)."""
    t = t_guess.copy()
    for _ in range(8):
        t = t - (psychro.sat_enthalpy(t, p) - i_air) / psychro.sat_enthalpy_slope(t, p)
    return t


def simulate_batch(
    t_in_c: ArrayLike,
    w_in: ArrayLike,
    dry_velocity_m_s: ArrayLike,
    working_air_fraction: ArrayLike,
    geometry: Geometry | None = None,
    grid: GridSpec | None = None,
    fan_power_w: float = 0.5,
    pressure_pa: float = psychro.P_ATM,
) -> dict[str, np.ndarray]:
    """Simulate a batch of operating points. All inputs broadcast to one shape.

    Returns a dict of 1-D arrays (see ``OUTPUT_KEYS``).
    """
    geo = geometry or Geometry()
    grd = grid or GridSpec()
    t_in, w0, vel, r = (
        np.atleast_1d(np.asarray(a, dtype=float)).ravel()
        for a in np.broadcast_arrays(t_in_c, w_in, dry_velocity_m_s, working_air_fraction)
    )
    nx, ny = grd.nx, grd.ny
    p = pressure_pa

    # --- flows per repeating channel pair ---
    m_primary = RHO_AIR * vel * geo.channel_gap_m * geo.wet_length_m  # kg/s dry-channel flow
    m_work = r * m_primary
    m_supply = (1.0 - r) * m_primary
    cp_air = psychro.moist_cp(w0)

    # --- heat transfer coefficients ---
    dh = geo.hydraulic_diameter_m
    re_dry = RHO_AIR * vel * dh / MU_AIR
    vel_wet = m_work / (RHO_AIR * geo.channel_gap_m * geo.dry_length_m)
    re_wet = RHO_AIR * vel_wet * dh / MU_AIR
    h_dry = _nusselt(re_dry) * K_AIR / dh
    h_wet = _nusselt(re_wet) * K_AIR / dh
    a_cell = geo.area_per_pair_m2 / (nx * ny)
    ua_dry = a_cell / (1.0 / h_dry + geo.wall_resistance)  # air to wetted surface, W/K
    hm_a = (h_wet / cp_air) * a_cell  # mass-transfer conductance x area, kg/s

    c_dry_row = m_primary / ny * cp_air  # W/K per dry-channel row
    m_wet_col = m_work / nx  # kg/s per wet-channel column

    t_dp = psychro.dew_point(w0, p)
    t_wb = psychro.wet_bulb(t_in, w0, p)

    t_dry_out = t_wb.copy()  # initial guess for the regenerative coupling
    converged = np.zeros_like(t_in, dtype=bool)
    iterations = 0
    for _ in range(grd.max_iterations):
        iterations += 1
        i_wet_in = psychro.enthalpy(t_dry_out, w0)
        t_dry = np.repeat(t_in[:, None], ny, axis=1)
        i_wet_out = np.zeros((t_in.size, nx))
        w_wet_out = np.zeros((t_in.size, nx))
        q_total = np.zeros_like(t_in)
        for ix in range(nx):
            i_w = i_wet_in.copy()
            w_w = w0.copy()
            t_surf = t_dry_out.copy()
            for jy in range(ny):
                t_d = t_dry[:, jy]
                t_w_air = psychro.temperature_from_enthalpy(i_w, w_w)
                for _ in range(2):  # re-linearise i_sat around the updated wall temperature
                    b = psychro.sat_enthalpy_slope(t_surf, p)
                    t_star = t_surf + (i_w - psychro.sat_enthalpy(t_surf, p)) / b
                    ua_wet = hm_a * b
                    ua = 1.0 / (1.0 / ua_dry + 1.0 / ua_wet)
                    c_wet = m_wet_col * b
                    c_min = np.minimum(c_dry_row, c_wet)
                    c_max = np.maximum(c_dry_row, c_wet)
                    eff = _crossflow_effectiveness(ua / c_min, c_min / c_max)
                    q = eff * c_min * (t_d - t_star)
                    t_star_mean = t_star + 0.5 * q / c_wet
                    t_surf = t_star_mean + q / ua_wet
                # --- dry side: sensible only ---
                t_dry[:, jy] = t_d - q / c_dry_row
                # --- wet side: enthalpy gain, humidity on the Le = 1 line ---
                di = q / m_wet_col
                i_new = i_w + di
                w_s = psychro.sat_humidity_ratio(t_surf, p)
                i_s = psychro.sat_enthalpy(t_surf, p)
                denom = np.where(np.abs(i_s - i_w) > 1e-9, i_s - i_w, 1e-9)
                w_new = w_w + di * (w_s - w_w) / denom
                t_new = psychro.temperature_from_enthalpy(i_new, w_new)
                supersat = w_new > psychro.sat_humidity_ratio(t_new, p)
                if np.any(supersat):
                    t_sat = _saturate_at_constant_enthalpy(i_new, np.maximum(t_new, t_w_air), p)
                    w_new = np.where(supersat, psychro.sat_humidity_ratio(t_sat, p), w_new)
                i_w, w_w = i_new, w_new
                q_total += q
            i_wet_out[:, ix] = i_w
            w_wet_out[:, ix] = w_w
        t_new_out = t_dry.mean(axis=1)
        change = np.abs(t_new_out - t_dry_out)
        t_dry_out = t_new_out
        converged = change < grd.tolerance_k
        if np.all(converged):
            break

    t_wet_cols = psychro.temperature_from_enthalpy(i_wet_out, w_wet_out)
    rh_wet_cols_max = psychro.relative_humidity(t_wet_cols, w_wet_out).max(axis=1)
    i_wo = i_wet_out.mean(axis=1)
    w_wo = w_wet_out.mean(axis=1)
    t_wo = psychro.temperature_from_enthalpy(i_wo, w_wo)
    n = geo.n_channel_pairs
    cooling_w = n * m_supply * cp_air * (t_in - t_dry_out)
    cop = cooling_w / fan_power_w
    return {
        "t_in_c": t_in,
        "w_in": w0,
        "t_supply_c": t_dry_out,
        "t_wet_out_c": t_wo,
        "w_wet_out": w_wo,
        "rh_wet_channel_outlet_max": rh_wet_cols_max,
        "t_wet_bulb_in_c": t_wb,
        "t_dew_point_in_c": t_dp,
        "eps_wb": (t_in - t_dry_out) / (t_in - t_wb),
        "eps_dp": (t_in - t_dry_out) / (t_in - t_dp),
        "supply_flow_kg_s": n * m_supply,
        "working_flow_kg_s": n * m_work,
        "cooling_capacity_w": cooling_w,
        "heat_removed_primary_w": n * q_total,
        "water_evaporated_kg_h": n * m_work * (w_wo - w0) * 3600.0,
        "cop": cop,
        "eer_btu_h_per_w": cop * BTU_H_PER_W,
        "dp_dry_pa": _pressure_drop(re_dry, vel, geo.dry_length_m, dh),
        "dp_wet_pa": _pressure_drop(re_wet, vel_wet, geo.wet_length_m, dh),
        "re_dry": re_dry,
        "re_wet": re_wet,
        "converged": converged,
        "iterations": np.full_like(t_in, iterations),
    }


OUTPUT_KEYS = (
    "t_supply_c",
    "t_wet_out_c",
    "w_wet_out",
    "t_wet_bulb_in_c",
    "t_dew_point_in_c",
    "eps_wb",
    "eps_dp",
    "supply_flow_kg_s",
    "working_flow_kg_s",
    "cooling_capacity_w",
    "heat_removed_primary_w",
    "water_evaporated_kg_h",
    "cop",
    "eer_btu_h_per_w",
)


def simulate(
    t_in_c: float,
    w_in: float,
    dry_velocity_m_s: float,
    working_air_fraction: float,
    geometry: Geometry | None = None,
    grid: GridSpec | None = None,
    fan_power_w: float = 0.5,
    pressure_pa: float = psychro.P_ATM,
) -> dict[str, float]:
    """Single operating point; returns plain floats."""
    out = simulate_batch(
        t_in_c,
        w_in,
        dry_velocity_m_s,
        working_air_fraction,
        geometry=geometry,
        grid=grid,
        fan_power_w=fan_power_w,
        pressure_pa=pressure_pa,
    )
    return {k: (bool(v[0]) if v.dtype == bool else float(v[0])) for k, v in out.items()}
