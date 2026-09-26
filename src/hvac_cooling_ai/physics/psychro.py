"""Moist-air psychrometrics (SI units, temperatures in degrees Celsius).

Formulation: ASHRAE Handbook - Fundamentals (2017), chapter 1.
  * saturation pressure over liquid water: Hyland-Wexler (eq. 6)
  * humidity ratio: W = 0.621945 * pw / (p - pw)             (eq. 20)
  * moist-air enthalpy: h = 1006 t + W (2501e3 + 1860 t)  J/kg dry air (eq. 30)
  * thermodynamic wet-bulb: ASHRAE eq. 35, solved by bisection
  * dew point: inverse of the saturation pressure, solved by bisection

All functions accept scalars or numpy arrays (vectorised), so the heat-exchanger
model can march a whole batch of operating points through its grid at once.
The Hyland-Wexler fit is used for liquid water only, which is fine for the
cooling range this project covers (dew points above 0 C).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

P_ATM = 101_325.0  # Pa, sea-level standard pressure
CP_DA = 1006.0  # J/(kg K), dry air
CP_V = 1860.0  # J/(kg K), water vapour
H_FG0 = 2501e3  # J/kg, latent heat of vaporisation at 0 C
MW_RATIO = 0.621945  # molar mass water / molar mass dry air

# Hyland-Wexler coefficients over liquid water (ASHRAE 2017, ch. 1, eq. 6)
_C8, _C9, _C10 = -5.8002206e3, 1.3914993, -4.8640239e-2
_C11, _C12, _C13 = 4.1764768e-5, -1.4452093e-8, 6.5459673

_BISECT_ITERS = 60  # halves a 150 K bracket to < 1e-15 K


def _as_array(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=float)


def sat_pressure(t_c: ArrayLike) -> np.ndarray:
    """Saturation vapour pressure over liquid water, Pa."""
    t = _as_array(t_c) + 273.15
    ln_p = _C8 / t + _C9 + _C10 * t + _C11 * t**2 + _C12 * t**3 + _C13 * np.log(t)
    return np.exp(ln_p)


def humidity_ratio_from_pw(pw: ArrayLike, p: float = P_ATM) -> np.ndarray:
    pw = _as_array(pw)
    return MW_RATIO * pw / (p - pw)


def sat_humidity_ratio(t_c: ArrayLike, p: float = P_ATM) -> np.ndarray:
    """Humidity ratio of saturated air at temperature t_c, kg/kg dry air."""
    return humidity_ratio_from_pw(sat_pressure(t_c), p)


def humidity_ratio(t_c: ArrayLike, rh: ArrayLike, p: float = P_ATM) -> np.ndarray:
    """Humidity ratio from dry-bulb temperature and relative humidity (0..1)."""
    return humidity_ratio_from_pw(_as_array(rh) * sat_pressure(t_c), p)


def relative_humidity(t_c: ArrayLike, w: ArrayLike, p: float = P_ATM) -> np.ndarray:
    """Relative humidity (0..1) from dry-bulb temperature and humidity ratio."""
    w = _as_array(w)
    pw = p * w / (MW_RATIO + w)
    return pw / sat_pressure(t_c)


def enthalpy(t_c: ArrayLike, w: ArrayLike) -> np.ndarray:
    """Moist-air specific enthalpy, J/kg dry air."""
    t = _as_array(t_c)
    w = _as_array(w)
    return CP_DA * t + w * (H_FG0 + CP_V * t)


def temperature_from_enthalpy(h: ArrayLike, w: ArrayLike) -> np.ndarray:
    """Dry-bulb temperature that gives enthalpy h at humidity ratio w."""
    h = _as_array(h)
    w = _as_array(w)
    return (h - H_FG0 * w) / (CP_DA + CP_V * w)


def moist_cp(w: ArrayLike) -> np.ndarray:
    """Humid specific heat, J/(kg dry air K)."""
    return CP_DA + CP_V * _as_array(w)


def sat_enthalpy(t_c: ArrayLike, p: float = P_ATM) -> np.ndarray:
    """Enthalpy of saturated air at t_c, J/kg dry air."""
    return enthalpy(t_c, sat_humidity_ratio(t_c, p))


def sat_enthalpy_slope(t_c: ArrayLike, p: float = P_ATM, dt: float = 1e-3) -> np.ndarray:
    """d(h_sat)/dT, J/(kg K), by central difference."""
    t = _as_array(t_c)
    return (sat_enthalpy(t + dt, p) - sat_enthalpy(t - dt, p)) / (2.0 * dt)


def _bisect(fn, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Vectorised bisection for a function that is increasing in its argument."""
    lo = lo.copy()
    hi = hi.copy()
    for _ in range(_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        positive = fn(mid) > 0.0
        hi = np.where(positive, mid, hi)
        lo = np.where(positive, lo, mid)
    return 0.5 * (lo + hi)


def dew_point(w: ArrayLike, p: float = P_ATM) -> np.ndarray:
    """Dew-point temperature, C, for humidity ratio w."""
    w = _as_array(w)
    pw = p * w / (MW_RATIO + w)
    lo = np.full_like(w, -60.0)
    hi = np.full_like(w, 100.0)
    return _bisect(lambda t: sat_pressure(t) - pw, lo, hi)


def wet_bulb(t_c: ArrayLike, w: ArrayLike, p: float = P_ATM) -> np.ndarray:
    """Thermodynamic wet-bulb temperature, C (ASHRAE 2017 ch. 1 eq. 35)."""
    t = _as_array(t_c)
    w = np.broadcast_to(_as_array(w), t.shape).astype(float)

    def residual(twb: np.ndarray) -> np.ndarray:
        ws_star = sat_humidity_ratio(twb, p)
        w_calc = ((2501.0 - 2.326 * twb) * ws_star - 1.006 * (t - twb)) / (2501.0 + 1.86 * t - 4.186 * twb)
        # w_calc rises with twb; zero where it equals the actual humidity ratio
        return w_calc - w

    lo = np.full_like(t, -40.0)
    hi = t.copy() + 1e-9
    return _bisect(residual, lo, hi)
