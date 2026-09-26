"""The operating envelope the surrogate was trained on and the assistant may answer inside.

Anything outside it is refused rather than extrapolated. The bounds are a
design choice (see README): inlet air 25-50 C covers the source report's
32.85-47.85 C sweep plus margin; channel velocity 0.5-3.0 m/s brackets the
1-3 m/s operating airflow the report names for the wet channel liner; the
working-air fraction 0.2-0.6 brackets typical M-cycle designs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from hvac_cooling_ai.physics import psychro


@dataclass(frozen=True)
class Envelope:
    t_in_c: tuple[float, float] = (25.0, 50.0)
    w_in_g_kg: tuple[float, float] = (4.0, 20.0)
    rh_max: float = 0.90
    velocity_m_s: tuple[float, float] = (0.5, 3.0)
    working_air_fraction: tuple[float, float] = (0.2, 0.6)

    def violations(
        self, t_in_c: float, w_in_g_kg: float, velocity_m_s: float, working_air_fraction: float
    ) -> list[str]:
        """Human-readable reasons the point is outside the envelope ([] if inside)."""
        problems: list[str] = []
        checks = (
            ("inlet dry-bulb", t_in_c, self.t_in_c, "C"),
            ("inlet humidity ratio", w_in_g_kg, self.w_in_g_kg, "g/kg"),
            ("channel air velocity", velocity_m_s, self.velocity_m_s, "m/s"),
            ("working-air fraction", working_air_fraction, self.working_air_fraction, ""),
        )
        for name, value, (lo, hi), unit in checks:
            if not lo <= value <= hi:
                problems.append(f"{name} {value:g}{unit} is outside the validated range {lo:g}-{hi:g}{unit}")
        if not problems:
            rh = float(psychro.relative_humidity(t_in_c, w_in_g_kg / 1000.0))
            if rh > self.rh_max:
                problems.append(
                    f"inlet relative humidity {rh:.0%} is above the validated maximum {self.rh_max:.0%}"
                )
        return problems

    def to_dict(self) -> dict:
        return asdict(self)


ENVELOPE = Envelope()
