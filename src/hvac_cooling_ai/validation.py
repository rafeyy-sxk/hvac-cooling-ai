"""Compare this port against the numbers reported by the source research project.

    python -m hvac_cooling_ai validate

The report gives results at inlet 306 K and 321 K but does not state the
inlet humidity, the air flow, the working-air fraction or the geometry. We do
NOT fit any of them to the reported outputs. Instead:

* inlet humidity ratio is back-calculated from the report's own dew-point and
  wet-bulb effectiveness values (all four agree on a dew point near 18 C);
* supply air flow is back-calculated from the report's own cooling capacity
  divided by its temperature drop (23.32 W / 37.03 K = 0.630 W/K);
* working-air fraction 1/3 and the plate geometry are our stated assumptions
  (see physics.iec.Geometry), fixed before this comparison was run.
"""

from __future__ import annotations

from hvac_cooling_ai.physics import iec, psychro

# Values as reported (inlet 306 K and 321 K). None = not reported at that point.
REPORT = {
    "t_supply_k": (None, 283.972),
    "delta_t_k": (29.8, 37.0),  # Table 4.1 range "29.8 - 37 K"
    "cooling_capacity_w": (18.77, 23.32),
    "eps_wb": (3.03, 1.81),
    "eps_dp": (2.00, 1.24),
    "cop": (37.53, 46.63),
    "eer": (117.9, 146.5),
}
INLET_K = (306.0, 321.0)
FAN_POWER_W = 0.5  # stated in the report
WORKING_AIR_FRACTION = 1.0 / 3.0  # our assumption


def inferred_operating_point() -> dict[str, float]:
    """Back-calculate humidity and flow from the report's own outputs."""
    dt_321 = 321.0 - REPORT["t_supply_k"][1]
    capacity_rate_w_k = REPORT["cooling_capacity_w"][1] / dt_321
    dt_306 = REPORT["cooling_capacity_w"][0] / capacity_rate_w_k
    # eps_dp = dT / (T_in - T_dp)  ->  T_dp = T_in - dT / eps_dp
    t_dp_306 = 306.0 - dt_306 / REPORT["eps_dp"][0] - 273.15
    t_dp_321 = 321.0 - dt_321 / REPORT["eps_dp"][1] - 273.15
    t_dp = 0.5 * (t_dp_306 + t_dp_321)
    w = float(psychro.sat_humidity_ratio(t_dp))
    m_supply = capacity_rate_w_k / float(psychro.moist_cp(w))
    geo = iec.Geometry()
    m_primary = m_supply / (1.0 - WORKING_AIR_FRACTION)
    velocity = m_primary / (iec.RHO_AIR * geo.channel_gap_m * geo.wet_length_m)
    return {
        "capacity_rate_w_k": capacity_rate_w_k,
        "delta_t_306_k": dt_306,
        "dew_point_c_from_306": t_dp_306,
        "dew_point_c_from_321": t_dp_321,
        "w_in_kg_kg": w,
        "supply_flow_kg_s": m_supply,
        "dry_channel_velocity_m_s": velocity,
    }


def run() -> dict:
    op = inferred_operating_point()
    rows = []
    for k, t_in_k in enumerate(INLET_K):
        t_in_c = t_in_k - 273.15
        ours = iec.simulate(
            t_in_c,
            op["w_in_kg_kg"],
            op["dry_channel_velocity_m_s"],
            WORKING_AIR_FRACTION,
            fan_power_w=FAN_POWER_W,
        )
        limit = iec.simulate(  # same inlet, exchanger 20x larger each way: the physical ceiling
            t_in_c,
            op["w_in_kg_kg"],
            op["dry_channel_velocity_m_s"],
            WORKING_AIR_FRACTION,
            geometry=iec.Geometry(dry_length_m=4.0, wet_length_m=4.0),
            fan_power_w=FAN_POWER_W,
        )
        rows.append(
            {
                "t_in_k": t_in_k,
                "report": {key: vals[k] for key, vals in REPORT.items()},
                "ours": {
                    "t_supply_k": ours["t_supply_c"] + 273.15,
                    "delta_t_k": t_in_c - ours["t_supply_c"],
                    "cooling_capacity_w": ours["cooling_capacity_w"],
                    "eps_wb": ours["eps_wb"],
                    "eps_dp": ours["eps_dp"],
                    "cop": ours["cop"],
                    "eer": ours["eer_btu_h_per_w"],
                },
                "large_exchanger_ceiling": {
                    "t_supply_k": limit["t_supply_c"] + 273.15,
                    "eps_dp": limit["eps_dp"],
                },
                "inlet_dew_point_k": ours["t_dew_point_in_c"] + 273.15,
                "inlet_wet_bulb_k": ours["t_wet_bulb_in_c"] + 273.15,
            }
        )
    eer_over_cop = [REPORT["eer"][i] / REPORT["cop"][i] for i in range(2)]
    return {"operating_point": op, "rows": rows, "report_eer_over_cop": eer_over_cop}


def format_table(result: dict) -> str:
    labels = {
        "t_supply_k": "Supply (outlet) temp, K",
        "delta_t_k": "Temperature drop, K",
        "cooling_capacity_w": "Cooling capacity, W",
        "eps_wb": "Wet-bulb effectiveness",
        "eps_dp": "Dew-point effectiveness",
        "cop": "COP (0.5 W fan)",
        "eer": "EER, Btu/h per W",
    }
    lines = [
        "| Metric | Report @306 K | Ours @306 K | Report @321 K | Ours @321 K |",
        "|---|---|---|---|---|",
    ]
    r306, r321 = result["rows"]
    for key, name in labels.items():

        def fmt(v):
            return "not stated" if v is None else f"{v:.2f}"

        lines.append(
            f"| {name} | {fmt(r306['report'][key])} | {fmt(r306['ours'][key])} | "
            f"{fmt(r321['report'][key])} | {fmt(r321['ours'][key])} |"
        )
    lines.append(
        f"| Inlet dew point, K (physical floor) | - | {r306['inlet_dew_point_k']:.2f} | - | "
        f"{r321['inlet_dew_point_k']:.2f} |"
    )
    lines.append(
        f"| Supply temp with a 20x larger exchanger, K | - | "
        f"{r306['large_exchanger_ceiling']['t_supply_k']:.2f} | - | "
        f"{r321['large_exchanger_ceiling']['t_supply_k']:.2f} |"
    )
    return "\n".join(lines)


def main() -> None:
    result = run()
    op = result["operating_point"]
    print("Operating point back-calculated from the report's own numbers:")
    for k, v in op.items():
        print(f"  {k} = {v:.6g}")
    print(
        f"  report EER / COP = {result['report_eer_over_cop'][0]:.4f}, {result['report_eer_over_cop'][1]:.4f}"
        f"  (standard conversion is {iec.BTU_H_PER_W:.4f})"
    )
    print()
    print(format_table(result))


if __name__ == "__main__":
    main()
