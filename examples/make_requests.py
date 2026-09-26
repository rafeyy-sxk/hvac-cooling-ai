"""Write two SYNTHETIC request logs for trying the input-drift check. Not real customer traffic.

    python examples/make_requests.py
    python -m hvac_cooling_ai drift --inputs examples/requests-training-like.csv   # passes
    python -m hvac_cooling_ai drift --inputs examples/requests-heatwave.csv        # flags drift

training-like: 500 points drawn the same way as the training data (a different seed).
heatwave:      the same 500 points, 8 C hotter, so some fall above the 50 C envelope limit.
"""

import csv
from pathlib import Path

from hvac_cooling_ai.surrogate import dataset

HERE = Path(__file__).parent
FIELDS = ["dry_bulb_c", "humidity_ratio_g_kg", "channel_velocity_m_s", "working_air_fraction"]


def write(path: Path, x) -> None:
    with path.open("w", newline="") as fh:
        out = csv.writer(fh)
        out.writerow(FIELDS)
        for row in x:
            out.writerow([f"{v:.3f}" for v in row])


if __name__ == "__main__":
    x = dataset.sample_envelope(500, seed=12345)
    write(HERE / "requests-training-like.csv", x)
    x[:, 0] += 8.0
    write(HERE / "requests-heatwave.csv", x)
