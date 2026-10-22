#!/usr/bin/env python
"""Generate a deterministic SYNTHETIC sensor CSV (Bengaluru-like assumptions; not real weather or plant data)."""
import argparse
import sys

import _bootstrap  # noqa: F401
from plant_ml.config import PROJECT_ROOT
from plant_ml.synthetic import check_generated, generate, n_days_for_rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="data/sample/synthetic_sensor_readings.csv", help="output CSV path")
    ap.add_argument("--rows", type=int, default=8000, help="minimum number of rows (rounded up to whole days x 16 plants x 12)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start-date", default="2026-09-21")
    ap.add_argument("--allow-small", action="store_true", help="skip the >= 8000 row assertion (for tests/smoke runs)")
    a = ap.parse_args()
    df = generate(n_days_for_rows(a.rows), a.seed, a.start_date)
    check_generated(df, min_rows=1 if a.allow_small else 8000)
    from pathlib import Path
    out = Path(a.output)
    out = out if out.is_absolute() else PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} SYNTHETIC rows to {out}\nstress_level counts: {df['stress_level'].value_counts().to_dict()}")
    print("NOTE: synthetic data - labels are simulation labels, class balance is not real-world prevalence.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
