#!/usr/bin/env python
"""Verify DEV-1: sending the last N=history_rows prior readings gives the same ML output as sending up to 500.

Needs the ML service running (ML_SERVICE_URL, default http://127.0.0.1:8001). Reads the dataset CSV directly
(no backend DB needed). Compares label, all probabilities, cluster id/distance, anomaly score/flag and the
recommendation ids; numeric differences must be <= 1e-9.

    backend/.venv/bin/python backend/scripts/verify_history_equivalence.py
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import pandas as pd  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.core.mappings import parse_timestamp, to_ist_iso  # noqa: E402
from app.integrations.ml_client import MLClient  # noqa: E402
from app.integrations.ml_types import MLReading, PredictRequest  # noqa: E402
from app.services.history_builder import history_rows_for, history_slice, read_ml_limits  # noqa: E402

TOL = 1e-9
PLANTS = (1, 6, 9, 14, 16)  # four experimental groups
POSITIONS = (3, 30, 60, 100, 150, 200, 280, 335)  # cold start .. deep in the 336-row series


def _num(value):
    return None if pd.isna(value) else float(value)


def load_series(csv: Path) -> dict[int, list]:
    df = pd.read_csv(csv)
    series: dict[int, list] = {}
    for plant_id, group in df.groupby("plant_id"):
        rows = []
        for r in group.itertuples():
            ts = parse_timestamp(r.timestamp)
            rows.append(SimpleNamespace(
                plant_id=int(plant_id), timestamp=ts,
                ml=MLReading(timestamp=to_ist_iso(ts), plant_id=int(plant_id), soil=_num(r.soil),
                             temperature=_num(r.temperature), humidity=_num(r.humidity), light=_num(r.light)),
            ))  # fmt: skip
        rows.sort(key=lambda x: x.timestamp)
        series[int(plant_id)] = rows
    return series


def pick_cases(series: dict[int, list]) -> list[tuple[int, int]]:
    cases = {(p, i) for p in PLANTS for i in POSITIONS}
    for p, rows in series.items():  # add readings with nulls (own or in the last 60 rows) deep in the series
        nulls = [i for i, r in enumerate(rows) if i > 60 and None in (r.ml.soil, r.ml.temperature, r.ml.humidity, r.ml.light)]
        cases.update((p, i) for i in nulls[:2] if p in PLANTS or len(cases) < 60)
    return sorted(cases)


def diff(a, b) -> list[str]:
    problems = []
    if a.stress_prediction.label != b.stress_prediction.label:
        problems.append("label")
    for k, v in a.stress_prediction.probabilities.items():
        if abs(v - b.stress_prediction.probabilities[k]) > TOL:
            problems.append(f"prob[{k}] {abs(v - b.stress_prediction.probabilities[k]):.2e}")
    if a.cluster.id != b.cluster.id:
        problems.append("cluster.id")
    if abs(a.cluster.distance - b.cluster.distance) > TOL:
        problems.append("cluster.distance")
    if abs(a.anomaly.score - b.anomaly.score) > TOL or a.anomaly.is_anomaly != b.anomaly.is_anomaly:
        problems.append("anomaly")
    if [r.id for r in a.recommendations] != [r.id for r in b.recommendations]:
        problems.append("recommendation ids")
    return problems


async def main() -> int:
    settings = get_settings()
    ml = MLClient(settings.ML_SERVICE_URL, settings.ML_TIMEOUT_S)
    try:
        metadata = await ml.metadata()
    except Exception as exc:
        print(f"ML service not usable at {settings.ML_SERVICE_URL}: {exc}")
        return 2
    max_hist, _ = read_ml_limits(settings.ML_CONFIG_PATH)
    n_small = history_rows_for(max_hist, metadata.required_history_rows_for_full_features, settings.HISTORY_ROWS_DEFAULT)
    n_big = max_hist
    series = load_series(settings.DATASET_PATH)
    cases = pick_cases(series)
    print(f"history_rows={n_small} vs {n_big}; {len(cases)} readings across {len({p for p, _ in cases})} plants")

    bad = 0
    with_nulls = 0
    for plant_id, i in cases:
        rows = series[plant_id]
        small = history_slice(rows, i, n_small)
        big = history_slice(rows, i, n_big)
        has_null = any(None in (r.ml.soil, r.ml.temperature, r.ml.humidity, r.ml.light) for r in [rows[i], *small])
        with_nulls += has_null
        a = await ml.predict(PredictRequest(reading=rows[i].ml, history=[r.ml for r in small], explain=False))
        b = await ml.predict(PredictRequest(reading=rows[i].ml, history=[r.ml for r in big], explain=False))
        problems = diff(a, b)
        if problems:
            bad += 1
            print(f"DIFF plant={plant_id} idx={i} (history {len(small)} vs {len(big)}): {problems}")
    print(f"{len(cases)} readings compared ({with_nulls} with nulls in the reading or its short history); "
          f"{'NOT equivalent: ' + str(bad) + ' differ' if bad else 'EQUIVALENT'} (tolerance {TOL})")
    await ml.aclose()
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
