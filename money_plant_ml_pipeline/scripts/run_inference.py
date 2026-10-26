#!/usr/bin/env python
"""Run inference on a CSV (labels NOT required). Predicts the latest row per plant, or every row with --rows all."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from plant_ml.artifact_manager import ArtifactError
from plant_ml.config import load_config
from plant_ml.data_loader import DataLoadError, load_csv
from plant_ml.inference import PlantPredictor
from plant_ml.validation import ValidationError, validate_frame, validation_messages


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="CSV with at least timestamp, plant_id, soil, temperature, humidity, light")
    ap.add_argument("--plant-id", type=int, help="only this plant")
    ap.add_argument("--rows", choices=["last", "all"], default="last")
    ap.add_argument("--run-id")
    ap.add_argument("--output", help="write results to this .json or .csv file")
    ap.add_argument("--no-explain", action="store_true")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    try:
        df = load_csv(a.input, cfg.get("timezone"), require_labels=False)
        res = validate_frame(df, mode="warn", require_labels=False)
        for m in validation_messages(res.report):
            print("validation warning:", m, file=sys.stderr)
        data = res.clean.reset_index(drop=True)
        reference = None
        if a.plant_id is not None:
            reference = data[data["plant_id"] != a.plant_id].reset_index(drop=True)   # reference plants for cross-plant features
            data = data[data["plant_id"] == a.plant_id].reset_index(drop=True)
            if data.empty:
                print(f"No rows for plant_id {a.plant_id}", file=sys.stderr)
                return 1
        predictor = PlantPredictor.from_config(cfg, a.run_id)
        results = predictor.predict_frame(data, rows=a.rows, explain=False if a.no_explain else None, reference=reference)
    except (ArtifactError, DataLoadError, ValidationError, ValueError, FileNotFoundError) as exc:
        print(f"INFERENCE FAILED: {exc}", file=sys.stderr)
        return 1
    flat = [{"plant_id": r["plant_id"], "timestamp": r["timestamp"], "label": r["stress_prediction"]["label"],
             **{f"p_{k}": round(v, 4) for k, v in r["stress_prediction"]["probabilities"].items()},
             "cluster": r["cluster"]["id"], "cluster_distance": round(r["cluster"]["distance"], 3),
             "is_anomaly": r["anomaly"]["is_anomaly"], "anomaly_score": round(r["anomaly"]["score"], 4),
             "n_recommendations": len(r["recommendations"]), "history_rows": r["history"]["rows_available"]} for r in results]
    table = pd.DataFrame(flat)
    print(table.to_string(index=False))
    print(f"\n{len(results)} prediction(s). trained_on={results[0]['model_metadata']['trained_on']} (experimental; not a validated diagnosis)")
    if a.output:
        out = Path(a.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix == ".csv":
            table.to_csv(out, index=False)
        else:
            out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        print("written:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
