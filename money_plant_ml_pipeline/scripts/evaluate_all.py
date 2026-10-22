#!/usr/bin/env python
"""Re-evaluate the saved models on a CSV using the split recorded at training time (optionally with SHAP reports)."""
import argparse
import json
import sys

import _bootstrap  # noqa: F401
from plant_ml.artifact_manager import ArtifactError
from plant_ml.config import load_config
from plant_ml.data_loader import DataLoadError
from plant_ml.evaluation_run import evaluate_run
from plant_ml.validation import ValidationError


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", help="CSV with labels (default: paths.default_input)")
    ap.add_argument("--run-id", help="artifact run id (default: latest)")
    ap.add_argument("--shap", action="store_true", help="also write SHAP summary plot + tabular explanations")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    try:
        out = evaluate_run(cfg, a.input or cfg.path("paths.default_input"), a.run_id, with_shap=a.shap)
    except (ArtifactError, DataLoadError, ValidationError, ValueError, FileNotFoundError) as exc:
        print(f"EVALUATION FAILED: {exc}", file=sys.stderr)
        return 1
    c = out["classifier"]
    print(f"Run {out['run_id']} | {c['metric_scope']}")
    print(f"split: {out['split']}")
    print(f"classifier: accuracy={c['accuracy']:.3f} macro-F1={c['macro_f1']:.3f} weighted-F1={c['weighted_f1']:.3f} balanced-acc={c['balanced_accuracy']:.3f}")
    if out.get("warning"):
        print("WARNING:", out["warning"])
    if out["anomaly"]:
        print("anomaly vs synthetic helper labels:", json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in out["anomaly"].items() if k != "note"}))
    print("clustering:", out["clustering"])
    print("reports:", out["reports_dir"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
