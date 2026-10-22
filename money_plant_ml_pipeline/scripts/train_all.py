#!/usr/bin/env python
"""Train all three models (XGBoost classifier, K-Means, Isolation Forest). Training is always an explicit command."""
import argparse
import sys

import _bootstrap  # noqa: F401
from plant_ml.config import load_config
from plant_ml.data_loader import DataLoadError
from plant_ml.training import run_training
from plant_ml.validation import ValidationError


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", help="training CSV (default: paths.default_input)")
    ap.add_argument("--config", help="config.yaml path")
    ap.add_argument("--split", choices=["chronological", "group_holdout"], help="override split.strategy")
    ap.add_argument("--provenance", choices=["synthetic", "real", "mixed"], help="override data.provenance")
    ap.add_argument("--validation-mode", choices=["fail", "warn", "quarantine"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--artifacts-dir", help="override paths.artifacts_dir")
    ap.add_argument("--reports-dir", help="override paths.reports_dir")
    ap.add_argument("--no-set-latest", action="store_true", help="do not move the artifacts/latest.json pointer to this run")
    ap.add_argument("--no-shap", action="store_true", help="skip SHAP summary generation")
    ap.add_argument("--save-processed", action="store_true", help="write the feature table to paths.processed_dir")
    a = ap.parse_args()
    cfg = load_config(a.config)
    for key, val in (("split.strategy", a.split), ("data.provenance", a.provenance), ("validation.mode", a.validation_mode),
                     ("seed", a.seed), ("paths.artifacts_dir", a.artifacts_dir), ("paths.reports_dir", a.reports_dir)):
        if val is not None:
            cfg.set(key, val)
    try:
        out = run_training(cfg, a.input, set_latest=not a.no_set_latest, save_processed=a.save_processed, with_shap=not a.no_shap)
    except (DataLoadError, ValidationError, ValueError, FileNotFoundError) as exc:
        print(f"TRAINING FAILED: {exc}", file=sys.stderr)
        return 1
    m = out["summary"]["classifier"]["metrics"]
    print(f"\nRun {out['run_id']} finished in {out['elapsed_s']:.1f}s")
    print(out["summary"]["provenance_banner"])
    print(f"XGBoost test: accuracy={m['accuracy']:.3f} macro-F1={m['macro_f1']:.3f} balanced-acc={m['balanced_accuracy']:.3f}")
    for name, b in out["summary"]["classifier"]["baselines"].items():
        print(f"  baseline {name}: accuracy={b['accuracy']:.3f} macro-F1={b['macro_f1']:.3f}")
    print(f"K-Means k={out['clustering']['selected_k']}; anomaly train flag rate={out['anomaly']['train_flag_rate']:.3f}")
    print(f"Artifacts: {out['run_dir']}\nReports:   {out['reports_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
