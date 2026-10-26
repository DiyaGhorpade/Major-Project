#!/usr/bin/env python
"""Cross-validated evaluation of the configured classifier (chronological hold-out, leave-one-session-out, unseen plants).

Reports mean +/- sd across folds for macro-F1, accuracy, balanced accuracy and per-class precision/recall/F1 plus the
confusion matrix, next to the logistic-regression and majority baselines. Read-only: writes a report under
reports/cv/, never touches artifacts/.
"""
import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
from plant_ml.classifier_variants import ClassifierSpec, spec_from_config
from plant_ml.config import load_config
from plant_ml.cv_evaluation import logreg_fit_predict, majority_fit_predict, run_cv, xgb_fit_predict
from plant_ml.data_loader import load_csv
from plant_ml.feature_engineering import build_features
from plant_ml.reports import write_json
from plant_ml.temporal_smoothing import smoothing_params
from plant_ml.validation import validate_frame

SCHEMES = ("chronological", "leave_one_session", "group_kfold_plants")


def main() -> int:
    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input")
    ap.add_argument("--config")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="seeds averaged inside each fold")
    ap.add_argument("--candidate", help="tuning report JSON (scripts/tune_classifier.py): also evaluate its candidate spec/features")
    ap.add_argument("--output", help="report path (default: reports/cv/cv_<timestamp>.json)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    raw = load_csv(a.input or cfg.path("paths.default_input"), cfg.get("timezone", "Asia/Kolkata"), require_labels=True)
    clean = validate_frame(raw, mode=cfg.get("validation.mode", "warn"), require_labels=True).clean.reset_index(drop=True)
    feat = build_features(clean, cfg)
    models = {"xgboost_configured": xgb_fit_predict(spec_from_config(cfg)),
              "xgboost_configured_unsmoothed": xgb_fit_predict(spec_from_config(cfg)),
              "logistic_regression": logreg_fit_predict, "majority_class": majority_fit_predict}
    smoothing = {"xgboost_configured": smoothing_params(cfg)}
    if a.candidate:
        cand = json.loads(Path(a.candidate).read_text(encoding="utf-8"))["candidate"]
        models["xgboost_candidate"] = xgb_fit_predict(ClassifierSpec.from_dict(cand["spec"]), cand["features"])
        smoothing["xgboost_candidate"] = smoothing_params(cfg)
    res = run_cv(feat, cfg, models, SCHEMES, seeds=a.seeds, smoothing=smoothing)
    out = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "seeds": a.seeds, "results": res,
           "note": "leave_one_session and group_kfold_plants folds train on later sessions: robustness, not forecasting. "
                   "macro_f1 is averaged over the classes present in each fold's test set. xgboost_configured applies the configured "
                   "temporal smoothing (transitions fitted on each fold's training labels); *_unsmoothed is the same model without it."}
    path = a.output or str(cfg.path("paths.reports_dir") / "cv" / f"cv_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_json(Path(path), out)
    for scheme, models_res in res.items():
        print(f"\n== {scheme} ==")
        for name, r in models_res.items():
            s = r["summary"]
            f = lambda k: f"{s[k]['mean']:.3f}+-{s[k]['sd']:.3f}"  # noqa: E731
            print(f"{name:22s} macroF1 {f('macro_f1')}  acc {f('accuracy')}  balAcc {f('balanced_accuracy')}  folds={s['n_folds']}")
    print("\nreport:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
