#!/usr/bin/env python
"""Validation-based tuning of the stress classifier. The chronological test session is NOT used for selection.

Stages
  1. random search (softprob, cumulative-ordinal, regression-ordinal; class_weight none/balanced) on inner folds built
     from the TRAIN sessions only, with early stopping to choose the number of rounds;
  2. optional feature-set ablation for the best spec;
  3. the winner (by VALIDATION macro-F1) and the currently configured spec are each retrained on the full train
     partition and evaluated ONCE on the test session for several seeds. A candidate is recommended only if it beats the
     configured spec by more than the margin in mean macro-F1 without losing balanced accuracy.
Writes reports/tuning/tuning_<timestamp>.json (including the candidate spec); does not change config or artifacts.
"""
import argparse
import sys
import warnings
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
from plant_ml import tuning
from plant_ml.classifier_variants import spec_from_config
from plant_ml.config import load_config
from plant_ml.data_loader import load_csv
from plant_ml.feature_engineering import build_features
from plant_ml.preprocessing import model_feature_list
from plant_ml.reports import write_json
from plant_ml.splitters import split_frame
from plant_ml.validation import validate_frame

ABLATIONS = {
    "all_features": lambda names: names,
    "no_lag1": lambda names: [n for n in names if not n.endswith("_lag1")],
    "no_lag1_diff1_trend3": lambda names: [n for n in names if not n.endswith(("_lag1", "_diff1", "_trend3"))],
    "no_roll3": lambda names: [n for n in names if "_roll3_" not in n],
    "no_calendar": lambda names: [n for n in names if n not in ("hour", "dayofweek", "hour_sin", "hour_cos")],
}


def main() -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input")
    ap.add_argument("--config")
    ap.add_argument("--trials", type=int, default=40, help="random hyper-parameter draws per (class_weight) for softprob")
    ap.add_argument("--ordinal-trials", type=int, default=15, help="draws for each ordinal kind")
    ap.add_argument("--seed", type=int, default=42, help="search seed")
    ap.add_argument("--confirm-seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--margin", type=float, default=0.02, help="min mean macro-F1 gain on the test session to recommend")
    ap.add_argument("--nthread", type=int, default=8)
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--output")
    a = ap.parse_args()

    cfg = load_config(a.config)
    raw = load_csv(a.input or cfg.path("paths.default_input"), cfg.get("timezone", "Asia/Kolkata"), require_labels=True)
    clean = validate_frame(raw, mode=cfg.get("validation.mode", "warn"), require_labels=True).clean.reset_index(drop=True)
    feat = build_features(clean, cfg)
    split = split_frame(feat, cfg)
    names = model_feature_list(cfg, "classifier")
    current = spec_from_config(cfg)
    base = replace(current, params={**current.params, "nthread": a.nthread})
    train_feat = feat[split.train_mask].reset_index(drop=True)          # test rows are physically absent from the search
    print(f"train rows {len(train_feat)}; inner folds: validate on sessions {[f['val_session'] for f in tuning.inner_folds(train_feat)]}")

    specs = [replace(base, class_weight=w, kind=k) for k in ("softprob", "cumulative", "regression") for w in ("none", "balanced")]
    specs += tuning.sample_specs(base, a.trials, a.seed, kinds=("softprob",))
    specs += tuning.sample_specs(base, a.ordinal_trials, a.seed + 1, kinds=("cumulative", "regression"))
    results = tuning.search(train_feat, names, specs, seed=a.seed)
    best = results[0]
    print("\nTop 5 by validation macro-F1:")
    for r in results[:5]:
        print(f"  {r['val_macro_f1']:.3f}  {r['spec'].label()}")
    by_kind = {}
    for r in results:
        by_kind.setdefault((r["spec"].kind, r["spec"].class_weight), r)
    print("Best per (kind, class_weight):", {f"{k[0]}/{k[1]}": round(v["val_macro_f1"], 3) for k, v in by_kind.items()})

    ablation = {}
    best_names = names
    if not a.no_ablation:
        print("\nFeature-set ablation (best spec):")
        for label, fn in ABLATIONS.items():
            sub = fn(names)
            r = tuning.search(train_feat, sub, [best["spec"]], seed=a.seed, log=None)[0]
            ablation[label] = {"n_features": len(sub), "val_macro_f1": r["val_macro_f1"]}
            print(f"  {label:24s} n={len(sub):3d} val macroF1 {r['val_macro_f1']:.3f}")
        top = max(ablation, key=lambda k: ablation[k]["val_macro_f1"])
        # prefer the smaller set only when it is not worse on validation
        if ablation[top]["val_macro_f1"] > ablation["all_features"]["val_macro_f1"] + 0.005:
            best_names = ABLATIONS[top](names)
            print(f"  -> using feature set {top}")

    cand_spec = best["spec"]
    print("\nConfirming on the test session (once per seed) ...")
    cur = tuning.confirm_on_test(feat, split, names, current, a.confirm_seeds)
    cand = tuning.confirm_on_test(feat, split, best_names, cand_spec, a.confirm_seeds)
    for label, r in (("current", cur), ("candidate", cand)):
        s = r["summary"]
        print(f"  {label:9s} macroF1 {s['macro_f1']['mean']:.3f}+-{s['macro_f1']['sd']:.3f}  acc {s['accuracy']['mean']:.3f}  "
              f"balAcc {s['balanced_accuracy']['mean']:.3f}+-{s['balanced_accuracy']['sd']:.3f}")
    gain = cand["summary"]["macro_f1"]["mean"] - cur["summary"]["macro_f1"]["mean"]
    bal_ok = cand["summary"]["balanced_accuracy"]["mean"] >= cur["summary"]["balanced_accuracy"]["mean"] - 1e-9
    ship = bool(gain > a.margin and bal_ok and cand_spec.kind == "softprob")
    print(f"\nmacro-F1 gain {gain:+.3f} (margin {a.margin}); balanced-accuracy not worse: {bal_ok}; "
          f"softprob (inference-compatible): {cand_spec.kind == 'softprob'} -> recommend shipping: {ship}")

    out = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "search_seed": a.seed,
           "inner_folds": [{k: v for k, v in f.items() if k not in ("train", "val")} for f in tuning.inner_folds(train_feat)],
           "n_trials": len(results),
           "trials": [{"val_macro_f1": r["val_macro_f1"], "per_fold": r["val_macro_f1_per_fold"], "best_rounds": r["best_rounds"],
                       "spec": r["spec"].to_dict()} for r in results],
           "best_per_kind": {f"{k[0]}/{k[1]}": {"val_macro_f1": v["val_macro_f1"], "spec": v["spec"].to_dict()} for k, v in by_kind.items()},
           "ablation": ablation, "candidate": {"spec": cand_spec.to_dict(), "features": best_names},
           "current": {"spec": current.to_dict()}, "test_session": {"current": cur, "candidate": cand},
           "macro_f1_gain": gain, "margin": a.margin, "balanced_accuracy_not_worse": bal_ok, "recommend_ship": ship}
    path = Path(a.output) if a.output else cfg.path("paths.reports_dir") / "tuning" / f"tuning_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, out)
    print("report:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
