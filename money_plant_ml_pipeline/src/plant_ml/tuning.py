"""Validation-based hyper-parameter search that never sees the test session.

The search receives ONLY the training rows (callers pass ``feat[train_mask]``), builds inner validation folds from the
training sessions, and scores every candidate by validation macro-F1. Early stopping chooses the number of rounds.
The chronological test session is evaluated once, afterwards, by ``confirm_on_test``.
"""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .classifier_variants import ClassifierSpec, fit_classifier, tune_cutpoints
from .cv_evaluation import session_ids
from .evaluate import classification_metrics
from .labels import encode_stress_level
from .preprocessing import select_matrix

N_CLASSES = 4
SEARCH_SPACE = {
    "max_depth": [2, 3, 4, 5, 6],
    "min_child_weight": [1, 3, 5, 10, 20],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.4, 0.6, 0.8, 1.0],
    "eta": [0.03, 0.05, 0.1],
    "reg_lambda": [1.0, 2.0, 5.0, 10.0, 20.0],
}


def inner_folds(train_feat: pd.DataFrame) -> list[dict]:
    """Validation folds over the training sessions. Session 1 is never a validation fold (it contains no 'high' rows).
    The fold validating on the LAST training session is strictly causal; the others train on a later session and are
    flagged ``uses_future_data`` (they stabilise the estimate)."""
    sess = session_ids(train_feat).to_numpy()
    ids = sorted(set(sess.tolist()))
    return [{"val_session": int(s), "train": sess != s, "val": sess == s, "uses_future_data": any(i > s for i in ids if i != s)}
            for s in ids[1:]]


def sample_specs(base: ClassifierSpec, n: int, seed: int, kinds=("softprob",), weights=("none", "balanced")) -> list[ClassifierSpec]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        hp = {k: v[int(rng.integers(len(v)))] for k, v in SEARCH_SPACE.items()}
        for kind in kinds:
            for w in weights:
                out.append(replace(base, kind=kind, class_weight=w, params={**base.params, **hp}))
    return out


def _macro_f1_present(y, pred) -> float:
    present = sorted(set(np.asarray(y).tolist()))
    return float(f1_score(y, pred, labels=present, average="macro", zero_division=0))


def score_spec(spec: ClassifierSpec, X: pd.DataFrame, y: np.ndarray, folds: list[dict], seed: int,
               max_rounds: int = 800, patience: int = 40) -> dict:
    """Validation score of one spec: mean over inner folds of present-class macro-F1 at the early-stopped iteration."""
    spec = replace(spec, num_boost_round=max_rounds)
    fits, scores_val = [], []
    for f in folds:
        m = fit_classifier(spec, X[f["train"]], y[f["train"]], seed, X[f["val"]], y[f["val"]], patience)
        fits.append(m)
        scores_val.append(m.raw(X[f["val"]]))
    out_spec = replace(spec, num_boost_round=int(np.mean([m.best_rounds for m in fits]) * 1.0) or max_rounds)
    if spec.kind == "regression":
        pooled_s = np.concatenate(scores_val)
        pooled_y = np.concatenate([y[f["val"]] for f in folds])
        cp = tune_cutpoints(pooled_s, pooled_y)
        out_spec = replace(out_spec, cutpoints=cp)
        preds = [np.digitize(s, cp) for s in scores_val]
    else:
        preds = [(m.predict_proba(X[f["val"]])).argmax(axis=1) for m, f in zip(fits, folds)]
    per_fold = [_macro_f1_present(y[f["val"]], p) for f, p in zip(folds, preds)]
    return {"spec": out_spec, "val_macro_f1": float(np.mean(per_fold)), "val_macro_f1_per_fold": per_fold,
            "best_rounds": [m.best_rounds for m in fits]}


def search(train_feat: pd.DataFrame, names: list[str], specs: list[ClassifierSpec], seed: int = 42, log=print) -> list[dict]:
    y_all = encode_stress_level(train_feat["stress_level"])
    ok = y_all.notna().to_numpy()
    tf = train_feat.loc[ok].reset_index(drop=True)
    X = select_matrix(tf, names)
    y = y_all[ok].astype(int).to_numpy()
    folds = inner_folds(tf)
    results = []
    t0 = time.time()
    for i, spec in enumerate(specs):
        r = score_spec(spec, X, y, folds, seed)
        r["trial"] = i
        results.append(r)
        if log:
            log(f"[{i + 1}/{len(specs)} {time.time() - t0:5.0f}s] val macroF1 {r['val_macro_f1']:.3f} "
                f"{[round(v, 3) for v in r['val_macro_f1_per_fold']]} rounds {r['best_rounds']} {r['spec'].label()}")
    return sorted(results, key=lambda r: -r["val_macro_f1"])


def confirm_on_test(feat, split, names, spec, seeds):
    """Fit on the full train partition (fixed rounds, no early stopping) and score the test session once per seed."""
    y_all = encode_stress_level(feat["stress_level"])
    ok = y_all.notna()
    tr, te = (split.train_mask & ok).to_numpy(), (split.test_mask & ok).to_numpy()
    X = select_matrix(feat, names)
    rows = []
    for s in seeds:
        m = fit_classifier(spec, X[tr], y_all[tr].astype(int).to_numpy(), s)
        met = classification_metrics(y_all[te].astype(int).to_numpy(), m.predict_proba(X[te]))
        rows.append({"seed": int(s), "macro_f1": met["macro_f1"], "accuracy": met["accuracy"],
                     "balanced_accuracy": met["balanced_accuracy"], "per_class_f1": {k: v["f1"] for k, v in met["per_class"].items()}})
    agg = {k: {"mean": float(np.mean([r[k] for r in rows])), "sd": float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else 0.0}
           for k in ("macro_f1", "accuracy", "balanced_accuracy")}
    return {"per_seed": rows, "summary": agg}
