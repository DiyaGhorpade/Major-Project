"""Cross-validated evaluation: fold construction, per-fold fitting and mean +/- sd aggregation.

Three schemes are reported next to each other:
* ``chronological``       the primary hold-out (train sessions 1-3, test session 4; definition unchanged from training);
* ``leave_one_session``   train on three sessions, test on the fourth. Folds that train on LATER sessions are
                          flagged ``uses_future_data`` - they measure robustness, not forecasting;
* ``group_kfold_plants``  GroupKFold by plant, stratified so every fold's test set holds one plant from every
                          experimental group (unseen-plant generalisation).

Every model is fitted INSIDE the fold on its training rows only (imputers/scalers included); the fold functions only
build boolean masks and never touch the labels' distribution.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .classifier_variants import ClassifierSpec, fit_classifier
from .config import Config
from .evaluate import classification_metrics
from .labels import encode_stress_level
from .preprocessing import make_numeric_pipeline, model_feature_list, select_matrix
from .schema import STRESS_LEVELS, plant_to_group
from .splitters import split_frame

N_CLASSES = len(STRESS_LEVELS)
SCALARS = ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy")


@dataclass
class Fold:
    name: str
    scheme: str
    train: np.ndarray                  # boolean mask aligned to the feature frame
    test: np.ndarray
    info: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- fold construction
def session_ids(feat: pd.DataFrame, n_sessions: int = 4) -> pd.Series:
    """The ``session_id`` column when present; otherwise ``n_sessions`` equal blocks of unique timestamps."""
    if "session_id" in feat.columns and feat["session_id"].notna().all():
        return feat["session_id"].astype(int)
    uniq = np.sort(feat["timestamp"].unique())
    block = np.minimum((np.arange(len(uniq)) * n_sessions) // len(uniq), n_sessions - 1) + 1
    return feat["timestamp"].map(dict(zip(uniq, block))).astype(int)


def chronological_folds(feat: pd.DataFrame, cfg: Config) -> list[Fold]:
    s = split_frame(feat, cfg, "chronological")
    return [Fold("chronological_holdout", "chronological", s.train_mask.to_numpy(), s.test_mask.to_numpy(),
                 {"uses_future_data": False, **s.info})]


def leave_one_session_out_folds(feat: pd.DataFrame) -> list[Fold]:
    sess = session_ids(feat).to_numpy()
    ids = sorted(set(sess.tolist()))
    folds = []
    for s in ids:
        later = [i for i in ids if i > s]
        folds.append(Fold(f"test_session_{s}", "leave_one_session", sess != s, sess == s,
                          {"test_session": int(s), "train_sessions": [i for i in ids if i != s],
                           "uses_future_data": bool(later),
                           "note": "trains on later sessions: robustness estimate, not a forecast" if later else
                                   "strictly causal (earlier sessions only)"}))
    return folds


def stratified_group_folds(feat: pd.DataFrame, n_splits: int = 4, seed: int = 42) -> list[Fold]:
    """GroupKFold by plant. Plants of each experimental group are shuffled (seeded) and dealt round-robin to the folds,
    so with 4 plants per group and 4 folds each test fold holds exactly one plant per group."""
    plants = sorted(int(p) for p in feat["plant_id"].unique())
    by_group: dict[str, list[int]] = {}
    for p in plants:
        by_group.setdefault(plant_to_group(p), []).append(p)
    rng = np.random.default_rng(seed)
    assign: dict[int, int] = {}
    for g in sorted(by_group):
        members = list(by_group[g])
        rng.shuffle(members)
        offset = int(rng.integers(n_splits))               # avoid always putting the first-shuffled plant in fold 0
        for i, p in enumerate(members):
            assign[p] = (i + offset) % n_splits
    fold_of = feat["plant_id"].map(assign).to_numpy()
    folds = []
    for k in range(n_splits):
        test_plants = sorted(p for p, f in assign.items() if f == k)
        if not test_plants:
            continue
        folds.append(Fold(f"plants_fold_{k}", "group_kfold_plants", fold_of != k, fold_of == k,
                          {"test_plants": test_plants, "test_groups": sorted({plant_to_group(p) for p in test_plants}),
                           "uses_future_data": True,
                           "note": "unseen plants; sessions are mixed (plants never are)"}))
    return folds


def build_folds(feat: pd.DataFrame, cfg: Config, schemes=("chronological", "leave_one_session", "group_kfold_plants"),
                seed: int | None = None) -> list[Fold]:
    seed = int(cfg.get("seed", 42)) if seed is None else seed
    out: list[Fold] = []
    if "chronological" in schemes:
        out += chronological_folds(feat, cfg)
    if "leave_one_session" in schemes:
        out += leave_one_session_out_folds(feat)
    if "group_kfold_plants" in schemes:
        out += stratified_group_folds(feat, 4, seed)
    return out


# --------------------------------------------------------------------------- model factories
FitPredict = Callable[[pd.DataFrame, np.ndarray, pd.DataFrame, int], np.ndarray]


def xgb_fit_predict(spec: ClassifierSpec, columns: list[str] | None = None) -> FitPredict:
    """``columns`` restricts the feature set (ablation candidates); default = every column passed in."""
    def fp(X_tr, y_tr, X_te, seed):
        if columns is not None:
            X_tr, X_te = X_tr[columns], X_te[columns]
        return fit_classifier(spec, X_tr, y_tr, seed).predict_proba(X_te)
    return fp


def fit_logreg(X_tr: pd.DataFrame, y_tr: np.ndarray, seed: int) -> Pipeline:
    """Median imputer + scaler + LogisticRegression, all fitted on the fold's training rows."""
    return Pipeline(make_numeric_pipeline(scale=True).steps +
                    [("clf", LogisticRegression(max_iter=3000, random_state=seed))]).fit(X_tr, y_tr)


def _full(proba: np.ndarray, classes) -> np.ndarray:
    out = np.zeros((proba.shape[0], N_CLASSES))
    for j, c in enumerate(classes):
        out[:, int(c)] = proba[:, j]
    return out


def logreg_fit_predict(X_tr, y_tr, X_te, seed):
    m = fit_logreg(X_tr, y_tr, seed)
    return _full(m.predict_proba(X_te), m.classes_)


def majority_fit_predict(X_tr, y_tr, X_te, seed):
    m = DummyClassifier(strategy="prior").fit(X_tr, y_tr)
    return _full(m.predict_proba(X_te), m.classes_)


# --------------------------------------------------------------------------- running + aggregation
def present_class_macro_f1(m: dict) -> float:
    f = [v["f1"] for v in m["per_class"].values() if v["support"] > 0]
    return float(np.mean(f)) if f else float("nan")


def _mean_sd(values) -> dict:
    a = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if a.size == 0:
        return {"mean": None, "sd": None, "n": 0}
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0, "n": int(a.size)}


def evaluate_folds(feat: pd.DataFrame, folds: list[Fold], fit_predict: FitPredict, names: list[str],
                   seeds=(42,), provenance: str = "synthetic") -> dict:
    """Run ``fit_predict`` in every fold. With several seeds, metrics are averaged over seeds per fold first, so the
    reported sd reflects fold-to-fold variation (seed noise is reported separately as ``seed_sd`` per scalar)."""
    y_all = encode_stress_level(feat["stress_level"])
    ok = y_all.notna().to_numpy()
    X_all = select_matrix(feat, names)
    fold_out = []
    for f in folds:
        tr, te = f.train & ok, f.test & ok
        if tr.sum() == 0 or te.sum() == 0:
            raise ValueError(f"fold {f.name}: empty train or test partition")
        if (tr & te).any():
            raise ValueError(f"fold {f.name}: train/test rows overlap")
        X_tr, y_tr = X_all[tr], y_all[tr].astype(int).to_numpy()
        X_te, y_te = X_all[te], y_all[te].astype(int).to_numpy()
        per_seed = []
        for sd in seeds:
            proba = fit_predict(X_tr, y_tr, X_te, int(sd))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
                m = classification_metrics(y_te, proba, provenance)
            m["macro_f1"] = present_class_macro_f1(m)
            per_seed.append(m)
        mean_m = _average_metrics(per_seed)
        mean_m["seed_sd"] = {k: float(np.std([m[k] for m in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0 for k in SCALARS}
        fold_out.append({"name": f.name, "scheme": f.scheme, "info": f.info, "n_train": int(tr.sum()), "n_test": int(te.sum()),
                         **{k: v for k, v in mean_m.items() if k not in ("metric_scope",)}})
    return {"folds": fold_out, "summary": summarise(fold_out), "seeds": [int(s) for s in seeds]}


def _average_metrics(ms: list[dict]) -> dict:
    out = dict(ms[0])
    for k in SCALARS:
        out[k] = float(np.mean([m[k] for m in ms]))
    out["per_class"] = {c: {k: (float(np.mean([m["per_class"][c][k] for m in ms])) if k != "support" else ms[0]["per_class"][c][k])
                            for k in ("precision", "recall", "f1", "support")} for c in STRESS_LEVELS}
    out["confusion_matrix"] = {**ms[0]["confusion_matrix"],
                               "matrix": np.round(np.mean([m["confusion_matrix"]["matrix"] for m in ms], axis=0), 2).tolist()}
    return out


def summarise(fold_out: list[dict]) -> dict:
    """mean +/- sd (across folds, ddof=1) for scalar metrics and per-class precision/recall/F1 (supported folds only),
    plus the pooled confusion matrix (sum over folds)."""
    summ = {k: _mean_sd([f[k] for f in fold_out]) for k in SCALARS}
    per_class = {}
    for c in STRESS_LEVELS:
        sup = [f for f in fold_out if f["per_class"][c]["support"] > 0]
        per_class[c] = {k: _mean_sd([f["per_class"][c][k] for f in sup]) for k in ("precision", "recall", "f1")}
        per_class[c]["n_folds_with_class"] = len(sup)
    cm = np.sum([np.asarray(f["confusion_matrix"]["matrix"], dtype=float) for f in fold_out], axis=0)
    return {"n_folds": len(fold_out), **summ, "per_class": per_class,
            "pooled_confusion_matrix": {"labels": list(STRESS_LEVELS), "matrix": np.round(cm, 1).tolist(),
                                        "rows": "true", "cols": "predicted"}}


def run_cv(feat: pd.DataFrame, cfg: Config, models: dict[str, FitPredict], schemes=("chronological", "leave_one_session", "group_kfold_plants"),
           seeds=(42,), names: list[str] | None = None) -> dict:
    """{scheme: {model_name: result}} for every model on every scheme's folds."""
    names = names or model_feature_list(cfg, "classifier")
    provenance = str(cfg.get("data.provenance", "synthetic"))
    folds = build_folds(feat, cfg, schemes)
    out: dict = {}
    for scheme in schemes:
        sf = [f for f in folds if f.scheme == scheme]
        out[scheme] = {name: evaluate_folds(feat, sf, fp, names, seeds, provenance) for name, fp in models.items()}
    return out
