"""Classifier variants used for tuning and cross-validation (softprob, cumulative-ordinal, regression-ordinal).

Every variant exposes the same ``predict_proba`` -> (n, 4) class probabilities so metrics, plots and (if ever shipped)
inference keep one 4-class structure. All fitting happens on the rows passed to ``fit_classifier`` only; early stopping
uses an explicit validation set that the caller must keep disjoint from any final test data.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score

from .config import Config
from .schema import STRESS_LEVELS

N_CLASSES = len(STRESS_LEVELS)
KINDS = ("softprob", "cumulative", "regression")


def balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency weights (computed from ``y`` only, i.e. from TRAIN labels)."""
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    counts[counts == 0] = np.nan
    cw = np.nan_to_num(len(y) / (N_CLASSES * counts), nan=0.0)
    return cw[y]


@dataclass(frozen=True)
class ClassifierSpec:
    kind: str = "softprob"
    params: dict = field(default_factory=dict)       # xgboost params (objective is set per kind)
    num_boost_round: int = 300
    class_weight: str = "none"                      # none | balanced
    cutpoints: tuple | None = None                  # regression kind only: 3 increasing thresholds on the 0..3 scale
    missing_reference_augmentation: float = 0.0     # fraction of training rows duplicated with cross-plant features blanked

    def to_dict(self) -> dict:
        return {"kind": self.kind, "params": dict(self.params), "num_boost_round": int(self.num_boost_round),
                "class_weight": self.class_weight, "cutpoints": list(self.cutpoints) if self.cutpoints else None,
                "missing_reference_augmentation": float(self.missing_reference_augmentation)}

    @classmethod
    def from_dict(cls, d: dict) -> "ClassifierSpec":
        return cls(kind=d.get("kind", "softprob"), params=dict(d.get("params", {})), num_boost_round=int(d.get("num_boost_round", 300)),
                   class_weight=d.get("class_weight", "none"), cutpoints=tuple(d["cutpoints"]) if d.get("cutpoints") else None,
                   missing_reference_augmentation=float(d.get("missing_reference_augmentation", 0.0)))

    def label(self) -> str:
        p = self.params
        return (f"{self.kind}/{self.class_weight} d{p.get('max_depth')} mcw{p.get('min_child_weight')} eta{p.get('eta')} "
                f"ss{p.get('subsample')} cs{p.get('colsample_bytree')} l{p.get('reg_lambda')} n{self.num_boost_round}")


def spec_from_config(cfg: Config) -> ClassifierSpec:
    params = dict(cfg.require("classifier.xgboost_params"))
    return ClassifierSpec(kind=str(cfg.get("classifier.kind", "softprob")), params=params,
                          num_boost_round=int(cfg.get("classifier.num_boost_round", 300)),
                          class_weight=str(cfg.get("classifier.class_weight", "none")).lower(),
                          missing_reference_augmentation=float(cfg.get("classifier.missing_reference_augmentation", 0.0) or 0.0))


def augment_missing_reference(X: pd.DataFrame, y: np.ndarray, fraction: float, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Append a random ``fraction`` of the TRAINING rows again with every cross-plant (``*_ref_dev*``) feature set to NaN,
    so the trees learn a fallback for readings that arrive without reference-plant context (controls offline). Test /
    validation rows are never augmented."""
    cols = [c for c in X.columns if "_ref_dev" in c]
    if fraction <= 0 or not cols or len(X) == 0:
        return X, y
    pick = np.random.default_rng(seed).random(len(X)) < fraction
    extra = X[pick].copy()
    extra[cols] = np.nan
    return pd.concat([X, extra]), np.concatenate([np.asarray(y), np.asarray(y)[pick]])


def _cum_probs(p_gt: np.ndarray) -> np.ndarray:
    """P(y>k) for k=0..2 -> 4 class probabilities (monotone-clipped so every probability is >= 0)."""
    p = np.clip(p_gt, 0.0, 1.0)
    p = np.minimum.accumulate(p, axis=1)             # P(y>0) >= P(y>1) >= P(y>2)
    out = np.empty((p.shape[0], N_CLASSES))
    out[:, 0] = 1.0 - p[:, 0]
    out[:, 1] = p[:, 0] - p[:, 1]
    out[:, 2] = p[:, 1] - p[:, 2]
    out[:, 3] = p[:, 2]
    return out


def regression_probs(score: np.ndarray, cutpoints, scale: float = 0.35) -> np.ndarray:
    """Soft 4-class probabilities from a 0..3 regression score: softmax of minus the distance to each cut-point bin
    (0 inside the bin). The arg-max equals the cut-point decision rule (ties resolved to the lowest class)."""
    cp = np.asarray(cutpoints, dtype=float)
    d = np.stack([np.where((score >= lo) & (score < hi), 0.0, np.minimum(np.abs(score - lo), np.abs(score - hi)))
                  for lo, hi in zip(np.concatenate([[-np.inf], cp]), np.concatenate([cp, [np.inf]]))], axis=1)
    logits = -d / scale
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


class FittedClassifier:
    def __init__(self, spec: ClassifierSpec, boosters: list, names: list[str], best: list):
        self.spec, self.boosters, self.names, self.best = spec, boosters, list(names), best
        known = [n for n in best if n]
        self.best_rounds = int(round(float(np.mean(known)))) if known else None

    def _predict(self, i: int, d) -> np.ndarray:
        n = self.best[i]
        return self.boosters[i].predict(d, iteration_range=(0, n)) if n else self.boosters[i].predict(d)

    def raw(self, X: pd.DataFrame) -> np.ndarray:
        d = xgb.DMatrix(X[self.names], feature_names=self.names)
        if self.spec.kind == "cumulative":
            return np.stack([self._predict(i, d) for i in range(len(self.boosters))], axis=1)
        return self._predict(0, d)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        r = self.raw(X)
        if self.spec.kind == "softprob":
            return r
        if self.spec.kind == "cumulative":
            return _cum_probs(r)
        return regression_probs(r, self.spec.cutpoints or (0.5, 1.5, 2.5))


def _train(params, dtrain, rounds, dval, early_stopping_rounds):
    if dval is not None and early_stopping_rounds:
        b = xgb.train(params, dtrain, num_boost_round=rounds, evals=[(dval, "val")],
                      early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
        return b, int(b.best_iteration) + 1
    return xgb.train(params, dtrain, num_boost_round=rounds, verbose_eval=False), None


def fit_classifier(spec: ClassifierSpec, X: pd.DataFrame, y: np.ndarray, seed: int,
                   X_val: pd.DataFrame | None = None, y_val: np.ndarray | None = None,
                   early_stopping_rounds: int | None = None) -> FittedClassifier:
    """Fit on (X, y). Early stopping (optional) monitors the validation loss of the variant's own objective.

    When early stopping is used, predictions use only the first ``best_iteration + 1`` trees (explicit
    ``iteration_range``, so behaviour does not depend on the xgboost version)."""
    if spec.kind not in KINDS:
        raise ValueError(f"unknown classifier kind {spec.kind!r}")
    names = list(X.columns)
    X, y = augment_missing_reference(X, y, spec.missing_reference_augmentation, seed)
    w = balanced_sample_weights(y) if spec.class_weight == "balanced" else None
    wv = (balanced_sample_weights(y_val) if (spec.class_weight == "balanced" and y_val is not None) else None)
    base = {k: v for k, v in spec.params.items() if k not in ("objective", "eval_metric", "num_class")}
    base["seed"] = seed
    use_val = X_val is not None and y_val is not None
    boosters, best = [], []
    if spec.kind == "softprob":
        p = {**base, "objective": "multi:softprob", "eval_metric": "mlogloss", "num_class": N_CLASSES}
        dtr = xgb.DMatrix(X, label=y, weight=w, feature_names=names)
        dva = xgb.DMatrix(X_val, label=y_val, weight=wv, feature_names=names) if use_val else None
        b, n = _train(p, dtr, spec.num_boost_round, dva, early_stopping_rounds)
        boosters.append(b); best.append(n)
    elif spec.kind == "cumulative":
        for k in range(N_CLASSES - 1):
            p = {**base, "objective": "binary:logistic", "eval_metric": "logloss"}
            dtr = xgb.DMatrix(X, label=(y > k).astype(float), weight=w, feature_names=names)
            dva = xgb.DMatrix(X_val, label=(y_val > k).astype(float), weight=wv, feature_names=names) if use_val else None
            b, n = _train(p, dtr, spec.num_boost_round, dva, early_stopping_rounds)
            boosters.append(b); best.append(n)
    else:
        p = {**base, "objective": "reg:squarederror", "eval_metric": "rmse"}
        dtr = xgb.DMatrix(X, label=y.astype(float), weight=w, feature_names=names)
        dva = xgb.DMatrix(X_val, label=y_val.astype(float), weight=wv, feature_names=names) if use_val else None
        b, n = _train(p, dtr, spec.num_boost_round, dva, early_stopping_rounds)
        boosters.append(b); best.append(n)
    return FittedClassifier(spec, boosters, names, best)


def tune_cutpoints(score: np.ndarray, y: np.ndarray, start=(0.5, 1.5, 2.5), span: float = 0.9, step: float = 0.05,
                   sweeps: int = 3) -> tuple:
    """Coordinate search of 3 increasing cut-points maximising macro-F1 on (score, y). Use VALIDATION data only."""
    cp = list(start)
    labels = list(range(N_CLASSES))

    def f1(c):
        pred = np.digitize(score, c)
        return f1_score(y, pred, labels=labels, average="macro", zero_division=0)

    best = f1(cp)
    for _ in range(sweeps):
        for i in range(3):
            lo = (cp[i - 1] + 0.05) if i > 0 else start[i] - span
            hi = (cp[i + 1] - 0.05) if i < 2 else start[i] + span
            for v in np.arange(max(lo, start[i] - span), min(hi, start[i] + span) + 1e-9, step):
                trial = cp.copy(); trial[i] = float(v)
                s = f1(trial)
                if s > best + 1e-12:
                    best, cp = s, trial
    return tuple(round(float(c), 4) for c in cp)
