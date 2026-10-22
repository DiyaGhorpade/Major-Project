"""Metric computation (no plotting, no I/O)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_recall_fscore_support, precision_score, recall_score, roc_auc_score,
)

from .schema import STRESS_LEVELS


def classification_metrics(y_true, proba, provenance: str = "synthetic") -> dict:
    """y_true: int class indices (0..3); proba: (n, 4) probabilities."""
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    y_pred = proba.argmax(axis=1)
    labels = list(range(len(STRESS_LEVELS)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    present = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "metric_scope": f"{provenance}-data metrics (NOT validated plant-health performance)" if provenance != "real"
        else "real-data metrics (verify labelling protocol before relying on them)",
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class": {
            STRESS_LEVELS[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i in labels
        },
        "confusion_matrix": {"labels": STRESS_LEVELS, "matrix": cm.tolist(), "rows": "true", "cols": "predicted"},
        "class_counts": {STRESS_LEVELS[i]: int((y_true == i).sum()) for i in labels},
        "classes_absent_from_eval_set": [STRESS_LEVELS[i] for i in labels if i not in present],
    }


def anomaly_metrics(y_helper, score, flag, provenance: str = "synthetic") -> dict | None:
    """Evaluate against the helper label ONLY when it exists and has both classes."""
    y = pd.Series(y_helper).dropna().astype(int)
    if len(y) == 0 or y.nunique() < 2:
        return None
    idx = y.index
    sc = np.asarray(pd.Series(score, index=pd.Series(y_helper).index).loc[idx], dtype=float)
    fl = np.asarray(pd.Series(flag, index=pd.Series(y_helper).index).loc[idx], dtype=int)
    yt = y.to_numpy()
    return {
        "note": "Evaluated against synthetic helper labels (is_synthetic_anomaly); the label was never a model input.",
        "n_samples": int(len(yt)),
        "n_helper_positives": int(yt.sum()),
        "precision": float(precision_score(yt, fl, zero_division=0)),
        "recall": float(recall_score(yt, fl, zero_division=0)),
        "f1": float(f1_score(yt, fl, zero_division=0)),
        "pr_auc": float(average_precision_score(yt, sc)),
        "roc_auc": float(roc_auc_score(yt, sc)),
        "n_flagged": int(fl.sum()),
        "positive_rate_baseline_pr_auc": float(yt.mean()),
    }
