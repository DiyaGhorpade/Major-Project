"""Model C: Isolation Forest anomaly detection (unsupervised; helper label used for evaluation only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .config import Config
from .evaluate import anomaly_metrics
from .preprocessing import make_numeric_pipeline, model_feature_list, select_matrix
from .splitters import SplitResult

SCORE_CONVENTION = ("anomaly_score = -IsolationForest.decision_function(x). HIGHER = MORE anomalous. "
                    "is_anomaly = anomaly_score > 0 (decision threshold 0.0, set by the contamination fraction on the "
                    "training data). Scores are relative to training data and are not probabilities.")
ANOMALY_NOTE = ("An anomaly flag means 'unusual combination of readings' - possibly a sensor / wiring / data-quality "
                "problem or an unusual environment.")


def train_anomaly(feat_df: pd.DataFrame, split: SplitResult, cfg: Config) -> dict:
    names = model_feature_list(cfg, "anomaly")
    seed = int(cfg.get("seed", 42))
    contamination = cfg.get("anomaly.contamination", 0.02)
    X_tr = select_matrix(feat_df.loc[split.train_mask], names)
    pipe = make_numeric_pipeline(scale=True).fit(X_tr)
    Z = pipe.transform(X_tr)
    model = IsolationForest(
        n_estimators=int(cfg.get("anomaly.n_estimators", 200)),
        max_samples=cfg.get("anomaly.max_samples", "auto"),
        contamination=contamination if contamination == "auto" else float(contamination),
        random_state=seed, n_jobs=1,
    ).fit(Z)

    def score(frame_mask) -> np.ndarray:
        Xm = select_matrix(feat_df.loc[frame_mask], names)
        return -model.decision_function(pipe.transform(Xm))

    train_scores = -model.decision_function(Z)
    label_col = cfg.get("anomaly.label_column", "is_synthetic_anomaly")
    eval_scope = cfg.get("anomaly.evaluate_on", "test")
    eval_mask = split.test_mask if eval_scope == "test" else (split.train_mask | split.test_mask)
    ev_scores = score(eval_mask)
    ev_flags = (ev_scores > 0).astype(int)

    metrics, metrics_train = None, None
    if label_col in feat_df.columns:
        helper_ev = feat_df.loc[eval_mask, label_col].reset_index(drop=True)
        metrics = anomaly_metrics(helper_ev, ev_scores, ev_flags, str(cfg.get("data.provenance", "synthetic")))
        helper_tr = feat_df.loc[split.train_mask, label_col].reset_index(drop=True)
        metrics_train = anomaly_metrics(helper_tr, train_scores, (train_scores > 0).astype(int))

    return {
        "pipeline": pipe, "model": model, "feature_names": names,
        "contamination": contamination, "threshold": {"decision_rule": "anomaly_score > 0", "value": 0.0,
                                                      "isolation_forest_offset": float(model.offset_)},
        "score_convention": SCORE_CONVENTION, "note": ANOMALY_NOTE,
        "train_score_quantiles": {q: float(np.quantile(train_scores, q)) for q in (0.5, 0.9, 0.98, 0.99)},
        "train_flag_rate": float((train_scores > 0).mean()),
        "eval_scores": ev_scores, "eval_scope": eval_scope, "eval_mask": eval_mask,
        "metrics_vs_helper_label": metrics, "metrics_vs_helper_label_train": metrics_train,
        "n_train": int(len(Z)),
        "preprocessing": {"imputation": "median (fit on train)", "scaling": "StandardScaler (fit on train)"},
    }
