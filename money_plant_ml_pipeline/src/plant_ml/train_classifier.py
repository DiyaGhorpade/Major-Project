"""Model A: XGBoost multiclass stress classifier (native xgboost.train API)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import Config
from .evaluate import classification_metrics
from .labels import encode_stress_level
from .preprocessing import make_numeric_pipeline, model_feature_list, select_matrix
from .schema import INDEX_TO_LABEL, LABEL_TO_INDEX, STRESS_LEVELS
from .splitters import SplitResult

N_CLASSES = len(STRESS_LEVELS)


def _full_proba(clf, X, classes) -> np.ndarray:
    raw = clf.predict_proba(X)
    out = np.zeros((raw.shape[0], N_CLASSES))
    for j, c in enumerate(classes):
        out[:, int(c)] = raw[:, j]
    return out


def train_classifier(feat_df: pd.DataFrame, split: SplitResult, cfg: Config) -> dict:
    names = model_feature_list(cfg, "classifier")
    y_all = encode_stress_level(feat_df["stress_level"])
    labelled = y_all.notna()
    tr = split.train_mask & labelled
    te = split.test_mask & labelled
    if tr.sum() == 0 or te.sum() == 0:
        raise ValueError("No labelled rows in the train or test partition; cannot train the classifier")

    X_tr, X_te = select_matrix(feat_df.loc[tr], names), select_matrix(feat_df.loc[te], names)
    y_tr, y_te = y_all[tr].astype(int).to_numpy(), y_all[te].astype(int).to_numpy()

    class_weight = str(cfg.get("classifier.class_weight", "none")).lower()
    weights = None
    if class_weight == "balanced":
        counts = np.bincount(y_tr, minlength=N_CLASSES).astype(float)
        counts[counts == 0] = np.nan
        cw = np.nan_to_num(len(y_tr) / (N_CLASSES * counts), nan=0.0)
        weights = cw[y_tr]
    elif class_weight != "none":
        raise ValueError("classifier.class_weight must be 'none' or 'balanced'")

    seed = int(cfg.get("seed", 42))
    params = dict(cfg.require("classifier.xgboost_params"))
    params.update({"num_class": N_CLASSES, "seed": seed})
    dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=weights, feature_names=names)
    dtest = xgb.DMatrix(X_te, feature_names=names)
    booster = xgb.train(params, dtrain, num_boost_round=int(cfg.get("classifier.num_boost_round", 300)))
    proba = booster.predict(dtest)
    provenance = str(cfg.get("data.provenance", "synthetic"))
    metrics = classification_metrics(y_te, proba, provenance)

    # --- simple baselines (same split, same features, fitted on TRAIN only) ---
    baselines: dict = {}
    for name in cfg.get("classifier.baselines", ["majority_class", "logistic_regression"]):
        if name == "majority_class":
            m = DummyClassifier(strategy="prior").fit(X_tr, y_tr)
            baselines[name] = classification_metrics(y_te, _full_proba(m, X_te, m.classes_), provenance)
        elif name == "logistic_regression":
            pipe = Pipeline(make_numeric_pipeline(scale=True).steps +
                            [("clf", LogisticRegression(max_iter=3000, random_state=seed))]).fit(X_tr, y_tr)
            baselines[name] = classification_metrics(y_te, _full_proba(pipe, X_te, pipe.classes_), provenance)
        else:
            raise ValueError(f"Unknown baseline {name!r}")

    gain = booster.get_score(importance_type="gain")
    total = float(sum(gain.values())) or 1.0
    importance = {k: v / total for k, v in sorted(gain.items(), key=lambda kv: -kv[1])}

    pred_frame = pd.DataFrame({
        "plant_id": feat_df.loc[te, "plant_id"].to_numpy(), "timestamp": feat_df.loc[te, "timestamp"].to_numpy(),
        "y_true": [INDEX_TO_LABEL[i] for i in y_te], "y_pred": [INDEX_TO_LABEL[i] for i in proba.argmax(axis=1)],
    })
    return {
        "booster": booster, "feature_names": names, "metrics": metrics, "baselines": baselines,
        "feature_importance_gain": importance, "predictions_test": pred_frame,
        "train_class_counts": {STRESS_LEVELS[i]: int((y_tr == i).sum()) for i in range(N_CLASSES)},
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "preprocessing": {"imputation": "none - XGBoost handles NaN natively (missing-value direction learned in training)",
                          "scaling": "none (tree model)", "class_weight": class_weight},
        "label_mapping": dict(LABEL_TO_INDEX),
        "xgboost_params": params, "num_boost_round": int(cfg.get("classifier.num_boost_round", 300)),
        "X_test": X_te,
    }
