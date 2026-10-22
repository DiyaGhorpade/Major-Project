"""Re-evaluate SAVED artifacts on a dataset using the split recorded at training time."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import silhouette_score

from . import reports as rp
from .artifact_manager import load_bundle
from .config import Config
from .data_loader import file_sha256, load_csv
from .evaluate import anomaly_metrics, classification_metrics
from .explainability import Explainer
from .feature_engineering import build_features
from .labels import encode_stress_level
from .preprocessing import select_matrix
from .splitters import split_frame
from .validation import validate_frame


def evaluate_run(cfg: Config, input_path: str | Path, run_id: str | None = None, with_shap: bool = False) -> dict:
    bundle = load_bundle(cfg, run_id)
    split_info = bundle.manifest["split"]
    provenance = bundle.manifest["training_data"]["provenance"]
    raw = load_csv(input_path, cfg.get("timezone", "Asia/Kolkata"), require_labels=True)
    clean = validate_frame(raw, mode=cfg.get("validation.mode", "warn"), require_labels=True).clean.reset_index(drop=True)
    feat = build_features(clean, cfg)

    cfg_eval = cfg.copy()
    if split_info["strategy"] == "group_holdout":
        cfg_eval.set("split.holdout_plants", split_info["holdout_plants"])
    split = split_frame(feat, cfg_eval, strategy=split_info["strategy"], cutoff=split_info.get("cutoff_timestamp"))

    same_data = file_sha256(input_path) == bundle.manifest["training_data"]["sha256"]
    out: dict = {"run_id": bundle.run_dir.name, "input": str(input_path), "same_file_as_training": same_data,
                 "split": split.info, "provenance": provenance}
    if not same_data:
        out["warning"] = ("Input differs from the training file; if it overlaps the training period the metrics are optimistic. "
                          "The recorded split cutoff/holdout is still applied.")

    names = bundle.meta["classifier"]["feature_names"]
    y = encode_stress_level(feat["stress_level"])
    te = split.test_mask & y.notna()
    X_te = select_matrix(feat.loc[te], names)
    proba = bundle.booster.predict(xgb.DMatrix(X_te, feature_names=names))
    out["classifier"] = classification_metrics(y[te].astype(int).to_numpy(), proba, provenance)

    an_names = bundle.meta["anomaly"]["feature_names"]
    score = -bundle.iforest.decision_function(bundle.anomaly_pipeline.transform(select_matrix(feat.loc[split.test_mask], an_names)))
    label_col = cfg.get("anomaly.label_column", "is_synthetic_anomaly")
    out["anomaly"] = (anomaly_metrics(feat.loc[split.test_mask, label_col].reset_index(drop=True), score, (score > 0).astype(int), provenance)
                      if label_col in feat.columns else None)
    out["anomaly_flag_rate_test"] = float((score > 0).mean())

    cl_names = bundle.meta["clustering"]["feature_names"]
    Zc = bundle.cluster_pipeline.transform(select_matrix(feat.loc[split.test_mask], cl_names))
    lab = bundle.kmeans.predict(Zc)
    out["clustering"] = {"selected_k": int(bundle.kmeans.n_clusters), "test_cluster_sizes": np.bincount(lab, minlength=bundle.kmeans.n_clusters).tolist(),
                         "test_silhouette": float(silhouette_score(Zc, lab)) if len(set(lab)) > 1 else None}

    rdir = cfg.path("paths.reports_dir") / bundle.run_dir.name / f"evaluation_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    rdir.mkdir(parents=True, exist_ok=True)
    rp.write_json(rdir / "evaluation.json", out)
    rp.plot_confusion(rdir / "confusion_matrix.png", out["classifier"]["confusion_matrix"]["matrix"], f"Re-evaluation ({provenance} data)")
    rp.plot_hist(rdir / "anomaly_score_hist.png", score, "Isolation Forest scores (test rows)", 0.0,
                 feat.loc[split.test_mask, label_col].to_numpy() if label_col in feat.columns else None)
    if with_shap:
        ex = Explainer(bundle.booster, names, int(cfg.get("explain.top_k", 5)))
        if ex.available:
            n = int(cfg.get("explain.background_rows", 500))
            Xs = X_te if len(X_te) <= n else X_te.sample(n, random_state=int(cfg.get("seed", 42)))
            gi = ex.global_importance(Xs)
            rp.write_json(rdir / "shap_global_importance.json", gi)
            rp.plot_bar(rdir / "shap_global_importance.png", gi["mean_abs_shap"], "Mean |SHAP| (model behaviour, not causation)", xlabel="mean |SHAP|")
            # tabular per-prediction explanations for a sample of test rows
            samp = Xs.iloc[:50]
            pred = bundle.booster.predict(xgb.DMatrix(samp, feature_names=names)).argmax(axis=1)
            rows = []
            for (idx, _), e in zip(samp.iterrows(), ex.explain_rows(samp, pred)):
                for t in e["top_features"]:
                    rows.append({"row": int(idx), "predicted": e["target_class"], **t})
            pd.DataFrame(rows).to_csv(rdir / "shap_sample_explanations.csv", index=False)
            out["shap"] = {"available": True, "report_dir": str(rdir)}
        else:
            out["shap"] = {"available": False, "reason": ex.reason}
    out["reports_dir"] = str(rdir)
    return out
