"""End-to-end training orchestration: load -> validate -> features -> split -> 3 models -> artifacts + reports."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import reports as rp
from .artifact_manager import ArtifactStore, dependency_versions
from .config import Config
from .data_loader import file_sha256, load_csv
from .explainability import Explainer
from .feature_engineering import build_features
from .labels import class_counts
from .logging_utils import get_logger
from .preprocessing import select_matrix
from .splitters import split_frame
from .train_anomaly import train_anomaly
from .train_classifier import train_classifier
from .train_clustering import train_clustering
from .validation import validate_frame, validation_messages

log = get_logger()

LIMITATIONS = {
    "classifier": [
        "Labels are simulation labels, not biological ground truth; metrics are synthetic-data metrics.",
        "Plant identity and experimental group are deliberately excluded, so the model sees only sensor-derived features.",
        "Small dataset (16 plants, a few weeks): estimates have wide uncertainty and may not generalise to real plants.",
        "Temporal features need ~49 consecutive readings per plant for full parity with training.",
    ],
    "clustering": ["Cluster IDs are arbitrary and not health states.", "k chosen by silhouette is a convenience, not discovered structure."],
    "anomaly": ["Flags unusual reading combinations (often sensor issues); not a plant-health predictor.",
                "Contamination is a configured assumption, not an estimate of the true anomaly rate."],
}
METRIC_GUIDE = (
    "Metrics are measured on a held-out later period (or unseen plants) of a SYNTHETIC dataset and describe how well the model "
    "recovers the simulator's labels - not how well it diagnoses real plants. Compare against the baselines rather than reading a "
    "single number in isolation; accuracy alone can mislead when classes are unevenly present in the test period. "
    "Treat per-class recall/F1 for classes with few test samples as very uncertain. Nothing here is experimentally validated."
)


def run_training(cfg: Config, input_path: str | Path | None = None, set_latest: bool = True,
                 save_processed: bool = False, with_shap: bool = True, run_tag: str | None = None) -> dict:
    t0 = datetime.now(timezone.utc)
    input_path = Path(input_path) if input_path else cfg.path("paths.default_input")
    tz = cfg.get("timezone", "Asia/Kolkata")
    seed = int(cfg.get("seed", 42))
    provenance = str(cfg.get("data.provenance", "synthetic"))
    if provenance not in ("synthetic", "real", "mixed"):
        raise ValueError("data.provenance must be synthetic | real | mixed")

    log.info("Loading %s", input_path)
    raw = load_csv(input_path, tz, require_labels=True)
    mode = cfg.get("validation.mode", "warn")
    vres = validate_frame(raw, mode=mode, require_labels=True)
    for msg in validation_messages(vres.report):
        log.warning("validation: %s", msg)
    clean = vres.clean.reset_index(drop=True)
    if len(vres.quarantined) and cfg.get("validation.save_quarantine", True):
        qdir = cfg.path("paths.processed_dir")
        qdir.mkdir(parents=True, exist_ok=True)
        qpath = qdir / f"quarantine_{t0.strftime('%Y%m%dT%H%M%SZ')}.csv"
        vres.quarantined.to_csv(qpath, index=False)
        log.warning("quarantined rows written to %s", qpath)

    feat = build_features(clean, cfg)
    split = split_frame(feat, cfg)
    log.info("split %s: train=%d test=%d", split.info["strategy"], split.info["train_rows"], split.info["test_rows"])

    log.info("Training classifier (XGBoost)")
    cls = train_classifier(feat, split, cfg)
    log.info("Training clustering (K-Means)")
    clu = train_clustering(feat, split, cfg)
    log.info("Training anomaly detector (Isolation Forest)")
    ano = train_anomaly(feat, split, cfg)

    # ---------------- artifacts ----------------
    version = str(cfg.get("project.model_version", "1.0.0"))
    store = ArtifactStore(cfg.path("paths.artifacts_dir"))
    run_dir = store.create_run(version)
    run_id = run_dir.name
    created = datetime.now(timezone.utc).isoformat()
    deps = dependency_versions()
    training_data = {
        "filename": input_path.name, "sha256": file_sha256(input_path), "rows": int(len(raw)),
        "rows_used": int(len(clean)), "rows_quarantined": int(len(vres.quarantined)),
        "provenance": provenance, "provenance_note": cfg.get("data.provenance_note"),
    }
    common = {"model_version": version, "created_at_utc": created, "random_seed": seed, "dependency_versions": deps,
              "training_data": training_data, "split": split.info}

    cpath = store.save_booster(run_dir, "classifier", cls["booster"])
    store.write_model_metadata(run_dir, "classifier", cpath, {
        **common, "model_name": "xgboost_stress_classifier", "feature_names": cls["feature_names"],
        "label_mapping": cls["label_mapping"], "preprocessing": cls["preprocessing"],
        "xgboost_params": cls["xgboost_params"], "num_boost_round": cls["num_boost_round"],
        "metrics": cls["metrics"], "baseline_metrics": {k: {m: v[m] for m in ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy")}
                                                         for k, v in cls["baselines"].items()},
        "train_class_counts": cls["train_class_counts"], "known_limitations": LIMITATIONS["classifier"]})
    kpath = store.save_joblib(run_dir, "clustering", {"pipeline": clu["pipeline"], "kmeans": clu["kmeans"]})
    store.write_model_metadata(run_dir, "clustering", kpath, {
        **common, "model_name": "kmeans_environment_clusters", "feature_names": clu["feature_names"],
        "selected_k": clu["selected_k"], "silhouette_by_k": clu["silhouette_by_k"], "cluster_summaries": clu["cluster_summaries"],
        "preprocessing": clu["preprocessing"], "note": clu["note"], "known_limitations": LIMITATIONS["clustering"]})
    apath = store.save_joblib(run_dir, "anomaly", {"pipeline": ano["pipeline"], "model": ano["model"]})
    store.write_model_metadata(run_dir, "anomaly", apath, {
        **common, "model_name": "isolation_forest_anomaly", "feature_names": ano["feature_names"],
        "contamination": ano["contamination"], "threshold": ano["threshold"], "score_convention": ano["score_convention"],
        "preprocessing": ano["preprocessing"], "metrics_vs_helper_label": ano["metrics_vs_helper_label"],
        "note": ano["note"], "known_limitations": LIMITATIONS["anomaly"]})
    manifest = {**common, "run_id": run_id, "validated_on_real_plants": bool(cfg.get("data.validated_on_real_plants", False)),
                "models": ["classifier", "clustering", "anomaly"], "config": cfg.snapshot(),
                "validation_mode": mode, "feature_engineering": {k: cfg.get(f"features.{k}") for k in
                                                                 ("rolling_windows", "baseline_window", "baseline_min_periods", "std_min_periods", "max_light")}}
    store.write_manifest(run_dir, manifest)
    if set_latest:
        store.set_latest(run_dir)

    # ---------------- reports ----------------
    rdir = cfg.path("paths.reports_dir") / run_id
    rdir.mkdir(parents=True, exist_ok=True)
    banner = (f"Training data provenance: {provenance.upper()}. All metrics below are {provenance}-data metrics, "
              "not experimentally validated plant-health performance.")
    summary = {"run_id": run_id, "provenance_banner": banner, "training_data": training_data, "data_quality": vres.report,
               "split": split.info, "metric_reading_guide": METRIC_GUIDE,
               "classifier": {"metrics": cls["metrics"], "baselines": cls["baselines"],
                              "feature_importance_gain": cls["feature_importance_gain"],
                              "train_class_counts": cls["train_class_counts"]},
               "clustering": {k: clu[k] for k in ("selected_k", "silhouette_by_k", "cluster_summaries", "crosstab", "note",
                                                  "silhouette_limitations", "train_distance_quantiles")},
               "anomaly": {k: ano[k] for k in ("contamination", "threshold", "score_convention", "note", "train_score_quantiles",
                                               "train_flag_rate", "metrics_vs_helper_label", "metrics_vs_helper_label_train", "eval_scope")}}
    rp.write_json(rdir / "data_quality_report.json", vres.report)
    rp.write_json(rdir / "class_distribution.json", {"all_rows": class_counts(clean["stress_level"]),
                  "train_labelled": cls["train_class_counts"], "test": cls["metrics"]["class_counts"],
                  "note": "Synthetic classes were balanced by the simulator; this does not reflect real-world prevalence."})
    rp.write_json(rdir / "classifier_metrics.json", {"xgboost": cls["metrics"], "baselines": cls["baselines"],
                  "split": split.info, "feature_importance_gain": cls["feature_importance_gain"]})
    rp.write_json(rdir / "clustering_report.json", summary["clustering"])
    rp.write_json(rdir / "anomaly_report.json", summary["anomaly"])
    rp.write_json(rdir / "artifact_manifest.json", manifest)
    cls["predictions_test"].to_csv(rdir / "classifier_test_predictions.csv", index=False)
    plots = {
        "confusion_matrix.png": rp.plot_confusion(rdir / "confusion_matrix.png", cls["metrics"]["confusion_matrix"]["matrix"],
                                                  f"XGBoost test confusion ({provenance} data)"),
        "feature_importance_gain.png": rp.plot_bar(rdir / "feature_importance_gain.png", cls["feature_importance_gain"],
                                                   "XGBoost feature importance (gain, normalised)", xlabel="share of total gain"),
        "silhouette_by_k.png": rp.plot_silhouette(rdir / "silhouette_by_k.png", clu["silhouette_by_k"]),
    }
    helper = None
    if cfg.get("anomaly.label_column", "is_synthetic_anomaly") in feat.columns:
        helper = feat.loc[ano["eval_mask"], cfg.get("anomaly.label_column", "is_synthetic_anomaly")].to_numpy()
    plots["anomaly_score_hist.png"] = rp.plot_hist(rdir / "anomaly_score_hist.png", ano["eval_scores"],
                                                   f"Isolation Forest scores ({ano['eval_scope']} rows)", 0.0, helper)

    shap_info = {"available": False}
    if with_shap and cfg.get("explain.enabled", True):
        ex = Explainer(cls["booster"], cls["feature_names"], int(cfg.get("explain.top_k", 5)))
        if ex.available:
            n = int(cfg.get("explain.background_rows", 500))
            Xs = cls["X_test"].iloc[: n] if len(cls["X_test"]) <= n else cls["X_test"].sample(n, random_state=seed)
            gi = ex.global_importance(Xs)
            rp.write_json(rdir / "shap_global_importance.json", gi)
            plots["shap_global_importance.png"] = rp.plot_bar(rdir / "shap_global_importance.png", gi["mean_abs_shap"],
                                                              "Mean |SHAP| (model-behaviour, not causation)", xlabel="mean |SHAP|")
            shap_info = {"available": True, "rows_used": int(len(Xs))}
        else:
            shap_info = {"available": False, "reason": ex.reason}
    summary["shap"] = shap_info
    summary["plots"] = plots
    (rdir / "REPORT.md").write_text(rp.markdown_summary(summary), encoding="utf-8")

    if save_processed:
        pdir = cfg.path("paths.processed_dir")
        pdir.mkdir(parents=True, exist_ok=True)
        feat.drop(columns=["_orig_index"]).to_csv(pdir / f"features_{run_id}.csv", index=False)

    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    log.info("Run %s complete in %.1fs. Artifacts: %s  Reports: %s", run_id, elapsed, run_dir, rdir)
    return {"run_id": run_id, "run_dir": run_dir, "reports_dir": rdir, "summary": summary, "elapsed_s": elapsed,
            "classifier": cls, "clustering": clu, "anomaly": ano, "features": feat, "split": split,
            "validation": vres}
