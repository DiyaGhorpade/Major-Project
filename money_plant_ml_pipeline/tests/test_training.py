import json

import numpy as np
import pandas as pd
import pytest

from plant_ml.artifact_manager import ArtifactError, ArtifactStore, load_bundle
from plant_ml.schema import FORBIDDEN_MODEL_COLUMNS, STRESS_LEVELS


def test_training_produces_all_artifacts_and_reports(trained, cfg_small):
    run_dir = trained["run_dir"]
    for f in ("classifier.xgb.json", "classifier.meta.json", "clustering.joblib", "clustering.meta.json",
              "anomaly.joblib", "anomaly.meta.json", "manifest.json"):
        assert (run_dir / f).exists(), f
    rdir = trained["reports_dir"]
    for f in ("REPORT.md", "data_quality_report.json", "class_distribution.json", "classifier_metrics.json",
              "clustering_report.json", "anomaly_report.json", "artifact_manifest.json"):
        assert (rdir / f).exists(), f
    assert "SYNTHETIC" in (rdir / "REPORT.md").read_text()


def test_metadata_contents(trained):
    meta = json.loads((trained["run_dir"] / "classifier.meta.json").read_text())
    for key in ("model_name", "model_version", "created_at_utc", "random_seed", "dependency_versions", "feature_names",
                "label_mapping", "preprocessing", "training_data", "split", "metrics", "known_limitations", "artifact_sha256"):
        assert key in meta, key
    assert meta["training_data"]["provenance"] == "synthetic" and len(meta["training_data"]["sha256"]) == 64
    assert meta["label_mapping"] == {n: i for i, n in enumerate(STRESS_LEVELS)}
    assert meta["metrics"]["metric_scope"] == "held-out evaluation set"
    assert "validated_on_real_plants" in json.loads((trained["run_dir"] / "manifest.json").read_text())


def test_artifact_metadata_carries_no_disclaimer_text(trained):
    banned = ("synthetic-data", "not validated", "not a plant-health", "not experimentally")
    for name in ("classifier", "clustering", "anomaly"):
        meta = json.loads((trained["run_dir"] / f"{name}.meta.json").read_text())
        for key in ("note", "metric_scope"):
            for v in (meta.get(key), (meta.get("metrics") or {}).get(key)):
                assert not v or not any(b in str(v).lower() for b in banned), (name, key, v)
        assert not any(b in " ".join(meta.get("known_limitations", [])).lower() for b in banned), name
    manifest = json.loads((trained["run_dir"] / "manifest.json").read_text())
    assert manifest["training_data"]["provenance"] == "synthetic" and manifest["training_data"]["provenance_note"] is None


def test_classifier_reports_required_metrics_and_baseline(trained):
    m = trained["classifier"]["metrics"]
    for k in ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "per_class", "confusion_matrix", "class_counts"):
        assert k in m
    assert set(trained["classifier"]["baselines"]) == {"majority_class", "logistic_regression"}
    assert 0 <= m["accuracy"] <= 1


def test_no_forbidden_columns_in_any_model(trained):
    for name in ("classifier", "clustering", "anomaly"):
        meta = json.loads((trained["run_dir"] / f"{name}.meta.json").read_text())
        assert not set(meta["feature_names"]) & set(FORBIDDEN_MODEL_COLUMNS), name
    assert "is_synthetic_anomaly" not in trained["anomaly"]["feature_names"]


def test_train_test_separation_is_recorded_and_chronological(trained):
    feat, split = trained["features"], trained["split"]
    assert feat.loc[split.train_mask, "timestamp"].max() < feat.loc[split.test_mask, "timestamp"].min()
    assert trained["summary"]["split"]["strategy"] == "chronological"


def test_artifact_round_trip_and_probabilities_sum_to_one(trained, cfg_small):
    import xgboost as xgb
    bundle = load_bundle(cfg_small, trained["run_id"])
    names = bundle.meta["classifier"]["feature_names"]
    X = trained["classifier"]["X_test"].iloc[:50]
    p_loaded = bundle.booster.predict(xgb.DMatrix(X, feature_names=names))
    p_orig = trained["classifier"]["booster"].predict(xgb.DMatrix(X, feature_names=names))
    assert p_loaded.shape == (50, 4)
    np.testing.assert_allclose(p_loaded.sum(axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(p_loaded, p_orig, atol=1e-6)


def test_clustering_output_shape_and_selected_k(trained, cfg_small):
    bundle = load_bundle(cfg_small, trained["run_id"])
    names = bundle.meta["clustering"]["feature_names"]
    X = trained["features"][names].iloc[:25]
    Z = bundle.cluster_pipeline.transform(X)
    d = bundle.kmeans.transform(Z)
    assert d.shape == (25, bundle.kmeans.n_clusters)
    assert 2 <= bundle.kmeans.n_clusters <= int(cfg_small.get("clustering.k_max"))
    assert set(bundle.kmeans.predict(Z)) <= set(range(bundle.kmeans.n_clusters))
    assert "arbitrary" in bundle.meta["clustering"]["note"]


def test_anomaly_score_convention_higher_is_more_anomalous(trained, cfg_small):
    bundle = load_bundle(cfg_small, trained["run_id"])
    names = bundle.meta["anomaly"]["feature_names"]
    X = trained["features"][names].iloc[:200]
    score = -bundle.iforest.decision_function(bundle.anomaly_pipeline.transform(X))
    flag = (score > 0)
    assert score.shape == (200,)
    assert score[flag].min() > score[~flag].max()            # flagged rows have strictly higher scores
    assert "HIGHER = MORE anomalous" in bundle.meta["anomaly"]["score_convention"]


def test_anomaly_label_used_only_for_evaluation(trained):
    m = trained["anomaly"]["metrics_vs_helper_label"]
    assert m is None or "never a model input" in m["note"]


def test_missing_artifacts_raise_helpful_error(base_cfg, tmp_path):
    cfg = base_cfg.copy()
    cfg.set("paths.artifacts_dir", str(tmp_path / "empty"))
    with pytest.raises(ArtifactError, match="train"):
        load_bundle(cfg)
    (tmp_path / "empty").mkdir()
    with pytest.raises(ArtifactError, match="latest.json"):
        load_bundle(cfg)


def test_corrupt_artifact_detected_by_checksum(trained, cfg_small):
    path = trained["run_dir"] / "clustering.joblib"
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"x")
        with pytest.raises(ArtifactError, match="Checksum"):
            load_bundle(cfg_small, trained["run_id"])
    finally:
        path.write_bytes(original)


def test_version_mismatch_is_reported(trained, cfg_small):
    from plant_ml.artifact_manager import check_compatibility
    manifest = json.loads((trained["run_dir"] / "manifest.json").read_text())
    manifest["dependency_versions"]["sklearn"] = "0.20.0"
    with pytest.raises(ArtifactError, match="mismatch"):
        check_compatibility(manifest, allow_mismatch=False)
    assert check_compatibility(manifest, allow_mismatch=True)


def test_new_training_run_never_overwrites_previous(cfg_small, trained):
    store = ArtifactStore(cfg_small.path("paths.artifacts_dir"))
    before = set(store.list_runs())
    d1, d2 = store.create_run("9.9.9"), store.create_run("9.9.9")
    assert d1 != d2 and before.issubset(set(p.name for p in store.root.iterdir()))


def test_injected_anomalies_score_higher_than_normal_rows(predictor, small_df):
    from plant_ml.data_loader import normalise_frame
    plant = normalise_frame(small_df[small_df["plant_id"] == 1].iloc[:120])
    normal = predictor.predict_frame(plant.iloc[:100], rows="last", explain=False)[0]
    inj = plant.iloc[:100].copy()
    inj.loc[inj.index[-1], ["soil", "temperature", "humidity", "light"]] = [99.5, 39.5, 20.5, 90000]
    bad = predictor.predict_frame(inj, rows="last", explain=False)[0]
    assert bad["anomaly"]["score"] > normal["anomaly"]["score"]
    assert bad["anomaly"]["is_anomaly"] is True
    assert bad["recommendations"][0]["id"] == "verify_sensor"
    # subtle contextual anomaly: darkness at midday with a sudden soil jump -> scored higher than the normal row
    sub = plant.iloc[:100].copy()
    sub.loc[sub.index[-1], ["light", "soil"]] = [0.0, sub["soil"].iloc[-2] + 25]
    subtle = predictor.predict_frame(sub, rows="last", explain=False)[0]
    assert subtle["anomaly"]["score"] > normal["anomaly"]["score"]
