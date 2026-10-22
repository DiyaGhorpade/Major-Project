"""Inference: shared feature engineering + the three models + explanations + recommendations.

Stateless design: the caller supplies chronological readings per plant (history + the reading of interest).
The result is produced for the LATEST row of each plant (``rows='last'``) or for every row (``rows='all'``,
used by offline scripts). Plants never share history: all temporal features are grouped by plant_id.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from .artifact_manager import ModelBundle, load_bundle
from .config import Config
from .explainability import Explainer
from .feature_engineering import build_features, required_history_rows
from .preprocessing import select_matrix
from .recommendations import DISCLAIMER, generate_recommendations
from .schema import SENSOR_COLUMNS, STRESS_LEVELS


class PlantPredictor:
    def __init__(self, bundle: ModelBundle, cfg: Config):
        self.bundle = bundle
        self.cfg = cfg
        self.cls_features = bundle.meta["classifier"]["feature_names"]
        self.cluster_features = bundle.meta["clustering"]["feature_names"]
        self.anomaly_features = bundle.meta["anomaly"]["feature_names"]
        self.provenance = bundle.manifest.get("training_data", {}).get("provenance", "unknown")
        self.explainer = Explainer(bundle.booster, self.cls_features, int(cfg.get("explain.top_k", 5)),
                                   bool(cfg.get("explain.enabled", True)))
        self.required_history = required_history_rows(cfg)

    @classmethod
    def from_config(cls, cfg: Config, run_id: str | None = None) -> "PlantPredictor":
        return cls(load_bundle(cfg, run_id), cfg)

    # ----------------------------------------------------------------- metadata
    def metadata(self) -> dict:
        m = self.bundle.manifest
        return {
            "run_id": self.bundle.run_dir.name,
            "model_version": m.get("model_version"),
            "trained_at": m.get("created_at_utc"),
            "training_data": m.get("training_data"),
            "trained_on": self.provenance,
            "validated_on_real_plants": bool(m.get("validated_on_real_plants", False)),
            "split": m.get("split"),
            "models": {name: {"feature_count": len(meta["feature_names"]), "feature_names": meta["feature_names"],
                              "artifact_sha256": meta["artifact_sha256"], "model_name": meta.get("model_name"),
                              "known_limitations": meta.get("known_limitations")}
                       for name, meta in self.bundle.meta.items()},
            "label_mapping": self.bundle.meta["classifier"].get("label_mapping"),
            "anomaly_score_convention": self.bundle.meta["anomaly"].get("score_convention"),
            "cluster_note": self.bundle.meta["clustering"].get("note"),
            "explainability": {"available": self.explainer.available, "reason": self.explainer.reason or None},
            "required_history_rows_for_full_features": self.required_history,
            "dependency_versions_at_training": m.get("dependency_versions"),
            "compatibility_warnings": self.bundle.warnings,
            "disclaimer": ("Experimental. Outputs are not validated plant-health diagnoses. "
                           + (self.bundle.manifest.get("training_data", {}).get("provenance_note") or "")),
        }

    # ----------------------------------------------------------------- prediction
    def predict_frame(self, df: pd.DataFrame, rows: str = "last", explain: bool | None = None) -> list[dict]:
        """``df`` must be normalised + validated (tz-aware timestamp, numeric sensors). Labels are ignored."""
        if rows not in ("last", "all"):
            raise ValueError("rows must be 'last' or 'all'")
        feat = build_features(df, self.cfg)
        feat["_pos"] = feat.groupby("plant_id", sort=False).cumcount()
        if rows == "last":
            sel = feat.groupby("plant_id", sort=False).tail(1)
        else:
            sel = feat
        sel_idx = sel.index.to_numpy()
        if len(sel_idx) == 0:
            return []

        X_cls = select_matrix(sel, self.cls_features)
        proba = self.bundle.booster.predict(xgb.DMatrix(X_cls, feature_names=self.cls_features))
        pred_idx = proba.argmax(axis=1)

        Zc = self.bundle.cluster_pipeline.transform(select_matrix(sel, self.cluster_features))
        cdist = self.bundle.kmeans.transform(Zc)
        cluster_id = cdist.argmin(axis=1)
        cluster_dist = cdist.min(axis=1)

        Za = self.bundle.anomaly_pipeline.transform(select_matrix(sel, self.anomaly_features))
        a_score = -self.bundle.iforest.decision_function(Za)

        want_explain = bool(self.cfg.get("explain.enabled", True)) if explain is None else explain
        explanations = (self.explainer.explain_rows(X_cls, pred_idx) if want_explain
                        else [{"available": False, "reason": "explanations not requested for this call", "top_features": []}
                              for _ in range(len(sel_idx))])

        results = []
        meta_block = {"trained_on": self.provenance,
                      "validated_on_real_plants": bool(self.bundle.manifest.get("validated_on_real_plants", False)),
                      "model_version": self.bundle.manifest.get("model_version"), "run_id": self.bundle.run_dir.name,
                      "disclaimer": "Experimental output; not a validated plant-health diagnosis."}
        for k, i in enumerate(sel_idx):
            row = feat.loc[i]
            pos = int(row["_pos"])
            recent = feat.iloc[max(0, i - min(pos, 5)): i + 1]
            probs = {name: float(proba[k, j]) for j, name in enumerate(STRESS_LEVELS)}
            label = STRESS_LEVELS[int(pred_idx[k])]
            is_anom = bool(a_score[k] > 0)
            missing = [c for c in SENSOR_COLUMNS if pd.isna(row[c])]
            warnings = []
            if pos + 1 < self.required_history:
                warnings.append(f"Only {pos + 1} reading(s) of history for this plant; {self.required_history} are needed for "
                                "fully-populated temporal features. Missing context is handled as missing values, so this "
                                "prediction is less reliable (cold start).")
            if missing:
                warnings.append(f"Missing sensor value(s): {missing}")
            recs = generate_recommendations(label, probs, is_anom, missing, recent, self.cfg)
            anomaly = {"is_anomaly": is_anom, "score": float(a_score[k]), "threshold": 0.0,
                       "interpretation": ("Possible sensor / data-quality anomaly (not a plant-health prediction)."
                                          if is_anom else "Within the range of readings seen in training.")}
            results.append({
                "plant_id": int(row["plant_id"]),
                "timestamp": row["timestamp"].isoformat(),
                "stress_prediction": {"label": label, "probabilities": probs},
                "cluster": {"id": int(cluster_id[k]), "distance": float(cluster_dist[k])},
                "anomaly": anomaly,
                "explanation": explanations[k],
                "recommendations": recs,
                "recommendations_disclaimer": DISCLAIMER,
                "history": {"rows_available": pos + 1, "rows_recommended": self.required_history,
                            "sufficient": pos + 1 >= self.required_history},
                "warnings": warnings,
                "model_metadata": meta_block,
            })
        return results
