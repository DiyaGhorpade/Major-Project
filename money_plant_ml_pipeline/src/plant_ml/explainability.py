"""Optional SHAP explainability for the XGBoost classifier.

SHAP values are additive feature contributions to the model's raw class score (log-odds margin). They describe
what the MODEL used, not what caused the plant's condition; they are never proof of causation.
If SHAP is not installed (or fails), an explicit 'unavailable' result with a reason is returned - nothing is invented.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import STRESS_LEVELS

CAUSATION_NOTE = ("SHAP contributions show how each input moved the model's score for the predicted class "
                  "(positive = pushed towards that class). They reflect model behaviour and correlation in the "
                  "training data, not biological causation.")


class Explainer:
    def __init__(self, booster, feature_names: list[str], top_k: int = 5, enabled: bool = True):
        self.feature_names = list(feature_names)
        self.top_k = int(top_k)
        self.available = False
        self.reason = "disabled in configuration (explain.enabled: false)"
        self._explainer = None
        if not enabled:
            return
        try:
            import shap  # optional dependency
            self._explainer = shap.TreeExplainer(booster)
            self.available = True
            self.reason = ""
        except ImportError:
            self.reason = "the optional 'shap' package is not installed (pip install -r requirements-optional.txt)"
        except Exception as exc:  # noqa: BLE001
            self.reason = f"SHAP could not be initialised: {exc}"

    # -------------------------------------------------------------- core
    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Return array (n_rows, n_features, n_classes)."""
        if not self.available:
            raise RuntimeError(self.reason)
        vals = self._explainer.shap_values(X[self.feature_names])
        if isinstance(vals, list):
            vals = np.stack(vals, axis=-1)
        vals = np.asarray(vals)
        if vals.ndim == 2:  # single-output fallback
            vals = vals[:, :, None]
        return vals

    def explain_rows(self, X: pd.DataFrame, class_idx) -> list[dict]:
        """Per-row explanation for the given predicted class indices."""
        if not self.available:
            return [self.unavailable() for _ in range(len(X))]
        try:
            vals = self.shap_values(X)
        except Exception as exc:  # noqa: BLE001
            return [{"available": False, "reason": f"SHAP computation failed: {exc}", "top_features": []} for _ in range(len(X))]
        out = []
        for i, c in enumerate(class_idx):
            contrib = vals[i, :, int(c)]
            order = np.argsort(-np.abs(contrib))[: self.top_k]
            top = []
            for j in order:
                v = X.iloc[i, X.columns.get_loc(self.feature_names[j])]
                top.append({
                    "feature": self.feature_names[j],
                    "value": None if pd.isna(v) else float(v),
                    "contribution": float(contrib[j]),
                    "direction": "towards_predicted_class" if contrib[j] > 0 else "away_from_predicted_class",
                })
            out.append({"available": True, "target_class": STRESS_LEVELS[int(c)], "top_features": top,
                        "units": "log-odds (raw model score) contribution", "note": CAUSATION_NOTE})
        return out

    def unavailable(self) -> dict:
        return {"available": False, "reason": self.reason, "top_features": []}

    # -------------------------------------------------------------- offline reports
    def global_importance(self, X: pd.DataFrame) -> dict:
        """Mean |SHAP| per feature (averaged over classes and rows) + per-class tables."""
        vals = self.shap_values(X)
        mean_abs = np.abs(vals).mean(axis=(0, 2))
        overall = {self.feature_names[j]: float(mean_abs[j]) for j in np.argsort(-mean_abs)}
        per_class = {}
        for c, name in enumerate(STRESS_LEVELS[: vals.shape[2]]):
            m = np.abs(vals[:, :, c]).mean(axis=0)
            per_class[name] = {self.feature_names[j]: float(m[j]) for j in np.argsort(-m)[:15]}
        return {"mean_abs_shap": overall, "per_class_top15": per_class, "note": CAUSATION_NOTE}
