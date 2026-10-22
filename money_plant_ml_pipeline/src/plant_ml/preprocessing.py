"""Model-input selection, leakage guards and fitted (train-only) preprocessing pipelines."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import Config
from .feature_engineering import all_feature_names
from .schema import FORBIDDEN_MODEL_COLUMNS


class LeakageError(ValueError):
    """A forbidden (label / identifier / timestamp) column was requested as a model input."""


def model_feature_list(cfg: Config, model: str) -> list[str]:
    """Resolve the ordered feature list for 'classifier' | 'clustering' | 'anomaly' from feature_config.yaml."""
    spec = cfg.features.get(model)
    if not spec:
        raise KeyError(f"feature_config.yaml has no '{model}' section")
    if "features" in spec:
        names = list(spec["features"])
    elif spec.get("include") == "all":
        exclude = set(spec.get("exclude", []) or [])
        names = [n for n in all_feature_names(cfg) if n not in exclude]
    else:
        raise KeyError(f"feature_config.yaml '{model}' needs either 'features: [...]' or 'include: all'")
    guard_feature_list(names, cfg, model)
    return names


def guard_feature_list(names: list[str], cfg: Config, model: str = "model") -> None:
    allowed_ids = set(cfg.features.get("allow_identifier_experiments", []) or [])
    bad = [n for n in names if n in FORBIDDEN_MODEL_COLUMNS and n not in allowed_ids]
    if bad:
        raise LeakageError(f"{model}: forbidden columns requested as inputs: {bad}")
    known = set(all_feature_names(cfg))
    unknown = [n for n in names if n not in known and n not in allowed_ids]
    if unknown:
        raise KeyError(f"{model}: unknown feature names {unknown}")
    if len(set(names)) != len(names):
        raise ValueError(f"{model}: duplicate feature names in list")


def select_matrix(feat_df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Return the float feature matrix in the stored order; +/-inf -> NaN. Never touches label columns."""
    missing = [n for n in names if n not in feat_df.columns]
    if missing:
        raise KeyError(f"Feature matrix is missing columns: {missing}")
    X = feat_df[names].astype("float64").replace([np.inf, -np.inf], np.nan)
    return X


def make_numeric_pipeline(scale: bool = True, strategy: str = "median") -> Pipeline:
    """Imputer (+ optional scaler). MUST be fit on training rows only."""
    steps = [("imputer", SimpleImputer(strategy=strategy, keep_empty_features=True))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)
