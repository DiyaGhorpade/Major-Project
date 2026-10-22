"""Model B: K-Means clustering of environmental states (no labels, no identifiers)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .config import Config
from .labels import add_group_column_if_missing
from .preprocessing import make_numeric_pipeline, model_feature_list, select_matrix
from .splitters import SplitResult

CLUSTER_NOTE = ("Cluster IDs are arbitrary integers. They describe groups of similar environmental readings and do NOT "
                "mean healthy / unhealthy. The number of clusters is chosen from the data (silhouette), not from the "
                "number of experimental groups.")


def train_clustering(feat_df: pd.DataFrame, split: SplitResult, cfg: Config) -> dict:
    names = model_feature_list(cfg, "clustering")
    seed = int(cfg.get("seed", 42))
    X_tr = select_matrix(feat_df.loc[split.train_mask], names)
    pipe = make_numeric_pipeline(scale=True).fit(X_tr)          # imputer + scaler fitted on TRAIN only
    Z = pipe.transform(X_tr)

    k_fixed = cfg.get("clustering.k")
    k_min, k_max = int(cfg.get("clustering.k_min", 2)), int(cfg.get("clustering.k_max", 8))
    n_init = int(cfg.get("clustering.n_init", 10))
    sample = int(cfg.get("clustering.silhouette_sample_size", 5000))
    sample_size = min(sample, len(Z)) if len(Z) > sample else None
    candidates = [int(k_fixed)] if k_fixed else list(range(k_min, min(k_max, len(Z) - 1) + 1))

    scores: dict[int, dict] = {}
    for k in candidates:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(Z)
        sil = float(silhouette_score(Z, km.labels_, sample_size=sample_size, random_state=seed)) if k > 1 else float("nan")
        scores[k] = {"silhouette": sil, "inertia": float(km.inertia_)}
    best_k = int(k_fixed) if k_fixed else max(scores, key=lambda k: scores[k]["silhouette"])
    kmeans = KMeans(n_clusters=best_k, n_init=n_init, random_state=seed).fit(Z)

    dist = np.min(kmeans.transform(Z), axis=1)
    centres_orig = pipe.named_steps["scaler"].inverse_transform(kmeans.cluster_centers_)
    summaries = []
    labels = kmeans.labels_
    for c in range(best_k):
        m = labels == c
        summaries.append({
            "cluster_id": c, "size": int(m.sum()), "share": float(m.mean()),
            "mean_distance_to_centroid": float(dist[m].mean()) if m.any() else None,
            "centroid_original_units": {n: float(v) for n, v in zip(names, centres_orig[c])},
        })

    # Offline description only: how clusters relate to experimental metadata (never used for fitting).
    crosstab = None
    try:
        sub = add_group_column_if_missing(feat_df.loc[split.train_mask].reset_index(drop=True))
        crosstab = {
            "experimental_group": pd.crosstab(labels, sub["experimental_group"]).to_dict(),
            "stress_level": pd.crosstab(labels, sub["stress_level"]).to_dict(),
            "note": "Descriptive cross-tab on TRAIN rows; metadata was not used to fit the clusters.",
        }
        crosstab = _jsonable(crosstab)
    except Exception:  # noqa: BLE001 - descriptive extra, never fatal
        crosstab = None

    return {
        "pipeline": pipe, "kmeans": kmeans, "feature_names": names, "selected_k": best_k,
        "silhouette_by_k": scores, "cluster_summaries": summaries, "crosstab": crosstab,
        "train_distance_quantiles": {q: float(np.quantile(dist, q)) for q in (0.5, 0.9, 0.99)},
        "n_train": int(len(Z)), "note": CLUSTER_NOTE,
        "preprocessing": {"imputation": "median (fit on train)", "scaling": "StandardScaler (fit on train)"},
        "silhouette_limitations": ("Silhouette favours compact, well-separated blobs; time-series environmental data forms "
                                   "continuous gradients, so scores are typically modest and the chosen k is a convenience, "
                                   "not a discovered biological structure."),
    }


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj
