"""Reproducible diagnostics for the greenhouse K-Means root-cause audit."""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Greenhouse Plant Growth Metrics.csv"
FEATURES = [
    "ACHP",
    "PHR",
    "AWWGV",
    "ALAP",
    "ANPL",
    "ARD",
    "ADWR",
    "PDMVG",
    "ARL",
    "AWWR",
    "ADWV",
    "PDMRG",
]
RANDOM_STATE = 42


def emit(message: str) -> None:
    print(message, flush=True)


def cluster_size_row(k: int, labels: np.ndarray) -> dict[str, int]:
    counts = np.bincount(labels, minlength=k)
    row: dict[str, int] = {
        "k": k,
        "minimum_size": int(counts.min()),
        "maximum_size": int(counts.max()),
        "singleton_count": int(np.sum(counts == 1)),
        "clusters_below_10": int(np.sum(counts < 10)),
        "clusters_below_1pct": int(np.sum(counts < len(labels) * 0.01)),
    }
    for index, count in enumerate(np.sort(counts)[::-1], start=1):
        row[f"size_{index}"] = int(count)
    return row


def line_distance_knee(k_values: np.ndarray, inertias: np.ndarray) -> int:
    x = (k_values - k_values.min()) / (k_values.max() - k_values.min())
    y = (inertias - inertias.min()) / (inertias.max() - inertias.min())
    chord = 1.0 - x
    return int(k_values[np.argmax(chord - y)])


def piecewise_knee(k_values: np.ndarray, inertias: np.ndarray) -> tuple[int, dict[int, float]]:
    errors: dict[int, float] = {}
    for split in range(2, len(k_values) - 1):
        left_coef = np.polyfit(k_values[: split + 1], inertias[: split + 1], 1)
        right_coef = np.polyfit(k_values[split:], inertias[split:], 1)
        left_error = np.square(
            inertias[: split + 1] - np.polyval(left_coef, k_values[: split + 1])
        ).sum()
        right_error = np.square(
            inertias[split:] - np.polyval(right_coef, k_values[split:])
        ).sum()
        errors[int(k_values[split])] = float(left_error + right_error)
    return min(errors, key=errors.get), errors


def main() -> None:
    started = time.perf_counter()
    df = pd.read_csv(DATA_PATH)
    X_df = df[FEATURES].copy()
    X = StandardScaler().fit_transform(X_df)
    emit(f"Loaded {len(df):,} rows and {len(FEATURES)} clustering features")

    correlations_pearson = X_df.corr(method="pearson")
    correlations_spearman = X_df.corr(method="spearman")

    def top_correlations(matrix: pd.DataFrame, limit: int = 30) -> list[dict[str, object]]:
        upper = matrix.where(np.triu(np.ones(matrix.shape), k=1).astype(bool))
        values = upper.stack().rename("correlation").reset_index()
        values.columns = ["feature_1", "feature_2", "correlation"]
        values["absolute_correlation"] = values["correlation"].abs()
        values = values.sort_values("absolute_correlation", ascending=False).head(limit)
        return values.to_dict(orient="records")

    rounded_unique = {
        str(decimals): int(X_df.round(decimals).drop_duplicates().shape[0])
        for decimals in (0, 1, 2, 3, 4, 6)
    }
    nearest = NearestNeighbors(n_neighbors=2).fit(X)
    nearest_distances = nearest.kneighbors(X, return_distance=True)[0][:, 1]

    metrics_rows: list[dict[str, object]] = []
    size_rows_default: list[dict[str, int]] = []
    size_rows_n10: list[dict[str, int]] = []
    labels_by_k_n10: dict[int, np.ndarray] = {}
    inertias_n10: list[float] = []

    for k in range(1, 31):
        fit_started = time.perf_counter()
        model_default = KMeans(n_clusters=k, random_state=RANDOM_STATE)
        labels_default = model_default.fit_predict(X)
        model_n10 = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels_n10 = model_n10.fit_predict(X)
        labels_by_k_n10[k] = labels_n10
        inertias_n10.append(float(model_n10.inertia_))

        row: dict[str, object] = {
            "k": k,
            "inertia_default": float(model_default.inertia_),
            "inertia_n10": float(model_n10.inertia_),
            "default_vs_n10_ari": float(adjusted_rand_score(labels_default, labels_n10)),
        }
        if k >= 2:
            row.update(
                {
                    "silhouette_sample_default": float(
                        silhouette_score(
                            X,
                            labels_default,
                            sample_size=10_000,
                            random_state=RANDOM_STATE,
                        )
                    ),
                    "silhouette_sample_n10": float(
                        silhouette_score(
                            X,
                            labels_n10,
                            sample_size=10_000,
                            random_state=RANDOM_STATE,
                        )
                    ),
                    "calinski_harabasz_n10": float(calinski_harabasz_score(X, labels_n10)),
                    "davies_bouldin_n10": float(davies_bouldin_score(X, labels_n10)),
                    "class_ari_n10": float(adjusted_rand_score(df["Class"], labels_n10)),
                    "class_nmi_n10": float(normalized_mutual_info_score(df["Class"], labels_n10)),
                    "random_ari_n10": float(adjusted_rand_score(df["Random"], labels_n10)),
                    "random_nmi_n10": float(normalized_mutual_info_score(df["Random"], labels_n10)),
                }
            )
            size_rows_default.append(cluster_size_row(k, labels_default))
            size_rows_n10.append(cluster_size_row(k, labels_n10))
        metrics_rows.append(row)
        emit(f"Fitted k={k:2d} in {time.perf_counter() - fit_started:.1f}s")

    metrics = pd.DataFrame(metrics_rows)
    sizes_default = pd.DataFrame(size_rows_default)
    sizes_n10 = pd.DataFrame(size_rows_n10)

    # Exact, full-data silhouettes reproduce the visible notebook configuration.
    exact_default: dict[int, float] = {}
    for k in range(2, 16):
        exact_started = time.perf_counter()
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE).fit_predict(X)
        exact_default[k] = float(silhouette_score(X, labels))
        emit(
            f"Exact silhouette k={k:2d}: {exact_default[k]:.6f} "
            f"({time.perf_counter() - exact_started:.1f}s)"
        )
    metrics["silhouette_exact_default"] = metrics["k"].map(exact_default)

    pca = PCA(n_components=len(FEATURES), random_state=RANDOM_STATE).fit(X)
    pca_scores = pca.transform(X)
    pca_cumulative = np.cumsum(pca.explained_variance_ratio_)

    k_values_15 = np.arange(1, 16)
    inertia_values_15 = metrics.loc[metrics["k"] <= 15, "inertia_n10"].to_numpy()
    distance_knee = line_distance_knee(k_values_15, inertia_values_15)
    piecewise_selected, piecewise_errors = piecewise_knee(k_values_15, inertia_values_15)
    relative_reductions = {
        int(k_values_15[index]): float(
            (inertia_values_15[index - 1] - inertia_values_15[index])
            / inertia_values_15[index - 1]
        )
        for index in range(1, len(k_values_15))
    }

    # Compact PCA diagnostics: color the same projection by k=3, k=15, and Class.
    sample_rng = np.random.default_rng(RANDOM_STATE)
    plot_indices = sample_rng.choice(len(df), size=min(10_000, len(df)), replace=False)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    axes[0].scatter(
        pca_scores[plot_indices, 0],
        pca_scores[plot_indices, 1],
        c=labels_by_k_n10[3][plot_indices],
        s=5,
        alpha=0.45,
        cmap="tab10",
    )
    axes[0].set_title("PCA colored by K-Means k=3")
    axes[1].scatter(
        pca_scores[plot_indices, 0],
        pca_scores[plot_indices, 1],
        c=labels_by_k_n10[15][plot_indices],
        s=5,
        alpha=0.45,
        cmap="tab20",
    )
    axes[1].set_title("PCA colored by K-Means k=15")
    class_codes = pd.Categorical(df["Class"]).codes
    axes[2].scatter(
        pca_scores[plot_indices, 0],
        pca_scores[plot_indices, 1],
        c=class_codes[plot_indices],
        s=5,
        alpha=0.45,
        cmap="tab10",
    )
    axes[2].set_title("PCA colored by experimental Class")
    for axis in axes:
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
    fig.savefig(ROOT / "clustering_pca_diagnostics.png", dpi=170)
    plt.close(fig)

    metrics.to_csv(ROOT / "clustering_validation_metrics.csv", index=False)
    sizes_default.to_csv(ROOT / "clustering_sizes_notebook_default.csv", index=False)
    sizes_n10.to_csv(ROOT / "clustering_sizes_n_init_10.csv", index=False)

    report = {
        "dataset": {
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "features": FEATURES,
            "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
            "missing_by_column": df.isna().sum().astype(int).to_dict(),
            "duplicate_rows": int(df.duplicated().sum()),
            "duplicate_feature_rows": int(X_df.duplicated().sum()),
            "unique_by_column": df.nunique().astype(int).to_dict(),
            "class_counts": df["Class"].value_counts().sort_index().astype(int).to_dict(),
            "random_counts": df["Random"].value_counts().sort_index().astype(int).to_dict(),
            "rounded_unique_feature_rows": rounded_unique,
            "numeric_describe": X_df.describe(
                percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
            ).to_dict(),
            "variance": X_df.var().to_dict(),
            "nearest_neighbor_distance_scaled_quantiles": {
                str(quantile): float(np.quantile(nearest_distances, quantile))
                for quantile in (0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1)
            },
            "top_pearson_correlations": top_correlations(correlations_pearson),
            "top_spearman_correlations": top_correlations(correlations_spearman),
        },
        "pca": {
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_explained_variance": pca_cumulative.tolist(),
            "components_for_80_percent": int(np.searchsorted(pca_cumulative, 0.80) + 1),
            "components_for_90_percent": int(np.searchsorted(pca_cumulative, 0.90) + 1),
            "components_for_95_percent": int(np.searchsorted(pca_cumulative, 0.95) + 1),
        },
        "elbow": {
            "line_distance_knee_1_to_15": distance_knee,
            "piecewise_linear_knee_1_to_15": piecewise_selected,
            "piecewise_sse": piecewise_errors,
            "relative_inertia_reduction": relative_reductions,
        },
        "exact_silhouette_notebook_default": exact_default,
        "runtime_seconds": float(time.perf_counter() - started),
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    with (ROOT / "clustering_root_cause_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    emit(f"Finished in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
