"""Write JSON / Markdown / PNG reports. Plots are skipped (with a note) when matplotlib is unavailable."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .artifact_manager import _json_default
from .schema import STRESS_LEVELS


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:  # noqa: BLE001
        return None


def write_json(path: Path, payload) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)


def plot_confusion(path: Path, cm: list[list[int]], title: str) -> bool:
    plt = _plt()
    if plt is None:
        return False
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    arr = np.asarray(cm)
    ax.imshow(arr, cmap="Blues")
    ax.set_xticks(range(len(STRESS_LEVELS)), STRESS_LEVELS, rotation=30)
    ax.set_yticks(range(len(STRESS_LEVELS)), STRESS_LEVELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=9)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, int(arr[i, j]), ha="center", va="center", color="white" if arr[i, j] > arr.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def plot_bar(path: Path, values: dict, title: str, top: int = 20, xlabel: str = "") -> bool:
    plt = _plt()
    if plt is None or not values:
        return False
    items = list(values.items())[:top][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(items) + 1.2))
    ax.barh([k for k, _ in items], [v for _, v in items], color="#3b7a57")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def plot_silhouette(path: Path, scores: dict) -> bool:
    plt = _plt()
    if plt is None:
        return False
    ks = sorted(scores)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(ks, [scores[k]["silhouette"] for k in ks], marker="o")
    ax.set_xlabel("k")
    ax.set_ylabel("silhouette")
    ax.set_title("K-Means silhouette by k (train)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def plot_hist(path: Path, values, title: str, vline: float | None = 0.0, helper=None) -> bool:
    plt = _plt()
    if plt is None:
        return False
    fig, ax = plt.subplots(figsize=(6, 3.8))
    values = np.asarray(values)
    if helper is not None and len(helper) == len(values) and np.nansum(helper) > 0:
        h = np.asarray(helper, dtype=float)
        ax.hist(values[h == 0], bins=40, alpha=0.7, label="helper label = 0")
        ax.hist(values[h == 1], bins=40, alpha=0.8, label="helper label = 1 (synthetic)")
        ax.legend()
    else:
        ax.hist(values, bins=40)
    if vline is not None:
        ax.axvline(vline, color="red", linestyle="--", label="threshold")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("anomaly_score (higher = more anomalous)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def markdown_summary(summary: dict) -> str:
    c = summary["classifier"]["metrics"]
    b = summary["classifier"]["baselines"]
    lines = [
        f"# Training report - {summary['run_id']}",
        "",
        f"> **{summary['provenance_banner']}**",
        "",
        "## Data",
        f"- file: `{summary['training_data']['filename']}` (sha256 `{summary['training_data']['sha256'][:16]}...`)",
        f"- rows: {summary['training_data']['rows']}, plants: {summary['data_quality'].get('plants')}, "
        f"time range: {summary['data_quality']['time_range']['start']} -> {summary['data_quality']['time_range']['end']}",
        f"- split: `{summary['split']['strategy']}` - train {summary['split']['train_rows']} rows / test {summary['split']['test_rows']} rows "
        f"({summary['split'].get('cutoff_timestamp') or summary['split'].get('holdout_plants')})",
        f"- stress_level counts: {summary['data_quality'].get('stress_level_counts')}",
        "",
        "## Model A - XGBoost stress classifier (synthetic-data metrics)",
        "| model | accuracy | macro-F1 | weighted-F1 | balanced acc. |",
        "|---|---|---|---|---|",
        f"| **XGBoost** | {fmt(c['accuracy'])} | {fmt(c['macro_f1'])} | {fmt(c['weighted_f1'])} | {fmt(c['balanced_accuracy'])} |",
    ]
    for name, m in b.items():
        lines.append(f"| baseline: {name} | {fmt(m['accuracy'])} | {fmt(m['macro_f1'])} | {fmt(m['weighted_f1'])} | {fmt(m['balanced_accuracy'])} |")
    lines += ["", "Per-class (test):", "| class | precision | recall | F1 | support |", "|---|---|---|---|---|"]
    for k, v in c["per_class"].items():
        lines.append(f"| {k} | {fmt(v['precision'])} | {fmt(v['recall'])} | {fmt(v['f1'])} | {v['support']} |")
    lines += ["", f"Confusion matrix (rows = true, cols = predicted; order {STRESS_LEVELS}):", "```"]
    lines += [str(r) for r in c["confusion_matrix"]["matrix"]]
    lines += ["```", ""]
    k = summary["clustering"]
    lines += ["## Model B - K-Means", f"- selected k = {k['selected_k']} (silhouette by k: "
              + ", ".join(f"{kk}: {fmt(v['silhouette'])}" for kk, v in k['silhouette_by_k'].items()) + ")",
              f"- {k['note']}", ""]
    a = summary["anomaly"]
    lines += ["## Model C - Isolation Forest", f"- contamination: {a['contamination']}; train flag rate: {fmt(a['train_flag_rate'])}",
              f"- {a['score_convention']}"]
    m = a.get("metrics_vs_helper_label")
    if m:
        lines.append(f"- vs synthetic helper labels ({a['eval_scope']} rows, {m['n_helper_positives']} positives): precision {fmt(m['precision'])}, "
                     f"recall {fmt(m['recall'])}, F1 {fmt(m['f1'])}, PR-AUC {fmt(m['pr_auc'])} (positive-rate baseline {fmt(m['positive_rate_baseline_pr_auc'])})")
    else:
        lines.append("- no suitable helper labels in the evaluation rows; no anomaly metrics reported")
    lines += ["", "## Reading these numbers", summary["metric_reading_guide"], ""]
    return "\n".join(lines)
