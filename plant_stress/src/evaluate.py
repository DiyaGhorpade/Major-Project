# src/evaluate.py

import numpy as np
import joblib
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from config import (
    DEVICE, MODEL_DIR, PLOT_DIR, REPORT_DIR,
    LABEL_NAMES, MLP_PARAMS, SHAP_MAX_SAMPLES, SEED
)
from train import StressClassifierMLP
from utils import set_seed

# ── Plot style ─────────────────────────────────────────────────────────────────

BG       = "#0f1117"
SURFACE  = "#1a1d27"
PALETTE  = ["#4caf6e", "#f5a623", "#e05252"]   # Healthy / Moderate / High
TEXT_CLR = "white"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    SURFACE,
    "axes.labelcolor":   TEXT_CLR,
    "xtick.color":       TEXT_CLR,
    "ytick.color":       TEXT_CLR,
    "text.color":        TEXT_CLR,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  False,
    "axes.spines.bottom":False,
    "grid.color":        "#2a2d3a",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
})


# ── Model loaders ──────────────────────────────────────────────────────────────

def load_lr():
    return joblib.load(MODEL_DIR / "logistic_regression.joblib")


def load_xgb():
    return joblib.load(MODEL_DIR / "xgboost.joblib")


def load_mlp(input_dim: int):
    ckpt = torch.load(MODEL_DIR / "mlp.pt", map_location="cpu")
    model = StressClassifierMLP(
        input_dim    = ckpt["input_dim"],
        hidden_dims  = ckpt["hidden_dims"],
        n_classes    = ckpt["n_classes"],
        dropout      = ckpt["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ── Predict helpers ────────────────────────────────────────────────────────────

def predict_mlp(model, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        return logits.argmax(dim=1).numpy()


def predict_proba_mlp(model, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        return torch.softmax(logits, dim=1).numpy()


# ── Per-model metrics ──────────────────────────────────────────────────────────

def evaluate_model(name: str, y_true: np.ndarray, y_pred: np.ndarray,
                   y_proba: np.ndarray = None) -> dict:
    """
    Compute and print full classification report + ROC-AUC.
    Returns dict of key metrics for comparison table.
    """
    print(f"\n{'='*55}")
    print(f" {name} — held-out test results")
    print(f"{'='*55}")

    report = classification_report(
        y_true, y_pred,
        target_names=LABEL_NAMES,
        digits=4,
    )
    print(report)

    # Save report to file
    report_path = REPORT_DIR / f"{name.lower().replace(' ', '_')}_report.txt"
    with open(report_path, "w") as f:
        f.write(f"{name} — Classification Report\n")
        f.write("="*55 + "\n")
        f.write(report)
    print(f"[saved] {report_path}")

    metrics = {
        "name":     name,
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "accuracy": (y_true == y_pred).mean(),
    }

    # ROC-AUC (one-vs-rest, requires probability scores)
    if y_proba is not None:
        y_bin = label_binarize(y_true, classes=[0, 1, 2])
        auc   = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")
        metrics["roc_auc"] = auc
        print(f"ROC-AUC (macro OvR): {auc:.4f}")
    else:
        metrics["roc_auc"] = None

    return metrics


# ── Confusion matrix plot ──────────────────────────────────────────────────────

def plot_confusion_matrices(results: list[dict], y_test: np.ndarray):
    """
    Side-by-side confusion matrices for all three models.
    Normalized by true label (rows sum to 1) so class imbalance
    doesn't make the majority class look better than it is.
    """
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6))
    fig.patch.set_facecolor(BG)

    for ax, res in zip(axes, results):
        cm = confusion_matrix(y_test, res["y_pred"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        sns.heatmap(
            cm_norm,
            annot=cm,                    # raw counts inside cells
            fmt="d",
            cmap="YlOrRd",
            vmin=0, vmax=1,
            linewidths=0.5,
            linecolor=BG,
            ax=ax,
            cbar=True,
            xticklabels=["Healthy", "Moderate", "High"],
            yticklabels=["Healthy", "Moderate", "High"],
            annot_kws={"size": 13, "weight": "bold"},
        )
        ax.set_facecolor(SURFACE)
        ax.set_title(
            f"{res['name']}\nF1-macro = {res['f1_macro']:.4f}",
            fontsize=12, pad=10
        )
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.figure.axes[-1].tick_params(colors=TEXT_CLR)

        # Colour-code diagonal cells green, off-diagonal red
        for i in range(3):
            for j in range(3):
                color = "#4caf6e" if i == j else "#e05252"
                ax.add_patch(plt.Rectangle(
                    (j, i), 1, 1,
                    fill=False,
                    edgecolor=color,
                    linewidth=2,
                    clip_on=False,
                ))

    plt.suptitle(
        "Confusion Matrices — Normalized by True Label (counts shown)",
        fontsize=14, y=1.02
    )
    plt.tight_layout()
    path = PLOT_DIR / "confusion_matrices.png"
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"\n[saved] {path}")


# ── ROC curves ─────────────────────────────────────────────────────────────────

def plot_roc_curves(results: list[dict], y_test: np.ndarray):
    """
    Per-class ROC curves for each model.
    One-vs-Rest: for each class, every other class is treated as negative.
    The curve tells you how well the model separates each stress level
    from the rest — useful for catching which class is hardest to detect.
    """
    y_bin = label_binarize(y_test, classes=[0, 1, 2])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor(BG)

    class_colors = PALETTE
    model_styles = ["-", "--", ":"]

    for cls_idx, (ax, cls_name) in enumerate(zip(axes, LABEL_NAMES)):
        ax.set_facecolor(SURFACE)
        ax.plot([0, 1], [0, 1], color="#555", linestyle="--",
                linewidth=1, label="Random (AUC=0.50)")

        for res, style in zip(results, model_styles):
            if res["y_proba"] is None:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, cls_idx], res["y_proba"][:, cls_idx])
            auc = roc_auc_score(y_bin[:, cls_idx], res["y_proba"][:, cls_idx])
            ax.plot(fpr, tpr, linestyle=style, linewidth=2,
                    color=class_colors[cls_idx],
                    label=f"{res['name']} (AUC={auc:.3f})")

        ax.set_title(f"ROC — {cls_name}", fontsize=12)
        ax.set_xlabel("False Positive Rate", fontsize=9)
        ax.set_ylabel("True Positive Rate", fontsize=9)
        ax.legend(facecolor=SURFACE, edgecolor="none", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)

    plt.suptitle("ROC Curves — One vs Rest per Stress Class", fontsize=14, y=1.02)
    plt.tight_layout()
    path = PLOT_DIR / "roc_curves.png"
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[saved] {path}")


# ── Model comparison bar chart ─────────────────────────────────────────────────

def plot_model_comparison(results: list[dict]):
    """
    Grouped bar chart comparing F1-macro, F1-weighted, Accuracy, ROC-AUC
    across all three models. Makes the performance gap visually immediate.
    """
    metrics  = ["f1_macro", "f1_weighted", "accuracy", "roc_auc"]
    labels   = ["F1-macro", "F1-weighted", "Accuracy", "ROC-AUC"]
    n        = len(metrics)
    x        = np.arange(n)
    w        = 0.22
    offsets  = [-w, 0, w]
    colors   = ["#7c9fd4", "#c49fd4", "#9fd4a5"]

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(SURFACE)

    for res, offset, color in zip(results, offsets, colors):
        vals = [res.get(m) or 0 for m in metrics]
        bars = ax.bar(x + offset, vals, width=w, label=res["name"],
                      color=color, alpha=0.88, edgecolor="none")
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", fontsize=8, color=TEXT_CLR
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison — Held-out Test Set", fontsize=14, pad=12)
    ax.legend(facecolor=SURFACE, edgecolor="none", fontsize=10)
    ax.axhline(1.0, color="#555", linewidth=0.8, linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = PLOT_DIR / "model_comparison.png"
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[saved] {path}")


# ── SHAP analysis — XGBoost ────────────────────────────────────────────────────

def run_shap_analysis(xgb_model, X_test: np.ndarray, feature_names: list[str]):
    """
    SHAP (SHapley Additive exPlanations) for XGBoost.

    WHY SHAP matters here:
        Our EDA showed Soil_Moisture dominates the label.
        SHAP will confirm this — if the model logic doesn't match
        the EDA insight, something is wrong in training.
        For production: when a plant is flagged stressed, SHAP
        tells you WHICH sensor reading caused it.

    We generate:
        1. Beeswarm plot  — global feature importance across all predictions
        2. Bar plot       — mean |SHAP| per feature (cleaner summary)
        3. Per-class heatmap — which features drive each stress level

    SHAP_MAX_SAMPLES: SHAP is O(n²) on background data.
    We subsample to keep it fast while still representative.
    """
    print(f"\n[shap] Running SHAP analysis (n={min(SHAP_MAX_SAMPLES, len(X_test))})...")

    rng        = np.random.default_rng(SEED)
    idx        = rng.choice(len(X_test), size=min(SHAP_MAX_SAMPLES, len(X_test)), replace=False)
    X_sample   = X_test[idx]

    explainer  = shap.TreeExplainer(xgb_model)
    shap_vals  = explainer.shap_values(X_sample)

    # SHAP 0.46 returns (n_samples, n_features, n_classes) for XGBoost multiclass
    # Older versions return a list of (n_samples, n_features) per class
    # Normalise to list format: [cls0_array, cls1_array, cls2_array]
    if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        # (n_samples, n_features, n_classes) → split along last axis
        shap_vals = [shap_vals[:, :, i] for i in range(shap_vals.shape[2])]
    # shap_vals is now a list of (n_samples, n_features) arrays — one per class
    # shap_vals shape: (n_classes, n_samples, n_features) for XGBoost multiclass

    # ── 1. Beeswarm — per class ────────────────────────────────────
    class_short = ["Healthy", "Moderate Stress", "High Stress"]

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.patch.set_facecolor(BG)

    for cls_idx, (ax, cls_name) in enumerate(zip(axes, class_short)):
        sv = shap_vals[cls_idx]           # (n_samples, n_features)

        # Sort features by mean |SHAP| for this class
        order = np.argsort(np.abs(sv).mean(axis=0))[::-1]

        ax.set_facecolor(SURFACE)
        for rank, feat_idx in enumerate(order[::-1]):   # bottom = most important
            vals   = sv[:, feat_idx]
            feat_v = X_sample[:, feat_idx]
            norm_v = (feat_v - feat_v.min()) / (feat_v.ptp() + 1e-9)
            colors = plt.cm.RdYlGn(norm_v)              # low value=red, high=green

            y_jitter = rank + np.random.uniform(-0.2, 0.2, size=len(vals))
            ax.scatter(vals, y_jitter, c=colors, s=14, alpha=0.7, edgecolors="none")

        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels([feature_names[i] for i in order[::-1]], fontsize=8)
        ax.axvline(0, color=TEXT_CLR, linewidth=0.8, alpha=0.4)
        ax.set_xlabel("SHAP value", fontsize=9)
        ax.set_title(f"SHAP — {cls_name}", fontsize=11, pad=8)

    plt.suptitle(
        "SHAP Beeswarm — Feature impact per stress class\n"
        "(colour = feature value: green=high, red=low)",
        fontsize=13, y=1.01
    )
    plt.tight_layout()
    path = PLOT_DIR / "shap_beeswarm.png"
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[saved] {path}")

    # ── 2. Mean |SHAP| bar — global summary ───────────────────────
    # Average absolute SHAP across all classes and all samples
    mean_abs_shap = np.mean([np.abs(shap_vals[c]).mean(axis=0)
                             for c in range(3)], axis=0)
    order_global  = np.argsort(mean_abs_shap)
    feat_sorted   = [feature_names[i] for i in order_global]
    vals_sorted   = mean_abs_shap[order_global]

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(SURFACE)

    bar_colors = [PALETTE[2] if v > vals_sorted.mean() else "#7c9fd4"
                  for v in vals_sorted]
    bars = ax.barh(feat_sorted, vals_sorted, color=bar_colors,
                   edgecolor="none", alpha=0.88)
    ax.axvline(vals_sorted.mean(), color="#f5a623", linewidth=1.5,
               linestyle="--", label=f"Mean = {vals_sorted.mean():.4f}")

    for bar, val in zip(bars, vals_sorted):
        ax.text(val + 0.0002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    ax.set_xlabel("Mean |SHAP value| (averaged across all classes)", fontsize=10)
    ax.set_title("Global Feature Importance — XGBoost SHAP", fontsize=13, pad=10)
    ax.legend(facecolor=SURFACE, edgecolor="none", fontsize=9)

    plt.tight_layout()
    path = PLOT_DIR / "shap_importance.png"
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[saved] {path}")

    # ── 3. Per-class SHAP heatmap ──────────────────────────────────
    # Mean SHAP (signed) per feature per class
    # Tells you: which features push TOWARD each class specifically
    mean_shap_signed = np.array([shap_vals[c].mean(axis=0) for c in range(3)])
    # shape: (3, n_features)

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(BG)

    im = ax.imshow(mean_shap_signed, cmap="RdYlGn", aspect="auto",
                   vmin=-np.abs(mean_shap_signed).max(),
                   vmax=np.abs(mean_shap_signed).max())

    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(3))
    ax.set_yticklabels(class_short, fontsize=10)
    ax.set_title(
        "Mean Signed SHAP per Class — green = pushes toward class, red = pushes away",
        fontsize=12, pad=10
    )
    ax.set_facecolor(SURFACE)
    plt.colorbar(im, ax=ax).ax.tick_params(colors=TEXT_CLR)

    for i in range(3):
        for j in range(len(feature_names)):
            ax.text(j, i, f"{mean_shap_signed[i, j]:.3f}",
                    ha="center", va="center", fontsize=7,
                    color="black" if abs(mean_shap_signed[i, j]) > 0.1 else TEXT_CLR)

    plt.tight_layout()
    path = PLOT_DIR / "shap_class_heatmap.png"
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[saved] {path}")

    # ── Print top features per class ───────────────────────────────
    print("\n[shap] Top 3 features per class:")
    for cls_idx, cls_name in enumerate(class_short):
        sv       = shap_vals[cls_idx]
        top_idx  = np.argsort(np.abs(sv).mean(axis=0))[::-1][:3]
        top_feats = [(feature_names[i], np.abs(sv).mean(axis=0)[i]) for i in top_idx]
        print(f"  {cls_name:20s}: " +
              " | ".join(f"{n} ({v:.4f})" for n, v in top_feats))

    return shap_vals


# ── Per-sample error analysis ──────────────────────────────────────────────────

def print_error_analysis(y_test: np.ndarray, y_pred: np.ndarray, name: str):
    """
    Break down where the model makes mistakes.
    The TYPE of error matters as much as the rate:
      Healthy → High Stress misclassification is dangerous
      Moderate → High Stress is a borderline case, less critical
    """
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n[error] {name} — error breakdown:")
    for true_cls in range(3):
        for pred_cls in range(3):
            if true_cls != pred_cls and cm[true_cls, pred_cls] > 0:
                n    = cm[true_cls, pred_cls]
                pct  = 100 * n / cm[true_cls].sum()
                flag = " ⚠ DANGEROUS" if abs(true_cls - pred_cls) == 2 else ""
                print(f"  True {LABEL_NAMES[true_cls]:18s} → "
                      f"Pred {LABEL_NAMES[pred_cls]:18s} : "
                      f"{n:3d} samples ({pct:.1f}%){flag}")


# ── Master evaluation runner ───────────────────────────────────────────────────

def run_full_evaluation(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    feature_names: list[str],
):
    set_seed()
    input_dim = X_test.shape[1]

    # Load models
    lr_model  = load_lr()
    xgb_model = load_xgb()
    mlp_model = load_mlp(input_dim)

    # Predictions
    lr_pred   = lr_model.predict(X_test)
    xgb_pred  = xgb_model.predict(X_test)
    mlp_pred  = predict_mlp(mlp_model, X_test)

    # Probabilities (for ROC-AUC)
    lr_proba  = lr_model.predict_proba(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)
    mlp_proba = predict_proba_mlp(mlp_model, X_test)

    # Per-model evaluation
    lr_metrics  = evaluate_model("Logistic Regression", y_test, lr_pred, lr_proba)
    xgb_metrics = evaluate_model("XGBoost",             y_test, xgb_pred, xgb_proba)
    mlp_metrics = evaluate_model("MLP",                 y_test, mlp_pred, mlp_proba)

    # Attach predictions for plotting
    lr_metrics["y_pred"]  = lr_pred;  lr_metrics["y_proba"]  = lr_proba
    xgb_metrics["y_pred"] = xgb_pred; xgb_metrics["y_proba"] = xgb_proba
    mlp_metrics["y_pred"] = mlp_pred; mlp_metrics["y_proba"] = mlp_proba

    all_results = [lr_metrics, xgb_metrics, mlp_metrics]

    # Error analysis
    for res in all_results:
        print_error_analysis(y_test, res["y_pred"], res["name"])

    # Plots
    print("\n[plots] Generating evaluation plots...")
    plot_confusion_matrices(all_results, y_test)
    plot_roc_curves(all_results, y_test)
    plot_model_comparison(all_results)

    # SHAP on XGBoost
    shap_vals = run_shap_analysis(xgb_model, X_test, feature_names)

    # Final summary table
    print(f"\n{'='*65}")
    print(f" FINAL EVALUATION SUMMARY — held-out test set (n={len(y_test)})")
    print(f"{'='*65}")
    print(f"{'Model':<24} {'F1-macro':>9} {'F1-weighted':>12} {'Accuracy':>9} {'ROC-AUC':>9}")
    print("-"*65)
    for res in all_results:
        auc = f"{res['roc_auc']:.4f}" if res["roc_auc"] else "  n/a  "
        print(f"{res['name']:<24} {res['f1_macro']:>9.4f} "
              f"{res['f1_weighted']:>12.4f} {res['accuracy']:>9.4f} {auc:>9}")
    print("="*65)

    return all_results, shap_vals


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from preprocess import load_and_preprocess

    X_train, X_test, y_train, y_test, scaler, class_weights, feature_names = \
        load_and_preprocess(verbose=False)

    run_full_evaluation(X_train, X_test, y_test, feature_names)
