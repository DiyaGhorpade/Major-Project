# src/train.py

import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier
from tqdm import tqdm

from config import (
    SEED, DEVICE, MODEL_DIR, LABEL_NAMES,
    XGB_PARAMS, LR_PARAMS, MLP_PARAMS,
    N_CV_FOLDS
)
from utils import set_seed


# ── MLP Architecture ───────────────────────────────────────────────────────────

class StressClassifierMLP(nn.Module):
    """
    Small feedforward network for tabular stress classification.

    Architecture: Input → [Linear → BN → ReLU → Dropout] × N → Output

    BatchNorm before ReLU stabilizes training on small tabular datasets.
    Dropout prevents overfitting — with only 960 training samples,
    this matters more than it would on larger data.
    """
    def __init__(self, input_dim: int, hidden_dims: list, n_classes: int, dropout: float):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── CV helper ──────────────────────────────────────────────────────────────────

def run_cv(model, X_train: np.ndarray, y_train: np.ndarray, label: str):
    """
    Run stratified k-fold CV and print per-fold + mean F1-macro.
    Stratified ensures each fold has representative class proportions.
    """
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)
    results = cross_validate(
        model, X_train, y_train,
        cv=cv,
        scoring=["f1_macro", "accuracy"],
        n_jobs=-1,
    )

    f1_scores  = results["test_f1_macro"]
    acc_scores = results["test_accuracy"]

    print(f"\n[cv] {label}")
    for i, (f1, acc) in enumerate(zip(f1_scores, acc_scores)):
        print(f"     Fold {i+1}: F1-macro = {f1:.4f}  |  Accuracy = {acc:.4f}")
    print(f"     Mean F1  : {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")
    print(f"     Mean Acc : {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")

    return f1_scores.mean()


# ── Logistic Regression ────────────────────────────────────────────────────────

def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray):
    """
    Linear baseline. Tells us how much of the problem is linearly separable.
    If LR hits 85%+ we know the boundary is mostly linear.
    If LR hits 60% we know we need nonlinear models.
    """
    set_seed()
    print("\n" + "="*55)
    print(" LOGISTIC REGRESSION — baseline")
    print("="*55)

    model = LogisticRegression(**LR_PARAMS)

    mean_f1 = run_cv(model, X_train, y_train, "Logistic Regression (CV)")

    # Fit on full train for final model
    model.fit(X_train, y_train)

    path = MODEL_DIR / "logistic_regression.joblib"
    joblib.dump(model, path)
    print(f"[saved] {path}")

    return model, mean_f1


# ── XGBoost ────────────────────────────────────────────────────────────────────

def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_weights: dict,
):
    """
    Primary model. Gradient boosted trees handle nonlinear feature
    interactions and are robust to the feature scale differences
    present even after StandardScaler (due to distribution shape).

    sample_weight: per-sample weights derived from class_weights dict.
    This is XGBoost's equivalent of class_weight='balanced' in sklearn.

    device='cuda' offloads tree construction to RTX 5060.
    For 960 samples this won't show dramatic speedup vs CPU,
    but it validates the GPU pipeline for when real hardware
    data scales this up.
    """
    set_seed()
    print("\n" + "="*55)
    print(" XGBOOST — primary model")
    print("="*55)

    # Build per-sample weight array from class weights
    sample_weights = np.array([class_weights[label] for label in y_train])

    model = XGBClassifier(**XGB_PARAMS)

    # CV — note: sample_weight passed via fit_params
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)
    f1_folds = []

    print(f"\n[cv] XGBoost (CV) — training on GPU: {DEVICE}")
    for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        sw_tr       = sample_weights[tr_idx]

        fold_model = XGBClassifier(**XGB_PARAMS)
        fold_model.fit(
            X_tr, y_tr,
            sample_weight=sw_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        from sklearn.metrics import f1_score
        preds = fold_model.predict(X_val)
        f1    = f1_score(y_val, preds, average="macro")
        f1_folds.append(f1)

        acc = (preds == y_val).mean()
        print(f"     Fold {fold+1}: F1-macro = {f1:.4f}  |  Accuracy = {acc:.4f}")

    f1_arr = np.array(f1_folds)
    print(f"     Mean F1  : {f1_arr.mean():.4f} ± {f1_arr.std():.4f}")

    # Final fit on full train
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        verbose=False,
    )

    path = MODEL_DIR / "xgboost.joblib"
    joblib.dump(model, path)
    print(f"[saved] {path}")

    return model, f1_arr.mean()


# ── PyTorch MLP ────────────────────────────────────────────────────────────────

def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_weights: dict,
):
    """
    Small neural reference model. On tabular data with ~1000 samples,
    MLPs often underperform XGBoost — we include it to:
      1. Quantify that gap explicitly
      2. Validate the PyTorch + CUDA pipeline for Step 2 (GRU)
         which uses the same training loop pattern

    Early stopping on validation loss prevents overfitting.
    We carve 10% of train as internal val for early stopping only —
    this is separate from the held-out test set.
    """
    set_seed()
    print("\n" + "="*55)
    print(" MLP — neural reference")
    print("="*55)
    print(f"[mlp] Device: {DEVICE}")

    p          = MLP_PARAMS
    input_dim  = X_train.shape[1]
    n_classes  = len(LABEL_NAMES)

    # Class weight tensor for CrossEntropyLoss
    weight_tensor = torch.tensor(
        [class_weights[i] for i in range(n_classes)],
        dtype=torch.float32
    ).to(DEVICE)

    # Internal val split (stratified, 10%) for early stopping
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.10,
        stratify=y_train,
        random_state=SEED,
    )

    def to_loader(X, y, shuffle=True):
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )
        return DataLoader(ds, batch_size=p["batch_size"], shuffle=shuffle)

    train_loader = to_loader(X_tr, y_tr, shuffle=True)
    val_loader   = to_loader(X_val, y_val, shuffle=False)

    model     = StressClassifierMLP(input_dim, p["hidden_dims"], n_classes, p["dropout"]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    patience_ctr  = 0
    best_state    = None

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>9}  {'Val F1':>8}")
    print("-" * 42)

    for epoch in range(1, p["epochs"] + 1):

        # ── train ──
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_batch)
        train_loss /= len(y_tr)

        # ── validate ──
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                logits = model(X_batch)
                val_loss += criterion(logits, y_batch).item() * len(y_batch)
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        val_loss /= len(y_val)

        from sklearn.metrics import f1_score
        val_f1 = f1_score(all_labels, all_preds, average="macro")

        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>6}  {train_loss:>10.4f}  {val_loss:>9.4f}  {val_f1:>8.4f}")

        # ── early stopping ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr  = 0
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= p["patience"]:
                print(f"\n[mlp] Early stopping at epoch {epoch} (patience={p['patience']})")
                break

    # Restore best weights
    model.load_state_dict(best_state)

    # Final val F1 with best weights
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(DEVICE)
            all_preds.extend(model(X_batch).argmax(dim=1).cpu().numpy())
            all_labels.extend(y_batch.numpy())
    final_f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"\n[mlp] Best val F1-macro: {final_f1:.4f}")

    path = MODEL_DIR / "mlp.pt"
    torch.save({
        "model_state": best_state,
        "input_dim":   input_dim,
        "n_classes":   n_classes,
        "hidden_dims": p["hidden_dims"],
        "dropout":     p["dropout"],
    }, path)
    print(f"[saved] {path}")

    return model, final_f1


# ── Train all & compare ────────────────────────────────────────────────────────

def train_all(X_train, y_train, class_weights):
    set_seed()

    lr_model,  lr_f1  = train_logistic_regression(X_train, y_train)
    xgb_model, xgb_f1 = train_xgboost(X_train, y_train, class_weights)
    mlp_model, mlp_f1 = train_mlp(X_train, y_train, class_weights)

    print("\n" + "="*55)
    print(" MODEL COMPARISON — CV F1-macro")
    print("="*55)
    results = {
        "Logistic Regression": lr_f1,
        "XGBoost":             xgb_f1,
        "MLP":                 mlp_f1,
    }
    for name, f1 in sorted(results.items(), key=lambda x: -x[1]):
        bar = "█" * int(f1 * 40)
        print(f"  {name:22s}: {f1:.4f}  {bar}")

    best_name = max(results, key=results.get)
    print(f"\n  Best model: {best_name} (F1 = {results[best_name]:.4f})")
    print("="*55)

    return lr_model, xgb_model, mlp_model, results


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from preprocess import load_and_preprocess

    X_train, X_test, y_train, y_test, scaler, class_weights, feature_names = load_and_preprocess()
    lr_model, xgb_model, mlp_model, results = train_all(X_train, y_train, class_weights)
