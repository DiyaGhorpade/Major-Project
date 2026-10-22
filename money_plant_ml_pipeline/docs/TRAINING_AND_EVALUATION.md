# Training and evaluation
`python scripts/train_all.py --input <csv>` -> load -> validate (mode from config) -> features -> split -> train A/B/C -> artifacts -> reports.
* **Splits:** `chronological` (default; last 25% of unique timestamps = test) or `group_holdout` (plants 4, 8, 12, 16 unseen). Random splits are intentionally unavailable. The split and cutoff are stored in the manifest and re-used by `evaluate_all.py`.
* **Imbalance:** `classifier.class_weight: none|balanced` (train sample weights only). The test set is never rebalanced.
* **Metrics:** accuracy, macro-F1, weighted-F1, balanced accuracy, per-class P/R/F1, confusion matrix, class counts, vs majority-class and logistic-regression baselines.
* **Clustering:** silhouette for each k (sampled to 5,000 rows); limitations in the report.
* **Anomaly:** precision/recall/F1/PR-AUC/ROC-AUC against `is_synthetic_anomaly` on test rows only (label never an input). With few positives (14 in test) these are very noisy.
* **Validation modes** (`validation.mode`): fail, warn, quarantine - see ARCHITECTURE.md.
* **Retraining:** always explicit; each run is a new immutable directory.
