# Model card
**Status: experimental; trained on SYNTHETIC data; not validated on real plants.**
| | XGBoost classifier | K-Means | Isolation Forest |
|---|---|---|---|
| Task | 4-class stress level | environmental state clustering | unusual sensor combinations |
| Inputs | 62 engineered features | 16 environmental features | 20 environmental/variability features |
| Preprocessing | none (native NaN) | median impute + standardise (train-fit) | median impute + standardise (train-fit) |
| Output | class probabilities | cluster id + centroid distance | score (higher = more anomalous), flag |
| Key setting | eta .05, depth 4, 300 rounds | k by silhouette 2-8 | contamination 0.02 |
**Intended use:** local monitoring prototype and pipeline development. **Not for:** diagnosis, automated watering, or claims about plant biology.
**Limitations:** simulator labels; 16 plants / 4 weeks; weak low-vs-moderate separation; contamination is an assumption; cluster ids arbitrary; anomalies may be sensor faults or unusual environments; SHAP shows model behaviour, not causation; recommendation thresholds are heuristic and configurable.
Actual metrics for the shipped run are in `BUILD_REPORT.md` and `reports/<run>/REPORT.md`.
Robustness (mean +- sd over folds, `scripts/cv_evaluate.py`): macro-F1 0.60 +- 0.07 leave-one-session-out, 0.55 +- 0.08 on unseen plants, against 0.58 on the primary chronological hold-out; logistic regression 0.47. A validation-based search (including class weighting and ordinal variants) found no configuration better than the shipped one beyond seed noise.
