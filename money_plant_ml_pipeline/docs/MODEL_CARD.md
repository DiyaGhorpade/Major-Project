# Model card
**Status: experimental; trained on SYNTHETIC data; not validated on real plants.**
| | XGBoost classifier | K-Means | Isolation Forest |
|---|---|---|---|
| Task | 4-class stress level | environmental state clustering | unusual sensor combinations |
| Inputs | 53 engineered features: deviations from the concurrent healthy-control plants, 24/48-reading context, time of day | 16 environmental features | 20 environmental/variability features |
| Preprocessing | none (native NaN) | median impute + standardise (train-fit) | median impute + standardise (train-fit) |
| Output | class probabilities (smoothed over the plant's last 12 readings; the single-reading output is also returned) | cluster id + centroid distance | score (higher = more anomalous), flag |
| Key setting | eta .05, depth 4, 300 rounds, balanced class weights, HMM forward filter (window 12, temperature 0.3), 25% missing-reference augmentation | k by silhouette 2-8 | contamination 0.02 |
**Intended use:** local monitoring prototype and pipeline development. **Not for:** diagnosis, automated watering, or claims about plant biology.
**Inputs at inference:** the plant's last 60 readings plus the concurrent readings of the reference plants 1-4 (`reference`). Without reference readings the model falls back to per-plant features and warns (session-4 macro-F1 about 0.72 instead of 0.74).
**Limitations:** simulator labels; 16 plants / 4 weeks; "low" is the weakest class (F1 about 0.5) and control plants' "low" episodes are close to unpredictable; the cross-plant features assume the experiment's design (healthy controls in the same room, read in the same 2-hour slot); contamination is an assumption; cluster ids arbitrary; anomalies may be sensor faults or unusual environments; SHAP shows model behaviour, not causation; recommendation thresholds are heuristic and configurable.
Actual metrics for the shipped run are in `BUILD_REPORT.md` and `reports/<run>/REPORT.md`.
Shipped run `v1.0.0_20261026T145547Z` (3 seeds, mean +- sd, `scripts/cv_evaluate.py`): macro-F1 **0.743 +- 0.006** on the primary chronological hold-out (previous model 0.582), 0.678 +- 0.155 leave-one-session-out (previously 0.601 +- 0.072), 0.627 +- 0.069 on unseen plants (previously 0.554 +- 0.082); logistic regression on the same features 0.46. A label oracle that knows the true neighbouring labels reaches about 0.80-0.82 on the hold-out, so roughly 0.06-0.08 of headroom remains. The gain comes mainly from the cross-plant deviations; see the ablation in `BUILD_REPORT.md`.
