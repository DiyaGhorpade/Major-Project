import numpy as np
import pandas as pd
import pytest

from plant_ml.cv_evaluation import Fold, evaluate_folds, smooth_in_fold
from plant_ml.temporal_smoothing import apply_from_meta, fit_transitions, forward_filter, smoothing_params, to_meta


def _proba(n, seed=0):
    p = np.random.default_rng(seed).random((n, 4)) + 0.05
    return p / p.sum(axis=1, keepdims=True)


A = np.array([[0.8, 0.1, 0.05, 0.05], [0.1, 0.7, 0.15, 0.05], [0.05, 0.15, 0.7, 0.1], [0.05, 0.05, 0.1, 0.8]])
PRIOR = np.array([0.3, 0.3, 0.2, 0.2])


def test_configured_smoothing(base_cfg):
    p = smoothing_params(base_cfg)
    assert p["method"] == "hmm_forward_filter" and p["window"] >= 1 and 0 < p["temperature"] <= 1


def test_filter_outputs_are_distributions():
    out = forward_filter(_proba(30), np.zeros(30), A, PRIOR, 0.3, 12)
    assert out.shape == (30, 4) and np.allclose(out.sum(axis=1), 1.0) and (out >= 0).all()


def test_filter_is_past_only():
    p = _proba(40)
    g = np.zeros(40)
    base = forward_filter(p, g, A, PRIOR, 0.5, 12)
    q = p.copy()
    q[25:] = _proba(15, seed=9)                                   # change the future only
    out = forward_filter(q, g, A, PRIOR, 0.5, 12)
    assert np.allclose(base[:25], out[:25]) and not np.allclose(base[25:], out[25:])


def test_window_makes_it_stateless():
    """Row k depends only on rows k-window+1..k: re-running on just that tail gives the same answer (inference path)."""
    p, w = _proba(50), 12
    full = forward_filter(p, np.zeros(50), A, PRIOR, 0.3, w)
    for k in (11, 20, 49):
        tail = forward_filter(p[k - w + 1:k + 1], np.zeros(w), A, PRIOR, 0.3, w)
        assert np.allclose(full[k], tail[-1])


def test_groups_do_not_mix():
    p = _proba(40)
    g = np.repeat([1, 2], 20)
    both = forward_filter(p, g, A, PRIOR, 0.3, 12)
    assert np.allclose(both[:20], forward_filter(p[:20], g[:20], A, PRIOR, 0.3, 12))
    assert np.allclose(both[20:], forward_filter(p[20:], g[20:], A, PRIOR, 0.3, 12))


def test_window_one_with_unit_temperature_returns_input():
    p = _proba(10)
    assert np.allclose(forward_filter(p, np.zeros(10), A, PRIOR, 1.0, 1), p)


def test_fit_transitions_counts_within_groups_only():
    y = np.array([0, 1, 1, 3, 3, 3])
    g = np.array([1, 1, 1, 2, 2, 2])
    T, prior = fit_transitions(y, g, pseudocount=1e-12)
    assert T[1, 3] == pytest.approx(0.0, abs=1e-9)                # 1 -> 3 happens only across the group boundary
    assert T[1, 1] == pytest.approx(1.0) and T[0, 1] == pytest.approx(1.0) and T[3, 3] == pytest.approx(1.0)
    assert np.allclose(T.sum(axis=1), 1.0) and prior.sum() == pytest.approx(1.0)


def test_meta_round_trip_and_none_method():
    p = _proba(20)
    meta = to_meta({"method": "hmm_forward_filter", "window": 5, "temperature": 0.4, "transition_pseudocount": 1.0}, A, PRIOR)
    assert np.allclose(apply_from_meta(meta, p, np.zeros(20)), forward_filter(p, np.zeros(20), A, PRIOR, 0.4, 5), atol=1e-5)
    assert apply_from_meta({"method": "none"}, p, np.zeros(20)) is p and apply_from_meta(None, p, np.zeros(20)) is p


# ------------------------------------------------------------------ per-fold fit (no test labels reach the smoother)
def test_smoothing_is_fitted_on_training_labels_only():
    p = _proba(30)
    y_tr = np.random.default_rng(1).integers(0, 4, 60)
    g_tr = np.repeat([1, 2], 30)
    cfg = {"method": "hmm_forward_filter", "window": 6, "temperature": 0.3, "transition_pseudocount": 1.0}
    a = smooth_in_fold(p, y_tr, g_tr, np.zeros(30), cfg)
    T, prior = fit_transitions(y_tr, g_tr, 1.0)
    assert np.allclose(a, forward_filter(p, np.zeros(30), T, prior, 0.3, 6))
    assert smooth_in_fold(p, y_tr, g_tr, np.zeros(30), {"method": "none"}) is p


def test_evaluate_folds_smoothing_ignores_test_labels():
    """Scrambling the TEST labels changes the score but not the smoothed predictions (transitions come from train)."""
    n = 120
    rng = np.random.default_rng(3)
    labels = np.array(["healthy", "low", "moderate", "high"])
    feat = pd.DataFrame({"plant_id": np.repeat([1, 2, 3, 4], n // 4), "x": rng.normal(size=n),
                         "stress_level": labels[rng.integers(0, 4, n)]})
    train = feat["plant_id"].isin([1, 2]).to_numpy()
    fold = Fold("f", "custom", train, ~train)
    captured = []

    def fp(X_tr, y_tr, X_te, seed):
        out = np.tile([0.4, 0.3, 0.2, 0.1], (len(X_te), 1)) + 0.05 * np.abs(X_te[["x"]].to_numpy())
        return out / out.sum(axis=1, keepdims=True)

    sm = {"method": "hmm_forward_filter", "window": 8, "temperature": 0.5, "transition_pseudocount": 1.0}
    res1 = evaluate_folds(feat, [fold], fp, ["x"], smoothing=sm)
    feat2 = feat.copy()
    feat2.loc[~train, "stress_level"] = labels[rng.integers(0, 4, int((~train).sum()))]
    res2 = evaluate_folds(feat2, [fold], fp, ["x"], smoothing=sm)
    pred1 = np.asarray(res1["folds"][0]["confusion_matrix"]["matrix"]).sum(axis=0)    # predicted-class counts
    pred2 = np.asarray(res2["folds"][0]["confusion_matrix"]["matrix"]).sum(axis=0)
    assert np.array_equal(pred1, pred2)
    feat3 = feat.copy()
    feat3.loc[train, "stress_level"] = "high"                     # TRAIN labels do drive the smoother
    res3 = evaluate_folds(feat3, [fold], fp, ["x"], smoothing=sm)
    assert not np.array_equal(pred1, np.asarray(res3["folds"][0]["confusion_matrix"]["matrix"]).sum(axis=0))


# ------------------------------------------------------------------ missing-reference augmentation (training only)
def test_missing_reference_augmentation_blanks_only_cross_plant_columns():
    from plant_ml.classifier_variants import augment_missing_reference
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"soil_ref_dev": rng.normal(size=400), "soil_ref_dev_roll3": rng.normal(size=400), "hour": rng.normal(size=400)})
    y = rng.integers(0, 4, 400)
    Xa, ya = augment_missing_reference(X, y, 0.25, seed=1)
    extra = Xa.iloc[len(X):]
    assert 60 < len(extra) < 140 and len(ya) == len(Xa)
    assert extra[["soil_ref_dev", "soil_ref_dev_roll3"]].isna().all().all() and extra["hour"].notna().all()
    pd.testing.assert_frame_equal(Xa.iloc[:len(X)], X)                      # originals untouched
    assert np.array_equal(ya[len(X):], y[X.index.get_indexer(extra.index)])  # labels follow their rows
    same, _ = augment_missing_reference(X[["hour"]], y, 0.25, seed=1)
    assert len(same) == len(X)                                               # nothing to blank -> no copies
    assert len(augment_missing_reference(X, y, 0.0, seed=1)[0]) == len(X)


def test_configured_spec_carries_augmentation(base_cfg):
    from plant_ml.classifier_variants import ClassifierSpec, spec_from_config
    s = spec_from_config(base_cfg)
    assert s.missing_reference_augmentation == pytest.approx(base_cfg.get("classifier.missing_reference_augmentation"))
    assert ClassifierSpec.from_dict(s.to_dict()) == s
