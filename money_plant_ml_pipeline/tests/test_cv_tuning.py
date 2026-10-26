import numpy as np
import pandas as pd
import pytest

from plant_ml import tuning
from plant_ml.classifier_variants import (ClassifierSpec, _cum_probs, fit_classifier, regression_probs, spec_from_config,
                                          tune_cutpoints)
from plant_ml.cv_evaluation import (build_folds, evaluate_folds, fit_logreg, leave_one_session_out_folds,
                                    logreg_fit_predict, majority_fit_predict, session_ids, stratified_group_folds,
                                    summarise, xgb_fit_predict)
from plant_ml.data_loader import normalise_frame
from plant_ml.feature_engineering import build_features
from plant_ml.labels import encode_stress_level
from plant_ml.preprocessing import model_feature_list, select_matrix
from plant_ml.schema import EXPERIMENTAL_GROUPS, plant_to_group
from plant_ml.splitters import split_frame
from plant_ml.synthetic import generate


@pytest.fixture(scope="module")
def feat(base_cfg):
    return build_features(normalise_frame(generate(n_days=28, seed=7)), base_cfg)


@pytest.fixture(scope="module")
def names(base_cfg):
    return model_feature_list(base_cfg, "classifier")


def _tiny_spec(base_cfg, **kw):
    s = spec_from_config(base_cfg)
    return ClassifierSpec(kind=kw.get("kind", "softprob"), params={**s.params, "max_depth": 3, "nthread": 2},
                          num_boost_round=kw.get("rounds", 15), class_weight=kw.get("class_weight", "none"))


# ------------------------------------------------------------------ fold construction
def test_sessions_are_four_blocks(feat):
    assert sorted(session_ids(feat).unique()) == [1, 2, 3, 4]


def test_leave_one_session_out_has_no_session_overlap(feat):
    folds = leave_one_session_out_folds(feat)
    sess = session_ids(feat).to_numpy()
    assert len(folds) == 4
    for f in folds:
        assert not (f.train & f.test).any() and (f.train | f.test).all()
        assert set(sess[f.test]) == {f.info["test_session"]}
        assert f.info["test_session"] not in set(sess[f.train])
        assert f.info["uses_future_data"] == (f.info["test_session"] < 4)
    assert not folds[-1].info["uses_future_data"]          # holding out the last session is strictly causal


def test_group_folds_are_plant_disjoint_and_stratified_by_group(feat):
    folds = stratified_group_folds(feat, 4, seed=3)
    assert len(folds) == 4
    tested = []
    for f in folds:
        train_plants, test_plants = set(feat.loc[f.train, "plant_id"]), set(feat.loc[f.test, "plant_id"])
        assert train_plants.isdisjoint(test_plants)
        assert {plant_to_group(p) for p in test_plants} == set(EXPERIMENTAL_GROUPS)       # every group present in every test fold
        assert len(test_plants) == len(EXPERIMENTAL_GROUPS)
        tested += sorted(test_plants)
    assert sorted(tested) == list(range(1, 17))                                            # each plant is tested exactly once


def test_group_folds_are_seeded(feat):
    a, b, c = (stratified_group_folds(feat, 4, seed=s) for s in (1, 1, 2))
    assert all((x.test == y.test).all() for x, y in zip(a, b))
    assert any((x.test != z.test).any() for x, z in zip(a, c))


def test_build_folds_includes_unchanged_chronological_holdout(feat, base_cfg):
    folds = build_folds(feat, base_cfg)
    chrono = [f for f in folds if f.scheme == "chronological"]
    ref = split_frame(feat, base_cfg, "chronological")
    assert len(chrono) == 1 and (chrono[0].train == ref.train_mask.to_numpy()).all() and (chrono[0].test == ref.test_mask.to_numpy()).all()
    assert set(f.scheme for f in folds) == {"chronological", "leave_one_session", "group_kfold_plants"}


# ------------------------------------------------------------------ preprocessing is fitted per fold
def test_logreg_preprocessing_uses_only_training_rows():
    rng = np.random.default_rng(0)
    X_tr = pd.DataFrame({"a": rng.normal(0, 1, 60), "b": rng.normal(5, 1, 60)})
    X_tr.loc[::7, "a"] = np.nan
    y_tr = rng.integers(0, 4, 60)
    X_te = pd.DataFrame({"a": rng.normal(1000, 1, 10), "b": rng.normal(-1000, 1, 10)})
    pipe = fit_logreg(X_tr, y_tr, seed=0)
    np.testing.assert_allclose(pipe.named_steps["imputer"].statistics_, X_tr.median().to_numpy())
    np.testing.assert_allclose(pipe.named_steps["scaler"].mean_[1], X_tr["b"].mean())
    out = logreg_fit_predict(X_tr, y_tr, X_te, 0)
    assert out.shape == (10, 4) and np.allclose(out.sum(axis=1), 1)


def test_every_fold_fits_on_its_own_training_rows_only(feat, names):
    seen = []

    def spy(X_tr, y_tr, X_te, seed):
        seen.append((set(X_tr.index), set(X_te.index)))
        return logreg_fit_predict(X_tr, y_tr, X_te, seed)

    folds = leave_one_session_out_folds(feat) + stratified_group_folds(feat, 4, seed=1)
    evaluate_folds(feat, folds, spy, names)
    assert len(seen) == len(folds)
    y_ok = encode_stress_level(feat["stress_level"]).notna().to_numpy()
    for (tr_idx, te_idx), f in zip(seen, folds):
        assert tr_idx.isdisjoint(te_idx)
        assert tr_idx == set(np.flatnonzero(f.train & y_ok)) and te_idx == set(np.flatnonzero(f.test & y_ok))


def test_evaluate_folds_reports_mean_sd_and_pooled_confusion(feat, names, base_cfg):
    folds = leave_one_session_out_folds(feat)
    res = evaluate_folds(feat, folds, xgb_fit_predict(_tiny_spec(base_cfg)), names, seeds=(1, 2))
    s = res["summary"]
    for k in ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy"):
        assert 0 <= s[k]["mean"] <= 1 and s[k]["sd"] >= 0 and s[k]["n"] == 4
    assert set(s["per_class"]) == {"healthy", "low", "moderate", "high"}
    assert all({"precision", "recall", "f1"} <= set(v) for v in s["per_class"].values())
    cm = np.asarray(s["pooled_confusion_matrix"]["matrix"])
    assert cm.shape == (4, 4) and cm.sum() == pytest.approx(sum(f["n_test"] for f in res["folds"]), abs=1)
    assert all("seed_sd" in f for f in res["folds"])


def test_majority_baseline_is_far_below_model(feat, names, base_cfg):
    folds = leave_one_session_out_folds(feat)[:2]
    maj = evaluate_folds(feat, folds, majority_fit_predict, names)["summary"]["macro_f1"]["mean"]
    mod = evaluate_folds(feat, folds, xgb_fit_predict(_tiny_spec(base_cfg)), names)["summary"]["macro_f1"]["mean"]
    assert mod > maj


def test_summarise_ignores_folds_without_the_class():
    def fold(f1_high, support_high):
        pc = {c: {"precision": .5, "recall": .5, "f1": .5, "support": 10} for c in ("healthy", "low", "moderate")}
        pc["high"] = {"precision": 0.0, "recall": 0.0, "f1": f1_high, "support": support_high}
        return {"accuracy": .5, "macro_f1": .5, "weighted_f1": .5, "balanced_accuracy": .5, "per_class": pc,
                "confusion_matrix": {"matrix": np.eye(4).tolist()}}
    s = summarise([fold(0.8, 5), fold(0.0, 0)])
    assert s["per_class"]["high"]["f1"]["mean"] == pytest.approx(0.8) and s["per_class"]["high"]["n_folds_with_class"] == 1


# ------------------------------------------------------------------ tuning
def test_inner_folds_use_train_sessions_only_and_skip_session_one(feat, base_cfg):
    split = split_frame(feat, base_cfg)
    train_feat = feat[split.train_mask].reset_index(drop=True)
    folds = tuning.inner_folds(train_feat)
    assert [f["val_session"] for f in folds] == [2, 3]
    sess = session_ids(train_feat).to_numpy()
    assert 4 not in set(sess)                                                  # the test session never reaches the search
    for f in folds:
        assert not (f["train"] & f["val"]).any() and (f["train"] | f["val"]).all()
    assert [f["uses_future_data"] for f in folds] == [True, False]            # validating on session 3 is causal


def test_search_scores_candidates_on_validation_only(feat, names, base_cfg):
    split = split_frame(feat, base_cfg)
    train_feat = feat[split.train_mask].reset_index(drop=True)
    specs = [_tiny_spec(base_cfg), _tiny_spec(base_cfg, class_weight="balanced"), _tiny_spec(base_cfg, kind="cumulative"),
             _tiny_spec(base_cfg, kind="regression")]
    res = tuning.search(train_feat, names, specs, seed=1, log=None)
    assert len(res) == 4 and res[0]["val_macro_f1"] >= res[-1]["val_macro_f1"]
    for r in res:
        assert len(r["val_macro_f1_per_fold"]) == 2 and all(0 <= v <= 1 for v in r["val_macro_f1_per_fold"])
        assert all(b and b >= 1 for b in r["best_rounds"])
    reg = next(r for r in res if r["spec"].kind == "regression")
    assert len(reg["spec"].cutpoints) == 3 and list(reg["spec"].cutpoints) == sorted(reg["spec"].cutpoints)


def test_sample_specs_is_seeded_and_stays_in_space(base_cfg):
    base = spec_from_config(base_cfg)
    a, b = tuning.sample_specs(base, 5, 7), tuning.sample_specs(base, 5, 7)
    assert [s.to_dict() for s in a] == [s.to_dict() for s in b] and len(a) == 10
    for s in a:
        assert s.params["max_depth"] in tuning.SEARCH_SPACE["max_depth"] and s.params["eta"] in tuning.SEARCH_SPACE["eta"]


def test_confirm_on_test_scores_the_test_partition(feat, names, base_cfg):
    split = split_frame(feat, base_cfg)
    r = tuning.confirm_on_test(feat, split, names, _tiny_spec(base_cfg), [1, 2])
    assert len(r["per_seed"]) == 2 and 0 <= r["summary"]["macro_f1"]["mean"] <= 1


# ------------------------------------------------------------------ variants keep the 4-class probability structure
@pytest.mark.parametrize("kind", ["softprob", "cumulative", "regression"])
def test_every_variant_outputs_valid_four_class_probabilities(feat, names, base_cfg, kind):
    y = encode_stress_level(feat["stress_level"])
    X = select_matrix(feat, names)
    idx = np.arange(0, len(X), 3)
    spec = _tiny_spec(base_cfg, kind=kind)
    m = fit_classifier(spec, X.iloc[idx], y.iloc[idx].astype(int).to_numpy(), seed=0)
    p = m.predict_proba(X.iloc[:50])
    assert p.shape == (50, 4) and (p >= 0).all() and np.allclose(p.sum(axis=1), 1.0, atol=1e-6)


def test_balanced_class_weight_changes_the_model(feat, names, base_cfg):
    y = encode_stress_level(feat["stress_level"]).astype(int).to_numpy()
    X = select_matrix(feat, names)
    p0 = fit_classifier(_tiny_spec(base_cfg), X, y, 0).predict_proba(X.iloc[:200])
    p1 = fit_classifier(_tiny_spec(base_cfg, class_weight="balanced"), X, y, 0).predict_proba(X.iloc[:200])
    assert not np.allclose(p0, p1)


def test_cumulative_probabilities_are_monotone_clipped():
    p = _cum_probs(np.array([[0.9, 0.95, 0.2], [0.1, 0.05, 0.01]]))
    assert (p >= 0).all() and np.allclose(p.sum(axis=1), 1)


def test_regression_probs_argmax_matches_cutpoints_and_cutpoint_search_improves():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 400)
    score = y + rng.normal(0, 0.3, 400) + 0.3          # biased scores: default cut-points are off
    cp = (0.5, 1.5, 2.5)
    assert (regression_probs(score, cp).argmax(axis=1) == np.digitize(score, cp)).all()
    from sklearn.metrics import f1_score
    tuned = tune_cutpoints(score, y)
    assert f1_score(y, np.digitize(score, tuned), average="macro") >= f1_score(y, np.digitize(score, cp), average="macro")
    assert list(tuned) == sorted(tuned)
