import pytest

from plant_ml.feature_engineering import build_features
from plant_ml.labels import encode_stress_level
from plant_ml.preprocessing import LeakageError, guard_feature_list, model_feature_list, select_matrix
from plant_ml.schema import FORBIDDEN_MODEL_COLUMNS
from plant_ml.splitters import split_frame


@pytest.fixture(scope="module")
def feat(small_df, base_cfg):
    from plant_ml.data_loader import normalise_frame
    return build_features(normalise_frame(small_df), base_cfg)


def test_chronological_split_has_no_time_overlap(feat, base_cfg):
    s = split_frame(feat, base_cfg, "chronological")
    assert feat.loc[s.train_mask, "timestamp"].max() < feat.loc[s.test_mask, "timestamp"].min()
    assert not (s.train_mask & s.test_mask).any()
    assert (s.train_mask | s.test_mask).all()
    assert 0.2 < s.test_mask.mean() < 0.3


def test_group_holdout_uses_unseen_plants(feat, base_cfg):
    s = split_frame(feat, base_cfg, "group_holdout")
    assert set(feat.loc[s.train_mask, "plant_id"]).isdisjoint(set(feat.loc[s.test_mask, "plant_id"]))
    assert set(feat.loc[s.test_mask, "plant_id"]) == {4, 8, 12, 16}


def test_unknown_strategy_and_random_split_are_rejected(feat, base_cfg):
    with pytest.raises(ValueError):
        split_frame(feat, base_cfg, "random")


def test_explicit_cutoff_reproduces_split(feat, base_cfg):
    s1 = split_frame(feat, base_cfg, "chronological")
    s2 = split_frame(feat, base_cfg, "chronological", cutoff=s1.info["cutoff_timestamp"])
    assert (s1.train_mask == s2.train_mask).all()


@pytest.mark.parametrize("model", ["classifier", "clustering", "anomaly"])
def test_model_feature_lists_exclude_labels_ids_and_timestamps(base_cfg, model):
    names = model_feature_list(base_cfg, model)
    assert not set(names) & set(FORBIDDEN_MODEL_COLUMNS)
    assert len(names) == len(set(names))


def test_classifier_excludes_elapsed_hours_by_default(base_cfg):
    assert "elapsed_hours" not in model_feature_list(base_cfg, "classifier")


@pytest.mark.parametrize("bad", FORBIDDEN_MODEL_COLUMNS)
def test_guard_rejects_each_forbidden_column(base_cfg, bad):
    with pytest.raises(LeakageError):
        guard_feature_list(["soil", bad], base_cfg)


def test_feature_matrices_contain_only_selected_numeric_columns(feat, base_cfg):
    for model in ("classifier", "clustering", "anomaly"):
        names = model_feature_list(base_cfg, model)
        X = select_matrix(feat, names)
        assert list(X.columns) == names
        assert not set(X.columns) & {"stress_level", "stress_type", "experimental_group", "is_synthetic_anomaly", "plant_id"}
        assert str(X.dtypes.unique()[0]) == "float64"


def test_labels_encode_to_four_classes(feat):
    y = encode_stress_level(feat["stress_level"])
    assert set(y.dropna().unique()) == {0, 1, 2, 3}
