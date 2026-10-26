import numpy as np
import pandas as pd
import pytest

from conftest import make_plant_frame
from plant_ml.feature_engineering import (all_feature_names, build_features, compute_vpd, cross_plant_feature_names,
                                         cross_plant_params, required_history_rows)
from plant_ml.preprocessing import model_feature_list


def two_plants(n=80):
    return pd.concat([make_plant_frame(n, 1, seed=1), make_plant_frame(n, 2, seed=2)], ignore_index=True)


def test_vpd_formula_and_units():
    # T=25 C, RH=50%: es = 3.169 kPa, vpd = es * 0.5
    assert compute_vpd(25.0, 50.0) == pytest.approx(1.5845, abs=2e-3)
    assert compute_vpd(25.0, 100.0) == pytest.approx(0.0, abs=1e-12)
    assert compute_vpd(30.0, 40.0) > compute_vpd(20.0, 40.0)          # warmer air -> higher deficit
    assert compute_vpd(np.array([20.0, 30.0]), np.array([50.0, 50.0])).shape == (2,)


def test_soil_rate_in_pct_points_per_hour(base_cfg):
    df = make_plant_frame(5, soil=[50.0, 48.0, 48.0, 44.0, 44.0])
    f = build_features(df, base_cfg)
    assert np.isnan(f.loc[0, "soil_rate_per_hr"])
    assert f.loc[1, "soil_rate_per_hr"] == pytest.approx(-1.0)        # -2 points over 2 h
    assert f.loc[3, "soil_rate_per_hr"] == pytest.approx(-2.0)


def test_soil_rate_safe_for_equal_or_reversed_timestamps(base_cfg):
    df = make_plant_frame(4, soil=[50.0, 49.0, 48.0, 47.0])
    df.loc[2, "timestamp"] = df.loc[1, "timestamp"]                 # zero elapsed time
    df.loc[3, "timestamp"] = df.loc[1, "timestamp"]                 # also sorted equal; no negative after sort
    f = build_features(df, base_cfg)
    assert np.isfinite(f["soil_rate_per_hr"].dropna()).all()
    assert not np.isinf(f["soil_rate_per_hr"]).any()
    assert f["soil_rate_per_hr"].iloc[2:].isna().all()               # NaN instead of inf / division by zero


def test_features_independent_of_input_row_order(base_cfg):
    df = two_plants(40)
    a = build_features(df, base_cfg).set_index(["plant_id", "timestamp"])
    b = build_features(df.sample(frac=1.0, random_state=3), base_cfg).set_index(["plant_id", "timestamp"]).loc[a.index]
    names = all_feature_names(base_cfg)
    pd.testing.assert_frame_equal(a[names], b[names])


def test_output_is_sorted_by_plant_then_time(base_cfg):
    f = build_features(two_plants(20).sample(frac=1.0, random_state=0), base_cfg)
    key = list(zip(f["plant_id"], f["timestamp"]))
    assert key == sorted(key)


def test_no_future_leakage_truncation_invariance(base_cfg):
    df = two_plants(90)
    full = build_features(df, base_cfg).set_index(["plant_id", "timestamp"])
    cut_ts = df["timestamp"].sort_values().iloc[100]
    part = build_features(df[df["timestamp"] <= cut_ts], base_cfg).set_index(["plant_id", "timestamp"])
    names = all_feature_names(base_cfg)
    pd.testing.assert_frame_equal(full.loc[part.index, names], part[names])   # earlier rows never depend on later rows


def test_changing_future_values_does_not_change_past_features(base_cfg):
    df = make_plant_frame(60, 1, seed=5)
    f1 = build_features(df, base_cfg)
    df2 = df.copy()
    df2.loc[40:, ["soil", "temperature", "humidity", "light"]] *= 1.5       # perturb the future only
    f2 = build_features(df2, base_cfg)
    names = all_feature_names(base_cfg)
    pd.testing.assert_frame_equal(f1.loc[:39, names], f2.loc[:39, names])


def test_no_cross_plant_contamination(base_cfg):
    # 5 and 6 are NOT reference plants: no feature of plant 5 may depend on plant 6 (references are tested below)
    df = pd.concat([make_plant_frame(60, 5, seed=1), make_plant_frame(60, 6, seed=2)], ignore_index=True)
    f1 = build_features(df, base_cfg)
    df2 = df.copy()
    df2.loc[df2["plant_id"] == 6, ["soil", "temperature", "humidity", "light"]] *= 3.0
    f2 = build_features(df2, base_cfg)
    names = all_feature_names(base_cfg)
    pd.testing.assert_frame_equal(f1[f1["plant_id"] == 5][names].reset_index(drop=True),
                                  f2[f2["plant_id"] == 5][names].reset_index(drop=True))
    first = f1[f1["plant_id"] == 6].iloc[0]
    assert np.isnan(first["soil_lag1"]) and np.isnan(first["soil_diff1"])   # plant 6 does not inherit plant 5's last row


# ------------------------------------------------------------------ cross-plant (reference) features
REF_PLANTS = (1, 2, 3, 4)


def room(n=80, target=9, offset_s=90, seed=0):
    """Reference plants 1-4 and one target plant sharing a 2-hourly schedule; the target is read ``offset_s`` later."""
    parts = [make_plant_frame(n, p, seed=seed + p) for p in REF_PLANTS]
    t = make_plant_frame(n, target, seed=seed + target)
    t["timestamp"] = t["timestamp"] + pd.Timedelta(seconds=offset_s)
    return pd.concat(parts + [t], ignore_index=True)


def ref_names(cfg):
    return cross_plant_feature_names(cfg)


def test_cross_plant_features_enabled_with_controls_as_reference(base_cfg):
    assert cross_plant_params(base_cfg)["reference_plants"] == list(REF_PLANTS)
    assert ref_names(base_cfg) and set(ref_names(base_cfg)) <= set(all_feature_names(base_cfg))


def test_ref_dev_is_value_minus_median_of_concurrent_references(base_cfg):
    df = room(10)
    f = build_features(df, base_cfg)
    t = f[f["plant_id"] == 9].iloc[5]
    ts = t["timestamp"] - pd.Timedelta(seconds=90)
    refs = df[(df["plant_id"].isin(REF_PLANTS)) & (df["timestamp"] == ts)]["soil"]
    assert t["soil_ref_dev"] == pytest.approx(t["soil"] - refs.median())


def test_reference_plant_is_never_its_own_reference(base_cfg):
    df = room(10)
    f = build_features(df, base_cfg)
    r = f[f["plant_id"] == 1].iloc[3]
    others = df[(df["plant_id"].isin([2, 3, 4])) & (df["timestamp"] == r["timestamp"])]["soil"]
    assert r["soil_ref_dev"] == pytest.approx(r["soil"] - others.median())
    assert r["_ref_plants_used"] == 3


def test_cross_plant_features_never_use_later_reference_readings(base_cfg):
    df = room(80)
    t9 = df[df["plant_id"] == 9]["timestamp"].sort_values().reset_index(drop=True)
    cut = t9.iloc[40]
    f1 = build_features(df, base_cfg)
    df2 = df.copy()
    later = df2["plant_id"].isin(REF_PLANTS) & (df2["timestamp"] > cut)        # reference readings after the cut ...
    df2.loc[later, ["soil", "temperature", "humidity", "light"]] *= 2.5      # ... are changed
    f2 = build_features(df2, base_cfg)
    keep = (f1["plant_id"] == 9) & (f1["timestamp"] <= cut)
    names = all_feature_names(base_cfg)
    pd.testing.assert_frame_equal(f1.loc[keep, names], f2.loc[keep, names])  # rows up to the cut are untouched
    assert not np.allclose(f1.loc[(f1["plant_id"] == 9) & (f1["timestamp"] > cut), "soil_ref_dev"],
                           f2.loc[(f2["plant_id"] == 9) & (f2["timestamp"] > cut), "soil_ref_dev"])   # later rows do use them


def test_reference_reading_just_after_the_row_is_not_used(base_cfg):
    # the target is read BEFORE the references in each slot -> it only sees the previous slot (2 h old > 30 min tolerance)
    df = room(20, offset_s=-60)
    f = build_features(df, base_cfg)
    t = f[f["plant_id"] == 9]
    assert t["soil_ref_dev"].isna().all() and (t["_ref_plants_used"] == 0).all()


def test_truncation_invariance_with_reference_plants(base_cfg):
    df = room(90)
    full = build_features(df, base_cfg).set_index(["plant_id", "timestamp"])
    cut_ts = df["timestamp"].sort_values().iloc[220]
    part = build_features(df[df["timestamp"] <= cut_ts], base_cfg).set_index(["plant_id", "timestamp"])
    names = all_feature_names(base_cfg)
    pd.testing.assert_frame_equal(full.loc[part.index, names], part[names])


def test_separate_reference_frame_equals_training_style_frame(base_cfg):
    """Serving (target history + `reference`) gives exactly the training features (all plants in one frame)."""
    df = room(120)
    train_style = build_features(df, base_cfg)
    train_style = train_style[train_style["plant_id"] == 9].set_index("timestamp")
    need = required_history_rows(base_cfg)
    target = df[df["plant_id"] == 9].iloc[-need:].reset_index(drop=True)
    reference = df[df["plant_id"].isin(REF_PLANTS)].reset_index(drop=True)
    served = build_features(target, base_cfg, reference=reference).iloc[-1]
    for n in model_feature_list(base_cfg, "classifier"):
        assert served[n] == pytest.approx(train_style.iloc[-1][n], nan_ok=True), n


def test_reference_rows_add_no_output_rows_and_target_rows_win(base_cfg):
    df = room(30)
    target = df[df["plant_id"] == 1].reset_index(drop=True)
    noisy = df[df["plant_id"].isin(REF_PLANTS)].copy()
    noisy.loc[noisy["plant_id"] == 1, "soil"] = 0.0                       # a conflicting copy of the target's own rows
    f = build_features(target, base_cfg, reference=noisy)
    assert len(f) == len(target) and set(f["plant_id"]) == {1}
    pd.testing.assert_series_equal(f["soil"].reset_index(drop=True), target["soil"].reset_index(drop=True), check_names=False)


def test_cross_plant_features_absent_without_references_are_nan(base_cfg):
    f = build_features(make_plant_frame(30, 9), base_cfg)
    assert f[ref_names(base_cfg)].isna().all().all() and (f["_ref_plants_used"] == 0).all()


def test_baseline_is_past_only_with_warmup(base_cfg):
    df = make_plant_frame(20, soil=np.arange(20, dtype=float))
    f = build_features(df, base_cfg)
    warm = base_cfg.get("features.baseline_min_periods")
    assert f["soil_base_dev"].iloc[:warm].isna().all()                       # warm-up: not enough past rows
    assert f.loc[10, "soil_base_dev"] == pytest.approx(10 - np.mean(np.arange(10)))   # excludes the current row


def test_rolling_windows_are_trailing_not_centered(base_cfg):
    df = make_plant_frame(20, soil=np.arange(20, dtype=float))
    f = build_features(df, base_cfg)
    assert f.loc[10, "soil_roll3_mean"] == pytest.approx(np.mean([8, 9, 10]))
    assert f.loc[10, "soil_roll6_mean"] == pytest.approx(np.mean(range(5, 11)))
    assert np.isnan(f.loc[0, "soil_roll3_std"])                              # min_periods=2 for std
    assert f.loc[1, "soil_roll3_std"] == pytest.approx(np.std([0, 1], ddof=1))


def test_lag_and_trend_features(base_cfg):
    df = make_plant_frame(10, soil=np.arange(10, dtype=float) * 2)
    f = build_features(df, base_cfg)
    assert f.loc[5, "soil_lag1"] == 8 and f.loc[5, "soil_diff1"] == 2 and f.loc[5, "soil_trend3"] == 6


def test_time_and_interaction_features(base_cfg):
    df = make_plant_frame(6, start="2026-10-04T06:00:00+05:30")             # a Sunday, 06:00 local
    f = build_features(df, base_cfg)
    assert f.loc[0, "hour"] == 6 and f.loc[0, "dayofweek"] == 6
    assert f.loc[0, "hour_sin"] == pytest.approx(1.0) and abs(f.loc[0, "hour_cos"]) < 1e-9
    assert f.loc[3, "elapsed_hours"] == pytest.approx(6.0)
    assert f.loc[0, "log_light"] == pytest.approx(np.log1p(df.loc[0, "light"]))
    assert f.loc[0, "light_soil_interaction"] == pytest.approx(np.log1p(df.loc[0, "light"]) / np.log1p(100000) * df.loc[0, "soil"])


def test_missing_sensor_values_stay_missing(base_cfg):
    df = make_plant_frame(10)
    df.loc[4, "soil"] = np.nan
    f = build_features(df, base_cfg)
    assert np.isnan(f.loc[4, "soil"]) and np.isnan(f.loc[4, "soil_diff1"]) and np.isnan(f.loc[5, "soil_diff1"])
    assert not np.isinf(f[all_feature_names(base_cfg)].to_numpy(dtype=float)).any()


def test_history_window_gives_same_features_as_full_history(base_cfg):
    df = make_plant_frame(200, seed=9)
    full = build_features(df, base_cfg).iloc[-1]
    need = required_history_rows(base_cfg)
    win = build_features(df.iloc[-need:].reset_index(drop=True), base_cfg).iloc[-1]
    for n in all_feature_names(base_cfg):
        if n == "elapsed_hours":
            continue                                                         # depends on window start; excluded from models
        assert win[n] == pytest.approx(full[n], nan_ok=True), n


def test_build_features_rejects_unusable_keys(base_cfg):
    df = make_plant_frame(5)
    df.loc[0, "timestamp"] = pd.NaT
    with pytest.raises(ValueError):
        build_features(df, base_cfg)


def test_long_window_features_are_trailing(base_cfg):
    df = make_plant_frame(60, soil=np.arange(60, dtype=float))
    f = build_features(df, base_cfg)
    assert f.loc[30, "soil_long24_mean"] == pytest.approx(np.mean(range(7, 31)))
    assert f.loc[30, "soil_long24_dev"] == pytest.approx(30 - np.mean(range(6, 30)))   # past-only, excludes the row
    assert "soil_long48_dev" not in all_feature_names(base_cfg)                      # = soil_base_dev (48-obs baseline)
