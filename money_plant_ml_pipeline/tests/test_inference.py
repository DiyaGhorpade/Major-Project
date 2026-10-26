import numpy as np
import pandas as pd
import pytest

from conftest import make_plant_frame
from plant_ml.data_loader import normalise_frame


def history(small_df, plant=3, n=80, start=0):
    sub = small_df[small_df["plant_id"] == plant].iloc[start:start + n]
    return normalise_frame(sub[["timestamp", "plant_id", "soil", "temperature", "humidity", "light"]])   # NO label columns


def reference(small_df, plants=(1, 2, 3, 4)):
    sub = small_df[small_df["plant_id"].isin(plants)]
    return normalise_frame(sub[["timestamp", "plant_id", "soil", "temperature", "humidity", "light"]])


def test_inference_works_without_any_label_columns(predictor, small_df):
    df = history(small_df)
    assert not {"stress_level", "stress_type", "experimental_group", "is_synthetic_anomaly"} & set(df.columns)
    r = predictor.predict_frame(df, rows="last")[0]
    probs = r["stress_prediction"]["probabilities"]
    assert set(probs) == {"healthy", "low", "moderate", "high"}
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-5)
    assert r["stress_prediction"]["label"] == max(probs, key=probs.get)
    assert r["plant_id"] == 3 and r["timestamp"].endswith("+05:30")
    assert set(r["cluster"]) == {"id", "distance"} and r["cluster"]["distance"] >= 0
    assert set(r["anomaly"]) >= {"is_anomaly", "score", "threshold"}
    assert r["model_metadata"]["trained_on"] == "synthetic" and r["model_metadata"]["validated_on_real_plants"] is False


def test_labels_in_input_are_ignored_by_predictions(predictor, small_df):
    plain = history(small_df)
    with_labels = normalise_frame(small_df[small_df["plant_id"] == 3].iloc[:80])
    a = predictor.predict_frame(plain, explain=False)[0]["stress_prediction"]["probabilities"]
    b = predictor.predict_frame(with_labels, explain=False)[0]["stress_prediction"]["probabilities"]
    assert a == pytest.approx(b)


def test_rows_all_returns_one_result_per_row(predictor, small_df):
    df = history(small_df, n=40)
    out = predictor.predict_frame(df, rows="all", explain=False)
    assert len(out) == 40
    assert out[0]["history"]["rows_available"] == 1 and out[-1]["history"]["rows_available"] == 40


def test_cold_start_is_flagged(predictor):
    r = predictor.predict_frame(make_plant_frame(1), explain=False)[0]
    assert r["history"]["sufficient"] is False
    assert any("cold start" in w for w in r["warnings"])


def test_sufficient_history_has_no_cold_start_warning(predictor, small_df):
    r = predictor.predict_frame(history(small_df, plant=11, n=70), explain=False, reference=reference(small_df))[0]
    assert r["history"]["sufficient"] is True and not r["warnings"]


def test_missing_reference_readings_are_flagged(predictor, small_df):
    r = predictor.predict_frame(history(small_df, plant=9, n=120), explain=False)[0]
    assert any("reference-plant" in w for w in r["warnings"])


def test_partial_reference_is_flagged_but_control_plants_are_not(predictor, small_df):
    full = reference(small_df)
    # treated plant: all four controls expected
    r = predictor.predict_frame(history(small_df, plant=11, n=70), explain=False, reference=full)[0]
    assert r["history"]["reference_plants_used"] == r["history"]["reference_plants_expected"] == 4
    assert not any("reference-plant" in w for w in r["warnings"])
    # losing one control is routine (staggered readings, as in training): recorded, not warned
    part = reference(small_df, plants=(1, 2, 3))
    r = predictor.predict_frame(history(small_df, plant=11, n=70), explain=False, reference=part)[0]
    assert r["history"]["reference_plants_used"] == 3 and r["history"]["reference_plants_expected"] == 4
    assert not any("reference-plant" in w for w in r["warnings"])
    # fewer than half of the expected controls is a real shortfall
    r = predictor.predict_frame(history(small_df, plant=11, n=70), explain=False, reference=reference(small_df, plants=(1,)))[0]
    assert r["history"]["reference_plants_used"] == 1
    assert any("Only 1 of 4 reference-plant" in w for w in r["warnings"])
    # a control plant never sees itself: 3 of 3 is complete, 2 of 3 is fine, 1 of 3 is a shortfall
    r = predictor.predict_frame(history(small_df, plant=1, n=120), explain=False, reference=full)[0]
    assert r["history"]["reference_plants_used"] == r["history"]["reference_plants_expected"] == 3
    assert not any("reference-plant" in w for w in r["warnings"])
    r = predictor.predict_frame(history(small_df, plant=1, n=120), explain=False,
                                reference=reference(small_df, plants=(2, 3)))[0]
    assert r["history"]["reference_plants_used"] == 2
    assert not any("reference-plant" in w for w in r["warnings"])
    r = predictor.predict_frame(history(small_df, plant=1, n=120), explain=False,
                                reference=reference(small_df, plants=(2,)))[0]
    assert r["history"]["reference_plants_used"] == 1
    assert any("Only 1 of 3 reference-plant" in w for w in r["warnings"])


def test_required_history_and_reference_reproduce_full_history_prediction(predictor, small_df):
    """Stateless serving: the required trailing rows + reference readings give the same (smoothed) output as the
    plant's whole history, i.e. what the smoother saw in training-style batch scoring."""
    need = predictor.required_history
    assert need == predictor.bundle.meta["classifier"]["temporal_smoothing"]["window"] - 1 + 49
    full = history(small_df, plant=9, n=200)
    ref = reference(small_df)
    a = predictor.predict_frame(full, explain=False, reference=ref)[0]["stress_prediction"]
    b = predictor.predict_frame(full.iloc[-need:].reset_index(drop=True), explain=False, reference=ref)[0]["stress_prediction"]
    assert a["probabilities"] == pytest.approx(b["probabilities"], abs=1e-6) and a["label"] == b["label"]


def test_rows_last_matches_rows_all_and_smoothing_is_reported(predictor, small_df):
    df = history(small_df, plant=9, n=80)
    ref = reference(small_df)
    last = predictor.predict_frame(df, explain=False, reference=ref)[0]["stress_prediction"]
    every = predictor.predict_frame(df, rows="all", explain=False, reference=ref)[-1]["stress_prediction"]
    assert last["probabilities"] == pytest.approx(every["probabilities"])
    assert set(last["unsmoothed_probabilities"]) == set(last["probabilities"])
    assert sum(last["unsmoothed_probabilities"].values()) == pytest.approx(1.0, abs=1e-5)


def test_future_reference_readings_do_not_change_the_prediction(predictor, small_df):
    df = history(small_df, plant=9, n=80)
    ref = reference(small_df)
    later = ref[ref["timestamp"] > df["timestamp"].max()].copy()
    later[["soil", "temperature", "humidity", "light"]] = later[["soil", "temperature", "humidity", "light"]] * 0 + 1
    upto = ref[ref["timestamp"] <= df["timestamp"].max()]
    a = predictor.predict_frame(df, explain=False, reference=upto)[0]["stress_prediction"]["probabilities"]
    b = predictor.predict_frame(df, explain=False, reference=pd.concat([upto, later]))[0]["stress_prediction"]["probabilities"]
    assert a == pytest.approx(b)


def test_plants_do_not_contaminate_each_other(predictor, small_df):
    a = history(small_df, plant=3, n=90)
    b = history(small_df, plant=9, n=90)
    solo = predictor.predict_frame(a, explain=False)[0]
    both = {r["plant_id"]: r for r in predictor.predict_frame(pd.concat([a, b], ignore_index=True), explain=False)}
    assert both[3]["stress_prediction"]["probabilities"] == pytest.approx(solo["stress_prediction"]["probabilities"])
    assert both[3]["anomaly"]["score"] == pytest.approx(solo["anomaly"]["score"])
    assert set(both) == {3, 9}


def test_missing_sensor_value_still_predicts_and_asks_for_retake(predictor, small_df):
    df = history(small_df, n=90)
    df.loc[df.index[-1], "soil"] = np.nan
    r = predictor.predict_frame(df, explain=False)[0]
    assert sum(r["stress_prediction"]["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
    assert any("soil" in w for w in r["warnings"])
    assert r["recommendations"][0]["id"] in ("retake_reading", "verify_sensor")


def test_explanations_available_or_honestly_unavailable(predictor, small_df):
    r = predictor.predict_frame(history(small_df, n=90), explain=True)[0]["explanation"]
    if r["available"]:
        assert r["top_features"] and {"feature", "contribution", "direction"} <= set(r["top_features"][0])
        assert "causation" in r["note"]
    else:
        assert r["reason"] and r["top_features"] == []


def test_invalid_rows_arg_rejected(predictor, small_df):
    with pytest.raises(ValueError):
        predictor.predict_frame(history(small_df, n=10), rows="middle")
