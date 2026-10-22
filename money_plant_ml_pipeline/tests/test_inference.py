import numpy as np
import pandas as pd
import pytest

from conftest import make_plant_frame
from plant_ml.data_loader import normalise_frame


def history(small_df, plant=3, n=80, start=0):
    sub = small_df[small_df["plant_id"] == plant].iloc[start:start + n]
    return normalise_frame(sub[["timestamp", "plant_id", "soil", "temperature", "humidity", "light"]])   # NO label columns


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
    r = predictor.predict_frame(history(small_df, n=120), explain=False)[0]
    assert r["history"]["sufficient"] is True and not r["warnings"]


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
