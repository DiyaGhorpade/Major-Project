import numpy as np
import pandas as pd
import pytest

from plant_ml.recommendations import DISCLAIMER, generate_recommendations

PROBS_OK = {"healthy": 0.05, "low": 0.1, "moderate": 0.25, "high": 0.6}


def recent(n=6, **over):
    base = {"soil": 50.0, "temperature": 26.0, "humidity": 65.0, "light": 3000.0, "vpd": 1.1, "hour": 11.0,
            "soil_rate_per_hr": 0.0, "soil_roll6_mean": 50.0}
    base.update(over)
    return pd.DataFrame({k: (list(v) if hasattr(v, "__iter__") else [v] * n) for k, v in base.items()})


def ids(recs):
    return [r["id"] for r in recs]


def test_anomaly_only_recommends_verifying_the_sensor(base_cfg):
    recs = generate_recommendations("high", PROBS_OK, True, [], recent(soil=[10] * 6), base_cfg)
    assert ids(recs) == ["verify_sensor"]                          # no stress inference from a suspect sample
    assert "sensor" in recs[0]["text"].lower() and recs[0]["reason"] and recs[0]["priority"] and recs[0]["supporting_features"]


def test_missing_inputs_request_another_reading(base_cfg):
    recs = generate_recommendations("high", PROBS_OK, False, ["soil"], recent(), base_cfg)
    assert ids(recs) == ["retake_reading"] and recs[0]["priority"] == "low"


def test_low_confidence_requests_another_reading_not_strong_advice(base_cfg):
    recs = generate_recommendations("moderate", {"healthy": 0.3, "low": 0.3, "moderate": 0.3, "high": 0.1}, False, [], recent(), base_cfg)
    assert ids(recs) == ["low_confidence"]


def test_persistent_low_soil_suggests_watering_check(base_cfg):
    recs = generate_recommendations("high", PROBS_OK, False, [], recent(soil=[20.0] * 6), base_cfg)
    assert "check_watering" in ids(recs)
    r = [x for x in recs if x["id"] == "check_watering"][0]
    assert r["priority"] == "high" and "soil_last" in r["supporting_features"] and r["type"] == "heuristic_suggestion"


def test_single_low_soil_sample_is_not_enough(base_cfg):
    soil = [50, 50, 50, 50, 50, 10.0]
    recs = generate_recommendations("moderate", PROBS_OK, False, [], recent(soil=soil), base_cfg)
    assert "check_watering" not in ids(recs)


def test_fast_soil_drop_suggestion(base_cfg):
    recs = generate_recommendations("moderate", PROBS_OK, False, [], recent(soil_rate_per_hr=[-2.5] * 6), base_cfg)
    assert "check_soil_drying" in ids(recs)


def test_low_daytime_light_and_night_is_ignored(base_cfg):
    low = generate_recommendations("moderate", PROBS_OK, False, [], recent(light=[50.0] * 6), base_cfg)
    assert "check_light_low" in ids(low)
    night = generate_recommendations("moderate", PROBS_OK, False, [], recent(light=[0.0] * 6, hour=[1.0] * 6), base_cfg)
    assert "check_light_low" not in ids(night)               # darkness at night is normal


def test_climate_suggestion_needs_persistence(base_cfg):
    hot = generate_recommendations("moderate", PROBS_OK, False, [], recent(temperature=[35.0] * 6), base_cfg)
    assert "check_climate" in ids(hot)
    blip = generate_recommendations("moderate", PROBS_OK, False, [], recent(temperature=[26, 26, 26, 26, 26, 36.0]), base_cfg)
    assert "check_climate" not in ids(blip)


def test_stressed_without_driver_gets_generic_monitor_advice(base_cfg):
    recs = generate_recommendations("high", PROBS_OK, False, [], recent(), base_cfg)
    assert ids(recs) == ["monitor_and_inspect"]


def test_healthy_normal_conditions_have_no_recommendations(base_cfg):
    assert generate_recommendations("healthy", {"healthy": 0.9, "low": 0.05, "moderate": 0.03, "high": 0.02}, False, [], recent(), base_cfg) == []


def test_no_watering_volumes_are_ever_recommended(base_cfg):
    import re
    for soil in ([20.0] * 6,):
        for r in generate_recommendations("high", PROBS_OK, False, [], recent(soil=soil), base_cfg):
            assert not re.search(r"\b\d+\s?(ml|l|litres?|liters?|cups?)\b", r["text"], re.I)


def test_disclaimer_is_labelled_heuristic():
    assert "Not an expert diagnosis" in DISCLAIMER
