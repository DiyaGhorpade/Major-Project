import pandas as pd
import pytest
from fastapi.testclient import TestClient

from conftest import rows_to_readings
from plant_ml.api.main import create_app
from plant_ml.data_loader import normalise_frame


@pytest.fixture(scope="module")
def client(cfg_small, predictor):
    with TestClient(create_app(cfg_small, predictor)) as c:
        yield c


def plant_rows(small_df, plant=3, n=70):
    sub = normalise_frame(small_df[small_df["plant_id"] == plant].iloc[:n])
    sub["timestamp"] = sub["timestamp"].map(lambda t: t.isoformat())
    return rows_to_readings(sub)


def body(small_df, plant=3, n=70):
    rows = plant_rows(small_df, plant, n)
    return {"reading": rows[-1], "history": rows[:-1], "explain": False}


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok" and j["models_loaded"] is True and j["trained_on"] == "synthetic"


def test_metadata_exposes_provenance_and_feature_schema(client):
    j = client.get("/metadata").json()
    assert j["trained_on"] == "synthetic" and j["validated_on_real_plants"] is False
    assert j["models"]["classifier"]["feature_names"] and j["required_history_rows_for_full_features"] >= 49
    assert "not validated" in j["disclaimer"].lower()


def test_predict_without_target_columns(client, small_df):
    b = body(small_df)
    assert "stress_level" not in b["reading"]
    r = client.post("/predict", json=b)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["plant_id"] == 3 and set(j["stress_prediction"]["probabilities"]) == {"healthy", "low", "moderate", "high"}
    assert sum(j["stress_prediction"]["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
    for key in ("cluster", "anomaly", "explanation", "recommendations", "model_metadata", "history"):
        assert key in j
    assert j["model_metadata"]["trained_on"] == "synthetic"


def test_predict_single_reading_without_history_is_cold_start(client, small_df):
    b = body(small_df)
    b["history"] = []
    j = client.post("/predict", json=b).json()
    assert j["history"]["sufficient"] is False and j["warnings"]


def test_predict_with_explanations(client, small_df):
    b = body(small_df)
    b["explain"] = True
    j = client.post("/predict", json=b).json()
    assert "available" in j["explanation"]


def test_predict_rejects_out_of_range_sensor(client, small_df):
    b = body(small_df)
    b["reading"]["temperature"] = 80
    r = client.post("/predict", json=b)
    assert r.status_code == 422 and r.json()["detail"]["error"] == "invalid_readings"
    assert "out_of_range_temperature" in r.json()["detail"]["issues"][0]["issues"]


def test_predict_rejects_history_from_another_plant(client, small_df):
    b = body(small_df, plant=3)
    other = plant_rows(small_df, plant=9, n=3)
    b["history"] = b["history"][:5] + other
    r = client.post("/predict", json=b)
    assert r.status_code == 422 and r.json()["detail"]["error"] == "mixed_plants"


def test_predict_rejects_future_history(client, small_df):
    rows = plant_rows(small_df, 3, 10)
    r = client.post("/predict", json={"reading": rows[0], "history": rows[1:]})
    assert r.status_code == 422 and r.json()["detail"]["error"] == "history_not_before_reading"


def test_predict_type_errors_give_422(client):
    r = client.post("/predict", json={"reading": {"timestamp": "not-a-date", "plant_id": "x"}})
    assert r.status_code == 422


def test_missing_sensor_value_is_allowed_and_flagged(client, small_df):
    b = body(small_df)
    b["reading"]["soil"] = None
    j = client.post("/predict", json=b).json()
    assert any("soil" in w for w in j["warnings"])


def test_validate_reports_without_predicting(client, small_df):
    rows = plant_rows(small_df, 3, 5)
    rows[2]["humidity"] = 3.0
    j = client.post("/validate", json={"readings": rows}).json()
    assert j["valid"] is False and j["issues"][0]["index"] == 2
    ok = client.post("/validate", json={"readings": plant_rows(small_df, 3, 5)}).json()
    assert ok["valid"] is True and ok["n_readings"] == 5
    assert client.post("/validate", json={"readings": []}).status_code == 422


def test_batch_predict_with_independent_plants_and_item_errors(client, small_df):
    good_a, good_b = body(small_df, 3), body(small_df, 11)
    bad = body(small_df, 5)
    bad["reading"]["soil"] = 500
    r = client.post("/predict/batch", json={"items": [good_a, bad, good_b]})
    assert r.status_code == 200
    res = r.json()["results"]
    assert [x["ok"] for x in res] == [True, False, True]
    assert res[0]["result"]["plant_id"] == 3 and res[2]["result"]["plant_id"] == 11
    assert res[1]["error"]["status"] == 422
    assert client.post("/predict/batch", json={"items": []}).status_code == 422


def test_service_reports_unavailable_when_artifacts_missing(base_cfg, tmp_path):
    cfg = base_cfg.copy()
    cfg.set("paths.artifacts_dir", str(tmp_path / "nothing"))
    with TestClient(create_app(cfg)) as c:                       # must NOT train on startup
        h = c.get("/health").json()
        assert h["status"] == "unavailable" and h["models_loaded"] is False and "train" in h["artifact_error"].lower()
        assert c.get("/metadata").status_code == 503
        r = c.post("/predict", json={"reading": {"timestamp": "2026-10-21T10:30:00+05:30", "plant_id": 1,
                                                  "soil": 40, "temperature": 25, "humidity": 60, "light": 1000}})
        assert r.status_code == 503 and r.json()["detail"]["error"] == "models_unavailable"
        assert not (tmp_path / "nothing").exists()               # startup never trains or creates artifacts


def ref_rows(small_df, upto_n=70, plants=(1, 2, 3, 4)):
    out = []
    for p in plants:
        out += plant_rows(small_df, p, upto_n)
    return out


def test_metadata_lists_reference_plants_and_smoothing(client):
    j = client.get("/metadata").json()
    assert j["cross_plant"]["enabled"] is True and j["cross_plant"]["reference_plants"] == [1, 2, 3, 4]
    assert j["temporal_smoothing"]["method"] == "hmm_forward_filter" and j["temporal_smoothing"]["window"] >= 1


def test_predict_with_reference_readings(client, small_df):
    b = body(small_df, plant=11, n=70)       # all four controls have a reading within the as-of tolerance here
    without = client.post("/predict", json=b).json()
    assert any("reference-plant" in w for w in without["warnings"])
    b["reference"] = ref_rows(small_df)
    j = client.post("/predict", json=b)
    assert j.status_code == 200, j.text
    j = j.json()
    assert not any("reference-plant" in w for w in j["warnings"])
    assert set(j["stress_prediction"]["unsmoothed_probabilities"]) == {"healthy", "low", "moderate", "high"}


def test_batch_reference_is_shared_and_equals_per_item_reference(client, small_df):
    a, b = body(small_df, 9), body(small_df, 13)
    shared = client.post("/predict/batch", json={"items": [a, b], "reference": ref_rows(small_df)}).json()["results"]
    per_item = client.post("/predict", json={**a, "reference": ref_rows(small_df)}).json()
    assert all(x["ok"] for x in shared)
    assert shared[0]["result"]["stress_prediction"]["probabilities"] == pytest.approx(per_item["stress_prediction"]["probabilities"])


def test_reference_size_limit(client, small_df, cfg_small):
    b = body(small_df, plant=9, n=5)
    limit = int(cfg_small.get("api.max_reference_rows"))
    b["reference"] = (ref_rows(small_df, 70) * (limit // 280 + 1))[: limit + 1]
    r = client.post("/predict", json=b)
    assert r.status_code == 422 and r.json()["detail"]["error"] == "reference_too_long"


def test_batch_total_reference_size_limit(client, small_df, cfg_small):
    total_limit = int(cfg_small.get("api.max_batch_reference_rows"))
    per_request = int(cfg_small.get("api.max_reference_rows"))
    ref = ref_rows(small_df, 70)

    def rows(n):
        return (ref * (n // len(ref) + 1))[:n]

    # the shared reference and the first item are each within the per-request cap; the second item pushes the sum over
    shared = rows(per_request)
    a, b = body(small_df, plant=9, n=5), body(small_df, plant=13, n=5)
    a["reference"] = rows(total_limit - per_request)
    b["reference"] = rows(1)
    r = client.post("/predict/batch", json={"items": [a, b], "reference": shared})
    assert r.status_code == 422 and r.json()["detail"]["error"] == "batch_reference_too_long"
    b["reference"] = []                                        # exactly at the cap is accepted
    assert client.post("/predict/batch", json={"items": [a, b], "reference": shared}).status_code == 200
