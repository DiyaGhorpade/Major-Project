"""Filters, pagination, CSV, SHAP reports, model performance, devices, sessions, settings, monitoring."""

import csv
import io
import json
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from app.api.deps import parse_time_param
from app.core import feature_dictionary as fd
from app.services import ai_service
from tests.conftest import REAL_ARTIFACTS, REAL_CSV, make_api_settings, open_app
from tests.fakes import FakeML
from tests.helpers import feature_names, write_shap_report

IST = ZoneInfo("Asia/Kolkata")
RUN_ID = json.loads((REAL_ARTIFACTS / "latest.json").read_text())["run_id"]
META = json.loads((REAL_ARTIFACTS / RUN_ID / "classifier.meta.json").read_text())


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    frame = pd.read_csv(REAL_CSV)
    frame["ts"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def total(response) -> int:
    return int(response.headers["x-total-count"])


# --- readings / data logs ---------------------------------------------------------------------------
def test_readings_default_order_limit_and_total(fresh_client, df):
    r = fresh_client.get("/api/sensors/readings")
    rows = r.json()
    assert total(r) == 5376 and len(rows) == 100
    stamps = [x["timestamp"] for x in rows]
    assert stamps == sorted(stamps, reverse=True)
    assert fresh_client.get("/api/sensors/readings?order=asc&limit=1").json()[0]["timestamp"] == "2026-09-21T00:00:00+05:30"
    assert fresh_client.get("/api/sensors/readings?limit=1000").json().__len__() == 1000


@pytest.mark.parametrize("query", ["limit=0", "limit=1001", "offset=-1", "order=sideways", "plantId=0", "plantId=17", "cluster=Cluster%205", "nodeId=3", "status=weird", "sessionId=0"])
def test_readings_bad_params_are_422(fresh_client, query):
    for path in ("/api/sensors/readings", "/api/data-logs"):
        r = fresh_client.get(f"{path}?{query}")
        assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"


def test_bad_time_is_422(fresh_client):
    r = fresh_client.get("/api/sensors/readings?from=notadate")
    assert r.status_code == 422 and "from" in r.json()["error"]["message"]


def test_filters(fresh_client, df):
    get = fresh_client.get
    assert total(get("/api/sensors/readings?plantId=3")) == 336
    assert total(get("/api/sensors/readings?cluster=Cluster 2")) == 4 * 336
    assert total(get("/api/sensors/readings?nodeId=2")) == 2688
    assert total(get("/api/sensors/readings?sessionId=2")) == 1344
    assert total(get("/api/sensors/readings?status=flagged")) == int(df[["soil", "temperature", "humidity", "light"]].isna().any(axis=1).sum())
    assert total(get("/api/sensors/readings?status=rejected")) == 0
    assert total(get("/api/sensors/readings?plantId=3&nodeId=2")) == 0  # plant 3 is on node 1
    assert total(get("/api/sensors/readings?plantId=3&sessionId=1&nodeId=1")) == 84


def test_time_range_naive_is_ist_and_bounds_are_inclusive(fresh_client, df):
    lo, hi = "2026-09-22T00:00:00", "2026-09-22T23:59:59"
    expected = df[(df.ts >= pd.Timestamp(lo, tz=IST)) & (df.ts <= pd.Timestamp(hi, tz=IST))]
    assert len(expected) > 0
    assert total(fresh_client.get("/api/sensors/readings", params={"from": lo, "to": hi})) == len(expected)
    # the same window written with an explicit offset, and with a space-decoded '+'
    assert total(fresh_client.get("/api/sensors/readings", params={"from": lo + "+05:30", "to": hi + "+05:30"})) == len(expected)
    assert total(fresh_client.get("/api/sensors/readings?from=2026-09-22T00:00:00 05:30&to=2026-09-22T23:59:59 05:30")) == len(expected)
    # UTC midnight is 05:30 IST: a different window
    utc_lo, utc_hi = datetime.fromisoformat("2026-09-22T00:00:00+00:00"), datetime.fromisoformat("2026-09-22T23:59:59+00:00")
    utc = fresh_client.get("/api/sensors/readings", params={"from": "2026-09-22T00:00:00Z", "to": "2026-09-22T23:59:59Z", "limit": 1000})
    in_utc = [x for x in utc.json() if utc_lo <= datetime.fromisoformat(x["timestamp"]) <= utc_hi]
    assert total(utc) == len(in_utc) == len(df[(df.ts >= pd.Timestamp(utc_lo)) & (df.ts <= pd.Timestamp(utc_hi))])
    assert {x["record_id"] for x in utc.json()} != {x["record_id"] for x in fresh_client.get("/api/sensors/readings", params={"from": lo, "to": hi, "limit": 1000}).json()}
    first = fresh_client.get("/api/sensors/readings", params={"from": "2026-10-18T22:00:00", "order": "asc"}).json()
    assert first and all(x["timestamp"] >= "2026-10-18T22:00:00+05:30" for x in first)
    assert parse_time_param("x", "2026-01-01T00:00:00").utcoffset().total_seconds() == 19800


def test_pagination_pages_are_disjoint_and_complete(fresh_client):
    ids = []
    for offset in range(0, 5376, 1000):
        ids += [x["record_id"] for x in fresh_client.get(f"/api/sensors/readings?limit=1000&offset={offset}&order=asc").json()]
    assert len(ids) == 5376 == len(set(ids))
    assert fresh_client.get("/api/sensors/readings?offset=5376").json() == []


def test_data_logs_search(fresh_client, df):
    get = fresh_client.get
    r = get("/api/data-logs?q=2026-09-21T00:00&limit=1000")
    assert r.status_code == 200 and total(r) == int(df.timestamp.str.contains("2026-09-21T00:00").sum())
    assert all("2026-09-21T00:00" in x["timestamp"] for x in r.json())
    assert total(get("/api/data-logs?q=zzz")) == 0
    one = get("/api/data-logs?limit=1&order=asc").json()[0]
    hit = get(f"/api/data-logs?q={one['record_id'].lower()}&limit=1000&order=asc")
    assert one["record_id"] in [x["record_id"] for x in hit.json()]
    # a % in the search text is literal, not a wildcard
    assert total(get("/api/data-logs?q=%25")) == 0
    # search composes with filters
    assert total(get("/api/data-logs?q=2026-09-21T00:00&plantId=1")) == 1


def test_csv_export_streams_filtered_rows(fresh_client, df):
    r = fresh_client.get("/api/data-logs?format=csv&plantId=3")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    disposition = r.headers["content-disposition"]
    assert disposition.startswith("attachment; filename=") and "plant_sensor_logs_" in disposition and disposition.rstrip('"').endswith(".csv")
    rows = list(csv.reader(io.StringIO(r.text)))
    assert rows[0] == [
        "record_id", "timestamp", "session_id", "node_id", "sampling_point", "plant_id", "soil", "temperature", "humidity", "light",
        "status", "validation_error", "ml_status", "ml_severity", "ml_confidence", "ml_is_anomaly", "ml_anomaly_score",
        "dataset_label_stress_level", "dataset_label_stress_type", "dataset_label_is_synthetic_anomaly",
    ]  # fmt: skip
    body = rows[1:]
    assert len(body) == 336 and all(row[5] == "3" for row in body)
    header = rows[0]
    first = dict(zip(header, body[0], strict=True))
    assert first["ml_status"] == "not_run" and first["ml_severity"] == "" and first["dataset_label_stress_level"] in {"healthy", "low", "moderate", "high"}
    # no pagination cap, descending by default, and `limit` is ignored
    everything = fresh_client.get("/api/data-logs?format=csv&limit=5")
    assert everything.text.count("\r\n") == 5377
    asc = list(csv.reader(io.StringIO(fresh_client.get("/api/data-logs?format=csv&plantId=3&order=asc").text)))
    assert asc[1][1] < asc[-1][1]
    flagged = fresh_client.get("/api/data-logs?format=csv&status=flagged").text.splitlines()
    assert len(flagged) - 1 == total(fresh_client.get("/api/data-logs?status=flagged&limit=1"))


def test_csv_has_ml_columns_after_a_run(ran_client):
    rows = list(csv.DictReader(io.StringIO(ran_client.get("/api/data-logs?format=csv&plantId=1").text)))
    assert {r["ml_status"] for r in rows} == {"ok"}
    assert {r["ml_severity"] for r in rows} <= {"healthy", "low", "moderate", "high"}
    assert all(r["ml_confidence"] for r in rows)
    assert {r["ml_is_anomaly"] for r in rows} <= {"0", "1"}


# --- predictions / anomalies / monitoring ---------------------------------------------------------
def test_predictions_filters_and_pagination(ran_client):
    get = ran_client.get
    assert total(get("/api/predictions")) == 5376
    page = get("/api/predictions?plantId=2&limit=5&offset=5").json()
    assert len(page) == 5 and {p["plant_id"] for p in page} == {2}
    first = page[0]
    assert set(first) >= {
        "id", "record_id", "plant_id", "timestamp", "severity", "probabilities", "confidence", "condition", "conditionSource",
        "envState", "anomaly", "warnings", "modelVersion", "runId", "status", "error",
    }  # fmt: skip
    assert first["conditionSource"] == "heuristic" and set(first["anomaly"]) == {"isAnomaly", "score"}
    assert first["envState"]["label"].startswith("Environmental state group ")
    sev = get("/api/predictions?severity=high&limit=1000")
    assert sev.json() and {p["severity"] for p in sev.json()} == {"high"}
    anomalies = get("/api/predictions?isAnomaly=true&limit=1000")
    assert total(anomalies) == total(get("/api/ai/anomalies")) and {p["anomaly"]["isAnomaly"] for p in anomalies.json()} == {True}
    assert total(get("/api/predictions?status=unavailable")) == 0
    assert total(get("/api/predictions?plantId=2&from=2026-10-18T00:00:00")) > 0
    assert total(get("/api/predictions?severity=bogus")) if False else get("/api/predictions?severity=bogus").status_code == 422


def test_anomalies_list(ran_client):
    r = ran_client.get("/api/ai/anomalies?limit=3")
    rows = r.json()
    assert total(r) > 3 and len(rows) == 3
    a = rows[0]
    assert set(a) == {"id", "plant_id", "plant_name", "record_id", "timestamp", "score", "threshold", "interpretation", "readings", "datasetLabelIsSyntheticAnomaly"}
    assert a["interpretation"] == "Unusual reading detected"
    assert set(a["readings"]) == {"soil", "temperature", "humidity", "light"}
    assert [x["timestamp"] for x in rows] == sorted((x["timestamp"] for x in rows), reverse=True)
    assert ran_client.get(f"/api/ai/anomalies?plantId={a['plant_id']}").json()
    assert ran_client.get("/api/ai/anomalies?limit=501").status_code == 422


def test_monitoring_plant_detail(ran_client):
    d = ran_client.get("/api/monitoring/plants/7?limit=10").json()
    assert d["plant"]["id"] == 7 and len(d["series"]) == 10
    stamps = [p["timestamp"] for p in d["series"]]
    assert stamps == sorted(stamps)
    assert set(d["series"][0]) == {"timestamp", "soil", "temperature", "humidity", "light"}
    assert len(d["predictionHistory"]) == 10
    h = d["predictionHistory"][0]
    assert set(h) == {"timestamp", "record_id", "status", "severity", "probabilities", "confidence", "condition", "isAnomaly", "anomalyScore", "envState"}
    assert [x["timestamp"] for x in d["predictionHistory"]] == stamps
    assert all(r["plant_id"] == 7 and r["status"] in ("new", "in_progress") for r in d["recommendations"])
    assert len(ran_client.get("/api/monitoring/plants/7").json()["series"]) == 84
    assert len(ran_client.get("/api/monitoring/plants/7?limit=336").json()["series"]) == 336
    assert ran_client.get("/api/monitoring/plants/7?limit=337").status_code == 422
    assert ran_client.get("/api/monitoring/plants/99").status_code == 404


def test_unknown_plant_is_404_envelope(fresh_client):
    for path in ("/api/plants/99", "/api/monitoring/plants/99", "/api/ai/shap/99"):
        r = fresh_client.get(path)
        assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"


def test_overview_details(ran_client):
    o = ran_client.get("/api/monitoring/overview").json()
    assert set(o) == {
        "totals", "plantStatusCounts", "severityCounts", "clusterSummaries", "recentAlerts", "topRecommendations",
        "latestSession", "devicesSummary", "pipeline", "modelVersion", "provenance", "disclaimer",
    }  # fmt: skip
    assert o["latestSession"]["session_id"] == 4
    assert o["devicesSummary"] == {"total": 3, "online": 0, "offline": 0, "unknown": 3, "lastTransmission": "2026-10-18T22:05:15+05:30"}
    assert o["totals"]["flaggedReadings"] > 0 and o["totals"]["anomalies"] > 0
    assert o["disclaimer"] == ""
    alerts = o["recentAlerts"]
    assert alerts == sorted(alerts, key=lambda a: a["timestamp"], reverse=True)
    anomaly_alerts = [a for a in alerts if a["kind"] == "anomaly"]
    assert not re.search(r"synthetic|verdict|diagnos", json.dumps(o["recentAlerts"] + o["topRecommendations"]), re.I)
    assert anomaly_alerts and all(a["message"].startswith("Unusual reading detected") and a["recommendationId"] is None for a in anomaly_alerts)
    assert all(a["plantId"] in range(1, 17) and a["timestamp"].endswith("+05:30") for a in alerts)
    for g in o["clusterSummaries"]:
        assert g["topCondition"] is None or isinstance(g["topCondition"], str)


# --- insights --------------------------------------------------------------------------------------
def test_insights_after_run(ran_client):
    i = ran_client.get("/api/ai/insights").json()
    assert i["hasPredictions"] and i["predictions"]["ok"] == 5376
    assert sum(i["severityCounts"].values()) == 5376 and sum(i["latestSeverityCounts"].values()) == 16
    assert sum(e["count"] for e in i["envStateCounts"]) == 5376
    assert all(e["label"] == f"Environmental state group {e['id']}" and e["note"] == "" for e in i["envStateCounts"])
    assert i["anomalies"]["count"] > 0 and 0 < i["anomalies"]["rate"] < 1
    assert i["explanationCoverage"] == {"plantsWithLocalShap": 16, "totalPlants": 16, "plantIds": list(range(1, 17))}
    assert i["topFeatures"]["localFrequency"][0]["count"] == 16  # the fake explains `soil` for each plant
    assert [f["name"] for f in i["topFeatures"]["global"][:2]] == feature_names()[:2]  # temp report from the fixture
    assert i["model"]["mlServiceOk"] is True and i["model"]["runId"] == "fake_run_1" and i["model"]["explainabilityAvailable"] is True
    assert i["lastPipelineRun"]["status"] == "completed" and i["provenance"] == "synthetic"
    assert not any("health" in e["label"].lower() for e in i["envStateCounts"])


# --- local SHAP ------------------------------------------------------------------------------------
def test_local_shap_after_run(ran_client):
    s = ran_client.get("/api/ai/shap/4").json()
    assert s["base_value"] is None and s["model_output"] is None
    assert s["target_class"] in {"healthy", "low", "moderate", "high"} and 0 < s["predicted_probability"] <= 1
    f = s["features"][0]
    assert f["name"] == "soil" and f["displayName"] == "Soil moisture" and f["unit"] == "%"
    assert f["direction"] == "increases_prediction" and isinstance(f["observed_value"], float)
    assert "base value not provided by the model service" in s["explanation_note"] and "log-odds" in s["explanation_note"]
    assert s["target_class"] in s["laymanSummary"] and "Soil moisture" in s["laymanSummary"]
    assert s["record_id"] == ran_client.get("/api/plants/4").json()["latestReading"]["record_id"]


def test_direction_mapping():
    assert ai_service.direction_of(0.3, "towards_predicted_class") == "increases_prediction"
    assert ai_service.direction_of(-0.3, "away_from_predicted_class") == "decreases_prediction"
    assert ai_service.direction_of(0.0, "towards_predicted_class") == "neutral"
    assert ai_service.direction_of(-0.2, None) == "decreases_prediction"


# --- feature dictionary ----------------------------------------------------------------------------
def test_feature_dictionary_resolves_every_classifier_feature():
    names = META["feature_names"]
    assert len(names) >= 50
    for name in names:
        info = fd.resolve(name)
        assert info.known, f"{name} fell back to the generic entry"
        assert info.display_name and info.description and info.display_name != name
    assert len({fd.resolve(n).display_name for n in names}) == len(names)  # no two features share a label
    old = ["soil_lag1", "light_diff1", "vpd_trend3", "humidity_roll12_std", "dayofweek", "light_soil_interaction", "soil_rate_per_hr"]
    assert all(fd.resolve(n).known for n in old)  # names of earlier models still resolve
    assert fd.resolve("soil_ref_dev").display_name == "Soil moisture vs. control plants"
    assert fd.resolve("temperature_ref_dev_roll24").unit == "°C" and fd.resolve("vpd_long24_dev").unit == "kPa"
    info = fd.resolve("soil_roll6_mean")
    assert (info.display_name, info.unit) == ("Soil moisture, 6-reading average", "%")
    assert fd.resolve("light_roll12_std").unit == "lux" and fd.resolve("temperature_base_dev").unit == "°C"
    assert fd.resolve("vpd_lag1").unit == "kPa"
    assert not fd.resolve("something_new").known


# --- global SHAP -----------------------------------------------------------------------------------
def test_global_shap_from_report(tmp_path):
    settings = make_api_settings(tmp_path)
    folder = write_shap_report(settings.ML_REPORTS_DIR, RUN_ID)
    names = feature_names()
    with open_app(settings, FakeML()) as c:
        r = c.get("/api/ai/shap/global")
        assert r.status_code == 200
        g = r.json()
        assert g["model_version"] == META["model_version"] and g["sample_count"] == 500
        assert len(g["features"]) == 15
        assert [f["name"] for f in g["features"]] == names[:15]
        assert [f["importance_rank"] for f in g["features"]] == list(range(1, 16))
        top = g["features"][0]
        assert top["displayName"] == fd.resolve(names[0]).display_name and top["mean_abs_shap"] > g["features"][1]["mean_abs_shap"]
        assert top["interpretation"] and top["laymanTip"]
        assert g["source"]["path"] == f"reports/{RUN_ID}/evaluation_20261025T085725Z/shap_global_importance.json" and g["source"]["generatedAt"].endswith("+05:30")
        assert g["approximate"]["beeswarm"] == "sample of 3 test rows, top-5 features each (not all rows or features)"
        assert set(g["perClassTop15"]) == {"healthy", "low", "moderate", "high"} and len(g["perClassTop15"]["low"]) == 15
        assert "log-odds" in g["units"]
        points = {p["id"]: p for p in g["beeswarm_points"]}
        assert len(points) == 7 and all(p["plant_id"] is None for p in points.values())
        a = [p for p in g["beeswarm_points"] if p["feature"] == names[0]]
        assert [p["feature_value_norm"] for p in a] == [0.0, 0.5, 1.0] and [p["shap_value"] for p in a] == [0.2, -0.1, 0.4]
        assert all(p["feature_value_norm"] is None for p in g["beeswarm_points"] if p["feature"] == names[1])  # constant -> null
        missing = [p for p in g["beeswarm_points"] if p["feature"] == names[2]]
        assert [p["feature_value_norm"] for p in missing] == [None, None]  # one value missing, one lone value (constant)


def test_global_shap_picks_newest_evaluation_then_flat_fallback(tmp_path):
    settings = make_api_settings(tmp_path)
    write_shap_report(settings.ML_REPORTS_DIR, RUN_ID, stamp="20261001T000000Z", scale=1.0)
    newest = write_shap_report(settings.ML_REPORTS_DIR, RUN_ID, stamp="20261030T000000Z", scale=2.0)
    assert ai_service.find_global_report(settings.ML_REPORTS_DIR, RUN_ID) == newest / "shap_global_importance.json"
    with open_app(settings, FakeML()) as c:
        assert c.get("/api/ai/shap/global").json()["features"][0]["mean_abs_shap"] == pytest.approx(2.0 * (1 - 0 / 63))
    flat = make_api_settings(tmp_path / "flat")
    write_shap_report(flat.ML_REPORTS_DIR, RUN_ID, nested=False)
    assert ai_service.find_global_report(flat.ML_REPORTS_DIR, RUN_ID).parent == flat.ML_REPORTS_DIR / RUN_ID
    # reports of another run are not used
    other = make_api_settings(tmp_path / "other")
    write_shap_report(other.ML_REPORTS_DIR, "some_other_run")
    assert ai_service.find_global_report(other.ML_REPORTS_DIR, RUN_ID) is None


def test_global_shap_missing_is_404_with_hint(fresh_client):
    r = fresh_client.get("/api/ai/shap/global")
    assert r.status_code == 404
    error = r.json()["error"]
    assert error["code"] == "global_shap_unavailable"
    assert error["details"]["hint"] == (
        "cd money_plant_ml_pipeline && .venv/bin/python scripts/generate_explanations.py "
        f"--input data/raw/plant_sensor_data.csv --run-id {RUN_ID}"
    )  # fmt: skip


def test_global_shap_unreadable_report_is_404(tmp_path):
    settings = make_api_settings(tmp_path)
    folder = write_shap_report(settings.ML_REPORTS_DIR, RUN_ID)
    (folder / "shap_global_importance.json").write_text("{not json")
    with open_app(settings, FakeML()) as c:
        assert c.get("/api/ai/shap/global").status_code == 404


# --- model performance -----------------------------------------------------------------------------
def test_model_performance_matches_classifier_meta(fresh_client):
    r = fresh_client.get("/api/models/performance")
    assert r.status_code == 200
    m, mm = r.json(), META["metrics"]
    assert m["model_name"] == META["model_name"] == "xgboost_stress_classifier" and m["model_version"] == META["model_version"]
    created = datetime.fromisoformat(META["created_at_utc"]).astimezone(ZoneInfo("Asia/Kolkata"))
    assert m["evaluated_at"] == created.isoformat(timespec="seconds")  # created_at_utc shown in IST
    assert m["dataset_name"] == "plant_sensor_data.csv" and m["total_samples"] == mm["n_samples"] == 1344
    assert m["split"] == f"chronological hold-out, last 25% (cutoff {META['split']['cutoff_timestamp']})"
    classes = ["healthy", "low", "moderate", "high"]
    assert m["overall"]["accuracy"] == mm["accuracy"] and round(m["overall"]["accuracy"], 4) == 0.8028  # shipped run (issue #5)
    assert m["overall"]["f1_score"] == mm["macro_f1"]
    assert m["overall"]["precision"] == pytest.approx(sum(mm["per_class"][c]["precision"] for c in classes) / 4)
    assert m["overall"]["recall"] == pytest.approx(sum(mm["per_class"][c]["recall"] for c in classes) / 4)
    assert [p["class_name"] for p in m["per_class"]] == classes
    for p in m["per_class"]:
        src = mm["per_class"][p["class_name"]]
        assert (p["precision"], p["recall"], p["f1_score"], p["support"]) == (src["precision"], src["recall"], src["f1"], src["support"])
        assert p["laymanStatus"]
    assert m["per_class"][0]["laymanStatus"] != m["per_class"][1]["laymanStatus"]  # F1 .82 vs .50
    assert m["confusion_matrix"] == {"labels": classes, "matrix": mm["confusion_matrix"]["matrix"]}
    assert m["provenance"] == "synthetic" and m["validatedOnRealPlants"] is False and m["note"] == ""
    assert m["balancedAccuracy"] == mm["balanced_accuracy"] and m["weightedF1"] == mm["weighted_f1"]
    assert m["baselines"]["majority_class"] == {"accuracy": META["baseline_metrics"]["majority_class"]["accuracy"], "macro_f1": META["baseline_metrics"]["majority_class"]["macro_f1"]}
    assert set(m["baselines"]) == {"majority_class", "logistic_regression"}
    assert m["anomalyModel"]["metricsVsHelperLabel"]["precision"] == pytest.approx(0.2857142857142857)
    assert m["anomalyModel"]["threshold"]["decision_rule"] == "anomaly_score > 0"
    assert m["clusteringModel"]["selectedK"] == 3 and m["clusteringModel"]["silhouette"] == pytest.approx(0.26267610513984396)
    assert m["anomalyModel"]["note"] is None and "note" not in m["anomalyModel"]["metricsVsHelperLabel"] and m["clusteringModel"]["note"] is None


def test_model_performance_is_cached_in_model_info_and_refreshes_on_new_run(tmp_path):
    from app.db.models import ModelInfo
    from sqlalchemy import select

    artifacts = tmp_path / "artifacts"
    for run in ("run_a", "run_b"):
        (artifacts / run).mkdir(parents=True)
        meta = json.loads(json.dumps(META))
        meta["model_version"] = run
        meta["metrics"]["accuracy"] = 0.5 if run == "run_a" else 0.9
        (artifacts / run / "classifier.meta.json").write_text(json.dumps(meta))
    (artifacts / "latest.json").write_text(json.dumps({"run_id": "run_a"}))
    settings = make_api_settings(tmp_path).model_copy(update={"ML_ARTIFACTS_DIR": artifacts})
    with open_app(settings, FakeML()) as c:
        assert c.get("/api/models/performance").json()["overall"]["accuracy"] == 0.5
        (artifacts / "run_a" / "classifier.meta.json").unlink()  # served from model_info now
        assert c.get("/api/models/performance").json()["overall"]["accuracy"] == 0.5
        with c.sf() as db:
            row = db.scalars(select(ModelInfo).where(ModelInfo.run_id == "run_a")).one()
            assert row.metrics["overall"]["accuracy"] == 0.5 and row.labels == ["healthy", "low", "moderate", "high"]
        (artifacts / "latest.json").write_text(json.dumps({"run_id": "run_b"}))  # retrained
        again = c.get("/api/models/performance").json()
        assert again["overall"]["accuracy"] == 0.9 and again["model_version"] == "run_b" and again["runId"] == "run_b"


def test_model_performance_unreadable_artifacts_is_503(tmp_path):
    settings = make_api_settings(tmp_path).model_copy(update={"ML_ARTIFACTS_DIR": tmp_path / "nothing"})
    with open_app(settings, FakeML()) as c:
        r = c.get("/api/models/performance")
        assert r.status_code == 503 and r.json()["error"]["code"] == "model_artifacts_unavailable"
    broken = tmp_path / "broken"
    (broken / "r1").mkdir(parents=True)
    (broken / "latest.json").write_text(json.dumps({"run_id": "r1"}))
    (broken / "r1" / "classifier.meta.json").write_text(json.dumps({"model_name": "x"}))
    with open_app(make_api_settings(tmp_path / "b").model_copy(update={"ML_ARTIFACTS_DIR": broken}), FakeML()) as c:
        assert c.get("/api/models/performance").status_code == 503


# --- devices / sessions ----------------------------------------------------------------------------
def test_devices_report_only_what_the_data_knows(fresh_client):
    devices = {d["id"]: d for d in fresh_client.get("/api/devices").json()}
    assert set(devices) == {"jetson-01", "esp32-node-1", "esp32-node-2"}
    for d in devices.values():
        assert d["status"] == "unknown" and d["ip_address"] is None and d["mac_address"] is None and d["last_heartbeat"] is None
        hw = d["hardware_details"]
        for key in ("cpu_usage", "gpu_usage", "ram_used_gb", "ram_total_gb", "disk_used_gb", "disk_total_gb", "temperature_c", "rssi_dbm", "uptime", "firmware_version"):
            assert hw[key] is None, key
        assert "demo" not in json.dumps(d).lower()
    n1, n2, jetson = devices["esp32-node-1"], devices["esp32-node-2"], devices["jetson-01"]
    assert n1["nodeNumber"] == "1" and n1["plantIds"] == [1, 3, 5, 7, 9, 11, 13, 15] and n2["plantIds"] == [2, 4, 6, 8, 10, 12, 14, 16]
    assert n1["readingCount"] == n2["readingCount"] == 2688 and jetson["readingCount"] == 5376
    assert n1["hardware_details"]["channels"] == 12 == n2["hardware_details"]["channels"]
    assert n1["last_transmission"].endswith("+05:30") and jetson["nodeNumber"] is None and jetson["hardware_details"]["channels"] is None
    newest = fresh_client.get("/api/sensors/readings?nodeId=1&limit=1").json()[0]["timestamp"]
    assert n1["last_transmission"] == newest


def test_sessions(fresh_client):
    sessions = fresh_client.get("/api/sessions").json()
    assert [s["session_id"] for s in sessions] == [4, 3, 2, 1]
    s = sessions[3]
    assert s["status"] == "completed" and s["trigger_type"] == "scheduled" and s["expected_records"] == 1344 == s["received_records"]
    assert s["rejected_records"] == 0 and s["flagged_records"] >= 0 and s["started_at"] == "2026-09-21T00:00:00+05:30"
    assert s["node_progress"] == {"1": {"expected": 672, "received": 672, "status": "completed"}, "2": {"expected": 672, "received": 672, "status": "completed"}}
    assert s["duration_seconds"] > 6 * 86400
    assert sum(x["flagged_records"] for x in sessions) == total(fresh_client.get("/api/sensors/readings?status=flagged&limit=1"))


# --- settings --------------------------------------------------------------------------------------
def test_settings_defaults_and_persistence(fresh_client):
    s = fresh_client.get("/api/settings").json()
    assert set(s) == {"polling_interval_seconds", "jetson_receiver_port", "ph_sensor_installed", "collection_times", "log_retention_days", "storage_path", "soil_adc_dry", "soil_adc_wet"}
    assert len(s["collection_times"]) == 12 and s["collection_times"][0] == "00:00"
    r = fresh_client.put("/api/settings", json={"polling_interval_seconds": 30, "ph_sensor_installed": True})
    assert r.status_code == 200 and r.json()["polling_interval_seconds"] == 30 and r.json()["ph_sensor_installed"] is True
    assert r.json()["jetson_receiver_port"] == s["jetson_receiver_port"]  # partial update leaves the rest alone
    assert fresh_client.get("/api/settings").json() == r.json()
    full = {**r.json(), "collection_times": ["08:00", "13:30"], "log_retention_days": 90, "jetson_receiver_port": 8080, "soil_adc_dry": 3000, "soil_adc_wet": 1200, "storage_path": "/data/x.db"}
    assert fresh_client.put("/api/settings", json=full).json() == full
    legacy = fresh_client.put("/api/settings", json={"api_base_url": "http://x", "demo_mode": True, "polling_interval_seconds": 5})
    assert legacy.status_code == 200 and "api_base_url" not in legacy.json() and "demo_mode" not in legacy.json()
    assert fresh_client.put("/api/settings", json={}).json()["polling_interval_seconds"] == 5


@pytest.mark.parametrize(
    "body",
    [
        {"polling_interval_seconds": 0}, {"polling_interval_seconds": 3601}, {"polling_interval_seconds": "5"}, {"polling_interval_seconds": 1.5},
        {"jetson_receiver_port": 0}, {"jetson_receiver_port": 65536},
        {"soil_adc_dry": -1}, {"soil_adc_wet": -5}, {"soil_adc_dry": "x"},
        {"log_retention_days": 0}, {"log_retention_days": 3651},
        {"collection_times": ["25:00"]}, {"collection_times": ["8:00"]}, {"collection_times": "08:00"}, {"collection_times": [800]},
        {"ph_sensor_installed": "yes"}, {"storage_path": ""},
    ],
)
def test_settings_validation(fresh_client, body):
    before = fresh_client.get("/api/settings").json()
    r = fresh_client.put("/api/settings", json=body)
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"
    assert fresh_client.get("/api/settings").json() == before


def test_settings_boundaries_are_accepted(fresh_client):
    ok = {"polling_interval_seconds": 1, "jetson_receiver_port": 65535, "soil_adc_dry": 0, "log_retention_days": 3650, "collection_times": ["00:00", "23:59"]}
    r = fresh_client.put("/api/settings", json=ok)
    assert r.status_code == 200 and all(r.json()[k] == v for k, v in ok.items())
    assert fresh_client.put("/api/settings", json={"polling_interval_seconds": 3600, "jetson_receiver_port": 1, "log_retention_days": 1}).status_code == 200


# --- plants ---------------------------------------------------------------------------------------
def test_plant_fields_after_run(ran_client):
    p = ran_client.get("/api/plants/6").json()
    assert (p["cluster"], p["clusterLabel"], p["stressType"], p["nodeId"]) == ("Cluster 2", "Cluster 2: Water Stress", "Water Stress", "2")
    assert p["samplingPoint"] == p["latestReading"]["sampling_point"]
    c = p["conditionPrediction"]
    assert c["conditionSource"] == "heuristic" and c["envState"]["label"].startswith("Environmental state group")
    assert c["severity"] in c["simpleExplanation"] and f"{c['probabilities'][c['severity']]:.2f}" in c["simpleExplanation"]
    assert "heuristic" not in c["simpleExplanation"] and "diagnosis" not in c["simpleExplanation"]
    assert p["status"] == {"healthy": "healthy", "low": "attention", "moderate": "attention", "high": "critical"}[c["severity"]]
    labels = p["latestReading"]["datasetLabels"]
    assert set(labels) == {"stressLevel", "stressType", "isSyntheticAnomaly"}
    assert "stressLevel" not in c and "datasetLabels" not in c  # dataset labels never mixed into ML fields
