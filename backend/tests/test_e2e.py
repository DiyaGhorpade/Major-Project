"""Fresh DB -> seed -> (empty AI state) -> Run with the fake ML service -> everything is populated."""

from tests.conftest import run_pipeline


def test_before_a_run_the_data_pages_have_rows_and_the_ai_pages_are_empty(fresh_client):
    c = fresh_client
    plants = c.get("/api/plants").json()
    assert len(plants) == 16
    assert all(p["latestReading"] and p["conditionPrediction"] is None and p["status"] == "unmeasured" for p in plants)
    assert {p["latestReading"]["status"] for p in plants} <= {"accepted", "flagged"}

    logs = c.get("/api/data-logs?limit=5")
    assert logs.headers["x-total-count"] == "5376" and len(logs.json()) == 5
    sessions = c.get("/api/sessions").json()
    assert [s["session_id"] for s in sessions] == [4, 3, 2, 1]
    assert all(s["received_records"] == 1344 == s["expected_records"] for s in sessions)

    # AI side: empty / 404 states, no fabricated numbers
    insights = c.get("/api/ai/insights").json()
    assert insights["hasPredictions"] is False
    assert insights["predictions"] == {"ok": 0, "unavailable": 0, "rejected": 0}
    assert all(v == 0 for v in insights["severityCounts"].values())
    assert insights["anomalies"]["count"] == 0 and insights["anomalies"]["rate"] is None
    assert insights["explanationCoverage"]["plantsWithLocalShap"] == 0
    assert c.get("/api/predictions").json() == [] and c.get("/api/predictions").headers["x-total-count"] == "0"
    assert c.get("/api/ai/anomalies").json() == []
    empty = c.get("/api/ai/anomalies/summary").json()
    assert empty["flagged"] == 0 and empty["totalScored"] == 0 and empty["rate"] is None and empty["plants"] == []
    assert empty["latestFlaggedAt"] is None and empty["plantsAffected"] == 0
    r = c.get("/api/ai/shap/1")
    assert r.status_code == 404 and r.json()["error"]["code"] == "explanation_unavailable"
    assert c.get("/api/recommendations").json() == []

    overview = c.get("/api/monitoring/overview").json()
    assert overview["totals"]["plants"] == 16 and overview["totals"]["readings"] == 5376
    assert overview["topRecommendations"] == [] and overview["recentAlerts"] == []
    assert overview["plantStatusCounts"]["unmeasured"] == 16
    assert overview["pipeline"]["state"] == "idle" and overview["pipeline"]["readyToRun"] is True
    detail = c.get("/api/monitoring/plants/1").json()
    assert len(detail["series"]) == 84 and detail["predictionHistory"] == [] and detail["recommendations"] == []


def test_run_populates_everything(fresh_client):
    c = fresh_client
    status = run_pipeline(c)
    assert status["state"] == "completed" and status["processed"] == 5376

    overview = c.get("/api/monitoring/overview").json()
    assert overview["totals"]["plants"] == 16
    assert overview["totals"]["predictions"] == {"ok": 5376, "unavailable": 0, "rejected": 0, "total": 5376}
    assert overview["pipeline"]["state"] == "completed"
    assert overview["topRecommendations"], "recommendations must be derived from the ML output"
    assert len(overview["topRecommendations"]) <= 5
    assert sum(overview["severityCounts"].values()) == 16
    assert sum(overview["plantStatusCounts"].values()) == 16 and overview["plantStatusCounts"]["unmeasured"] == 0
    assert [g["cluster"] for g in overview["clusterSummaries"]] == ["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4"]
    assert all(g["plants"] == 4 and g["averages"]["soil"] is not None for g in overview["clusterSummaries"])
    assert overview["modelVersion"] == "1.0.0" and overview["provenance"] == "synthetic"
    assert len(overview["recentAlerts"]) <= 10
    kinds = {a["kind"] for a in overview["recentAlerts"]}
    assert kinds <= {"anomaly", "recommendation"}

    plants = c.get("/api/plants").json()
    assert all(p["conditionPrediction"] and p["status"] != "unmeasured" for p in plants)
    assert c.get("/api/predictions").headers["x-total-count"] == "5376"
    assert c.get("/api/ai/insights").json()["hasPredictions"] is True
    assert c.get("/api/ai/shap/16").status_code == 200
    anomalies = c.get("/api/ai/anomalies")
    assert int(anomalies.headers["x-total-count"]) == c.get("/api/ai/insights").json()["anomalies"]["count"] > 0

    # pressing Run again changes nothing
    again = run_pipeline(c)
    assert again["state"] == "completed" and again["total"] == 0
    assert c.get("/api/predictions").headers["x-total-count"] == "5376"
