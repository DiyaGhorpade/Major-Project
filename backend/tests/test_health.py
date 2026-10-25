"""/api/health and the error envelope."""


def test_health_reports_seeded_counts(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok" and body["seeded"] is True
    assert body["db"] == {"ok": True, "readings": 5376, "plants": 16}
    assert body["mlService"]["ok"] is False
    assert body["modelVersion"] is None
    assert body["datasetName"] == "plant_sensor_data.csv"  # file name only, never a filesystem path
    assert response.headers["X-Request-ID"]


def test_error_envelope(client):
    missing = client.get("/api/nope", headers={"X-Request-ID": "abc-123"})

    assert missing.status_code == 404
    assert missing.json() == {"error": {"code": "not_found", "message": "Not Found", "details": None}}
    assert missing.headers["X-Request-ID"] == "abc-123"
