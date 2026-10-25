"""ML client against httpx.MockTransport: error mapping, retries, batch normalisation and fallback."""

import json

import httpx
import pytest

from app.integrations.ml_client import MLClient, MLRejected, MLUnavailable
from app.integrations.ml_types import MLReading, PredictRequest

READING = MLReading(timestamp="2026-09-21T00:00:00+05:30", plant_id=1, soil=40.0, temperature=25.0, humidity=60.0, light=500.0)
REQ = PredictRequest(reading=READING, history=[], explain=False)


def ok_result(plant_id=1):
    return {
        "plant_id": plant_id, "timestamp": "2026-09-21T00:00:00+05:30",
        "stress_prediction": {"label": "healthy", "probabilities": {"healthy": 0.9, "low": 0.05, "moderate": 0.03, "high": 0.02}},
        "cluster": {"id": 1, "distance": 1.0}, "anomaly": {"is_anomaly": False, "score": -0.2, "threshold": 0.0, "interpretation": "x"},
        "explanation": {"available": False, "reason": "off", "top_features": []}, "recommendations": [],
        "recommendations_disclaimer": "d", "history": {"rows_available": 1, "rows_recommended": 49, "sufficient": False},
        "warnings": [], "model_metadata": {"model_version": "1.0.0", "run_id": "r"},
    }  # fmt: skip


def client(handler, **kw) -> MLClient:
    return MLClient("http://ml.test", 1, transport=httpx.MockTransport(handler), backoff_s=0, **kw)


async def test_predict_ok_parses_response():
    ml = client(lambda r: httpx.Response(200, json=ok_result()))
    out = await ml.predict(REQ)
    assert out.stress_prediction.label == "healthy" and out.model_metadata.run_id == "r"


async def test_422_dict_detail_becomes_rejected():
    body = {"detail": {"error": "invalid_readings", "message": "bad", "issues": [{"index": 0, "issues": ["soil_out_of_range"]}]}}
    ml = client(lambda r: httpx.Response(422, json=body))
    with pytest.raises(MLRejected) as exc:
        await ml.predict(REQ)
    assert exc.value.code == "invalid_readings" and exc.value.message == "bad" and exc.value.issues[0]["index"] == 0


async def test_422_pydantic_list_detail_becomes_rejected():
    body = {"detail": [{"loc": ["body", "reading", "plant_id"], "msg": "Field required", "type": "missing"}]}
    ml = client(lambda r: httpx.Response(422, json=body))
    with pytest.raises(MLRejected) as exc:
        await ml.predict(REQ)
    assert exc.value.code == "invalid_request" and "reading.plant_id" in exc.value.message and exc.value.issues == body["detail"]


async def test_503_then_success_is_retried():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(503, json={"detail": {"error": "models_unavailable", "message": "loading"}})
        return httpx.Response(200, json=ok_result())

    out = await client(handler).predict(REQ)
    assert out.plant_id == 1 and len(calls) == 3


async def test_persistent_503_raises_unavailable_after_three_tries():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503, json={"detail": {"error": "models_unavailable", "message": "no artifacts"}})

    with pytest.raises(MLUnavailable) as exc:
        await client(handler).predict(REQ)
    assert len(calls) == 3 and "no artifacts" in exc.value.reason


@pytest.mark.parametrize("error", [httpx.ReadTimeout("slow"), httpx.ConnectError("refused")])
async def test_timeouts_and_connection_errors_raise_unavailable_never_httpx(error):
    calls = []

    def handler(request):
        calls.append(1)
        raise error

    with pytest.raises(MLUnavailable) as exc:
        await client(handler).predict(REQ)
    assert len(calls) == 3
    assert isinstance(exc.value, MLUnavailable)


async def test_health_is_single_attempt_unless_retry():
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx.ConnectError("refused")

    ml = client(handler)
    with pytest.raises(MLUnavailable, match="not reachable at http://ml.test"):
        await ml.health()
    assert len(calls) == 1
    with pytest.raises(MLUnavailable):
        await ml.health(retry=True)
    assert len(calls) == 4


async def test_cached_health_reports_failure_reason():
    ml = client(lambda r: (_ for _ in ()).throw(httpx.ConnectError("refused")))
    health, reason = await ml.cached_health()
    assert health is None and "not reachable" in reason


async def test_batch_maps_per_item_ok_422_503():
    def handler(request):
        items = json.loads(request.content)["items"]
        assert len(items) == 3 and request.url.path == "/predict/batch"
        return httpx.Response(200, json={"results": [
            {"index": 0, "ok": True, "result": ok_result(), "error": None},
            {"index": 1, "ok": False, "result": None, "error": {"status": 422, "error": "invalid_readings", "message": "bad row", "issues": [{"index": 0}]}},
            {"index": 2, "ok": False, "result": None, "error": {"status": 503, "error": "models_unavailable", "message": "gone"}},
        ]})  # fmt: skip

    res = await client(handler).predict_batch([REQ, REQ, REQ])
    assert [r.status for r in res] == ["ok", "rejected", "unavailable"]
    assert res[1].code == "invalid_readings" and res[1].issues == [{"index": 0}] and "bad row" in res[1].error_text
    assert res[2].message == "gone" and res[0].result.plant_id == 1


async def test_whole_batch_pydantic_422_falls_back_to_per_item_predict():
    paths = []

    def handler(request):
        paths.append(request.url.path)
        if request.url.path == "/predict/batch":
            return httpx.Response(422, json={"detail": [{"loc": ["body", "items", 1, "reading", "timestamp"], "msg": "bad datetime", "type": "x"}]})
        body = json.loads(request.content)
        if body["reading"]["soil"] == 13.0:
            return httpx.Response(422, json={"detail": {"error": "invalid_readings", "message": "bad"}})
        return httpx.Response(200, json=ok_result())

    bad = PredictRequest(reading=READING.model_copy(update={"soil": 13.0}))
    res = await client(handler).predict_batch([REQ, bad, REQ])
    assert [r.status for r in res] == ["ok", "rejected", "ok"]
    assert paths == ["/predict/batch", "/predict", "/predict", "/predict"]


async def test_batch_503_whole_call_raises_unavailable():
    ml = client(lambda r: httpx.Response(503, json={"detail": {"error": "models_unavailable", "message": "x"}}))
    with pytest.raises(MLUnavailable):
        await ml.predict_batch([REQ])


async def test_bad_batch_size_is_a_rejection_not_a_fallback():
    ml = client(lambda r: httpx.Response(422, json={"detail": {"error": "bad_batch_size", "message": "1..64"}}))
    with pytest.raises(MLRejected) as exc:
        await ml.predict_batch([REQ])
    assert exc.value.code == "bad_batch_size"


async def test_requests_carry_only_core_reading_fields():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=ok_result())

    await client(handler).predict(REQ)
    assert set(seen["reading"]) == {"timestamp", "plant_id", "soil", "temperature", "humidity", "light"}


async def test_batch_5xx_falls_back_and_per_item_500_is_unavailable():
    paths = []

    def handler(request):
        paths.append(request.url.path)
        if request.url.path == "/predict/batch":
            return httpx.Response(500, json={"detail": "boom"})
        if json.loads(request.content)["reading"]["soil"] == 13.0:
            return httpx.Response(500, json={"detail": "boom"})
        return httpx.Response(200, json=ok_result())

    bad = PredictRequest(reading=READING.model_copy(update={"soil": 13.0}))
    res = await client(handler).predict_batch([REQ, bad, REQ])
    assert [r.status for r in res] == ["ok", "unavailable", "ok"]
    assert "ML internal error" in res[1].message and paths == ["/predict/batch", "/predict", "/predict", "/predict"]


async def test_non_transport_httpx_errors_never_escape():
    def handler(request):
        raise httpx.DecodingError("bad gzip")

    with pytest.raises(MLUnavailable):
        await client(handler).predict(REQ)


async def test_cached_metadata_is_fetched_once():
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, json={"run_id": "r", "model_version": "1.0.0", "required_history_rows_for_full_features": 49})

    ml = client(handler)
    assert (await ml.cached_metadata()).run_id == "r" and (await ml.cached_metadata()).model_version == "1.0.0"
    assert calls == ["/metadata"]
