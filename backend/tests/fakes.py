"""A scriptable stand-in for the ML service that speaks the real wire format.

`FakeML` is a real `MLClient` whose HTTP transport is an `httpx.MockTransport`, so the client's retry,
error mapping and batch normalisation all run. The handler mimics the real service: per-item
`{index, ok, result, error}` batch results, 422 detail dicts, condition recommendations only after 3
consecutive rows, etc.
"""

import asyncio
import json
import threading

import httpx

from app.integrations.ml_client import MLClient
from app.integrations.ml_types import PredictRequest

LABELS = ("healthy", "low", "moderate", "high")
RUN_ID = "fake_run_1"


def make_response(req: PredictRequest, *, anomaly: bool, explain: bool, run_id: str = RUN_ID) -> dict:
    cur = req.reading
    soil = cur.soil if cur.soil is not None else 50.0
    label = LABELS[int(soil // 26) % 4]
    probs = {k: 0.1 for k in LABELS}
    probs[label] = 0.7
    recs: list[dict] = []
    missing = None in (cur.soil, cur.temperature, cur.humidity, cur.light)
    if anomaly:  # precedence like the real service: an anomaly yields only verify_sensor
        recs = [{"id": "verify_sensor", "category": "data_quality", "text": "Check the sensor", "reason": "anomaly", "priority": "medium"}]
    elif missing:
        recs = [{"id": "retake_reading", "category": "data_quality", "text": "Retake", "reason": "missing", "priority": "low"}]
    else:
        last3 = [*req.history[-2:], cur]
        if len(last3) == 3 and all(r.soil is not None and r.soil < 25 for r in last3):  # persistent condition
            recs = [{"id": "check_watering", "category": "soil", "text": "Check watering", "reason": f"soil {soil}", "priority": "high"}]
    return {
        "plant_id": cur.plant_id,
        "timestamp": cur.timestamp,
        "stress_prediction": {"label": label, "probabilities": probs},
        "cluster": {"id": cur.plant_id % 3, "distance": 1.5},
        "anomaly": {"is_anomaly": anomaly, "score": 0.2 if anomaly else -0.3, "threshold": 0.0, "interpretation": "fake"},
        "explanation": (
            {"available": True, "target_class": label, "top_features": [{"feature": "soil", "value": soil, "contribution": 0.4, "direction": "towards_predicted_class"}]}
            if explain else {"available": False, "reason": "not requested", "top_features": []}
        ),
        "recommendations": recs,
        "recommendations_disclaimer": "fake",
        "history": {"rows_available": len(req.history) + 1, "rows_recommended": 49, "sufficient": len(req.history) >= 48},
        "warnings": [] if len(req.history) >= 48 else ["cold start"],
        "model_metadata": {"trained_on": "synthetic", "validated_on_real_plants": False, "model_version": "1.0.0", "run_id": run_id},
    }  # fmt: skip


class FakeML(MLClient):
    """Knobs: `up`, `models_error`, `die_after_batches`, `reject_if(req)` (+ `reject_code`), `gate`, `batch_status`,
    `batch_error` (a 422 detail dict that refuses the whole /predict/batch call), `ml_run_id`."""

    def __init__(self):
        self.up = True
        self.models_error: str | None = None
        self.ml_run_id = RUN_ID
        self.die_after_batches: int | None = None
        self.die_after_predicts: int | None = None  # the service goes away after this many successful /predict calls
        self.batch_status: int | None = None  # force /predict/batch to answer with this HTTP status
        self.item_500_if = lambda req: False  # per-item 500 on /predict (batch fallback path)
        self.reject_if = lambda req: False
        self.reject_code = "invalid_readings"  # the 422 code `reject_if` answers with
        self.batch_error: dict | None = None  # e.g. {"error": "bad_batch_size", "message": "..."}: refuses the whole call
        self.gate = threading.Event()  # clear it to hold batches
        self.gate.set()
        self.batches = 0
        self.max_items_seen = 0
        self.predict_calls = 0
        self.seen: list[PredictRequest] = []  # every item received, via batch or single predict
        self.cross_plant: dict | None = None  # advertised in /metadata when set (e.g. {"enabled": True, "reference_plants": [...]})
        self.batch_payloads: list[dict] = []  # raw /predict/batch bodies (to inspect the shared `reference`)
        super().__init__("http://fake-ml:8001", 1, transport=httpx.MockTransport(self._handle), attempts=3, backoff_s=0)

    def _item_result(self, i: int, req: PredictRequest) -> dict:
        self.seen.append(req)
        if self.reject_if(req):
            err = {"status": 422, "error": self.reject_code, "message": "fake rejection", "issues": [{"index": 0, "issues": ["soil_out_of_range"]}]}
            return {"index": i, "ok": False, "result": None, "error": err}
        anomaly = (req.reading.plant_id + len(req.history)) % 40 == 0
        return {"index": i, "ok": True, "result": make_response(req, anomaly=anomaly, explain=bool(req.explain), run_id=self.ml_run_id), "error": None}

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if not self.up:
            raise httpx.ConnectError("refused")
        if path == "/health":
            if self.models_error:
                return httpx.Response(200, json={"status": "unavailable", "models_loaded": False, "artifact_error": self.models_error})
            return httpx.Response(200, json={"status": "ok", "models_loaded": True, "artifact_error": None, "run_id": self.ml_run_id,
                                             "trained_on": "synthetic", "explainability_available": True, "warnings": []})  # fmt: skip
        if path == "/metadata":
            if self.models_error:
                return httpx.Response(503, json={"detail": {"error": "models_unavailable", "message": self.models_error}})
            return httpx.Response(200, json={
                "run_id": self.ml_run_id, "model_version": "1.0.0", "trained_at": "2026-10-21T14:52:44+00:00", "trained_on": "synthetic",
                "validated_on_real_plants": False, "label_mapping": {k: i for i, k in enumerate(LABELS)},
                "explainability": {"available": True, "reason": None}, "required_history_rows_for_full_features": 49, "disclaimer": "fake",
                **({"cross_plant": self.cross_plant} if self.cross_plant is not None else {}),
            })  # fmt: skip
        body = json.loads(request.content) if request.content else {}
        if path == "/validate":
            n = len(body["readings"])
            return httpx.Response(200, json={"valid": True, "n_readings": n, "issues": [], "summary": {"rows": n}})
        if path == "/predict":
            req = PredictRequest.model_validate(body)
            if self.die_after_predicts is not None and self.predict_calls >= self.die_after_predicts:
                self.up = False
                raise httpx.ConnectError("refused")
            self.predict_calls += 1
            if self.item_500_if(req):
                return httpx.Response(500, json={"detail": "boom"})
            item = self._item_result(0, req)
            if not item["ok"]:
                return httpx.Response(422, json={"detail": {"error": self.reject_code, "message": "fake rejection", "issues": item["error"]["issues"]}})
            return httpx.Response(200, json=item["result"])
        if path == "/predict/batch":
            while not self.gate.is_set():
                await asyncio.sleep(0.005)
            if self.die_after_batches is not None and self.batches >= self.die_after_batches:
                self.up = False
                raise httpx.ConnectError("refused")
            if self.batch_status:
                return httpx.Response(self.batch_status, json={"detail": "forced"})
            if self.batch_error:
                return httpx.Response(422, json={"detail": self.batch_error})
            self.batches += 1
            self.batch_payloads.append(body)
            items = [PredictRequest.model_validate(i) for i in body["items"]]
            self.max_items_seen = max(self.max_items_seen, len(items))
            return httpx.Response(200, json={"results": [self._item_result(i, req) for i, req in enumerate(items)]})
        return httpx.Response(404, json={"detail": "not found"})
