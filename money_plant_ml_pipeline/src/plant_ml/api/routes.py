"""HTTP routes. The service is stateless: each request carries the plant's recent history."""
from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from ..data_loader import normalise_frame
from ..validation import validate_frame
from .schemas import (
    BatchItemResult, BatchPredictRequest, BatchPredictResponse, HealthResponse, PredictRequest, PredictResponse,
    Reading, ValidateRequest, ValidateResponse,
)

router = APIRouter()


def _frame(readings: list[Reading], tz: str) -> pd.DataFrame:
    rows = []
    for r in readings:
        d = r.model_dump()
        d["timestamp"] = r.timestamp.isoformat()
        rows.append(d)
    df = pd.DataFrame(rows)
    return normalise_frame(df, tz)


def _issues(df: pd.DataFrame) -> tuple[list[dict], dict]:
    res = validate_frame(df, mode="quarantine", require_labels=False)
    issues = [{"index": int(i), "issues": s.split(",")} for i, s in res.row_issues.items() if s]
    rep = res.report
    summary = {"rows": rep["rows"], "plants": rep["plants"], "time_range": rep["time_range"],
               "missing_sensor_values": {c: rep["missingness"][c]["count"] for c in ("soil", "temperature", "humidity", "light")},
               "sensor_range_violations": rep["sensor_range_violations"], "duplicate_plant_timestamp": rep["duplicate_plant_timestamp"]}
    return issues, summary


def _reference_frame(readings: list[Reading], request: Request) -> pd.DataFrame | None:
    """Normalised reference-plant readings (invalid rows are dropped: they only feed the cross-plant features)."""
    if not readings:
        return None
    cfg = request.app.state.cfg
    limit = int(cfg.get("api.max_reference_rows", 5000))
    if len(readings) > limit:
        raise HTTPException(status_code=422, detail={"error": "reference_too_long",
                            "message": f"reference has {len(readings)} rows; limit is {limit}. Send the history's time span only."})
    df = _frame(readings, cfg.get("timezone", "Asia/Kolkata"))
    clean = validate_frame(df, mode="quarantine", require_labels=False).clean
    plants = set(int(p) for p in (cfg.get("features.cross_plant.reference_plants", []) or []))
    return clean[clean["plant_id"].isin(plants)].reset_index(drop=True)


def _predict_one(req: PredictRequest, request: Request, shared_reference: pd.DataFrame | None = None) -> dict:
    state = request.app.state
    cfg = state.cfg
    if state.predictor is None:
        raise HTTPException(status_code=503, detail={"error": "models_unavailable", "message": state.artifact_error})
    max_hist = int(cfg.get("api.max_history_rows", 500))
    if len(req.history) > max_hist:
        raise HTTPException(status_code=422, detail={"error": "history_too_long",
                            "message": f"history has {len(req.history)} rows; limit is {max_hist}. Send the most recent rows only."})
    pid = req.reading.plant_id
    foreign = [i for i, h in enumerate(req.history) if h.plant_id != pid]
    if foreign:
        raise HTTPException(status_code=422, detail={"error": "mixed_plants",
                            "message": f"history rows {foreign} belong to a different plant than the reading (plant_id={pid}). "
                                       "Send one plant's history per item."})
    tz = cfg.get("timezone", "Asia/Kolkata")
    df = _frame(list(req.history) + [req.reading], tz)
    issues, summary = _issues(df)
    latest_idx = len(df) - 1
    if issues:
        raise HTTPException(status_code=422, detail={"error": "invalid_readings", "message":
                            "One or more readings failed validation (history rows are indexed 0..n-1, the reading is last).",
                            "issues": issues})
    if (df["timestamp"].iloc[:-1] >= df["timestamp"].iloc[latest_idx]).any():
        raise HTTPException(status_code=422, detail={"error": "history_not_before_reading",
                            "message": "All history timestamps must be strictly earlier than the reading's timestamp."})
    reference = _reference_frame(list(req.reference), request)
    if shared_reference is not None:
        reference = shared_reference if reference is None else pd.concat([reference, shared_reference], ignore_index=True)
    out = state.predictor.predict_frame(df, rows="last", explain=req.explain, reference=reference)
    return out[0]


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    st = request.app.state
    if st.predictor is None:
        return HealthResponse(status="unavailable", models_loaded=False, artifact_error=st.artifact_error)
    md = st.predictor.metadata()
    return HealthResponse(status="ok", models_loaded=True, run_id=md["run_id"], trained_on=md["trained_on"],
                          explainability_available=md["explainability"]["available"], warnings=md["compatibility_warnings"])


@router.get("/metadata")
def metadata(request: Request) -> dict[str, Any]:
    st = request.app.state
    if st.predictor is None:
        raise HTTPException(status_code=503, detail={"error": "models_unavailable", "message": st.artifact_error})
    return st.predictor.metadata()


@router.post("/validate", response_model=ValidateResponse)
def validate(body: ValidateRequest, request: Request):
    if not body.readings:
        raise HTTPException(status_code=422, detail={"error": "empty_request", "message": "readings must not be empty"})
    df = _frame(body.readings, request.app.state.cfg.get("timezone", "Asia/Kolkata"))
    issues, summary = _issues(df)
    return ValidateResponse(valid=not issues, n_readings=len(df), issues=issues, summary=summary)


@router.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request):
    return _predict_one(body, request)


@router.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(body: BatchPredictRequest, request: Request):
    st = request.app.state
    if st.predictor is None:
        raise HTTPException(status_code=503, detail={"error": "models_unavailable", "message": st.artifact_error})
    limit = int(st.cfg.get("api.max_batch_items", 64))
    if not body.items or len(body.items) > limit:
        raise HTTPException(status_code=422, detail={"error": "bad_batch_size", "message": f"items must contain 1..{limit} entries"})
    total_ref = len(body.reference) + sum(len(item.reference) for item in body.items)
    total_limit = int(st.cfg.get("api.max_batch_reference_rows", 10000))
    if total_ref > total_limit:
        raise HTTPException(status_code=422, detail={"error": "batch_reference_too_long",
                            "message": f"batch carries {total_ref} reference rows in total (shared + per item); limit is "
                                       f"{total_limit}. Send the shared `reference` once per batch, covering the history's "
                                       "time span only."})
    shared = _reference_frame(list(body.reference), request)
    results = []
    for i, item in enumerate(body.items):
        try:
            results.append(BatchItemResult(index=i, ok=True, result=PredictResponse(**_predict_one(item, request, shared))))
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
            results.append(BatchItemResult(index=i, ok=False, error={"status": exc.status_code, **detail}))
    return BatchPredictResponse(results=results)
