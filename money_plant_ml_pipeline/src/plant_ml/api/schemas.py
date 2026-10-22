"""Pydantic request / response schemas. Range checks are done by plant_ml.validation so that
/validate can REPORT violations (HTTP 200) while /predict rejects them (HTTP 422) with details."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class Reading(BaseModel):
    """One sensor reading. Target labels are NOT required (and are ignored if supplied)."""
    model_config = ConfigDict(extra="ignore", json_schema_extra={"example": {
        "timestamp": "2026-10-21T10:30:00+05:30", "plant_id": 3, "node_id": "1", "sampling_point": 6, "session_id": 4,
        "soil": 48.2, "temperature": 26.4, "humidity": 68.0, "light": 3200}})

    timestamp: datetime = Field(..., description="ISO 8601; +05:30 preferred. Naive values are assumed Asia/Kolkata.")
    plant_id: int = Field(..., description="1-16")
    soil: Optional[float] = Field(None, description="soil moisture %, 0-100; null = missing")
    temperature: Optional[float] = Field(None, description="deg C, 15-40; null = missing")
    humidity: Optional[float] = Field(None, description="% RH, 20-100; null = missing")
    light: Optional[float] = Field(None, description="lux, 0-100000; null = missing")
    session_id: Optional[int] = None
    node_id: Optional[Union[str, int]] = None
    sampling_point: Optional[int] = None
    # Accepted for convenience when replaying training rows; NEVER used by the models.
    stress_level: Optional[str] = Field(None, description="ignored")
    stress_type: Optional[str] = Field(None, description="ignored")
    experimental_group: Optional[str] = Field(None, description="ignored")
    is_synthetic_anomaly: Optional[int] = Field(None, description="ignored")


class PredictRequest(BaseModel):
    reading: Reading
    history: List[Reading] = Field(default_factory=list,
                                   description="Earlier readings of the SAME plant (any order). "
                                               "Provide >= required_history_rows_for_full_features (see /metadata) for best accuracy.")
    explain: Optional[bool] = Field(None, description="Override explain.enabled for this call")


class BatchPredictRequest(BaseModel):
    items: List[PredictRequest]


class ValidateRequest(BaseModel):
    readings: List[Reading]


class StressPrediction(BaseModel):
    label: str
    probabilities: Dict[str, float]


class ClusterOut(BaseModel):
    id: int
    distance: float


class AnomalyOut(BaseModel):
    is_anomaly: bool
    score: float
    threshold: float
    interpretation: str


class PredictResponse(BaseModel):
    plant_id: int
    timestamp: str
    stress_prediction: StressPrediction
    cluster: ClusterOut
    anomaly: AnomalyOut
    explanation: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    recommendations_disclaimer: str
    history: Dict[str, Any]
    warnings: List[str]
    model_metadata: Dict[str, Any]


class BatchItemResult(BaseModel):
    index: int
    ok: bool
    result: Optional[PredictResponse] = None
    error: Optional[Dict[str, Any]] = None


class BatchPredictResponse(BaseModel):
    results: List[BatchItemResult]


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    artifact_error: Optional[str] = None
    run_id: Optional[str] = None
    trained_on: Optional[str] = None
    explainability_available: Optional[bool] = None
    warnings: List[str] = Field(default_factory=list)


class ValidateResponse(BaseModel):
    valid: bool
    n_readings: int
    issues: List[Dict[str, Any]]
    summary: Dict[str, Any]
