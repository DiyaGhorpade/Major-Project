"""Prediction models. Phase 2 ships the minimal on-demand shape; Phase 3 adds list/detail views."""

from typing import Any

from pydantic import BaseModel, Field

from app.schemas.common import CamelModel


class PredictionRequest(BaseModel):
    plant_id: int = Field(alias="plantId")
    reading_id: int | None = Field(default=None, alias="readingId")
    explain: bool = True

    model_config = {"populate_by_name": True}


class EnvState(CamelModel):
    id: int
    distance: float | None
    label: str  # "Environmental state group N"; not a stress level


class PredictionOut(CamelModel):
    id: int
    reading_id: int
    record_id: str
    plant_id: int
    timestamp: str
    status: str
    error: str | None
    warnings: list[str]
    model_version: str | None
    run_id: str | None
    severity: str | None
    probabilities: dict[str, float] | None
    confidence: float | None
    env_state: EnvState | None
    anomaly_score: float | None
    is_anomaly: bool | None
    condition: str | None  # heuristic stress-type guess, not a diagnosis
    condition_source: str = "heuristic"
    explanation: dict[str, Any] | None
    ml_recommendations: list[dict[str, Any]]
    history_rows_sent: int | None
