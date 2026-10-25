"""Prediction models: the request body of POST /api/predictions and the one response shape shared by
GET and POST (`PredictionListItem`, DEV-38)."""

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


class AnomalyInfo(CamelModel):
    """Isolation Forest output: unusual reading, NOT a plant-health verdict."""

    is_anomaly: bool | None
    score: float | None


class PredictionListItem(CamelModel):
    """One row of `GET /api/predictions` (no explanation / ML recommendations: those are in the detail views)."""

    id: int
    reading_id: int
    record_id: str = Field(alias="record_id")  # snake_case on purpose (DEV-38): the contract names these two literally
    plant_id: int = Field(alias="plant_id")
    timestamp: str
    status: str
    error: str | None
    warnings: list[str]
    model_version: str | None
    run_id: str | None
    severity: str | None
    probabilities: dict[str, float] | None
    confidence: float | None
    condition: str | None
    condition_source: str = "heuristic"
    env_state: EnvState | None
    anomaly: AnomalyInfo
