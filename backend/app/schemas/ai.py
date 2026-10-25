"""AI insights and anomaly list models (camelCase), DEV-37."""

from typing import Any

from pydantic import BaseModel

from app.schemas.common import CamelModel


class AnomalyOut(BaseModel):
    """An Isolation Forest flag: an unusual reading (check the sensor), not a plant-health verdict.
    Field names are literal (snake_case except the last), as in the contract."""

    id: str
    plant_id: int
    plant_name: str
    record_id: str
    timestamp: str
    score: float
    threshold: float
    interpretation: str
    readings: dict[str, float | None]
    datasetLabelIsSyntheticAnomaly: bool | None


class EnvStateCount(CamelModel):
    id: int
    label: str
    count: int
    note: str = "Environmental state group. Not a stress level."


class InsightsModelStatus(CamelModel):
    ml_service_ok: bool
    model_version: str | None
    run_id: str | None
    explainability_available: bool | None
    trained_at: str | None


class InsightsOut(CamelModel):
    has_predictions: bool
    predictions: dict[str, int]  # by status
    severity_counts: dict[str, int]  # all ok predictions
    latest_severity_counts: dict[str, int]  # latest ok prediction per plant
    condition_guess_counts: dict[str, int]  # heuristic stress-type guesses, all ok predictions
    latest_condition_guess_counts: dict[str, int]
    env_state_counts: list[EnvStateCount]
    anomalies: dict[str, Any]  # {count, rate, rateBasis, interpretation}
    explanation_coverage: dict[str, Any]  # {plantsWithLocalShap, totalPlants, plantIds}
    top_features: dict[str, Any]  # {global: [...]|None, globalSource, localFrequency: [...]}
    model: InsightsModelStatus
    last_pipeline_run: dict[str, Any] | None
    provenance: str
    disclaimer: str
