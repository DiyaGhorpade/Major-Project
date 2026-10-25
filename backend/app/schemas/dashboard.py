"""Overview and plant-monitoring models (camelCase), DEV-36."""

from typing import Literal

from pydantic import ConfigDict

from app.schemas.common import CamelModel
from app.schemas.pipeline import PipelineStatus
from app.schemas.plant import PlantOut
from app.schemas.prediction import EnvState
from app.schemas.recommendation import RecommendationOut
from app.schemas.session import SessionOut


class PredictionTotals(CamelModel):
    ok: int
    unavailable: int
    rejected: int
    total: int


class Totals(CamelModel):
    plants: int
    readings: int
    flagged_readings: int
    predictions: PredictionTotals
    anomalies: int


class SensorAverages(CamelModel):
    soil: float | None
    temperature: float | None
    humidity: float | None
    light: float | None


class ClusterSummary(CamelModel):
    """One experimental group (not a KMeans cluster). Averages use each plant's latest reading, ignoring nulls."""

    cluster: str
    cluster_label: str
    plant_ids: list[int]
    plants: int
    averages: SensorAverages
    severity_counts: dict[str, int]
    top_condition: str | None  # most common heuristic stress-type guess among the group's latest predictions


class Alert(CamelModel):
    kind: Literal["anomaly", "recommendation"]
    plant_id: int
    plant_name: str
    timestamp: str
    message: str
    severity: str | None = None
    record_id: str | None = None
    recommendation_id: str | None = None


class DevicesSummary(CamelModel):
    total: int
    online: int
    offline: int
    unknown: int
    last_transmission: str | None


class OverviewCore(CamelModel):
    """The cached part of the overview (everything except the live pipeline status)."""

    totals: Totals
    plant_status_counts: dict[str, int]
    severity_counts: dict[str, int]  # latest ok prediction per plant; `unmeasured` = no prediction yet
    cluster_summaries: list[ClusterSummary]
    recent_alerts: list[Alert]
    top_recommendations: list[RecommendationOut]
    latest_session: SessionOut | None
    devices_summary: DevicesSummary
    model_version: str | None
    provenance: str
    disclaimer: str


class OverviewOut(OverviewCore):
    pipeline: PipelineStatus


class SeriesPoint(CamelModel):
    timestamp: str
    soil: float | None
    temperature: float | None
    humidity: float | None
    light: float | None


class PredictionHistoryPoint(CamelModel):
    """Mixed case on purpose: `record_id` is snake_case, the rest camelCase (the contract in the task)."""

    model_config = ConfigDict(alias_generator=None, populate_by_name=True)

    timestamp: str
    record_id: str
    status: str
    severity: str | None
    probabilities: dict[str, float] | None
    confidence: float | None
    condition: str | None
    isAnomaly: bool | None
    anomalyScore: float | None
    envState: EnvState | None


class PlantMonitoringOut(CamelModel):
    plant: PlantOut
    series: list[SeriesPoint]
    prediction_history: list[PredictionHistoryPoint]
    recommendations: list[RecommendationOut]
