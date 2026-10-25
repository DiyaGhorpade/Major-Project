"""Plant models (TS `Plant`, camelCase) with the labelled ML sub-objects (DEV-30)."""

from typing import Any, Literal

from app.core.mappings import NodeId, PlantHealthStatus, StressType
from app.schemas.common import CamelModel
from app.schemas.prediction import AnomalyInfo, EnvState
from app.schemas.sensor import SensorRecordOut


class PlantPredictionOut(CamelModel):
    """Latest `ok` ML prediction of a plant. `condition` is a heuristic stress-type guess, not a diagnosis;
    `envState` (KMeans) and `anomaly` are separate labelled fields and never mean health."""

    condition: StressType | None
    condition_source: Literal["heuristic"] = "heuristic"
    severity: str | None
    probabilities: dict[str, float] | None
    confidence: float | None
    timestamp: str
    model_version: str | None
    record_id: str
    simple_explanation: str
    env_state: EnvState | None
    anomaly: AnomalyInfo


class PlantOut(CamelModel):
    id: int
    name: str
    species: str
    variety: str
    cluster: str  # experimental group "Cluster 1".."Cluster 4"
    cluster_label: str
    stress_type: StressType  # the group's designed treatment
    node_id: NodeId
    sampling_point: int | None  # multiplexer channel of the latest reading
    optimal_ranges: dict[str, Any]  # `light` also carries `daytimeHours: [first, last]` (IST hours, inclusive)
    latest_reading: SensorRecordOut | None
    status: PlantHealthStatus
    condition_prediction: PlantPredictionOut | None
    description: str
    simple_care_tip: str
    location: str
