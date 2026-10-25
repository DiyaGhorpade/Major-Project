"""Sensor reading models. `SensorRecordOut` keeps the snake_case field names of the TS `SensorRecord` and adds
`datasetLabels`, the dataset's ground-truth labels, kept apart from any ML output (DEV-6, DEV-10, DEV-31).
Ingestion models validate the shape of `POST /api/sensors/readings` payloads."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.mappings import NodeId, normalise_node_id
from app.schemas.common import CamelModel

MAX_BATCH = 500


class DatasetLabels(CamelModel):
    """Ground-truth labels from the bundled (synthetic) dataset. NOT model output."""

    stress_level: str | None
    stress_type: str | None
    is_synthetic_anomaly: bool | None


class SensorRecordOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    record_id: str
    timestamp: str
    session_id: int | None
    node_id: NodeId
    sampling_point: int | None
    plant_id: int
    soil: float | None
    temperature: float | None
    humidity: float | None
    light: float | None
    received_at: str | None
    accepted_at: str | None
    status: Literal["accepted", "rejected", "flagged"]
    validation_error: str | None
    rssi: float | None = None  # the dataset carries no radio metrics
    crc_ok: bool | None = None
    dataset_labels: DatasetLabels = Field(alias="datasetLabels")


class IngestReading(BaseModel):
    """One reading sent by a sensor node. Unknown extra fields (e.g. `rssi`) are ignored, not stored."""

    model_config = ConfigDict(extra="ignore")

    timestamp: datetime  # ISO-8601; naive = IST
    plant_id: int = Field(ge=1, le=16)
    session_id: int | None = Field(default=None, ge=1)
    node_id: str | int | None = None
    sampling_point: int | None = Field(default=None, ge=1, le=12)
    soil: float | None = Field(default=None, allow_inf_nan=False)
    temperature: float | None = Field(default=None, allow_inf_nan=False)
    humidity: float | None = Field(default=None, allow_inf_nan=False)
    light: float | None = Field(default=None, allow_inf_nan=False)

    @field_validator("timestamp", mode="before")
    @classmethod
    def _timestamp_is_text(cls, value: object) -> object:
        if not isinstance(value, str):
            raise ValueError("timestamp must be an ISO-8601 string")
        return value

    @field_validator("node_id")
    @classmethod
    def _node(cls, value: str | int | None) -> str | None:
        if value is None:
            return None
        node = normalise_node_id(value)
        if node is None:
            raise ValueError("node_id must be 1 or 2")
        return node


class IngestBatch(BaseModel):
    readings: list[IngestReading] = Field(min_length=1, max_length=MAX_BATCH)


class IngestItem(BaseModel):
    record_id: str | None
    plant_id: int
    timestamp: str
    status: Literal["accepted", "flagged", "duplicate"]


class IngestResponse(BaseModel):
    accepted: int
    flagged: int
    duplicates: int
    items: list[IngestItem]
