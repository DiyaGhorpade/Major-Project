"""Recommendation models. Field names match `frontend/src/types/index.ts` `Recommendation` exactly
(mixed snake_case and camelCase), plus the new fields `source`, `confidence`, `basedOn`, `disclaimer`, `mlRecId`.
Optional fields are emitted as `null` when empty."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Priority = Literal["high", "medium", "low"]
RecStatus = Literal["new", "in_progress", "completed", "dismissed", "superseded"]
UserStatus = Literal["new", "in_progress", "completed", "dismissed"]  # what a user may set
Category = Literal["irrigation", "lighting", "nutrition", "sensor_check", "climate", "monitoring"]
Source = Literal["ml", "heuristic"]


class SensorValue(BaseModel):
    metric: str
    value: str  # formatted with its unit, or "n/a" when the sensor value is missing
    target: str  # e.g. "25–75 %"
    status: Literal["good", "low", "high", "attention"]


class BasedOn(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    timestamp: str
    model_version: str | None = Field(alias="modelVersion")
    run_id: str | None = Field(alias="runId")  # the ML artifact run id


class RecommendationOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str  # "rec-<db id>"
    plant_id: int
    plant_name: str
    cluster: str
    title: str
    simple_action: str = Field(alias="simpleAction")
    priority: Priority
    status: RecStatus
    category: Category
    supporting_sensor_values: list[SensorValue]
    reason: str
    scientific_context: str = Field(alias="scientificContext")
    guidance_source: str
    created_at: str
    updated_at: str | None
    resolved_at: str | None
    source: Source
    confidence: float | None
    based_on: BasedOn | None = Field(alias="basedOn")
    disclaimer: Literal[True] = True
    ml_rec_id: str | None = Field(alias="mlRecId")


class RecommendationStatusUpdate(BaseModel):
    status: UserStatus
