"""Pipeline run request/response models (camelCase JSON)."""

from typing import Literal

from pydantic import BaseModel

from app.schemas.common import CamelModel

PipelineState = Literal["idle", "queued", "running", "completed", "failed"]


class RunRequest(BaseModel):
    reset: bool = False


class RunResponse(CamelModel):
    run_id: int
    state: PipelineState


class PipelineStatus(CamelModel):
    run_id: int | None
    state: PipelineState
    total: int
    processed: int
    failed: int
    percent: float
    current_plant_id: int | None
    started_at: str | None
    finished_at: str | None
    message: str | None
    ml_service_ok: bool
    model_version: str | None
    ready_to_run: bool
