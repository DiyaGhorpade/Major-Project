"""Shared schema building blocks: camelCase base model, error envelope, health response."""

from typing import Any

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    """Base for API models that serialise to camelCase JSON."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Any = None


class ErrorEnvelope(BaseModel):
    error: ErrorDetail


class DbHealth(CamelModel):
    ok: bool
    readings: int
    plants: int


class MlServiceHealth(CamelModel):
    ok: bool
    url: str
    detail: str


class HealthResponse(CamelModel):
    status: str
    db: DbHealth
    ml_service: MlServiceHealth
    model_version: str | None
    dataset_name: str
    seeded: bool
