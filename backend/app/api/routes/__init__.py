"""Registers all routers under the /api prefix."""

from fastapi import APIRouter

from app.api.routes import (
    ai,
    data_logs,
    devices,
    health,
    models,
    monitoring,
    pipeline,
    plants,
    predictions,
    recommendations,
    sensors,
    sessions,
    settings,
)

api_router = APIRouter(prefix="/api")
for module in (
    health, pipeline, plants, sensors, data_logs, monitoring, predictions,
    ai, recommendations, devices, sessions, models, settings,
):  # fmt: skip
    api_router.include_router(module.router)
