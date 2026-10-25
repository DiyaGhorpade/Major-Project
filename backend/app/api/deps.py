"""FastAPI dependencies: DB session, settings, ML client, pipeline service."""

from collections.abc import Iterator

from fastapi import Request
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.integrations.ml_client import MLClient
from app.services.pipeline_service import PipelineService


def get_db(request: Request) -> Iterator[Session]:
    """One SQLAlchemy session per request, taken from the app's session factory."""
    db = request.app.state.session_factory()
    try:
        yield db
    finally:
        db.close()


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_ml_client(request: Request) -> MLClient:
    """The app-wide ML client (tests replace `app.state.ml_client` with a fake)."""
    return request.app.state.ml_client


def get_pipeline_service(request: Request) -> PipelineService:
    return request.app.state.pipeline


def get_session_factory(request: Request):
    """The app's session factory, for code that must not hold a session across an `await`."""
    return request.app.state.session_factory
