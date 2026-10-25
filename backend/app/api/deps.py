"""FastAPI dependencies. (ML client and pagination dependencies arrive in later phases.)"""

from collections.abc import Iterator

from fastapi import Request
from sqlalchemy.orm import Session

from app.core.config import Settings


def get_db(request: Request) -> Iterator[Session]:
    """One SQLAlchemy session per request, taken from the app's session factory."""
    db = request.app.state.session_factory()
    try:
        yield db
    finally:
        db.close()


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings
