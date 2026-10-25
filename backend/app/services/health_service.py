"""Database part of GET /api/health."""

from sqlalchemy.orm import Session

from app.db import repositories as repo
from app.db.models import ModelInfo


def db_part(db: Session) -> tuple[bool, int, int, ModelInfo | None]:
    """`(db_ok, readings, plants, cached model row)`; the model row is None if the database is unreachable."""
    db_ok, readings, plants = repo.db_summary(db)
    return db_ok, readings, plants, repo.latest_model_info(db) if db_ok else None
