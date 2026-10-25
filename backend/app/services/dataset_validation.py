"""Best-effort, non-blocking advisory check of the seeded dataset against the ML service's `/validate`.

Runs as a background task after startup seeding. If the ML service is unreachable it logs and gives up;
it never raises into the caller. The summary is logged and stored in `model_info.metadata_json
["dataset_validation"]` (when `/metadata` is available). The local checks in the seed stay authoritative.
"""

import asyncio
import logging
from typing import Any

from sqlalchemy.orm import sessionmaker

from app.db import repositories as repo
from app.integrations.ml_client import MLClient, MLError
from app.services import model_service
from app.services.prediction_service import to_ml_reading

logger = logging.getLogger("app.dataset_validation")


def _load_readings(session_factory: sessionmaker) -> list:
    with session_factory() as db:
        return [to_ml_reading(r) for r in repo.all_readings_ordered(db)]


def _store(session_factory: sessionmaker, metadata, summary: dict[str, Any]) -> None:
    with session_factory() as db:
        model_service.cache_model_info(db, metadata, {"dataset_validation": summary})


async def validate_with_ml(session_factory: sessionmaker, ml: MLClient) -> dict[str, Any] | None:
    """Send every seeded reading to `/validate` in one request (the endpoint has no size limit).
    Returns the stored summary, or None if skipped or failed."""
    try:
        readings = await asyncio.to_thread(_load_readings, session_factory)
        if not readings:
            logger.info("dataset validation skipped: no readings")
            return None
        response = await ml.validate(readings)
        summary = {
            "valid": response.valid,
            "n_readings": response.n_readings,
            "rows_with_issues": len(response.issues),
            "summary": response.summary,
        }
        logger.info("ML /validate: %s", summary)
        try:
            metadata = await ml.metadata()
            await asyncio.to_thread(_store, session_factory, metadata, summary)
        except MLError as exc:
            logger.info("validation summary not stored (no /metadata: %s)", exc)
        return summary
    except MLError as exc:
        logger.info("dataset validation skipped: ML service unavailable or refused (%s)", exc)
    except Exception:
        logger.exception("dataset validation failed")
    return None
