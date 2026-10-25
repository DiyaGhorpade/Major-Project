"""Model metadata cached from the ML service into `model_info` (Phase 3 adds the read APIs)."""

import logging
from typing import Any

from sqlalchemy.orm import Session

from app.core.mappings import parse_timestamp
from app.db import repositories as repo
from app.db.models import ModelInfo, utcnow
from app.integrations.ml_client import MLClient, MLError
from app.integrations.ml_types import HealthResponse, MetadataResponse

logger = logging.getLogger("app.model")


def _labels(label_mapping: Any) -> list[str] | None:
    if isinstance(label_mapping, dict):
        values = list(label_mapping.values())
        if values and all(isinstance(v, str) for v in values):
            return values
        return [str(k) for k in label_mapping]
    if isinstance(label_mapping, list):
        return [str(v) for v in label_mapping]
    return None


def cache_model_info(db: Session, metadata: MetadataResponse, extra: dict[str, Any] | None = None) -> ModelInfo:
    """Upsert the model_info row for the ML run. The full `/metadata` document goes into
    `metadata_json` (run_id, model_version, trained_at, validated_on_real_plants, label_mapping,
    feature names, required history rows, disclaimer...); existing extra keys (e.g. the dataset
    validation summary) are preserved unless overridden by `extra`."""
    document = metadata.model_dump(mode="json")
    existing = repo.latest_model_info(db)
    kept = dict(existing.metadata_json or {}) if existing and existing.run_id == metadata.run_id else {}
    kept.pop("run_id", None)
    document = {**{k: v for k, v in kept.items() if k in ("dataset_validation",)}, **document, **(extra or {})}
    row = repo.upsert_model_info(
        db,
        {
            "run_id": metadata.run_id,
            "model_version": metadata.model_version,
            "trained_at": parse_timestamp(metadata.trained_at) if metadata.trained_at else None,
            "provenance": metadata.trained_on,
            "labels": _labels(metadata.label_mapping),
            "metadata_json": document,
            "cached_at": utcnow(),
        },
    )
    db.commit()
    return row


def current_model(db: Session) -> ModelInfo | None:
    return repo.latest_model_info(db)


async def resolve_model(model: ModelInfo | None, health: HealthResponse | None, ml: MLClient) -> tuple[str | None, str | None]:
    """`(run_id, model_version)` for health/status: the live ML run id wins; the version comes from the
    cached `model_info` row of that run, else from (cached) `/metadata`."""
    run_id = (health.run_id if health and health.run_id else None) or (model.run_id if model else None)
    if model is not None and (run_id is None or model.run_id == run_id):
        return run_id, model.model_version
    if health is not None and health.status == "ok" and health.models_loaded:
        try:
            return run_id, (await ml.cached_metadata()).model_version
        except MLError:
            pass
    return run_id, None
