"""Model metadata cached into `model_info`: the ML service's `/metadata`, and the evaluation metrics read from
the run's `classifier.meta.json` (the run is resolved through `artifacts/latest.json`; DEV-35)."""

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import display_path
from app.core.mappings import parse_timestamp, to_ist_iso
from app.db import repositories as repo
from app.db.models import MlPrediction, ModelInfo, utcnow
from app.integrations.ml_client import MLClient, MLError
from app.integrations.ml_types import HealthResponse, MetadataResponse
from app.schemas.model import ModelMetricsOut

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


# --- model performance (file read, cached in model_info) ----------------------------------------
class ArtifactsUnavailable(Exception):
    """latest.json or the run's classifier.meta.json cannot be read."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactsUnavailable(f"cannot read {path.name}: {type(exc).__name__}") from exc
    if not isinstance(data, dict):
        raise ArtifactsUnavailable(f"{path.name} is not a JSON object")
    return data


def current_run_id(artifacts_dir: Path) -> str:
    """The newest trained run according to `artifacts/latest.json`."""
    run_id = _read_json(artifacts_dir / "latest.json").get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ArtifactsUnavailable("latest.json has no run_id")
    return run_id


def read_optional_json(path: Path) -> dict[str, Any] | None:
    try:
        return _read_json(path)
    except ArtifactsUnavailable:
        return None


def layman_status(f1: float) -> str:
    """Honest wording from F1 bands (synthetic test set; not a statement about real plants)."""
    if f1 >= 0.75:
        return "Recognised fairly reliably on the synthetic test set"
    if f1 >= 0.5:
        return "Partly reliable: expect some mistakes"
    return "Often confused with other classes"


def build_metrics(run_id: str, meta: dict[str, Any], anomaly: dict[str, Any] | None, clustering: dict[str, Any] | None) -> ModelMetricsOut:
    try:
        m = meta["metrics"]
        labels = list(m["confusion_matrix"]["labels"])
        per_class = m["per_class"]
        rows = [
            {
                "class_name": label,
                "precision": per_class[label]["precision"],
                "recall": per_class[label]["recall"],
                "f1_score": per_class[label]["f1"],
                "support": int(per_class[label]["support"]),
                "laymanStatus": layman_status(per_class[label]["f1"]),
            }
            for label in labels
        ]
        split = meta.get("split") or {}
        frac = split.get("test_fraction")
        cutoff = split.get("cutoff_timestamp")
        split_text = "chronological hold-out" + (f", last {round(frac * 100)}%" if frac else "") + (f" (cutoff {cutoff})" if cutoff else "")
        base = meta.get("baseline_metrics") or m.get("baseline_metrics") or {}
        anomaly_block = None
        if anomaly:
            anomaly_block = {
                "metricsVsHelperLabel": anomaly.get("metrics_vs_helper_label"),
                "threshold": anomaly.get("threshold"),
                "scoreConvention": anomaly.get("score_convention"),
                "note": anomaly.get("note"),
            }
        clustering_block = None
        if clustering:
            k = clustering.get("selected_k")
            clustering_block = {
                "selectedK": k,
                "silhouette": ((clustering.get("silhouette_by_k") or {}).get(str(k)) or {}).get("silhouette"),
                "note": clustering.get("note"),
            }
        return ModelMetricsOut(
            model_name=meta["model_name"],
            model_version=meta["model_version"],
            evaluated_at=to_ist_iso(parse_timestamp(meta["created_at_utc"]).replace(microsecond=0)) or "",
            dataset_name=(meta.get("training_data") or {}).get("filename") or "plant_sensor_data.csv",
            total_samples=int(m["n_samples"]),
            split=split_text,
            overall={
                "accuracy": m["accuracy"],
                "precision": sum(r["precision"] for r in rows) / len(rows),
                "recall": sum(r["recall"] for r in rows) / len(rows),
                "f1_score": m["macro_f1"],
            },
            per_class=rows,
            confusion_matrix={"labels": labels, "matrix": m["confusion_matrix"]["matrix"]},
            provenance=(meta.get("training_data") or {}).get("provenance") or "synthetic",
            runId=run_id,
            balancedAccuracy=m.get("balanced_accuracy"),
            weightedF1=m.get("weighted_f1"),
            baselines={
                name: {"accuracy": vals["accuracy"], "macro_f1": vals["macro_f1"]}
                for name, vals in base.items()
                if isinstance(vals, dict) and "accuracy" in vals and "macro_f1" in vals
            },
            anomalyModel=anomaly_block,
            clusteringModel=clustering_block,
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ArtifactsUnavailable(f"classifier.meta.json has an unexpected shape ({type(exc).__name__}: {exc})") from exc


def model_performance(db: Session, artifacts_dir: Path) -> ModelMetricsOut:
    """Metrics of the current run. Served from `model_info` when that run is already cached; otherwise read
    from the artifact files and cached (so a retrained model, i.e. a new `run_id`, refreshes automatically)."""
    run_id = current_run_id(artifacts_dir)
    row = repo.get_model_info_by_run(db, run_id)
    if row is not None and row.metrics:
        try:
            return ModelMetricsOut.model_validate(row.metrics)
        except ValueError:
            logger.warning("cached metrics for run %s are unreadable; re-reading the artifact", run_id)
    run_dir = artifacts_dir / run_id
    meta = _read_json(run_dir / "classifier.meta.json")
    result = build_metrics(run_id, meta, read_optional_json(run_dir / "anomaly.meta.json"), read_optional_json(run_dir / "clustering.meta.json"))
    repo.upsert_model_info(
        db,
        {
            "run_id": run_id,
            "model_version": result.model_version,
            "trained_at": parse_timestamp(meta["created_at_utc"]),
            "provenance": result.provenance,
            "metrics": result.model_dump(mode="json"),
            "confusion_matrix": meta["metrics"]["confusion_matrix"],
            "labels": result.confusion_matrix.labels,
        },
    )
    db.commit()
    return result


def display_model(db: Session, latest_predictions: Iterable[MlPrediction]) -> tuple[str | None, str | None]:
    """`(model_version, run_id)` to show in the UI: those of the newest-made ok prediction, else of the cached
    `model_info` row. The one fallback rule for the overview and the insights."""
    newest = max(latest_predictions, key=lambda p: p.created_at, default=None)
    model = repo.latest_model_info(db)
    return (
        (newest.model_version if newest else None) or (model.model_version if model else None),
        (newest.run_id if newest else None) or (model.run_id if model else None),
    )


def artifacts_location(artifacts_dir: Path) -> str:
    return display_path(artifacts_dir, artifacts_dir)
