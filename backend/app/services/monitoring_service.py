"""Overview and plant-detail aggregates for the dashboard (camelCase), built only from stored data."""

from collections import Counter
from collections.abc import Callable
from pathlib import Path

from sqlalchemy.orm import Session, sessionmaker

from app.core.cache import ResponseCache
from app.core.mappings import EXPERIMENTAL_GROUPS, to_ist_iso
from app.core.provenance import DISCLAIMER, PROVENANCE
from app.db import repositories as repo
from app.schemas.dashboard import (
    Alert,
    ClusterSummary,
    DevicesSummary,
    OverviewCore,
    OverviewOut,
    PlantMonitoringOut,
    PredictionHistoryPoint,
    PredictionTotals,
    SensorAverages,
    SeriesPoint,
    Totals,
)
from app.schemas.pipeline import PipelineStatus
from app.services import device_service, model_service, plant_service, prediction_service, recommendation_service, session_service
from app.services.ai_service import ANOMALY_TEXT, SEVERITIES
from app.services.sensor_values import valid_values

PLANT_STATUSES = ("healthy", "attention", "critical", "stale", "unmeasured")
MAX_ALERTS = 10
TOP_RECOMMENDATIONS = 5
SENSORS = ("soil", "temperature", "humidity", "light")


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def _cluster_summaries(db: Session) -> list[ClusterSummary]:
    readings = {r.plant_id: r for r in repo.latest_reading_per_plant(db)}
    preds = repo.latest_ok_prediction_per_plant(db)
    out = []
    for group in EXPERIMENTAL_GROUPS:
        ids = list(group.plant_ids)
        latest = [readings[i] for i in ids if i in readings]
        sev = Counter(preds[i].severity for i in ids if i in preds)
        guesses = Counter(preds[i].stress_type_guess for i in ids if i in preds and preds[i].stress_type_guess)
        out.append(
            ClusterSummary(
                cluster=group.cluster,
                cluster_label=group.label,
                plant_ids=ids,
                plants=len(ids),
                averages=SensorAverages(**{s: _mean(valid_values(latest, s)) for s in SENSORS}),
                severity_counts={**{s: sev.get(s, 0) for s in SEVERITIES}, "unmeasured": sum(1 for i in ids if i not in preds)},
                top_condition=guesses.most_common(1)[0][0] if guesses else None,
            )
        )
    return out


def _recent_alerts(db: Session) -> list[Alert]:
    alerts: list[Alert] = []
    anomalies, _ = repo.list_anomalies(db, plant_id=None, limit=MAX_ALERTS, offset=0)
    for a, plant, reading in anomalies:
        alerts.append(
            Alert(
                kind="anomaly",
                plant_id=plant.id,
                plant_name=plant.name,
                timestamp=to_ist_iso(a.reading_timestamp) or "",
                message=f"{ANOMALY_TEXT} (anomaly score {a.score:.2f}, threshold {a.threshold:g})",
                record_id=reading.record_id,
            )
        )
    recs, _ = repo.list_recommendations(db, priority="high", active_only=True, limit=MAX_ALERTS)
    for rec, plant, reading in recs:
        alerts.append(
            Alert(
                kind="recommendation",
                plant_id=plant.id,
                plant_name=plant.name,
                timestamp=to_ist_iso(reading.timestamp) or "",
                message=f"{rec.title}: {rec.simple_action}",
                severity=rec.priority,
                record_id=reading.record_id,
                recommendation_id=f"rec-{rec.id}",
            )
        )
    alerts.sort(key=lambda a: a.timestamp, reverse=True)  # same offset (+05:30) everywhere, so text order is time order
    return alerts[:MAX_ALERTS]


def overview(db: Session, config_path: Path) -> OverviewCore:
    status_counts = repo.prediction_status_counts(db)
    latest_preds = repo.latest_ok_prediction_per_plant(db)
    plants = repo.count_plants(db)
    sev = Counter(p.severity for p in latest_preds.values())
    plant_status = repo.plant_status_counts(db)
    devices, _ = device_service.list_devices(db)
    transmissions = [d.last_transmission for d in devices if d.last_transmission]
    model_version, _ = model_service.display_model(db, latest_preds.values())
    top, _ = recommendation_service.list_out(db, config_path, active_only=True, limit=TOP_RECOMMENDATIONS)
    return OverviewCore(
        totals=Totals(
            plants=plants,
            readings=repo.count_readings(db),
            flagged_readings=repo.count_flagged_readings(db),
            predictions=PredictionTotals(
                ok=status_counts.get("ok", 0),
                unavailable=status_counts.get("unavailable", 0),
                rejected=status_counts.get("rejected", 0),
                total=sum(status_counts.values()),
            ),
            anomalies=repo.count_anomalies(db),
        ),
        plant_status_counts={s: plant_status.get(s, 0) for s in PLANT_STATUSES},
        severity_counts={**{s: sev.get(s, 0) for s in SEVERITIES}, "unmeasured": max(0, plants - len(latest_preds))},
        cluster_summaries=_cluster_summaries(db),
        recent_alerts=_recent_alerts(db),
        top_recommendations=top,
        latest_session=session_service.latest_session(db),
        devices_summary=DevicesSummary(
            total=len(devices),
            online=sum(d.status == "online" for d in devices),
            offline=sum(d.status == "offline" for d in devices),
            unknown=sum(d.status == "unknown" for d in devices),
            last_transmission=max(transmissions) if transmissions else None,
        ),
        model_version=model_version,
        provenance=PROVENANCE,
        disclaimer=DISCLAIMER,
    )


def plant_monitoring(db: Session, config_path: Path, plant_id: int, limit: int) -> PlantMonitoringOut | None:
    plant = plant_service.get_plant(db, plant_id)
    if plant is None:
        return None
    readings = repo.readings_series(db, plant_id, limit)
    preds = repo.predictions_by_reading(db, [r.id for r in readings])
    history = []
    for r in readings:
        p = preds.get(r.id)
        if p is None:
            continue
        history.append(
            PredictionHistoryPoint(
                timestamp=to_ist_iso(r.timestamp) or "",
                record_id=r.record_id,
                status=p.status,
                severity=p.severity,
                probabilities=p.probabilities,
                confidence=p.confidence,
                condition=p.stress_type_guess,
                isAnomaly=p.is_anomaly,
                anomalyScore=p.anomaly_score,
                envState=prediction_service.env_state(p),
            )
        )
    recs, _ = recommendation_service.list_out(db, config_path, plant_id=plant_id, active_only=True, limit=50)
    return PlantMonitoringOut(
        plant=plant,
        series=[
            SeriesPoint(timestamp=to_ist_iso(r.timestamp) or "", soil=r.soil, temperature=r.temperature, humidity=r.humidity, light=r.light)
            for r in readings
        ],
        prediction_history=history,
        recommendations=recs,
    )


def overview_cached(
    session_factory: sessionmaker, cache: ResponseCache, config_path: Path, pipeline_status: Callable[[Session], PipelineStatus]
) -> OverviewOut:
    """The overview: the heavy part is cached until the data changes (`repositories.state_token`); the
    pipeline status is always live (`pipeline_status(db)`)."""
    with session_factory() as db:
        core: OverviewCore = cache.get_or_compute("overview", repo.state_token(db), lambda: overview(db, config_path))
        status = pipeline_status(db)
    return OverviewOut(**{name: getattr(core, name) for name in OverviewCore.model_fields}, pipeline=status)
