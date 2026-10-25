"""Reading queries (list / data logs) and ingestion of new readings from sensor nodes.

Ingestion rules: the payload shape is validated by Pydantic (422 on failure); each reading is then checked with
the same local rules as the seed (`seed_service.validation_issues`: missing or out-of-range values are *flagged*,
never filled); a repeated `(plant_id, timestamp)` is reported as a duplicate and not inserted; rows are stored with
`source='ingest'` and `received_at=now`. A `session_id` that is not a known session is stored as NULL and the
reading is flagged (`unknown_session`), because sessions are only created from the dataset; an explicit `node_id`
that disagrees with the plant's node is kept as sent and flagged (`node_mismatch`) (DEV-31).
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.core.mappings import ensure_aware, node_for_plant, to_ist_iso
from app.db import repositories as repo
from app.db.models import SensorReading, utcnow
from app.integrations.ml_client import MLClient, MLError, MLUnavailable
from app.schemas.sensor import DatasetLabels, IngestItem, IngestReading, IngestResponse, SensorRecordOut
from app.services import prediction_service
from app.services.seed_service import SENSOR_RANGES, validation_issues

logger = logging.getLogger("app.sensors")


def to_record(reading: SensorReading) -> SensorRecordOut:
    return SensorRecordOut(
        record_id=reading.record_id,
        timestamp=to_ist_iso(reading.timestamp) or "",
        session_id=reading.session_id,
        node_id=reading.node_id,
        sampling_point=reading.sampling_point,
        plant_id=reading.plant_id,
        soil=reading.soil,
        temperature=reading.temperature,
        humidity=reading.humidity,
        light=reading.light,
        received_at=to_ist_iso(reading.received_at),
        accepted_at=to_ist_iso(reading.accepted_at),
        status=reading.status,
        validation_error=reading.validation_error,
        datasetLabels=DatasetLabels(
            stress_level=reading.label_stress_level,
            stress_type=reading.label_stress_type,
            is_synthetic_anomaly=reading.label_is_synthetic_anomaly,
        ),
    )


def list_records(db: Session, filters: repo.ReadingFilters, limit: int, offset: int) -> tuple[list[SensorRecordOut], int]:
    rows, total = repo.list_readings(db, filters, limit, offset)
    return [to_record(r) for r in rows], total


# --- ingestion --------------------------------------------------------------------------------
@dataclass
class IngestTarget:
    """A freshly inserted reading to predict. `explain` is True only for its plant's newest reading."""

    reading_id: int
    plant_id: int
    explain: bool


@dataclass
class IngestResult:
    response: IngestResponse
    targets: list[IngestTarget]


def ingest(db: Session, readings: list[IngestReading]) -> IngestResult:
    """Validate, dedupe and store the readings (one transaction). Does not call the ML service.

    A reading is a duplicate only when its `(plant_id, timestamp)` already exists: the pre-check catches the usual
    case and the insert (`ON CONFLICT (plant_id, timestamp) DO NOTHING`) catches a concurrent writer. Any other
    integrity failure propagates (a 500 envelope; the whole request is rolled back, nothing is committed).
    """
    now = utcnow()
    stamps = [ensure_aware(r.timestamp).astimezone(UTC) for r in readings]
    seen = repo.existing_reading_keys(db, [r.plant_id for r in readings], stamps)
    sessions = repo.known_session_ids(db)
    nodes = repo.plant_nodes(db)
    items: list[IngestItem] = []
    inserted: list[tuple[int, int, int | None]] = []  # (reading id, plant id, session id)
    for reading, stamp in zip(readings, stamps, strict=True):
        key = (reading.plant_id, stamp)
        iso = to_ist_iso(stamp) or ""
        if key in seen:
            items.append(IngestItem(record_id=None, plant_id=reading.plant_id, timestamp=iso, status="duplicate"))
            continue
        values = {c: getattr(reading, c) for c in SENSOR_RANGES}
        problems = [validation_issues(values, set())]
        session_id = reading.session_id
        if session_id is not None and session_id not in sessions:
            problems.append(f"unknown_session: {session_id}")
            session_id = None
        expected_node = nodes.get(reading.plant_id) or node_for_plant(reading.plant_id)
        if reading.node_id is not None and reading.node_id != expected_node:
            problems.append(f"node_mismatch: expected {expected_node}")
        issues = "; ".join(p for p in problems if p) or None
        status = "flagged" if issues else "accepted"
        new_id = repo.insert_reading_if_new(
            db,
            dict(
                timestamp=stamp, plant_id=reading.plant_id, node_id=reading.node_id or expected_node,  # as sent; flagged above
                sampling_point=reading.sampling_point, session_id=session_id, status=status, validation_error=issues,
                source="ingest", received_at=now, accepted_at=now, **values,
            ),
        )  # fmt: skip
        seen.add(key)
        if new_id is None:  # lost a race with a concurrent writer of the same key
            items.append(IngestItem(record_id=None, plant_id=reading.plant_id, timestamp=iso, status="duplicate"))
            continue
        inserted.append((new_id, reading.plant_id, session_id))
        items.append(IngestItem(record_id=f"R-{new_id}", plant_id=reading.plant_id, timestamp=iso, status=status))  # type: ignore[arg-type]
    touched = {sid for _, _, sid in inserted if sid is not None}
    if touched:
        repo.refresh_session_counts(db, sorted(touched))
    newest = {r.plant_id: r.id for r in repo.latest_reading_per_plant(db, sorted({pid for _, pid, _ in inserted}))}
    db.commit()
    response = IngestResponse(
        accepted=sum(i.status == "accepted" for i in items),
        flagged=sum(i.status == "flagged" for i in items),
        duplicates=sum(i.status == "duplicate" for i in items),
        items=items,
    )
    return IngestResult(response, [IngestTarget(rid, pid, newest.get(pid) == rid) for rid, pid, _ in inserted])


async def ingest_async(session_factory: sessionmaker, readings: list[IngestReading]) -> IngestResult:
    def work() -> IngestResult:
        with session_factory() as db:
            return ingest(db, readings)

    return await asyncio.to_thread(work)


async def predict_new_readings(
    session_factory: sessionmaker, ml: MLClient, settings: Settings, targets: list[IngestTarget], on_done=None
) -> None:
    """Best-effort background prediction of freshly ingested readings (everything needed is in `targets`, no
    per-reading lookups). Never raises: if the ML service is down the readings simply stay without a prediction
    (a later Run picks them up); the first outage stops the remaining ones."""
    try:
        health, reason = await ml.cached_health()
        if health is None or health.status != "ok" or not health.models_loaded:
            logger.info("ingest: ML service unavailable (%s); %d readings left unpredicted", reason or "models not loaded", len(targets))
            return
        gate = asyncio.Semaphore(max(1, settings.ML_CONCURRENCY))
        stop = False

        async def one(target: IngestTarget) -> None:
            nonlocal stop
            async with gate:
                if stop:
                    return
                try:
                    await prediction_service.predict_one(
                        session_factory, ml, settings, target.plant_id, target.reading_id, explain=target.explain
                    )
                except MLUnavailable as exc:
                    logger.info("ingest: ML service went away (%s); skipping the remaining readings", exc)
                    stop = True
                except MLError as exc:
                    logger.info("ingest: prediction for reading %s skipped (%s)", target.reading_id, exc)
                except Exception:
                    logger.exception("ingest: prediction for reading %s failed", target.reading_id)

        await asyncio.gather(*(one(t) for t in targets))
    except Exception:
        logger.exception("ingest: background prediction aborted")
    finally:
        if on_done is not None:
            on_done()
