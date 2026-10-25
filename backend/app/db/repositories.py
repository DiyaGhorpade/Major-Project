"""All database queries and writes. Services and routes never build SQL themselves."""

from collections.abc import Sequence
from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from app.db.models import AcquisitionSession, AppSettings, Device, Plant, SensorReading

# Keep (rows x columns) below SQLite's bound-variable limit (32766 on modern builds).
_INSERT_CHUNK = 500


def _upsert(db: Session, model: type, rows: Sequence[dict[str, Any]], key: str, update_cols: Sequence[str]) -> None:
    if not rows:
        return
    stmt = sqlite_insert(model).values(list(rows))
    stmt = stmt.on_conflict_do_update(
        index_elements=[key], set_={c: stmt.excluded[c] for c in update_cols}
    )
    db.execute(stmt)


# --- seeding writes -------------------------------------------------------------------------
def bulk_insert_readings(db: Session, rows: Sequence[dict[str, Any]]) -> int:
    """Insert readings, silently skipping existing `(plant_id, timestamp)` pairs. Returns rows inserted."""
    inserted = 0
    for start in range(0, len(rows), _INSERT_CHUNK):
        chunk = list(rows[start : start + _INSERT_CHUNK])
        result = db.execute(
            sqlite_insert(SensorReading)
            .values(chunk)
            .on_conflict_do_nothing(index_elements=["plant_id", "timestamp"])
        )
        inserted += max(result.rowcount, 0)
    return inserted


def upsert_plants(db: Session, rows: Sequence[dict[str, Any]]) -> None:
    """Insert plants; on conflict refresh catalog/derived fields but keep the computed `status`."""
    cols = [c for c in (rows[0] if rows else {}) if c not in ("id", "status", "created_at")]
    _upsert(db, Plant, rows, "id", cols)


def upsert_devices(db: Session, rows: Sequence[dict[str, Any]]) -> None:
    """Insert devices; on conflict refresh only data-derived fields (never overwrite a live status)."""
    _upsert(db, Device, rows, "id", ["last_transmission", "channels"])


def upsert_sessions(db: Session, rows: Sequence[dict[str, Any]]) -> None:
    cols = [c for c in (rows[0] if rows else {}) if c not in ("session_id", "trigger_type")]
    _upsert(db, AcquisitionSession, rows, "session_id", cols)


def insert_default_settings(db: Session, values: dict[str, Any]) -> bool:
    """Create the single settings row (id=1) if absent. Returns True when created."""
    result = db.execute(
        sqlite_insert(AppSettings).values(id=1, **values).on_conflict_do_nothing(index_elements=["id"])
    )
    return result.rowcount > 0


# --- counts and aggregates ------------------------------------------------------------------
def _count(db: Session, stmt: Select) -> int:
    return db.scalar(select(func.count()).select_from(stmt.subquery())) or 0


def count_readings(db: Session) -> int:
    return _count(db, select(SensorReading.id))


def count_flagged_readings(db: Session) -> int:
    return _count(db, select(SensorReading.id).where(SensorReading.status == "flagged"))


def count_plants(db: Session) -> int:
    return _count(db, select(Plant.id))


def count_devices(db: Session) -> int:
    return _count(db, select(Device.id))


def count_sessions(db: Session) -> int:
    return _count(db, select(AcquisitionSession.id))


def db_summary(db: Session) -> tuple[bool, int, int]:
    """Connectivity probe: `(ok, readings, plants)`; `(False, 0, 0)` if the database cannot be queried."""
    try:
        db.execute(select(1))
        return True, count_readings(db), count_plants(db)
    except Exception:
        return False, 0, 0


def session_stats(db: Session) -> dict[int, dict[str, Any]]:
    """Per session: received/flagged/rejected counts, first/last timestamp, and per-node received."""
    stats: dict[int, dict[str, Any]] = {}
    rows = db.execute(
        select(
            SensorReading.session_id,
            func.count(),
            func.sum(func.iif(SensorReading.status == "flagged", 1, 0)),
            func.sum(func.iif(SensorReading.status == "rejected", 1, 0)),
            func.min(SensorReading.timestamp),
            func.max(SensorReading.timestamp),
        )
        .where(SensorReading.session_id.is_not(None))
        .group_by(SensorReading.session_id)
    )
    for sid, received, flagged, rejected, first, last in rows:
        stats[sid] = {
            "received": received,
            "flagged": flagged or 0,
            "rejected": rejected or 0,
            "first": first,
            "last": last,
        }
    return stats


def node_stats(db: Session) -> dict[str, dict[str, Any]]:
    """Per node: last reading timestamp and number of distinct sampling points seen."""
    rows = db.execute(
        select(
            SensorReading.node_id,
            func.max(SensorReading.timestamp),
            func.count(func.distinct(SensorReading.sampling_point)),
        ).group_by(SensorReading.node_id)
    )
    return {node: {"last": last, "channels": channels} for node, last, channels in rows}


# --- reads ----------------------------------------------------------------------------------
def readings_for_plant_ordered(db: Session, plant_id: int) -> list[SensorReading]:
    """All readings of one plant, oldest first."""
    return list(
        db.scalars(
            select(SensorReading)
            .where(SensorReading.plant_id == plant_id)
            .order_by(SensorReading.timestamp, SensorReading.id)
        )
    )


def latest_reading_per_plant(db: Session) -> list[SensorReading]:
    """The most recent reading of every plant, ordered by plant id."""
    latest = (
        select(SensorReading.plant_id, func.max(SensorReading.timestamp).label("ts"))
        .group_by(SensorReading.plant_id)
        .subquery()
    )
    return list(
        db.scalars(
            select(SensorReading)
            .join(
                latest,
                (SensorReading.plant_id == latest.c.plant_id) & (SensorReading.timestamp == latest.c.ts),
            )
            .order_by(SensorReading.plant_id)
        )
    )
