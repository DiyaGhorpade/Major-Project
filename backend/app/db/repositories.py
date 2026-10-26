"""All database queries and writes. Services and routes never build SQL themselves."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import Select, String, and_, case, cast, delete, func, literal, or_, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from app.core.mappings import EXPERIMENTAL_GROUPS, PRIORITY_RANK
from app.db.models import (
    AcquisitionSession,
    AnomalyRecord,
    AppSettings,
    Device,
    MlPrediction,
    ModelInfo,
    PipelineRun,
    Plant,
    Recommendation,
    SensorReading,
)

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


def session_stats(db: Session, session_ids: Sequence[int] | None = None) -> dict[int, dict[str, Any]]:
    """Per session: received/flagged/rejected counts and first/last timestamp (only `session_ids` if given)."""
    stats: dict[int, dict[str, Any]] = {}
    where = [SensorReading.session_id.is_not(None)]
    if session_ids is not None:
        where.append(SensorReading.session_id.in_(list(session_ids)))
    rows = db.execute(
        select(
            SensorReading.session_id,
            func.count(),
            func.sum(func.iif(SensorReading.status == "flagged", 1, 0)),
            func.sum(func.iif(SensorReading.status == "rejected", 1, 0)),
            func.min(SensorReading.timestamp),
            func.max(SensorReading.timestamp),
        )
        .where(*where)
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


def latest_reading_per_plant(db: Session, plant_ids: Sequence[int] | None = None) -> list[SensorReading]:
    """The most recent reading of every plant (or of `plant_ids`), ordered by plant id."""
    where = [] if plant_ids is None else [SensorReading.plant_id.in_(list(plant_ids))]
    latest = (
        select(SensorReading.plant_id, func.max(SensorReading.timestamp).label("ts"))
        .where(*where)
        .group_by(SensorReading.plant_id)
        .subquery()
    )
    return list(
        db.scalars(
            select(SensorReading)
            .join(latest, (SensorReading.plant_id == latest.c.plant_id) & (SensorReading.timestamp == latest.c.ts))
            .order_by(SensorReading.plant_id)
        )
    )


def readings_before(db: Session, plant_id: int, before: Any, limit: int) -> list[SensorReading]:
    """Up to `limit` readings of one plant strictly before `before`, oldest first."""
    if limit <= 0:
        return []
    rows = list(
        db.scalars(
            select(SensorReading)
            .where(SensorReading.plant_id == plant_id, SensorReading.timestamp < before)
            .order_by(SensorReading.timestamp.desc(), SensorReading.id.desc())
            .limit(limit)
        )
    )
    rows.reverse()
    return rows


def readings_between(db: Session, plant_ids: Sequence[int], start: Any, end: Any) -> list[SensorReading]:
    """Readings of `plant_ids` with `start <= timestamp <= end`, oldest first (reference-plant context for the ML)."""
    if not plant_ids:
        return []
    return list(
        db.scalars(
            select(SensorReading)
            .where(SensorReading.plant_id.in_(list(plant_ids)), SensorReading.timestamp >= start, SensorReading.timestamp <= end)
            .order_by(SensorReading.timestamp, SensorReading.plant_id, SensorReading.id)
        )
    )


def get_reading(db: Session, reading_id: int) -> SensorReading | None:
    return db.get(SensorReading, reading_id)


def latest_reading_for_plant(db: Session, plant_id: int) -> SensorReading | None:
    rows = latest_reading_per_plant(db, [plant_id])
    return rows[0] if rows else None


def all_readings_ordered(db: Session) -> list[SensorReading]:
    """Every reading ordered by plant, then time (the pipeline's work order)."""
    return list(db.scalars(select(SensorReading).order_by(SensorReading.plant_id, SensorReading.timestamp, SensorReading.id)))


# --- predictions / anomalies ----------------------------------------------------------------
def predictions_by_reading(db: Session, reading_ids: Sequence[int]) -> dict[int, MlPrediction]:
    out: dict[int, MlPrediction] = {}
    for start in range(0, len(reading_ids), _INSERT_CHUNK):
        chunk = list(reading_ids[start : start + _INSERT_CHUNK])
        for p in db.scalars(select(MlPrediction).where(MlPrediction.reading_id.in_(chunk))):
            out[p.reading_id] = p
    return out


def all_predictions_light(db: Session) -> dict[int, tuple[str, str | None, bool]]:
    """`reading_id -> (status, run_id, has_explanation)` for every stored prediction."""
    rows = db.execute(
        select(MlPrediction.reading_id, MlPrediction.status, MlPrediction.run_id, MlPrediction.explanation_json.is_not(None))
    )
    return {rid: (status, run_id, bool(has_expl)) for rid, status, run_id, has_expl in rows}


def delete_anomalies_for_predictions(db: Session, prediction_ids: Sequence[int]) -> None:
    for start in range(0, len(prediction_ids), _INSERT_CHUNK):
        db.execute(delete(AnomalyRecord).where(AnomalyRecord.prediction_id.in_(list(prediction_ids[start : start + _INSERT_CHUNK]))))


def delete_all_predictions(db: Session) -> tuple[int, int]:
    """Remove every anomaly record and prediction. Returns `(predictions, anomalies)` deleted."""
    anomalies = db.execute(delete(AnomalyRecord)).rowcount
    predictions = db.execute(delete(MlPrediction)).rowcount
    return predictions, anomalies


def reset_plant_statuses(db: Session) -> int:
    """Set every plant's status to `unmeasured` (their predictions are gone). The caller commits."""
    return db.execute(update(Plant).values(status="unmeasured")).rowcount


# --- recommendations ------------------------------------------------------------------------
_PRIORITY_RANK = case(PRIORITY_RANK, value=Recommendation.priority, else_=len(PRIORITY_RANK))


def latest_ok_reading(db: Session, plant_id: int) -> SensorReading | None:
    """The plant's newest reading that has an `ok` prediction."""
    return db.scalars(
        select(SensorReading)
        .join(MlPrediction, MlPrediction.reading_id == SensorReading.id)
        .where(SensorReading.plant_id == plant_id, MlPrediction.status == "ok")
        .order_by(SensorReading.timestamp.desc(), SensorReading.id.desc())
        .limit(1)
    ).first()


def readings_with_predictions_upto(
    db: Session, plant_id: int, upto: SensorReading, limit: int
) -> list[tuple[SensorReading, MlPrediction | None]]:
    """Up to `limit` readings of the plant ending at `upto` (newest first, `upto` included) with their prediction row."""
    rows = db.execute(
        select(SensorReading, MlPrediction)
        .outerjoin(MlPrediction, MlPrediction.reading_id == SensorReading.id)
        .where(SensorReading.plant_id == plant_id, SensorReading.timestamp <= upto.timestamp)
        .order_by(SensorReading.timestamp.desc(), SensorReading.id.desc())
        .limit(limit)
    )
    return [(reading, pred) for reading, pred in rows]


def get_prediction_by_reading(db: Session, reading_id: int) -> MlPrediction | None:
    return db.scalars(select(MlPrediction).where(MlPrediction.reading_id == reading_id)).first()


def list_plants(db: Session, limit: int | None = None, offset: int = 0) -> list[Plant]:
    stmt = select(Plant).order_by(Plant.id).offset(offset)
    return list(db.scalars(stmt if limit is None else stmt.limit(limit)))


def plant_nodes(db: Session) -> dict[int, str]:
    """`plant_id -> node_id` as stored (the node each plant belongs to)."""
    return dict(db.execute(select(Plant.id, Plant.node_id)).all())


def recommendations_for_plant(db: Session, plant_id: int) -> list[Recommendation]:
    return list(db.scalars(select(Recommendation).where(Recommendation.plant_id == plant_id).order_by(Recommendation.id)))


def other_active_recommendation(db: Session, rec: Recommendation) -> Recommendation | None:
    """Another active (`new`/`in_progress`) recommendation for the same plant and category, if any."""
    return db.scalars(
        select(Recommendation)
        .where(
            Recommendation.plant_id == rec.plant_id,
            Recommendation.category == rec.category,
            Recommendation.id != rec.id,
            Recommendation.status.in_(("new", "in_progress")),
        )
        .order_by(Recommendation.id.desc())
        .limit(1)
    ).first()


def recommendations_by_plant(db: Session) -> dict[int, list[Recommendation]]:
    out: dict[int, list[Recommendation]] = {}
    for rec in db.scalars(select(Recommendation).order_by(Recommendation.id)):
        out.setdefault(rec.plant_id, []).append(rec)
    return out


def delete_generated_recommendations(db: Session) -> int:
    """Delete every recommendation the user has not touched. Returns how many were removed."""
    return db.execute(delete(Recommendation).where(Recommendation.user_modified.is_(False))).rowcount


def detach_recommendations_from_predictions(db: Session) -> int:
    """Clear `prediction_id` on the remaining recommendations (their predictions are being deleted)."""
    return db.execute(
        update(Recommendation).where(Recommendation.prediction_id.is_not(None)).values(prediction_id=None)
    ).rowcount


def count_recommendations(db: Session) -> int:
    return _count(db, select(Recommendation.id))


def list_recommendations(
    db: Session,
    *,
    plant_id: int | None = None,
    status: str | None = None,
    priority: str | None = None,
    category: str | None = None,
    source: str | None = None,
    q: str | None = None,
    include_superseded: bool = False,
    active_only: bool = False,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[tuple[Recommendation, Plant, SensorReading]], int]:
    """Filtered page of `(recommendation, plant, source reading)` ordered high -> low priority, then
    newest first; plus the total number of matches. `q` is a case-insensitive substring over the title, action,
    reason, plant name and plant id."""
    where = []
    for column, value in (
        (Recommendation.plant_id, plant_id),
        (Recommendation.status, status),
        (Recommendation.priority, priority),
        (Recommendation.category, category),
        (Recommendation.source, source),
    ):
        if value is not None:
            where.append(column == value)
    if not include_superseded and status != "superseded":
        where.append(Recommendation.status != "superseded")
    if active_only:
        where.append(Recommendation.status.in_(("new", "in_progress")))
    if q and q.strip():
        like = _like_pattern(q)
        where.append(
            or_(
                func.lower(Recommendation.title).like(like, escape="\\"),
                func.lower(Recommendation.simple_action).like(like, escape="\\"),
                func.lower(Recommendation.reason).like(like, escape="\\"),
                func.lower(Plant.name).like(like, escape="\\"),
                cast(Recommendation.plant_id, String).like(like, escape="\\"),
            )
        )
    total = _count(db, select(Recommendation.id).join(Plant, Plant.id == Recommendation.plant_id).where(*where))
    rows = db.execute(
        select(Recommendation, Plant, SensorReading)
        .join(Plant, Plant.id == Recommendation.plant_id)
        .join(SensorReading, SensorReading.id == Recommendation.source_reading_id)
        .where(*where)
        .order_by(_PRIORITY_RANK, Recommendation.created_at.desc(), Recommendation.id.desc())
        .limit(limit)
        .offset(offset)
    )
    return [(rec, plant, reading) for rec, plant, reading in rows], total


def get_recommendation(db: Session, rec_id: int) -> Recommendation | None:
    return db.get(Recommendation, rec_id)


def get_recommendation_row(db: Session, rec_id: int) -> tuple[Recommendation, Plant, SensorReading] | None:
    row = db.execute(
        select(Recommendation, Plant, SensorReading)
        .join(Plant, Plant.id == Recommendation.plant_id)
        .join(SensorReading, SensorReading.id == Recommendation.source_reading_id)
        .where(Recommendation.id == rec_id)
    ).first()
    return None if row is None else (row[0], row[1], row[2])


# --- pipeline runs --------------------------------------------------------------------------
def get_run(db: Session, run_id: int) -> PipelineRun | None:
    return db.get(PipelineRun, run_id)


def active_run(db: Session) -> PipelineRun | None:
    return db.scalars(
        select(PipelineRun).where(PipelineRun.status.in_(("queued", "running"))).order_by(PipelineRun.id.desc()).limit(1)
    ).first()


def latest_run(db: Session) -> PipelineRun | None:
    return db.scalars(select(PipelineRun).order_by(PipelineRun.id.desc()).limit(1)).first()


def fail_stale_runs(db: Session, message: str, finished_at: Any) -> int:
    """Mark every queued/running run failed (used at startup). Returns the number changed."""
    runs = list(db.scalars(select(PipelineRun).where(PipelineRun.status.in_(("queued", "running")))))
    for run in runs:
        run.status, run.message, run.finished_at = "failed", message, finished_at
    db.commit()
    return len(runs)


# --- model info -----------------------------------------------------------------------------
def latest_model_info(db: Session) -> ModelInfo | None:
    return db.scalars(select(ModelInfo).order_by(ModelInfo.cached_at.desc(), ModelInfo.id.desc()).limit(1)).first()


def get_model_info_by_run(db: Session, run_id: str) -> ModelInfo | None:
    return db.scalars(select(ModelInfo).where(ModelInfo.run_id == run_id)).first()


def upsert_model_info(db: Session, values: dict[str, Any]) -> ModelInfo:
    """Insert or refresh the model_info row for `values['run_id']` (unique)."""
    row = db.scalars(select(ModelInfo).where(ModelInfo.run_id == values["run_id"])).first()
    if row is None:
        row = ModelInfo(**values)
        db.add(row)
    else:
        for key, value in values.items():
            setattr(row, key, value)
    db.flush()
    return row


# --- Phase 3 read queries ---------------------------------------------------------------------
@dataclass
class ReadingFilters:
    """Filters shared by `/api/sensors/readings` and `/api/data-logs`. Times are tz-aware datetimes."""

    plant_id: int | None = None
    cluster: str | None = None  # "Cluster 1".."Cluster 4" (experimental group)
    node_id: str | None = None
    status: str | None = None
    session_id: int | None = None
    start: datetime | None = None
    end: datetime | None = None
    q: str | None = None
    order: str = "desc"


_IST_TEXT = func.strftime("%Y-%m-%dT%H:%M:%S", SensorReading.timestamp, "+330 minutes")


def _like_pattern(text: str) -> str:
    """A case-insensitive substring pattern for `LIKE ... ESCAPE '\\'` (wildcards in `text` are literal)."""
    needle = text.strip().lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{needle}%"


def _reading_where(f: ReadingFilters) -> list[Any]:
    where: list[Any] = []
    if f.plant_id is not None:
        where.append(SensorReading.plant_id == f.plant_id)
    if f.cluster is not None:
        group = next((g for g in EXPERIMENTAL_GROUPS if g.cluster == f.cluster), None)
        where.append(SensorReading.plant_id.in_(list(group.plant_ids) if group else []))
    if f.node_id is not None:
        where.append(SensorReading.node_id == f.node_id)
    if f.status is not None:
        where.append(SensorReading.status == f.status)
    if f.session_id is not None:
        where.append(SensorReading.session_id == f.session_id)
    if f.start is not None:
        where.append(SensorReading.timestamp >= f.start)
    if f.end is not None:
        where.append(SensorReading.timestamp <= f.end)
    if f.q:
        like = _like_pattern(f.q)
        where.append(
            or_(
                cast(SensorReading.plant_id, String).like(like, escape="\\"),
                cast(SensorReading.session_id, String).like(like, escape="\\"),
                cast(SensorReading.node_id, String).like(like, escape="\\"),
                cast(SensorReading.sampling_point, String).like(like, escape="\\"),
                _IST_TEXT.like(like, escape="\\"),
                func.lower(literal("R-") + cast(SensorReading.id, String)).like(like, escape="\\"),
            )
        )
    return where


def _reading_order(f: ReadingFilters) -> list[Any]:
    if f.order == "asc":
        return [SensorReading.timestamp.asc(), SensorReading.id.asc()]
    return [SensorReading.timestamp.desc(), SensorReading.id.desc()]


def list_readings(db: Session, f: ReadingFilters, limit: int, offset: int) -> tuple[list[SensorReading], int]:
    """A page of readings matching the filters, plus the total number of matches."""
    where = _reading_where(f)
    total = db.scalar(select(func.count()).select_from(SensorReading).where(*where)) or 0
    rows = list(db.scalars(select(SensorReading).where(*where).order_by(*_reading_order(f)).limit(limit).offset(offset)))
    return rows, total


def iter_readings_with_predictions(db: Session, f: ReadingFilters) -> Iterator[tuple[SensorReading, MlPrediction | None]]:
    """Every reading matching the filters with its prediction (or None), streamed in batches."""
    stmt = (
        select(SensorReading, MlPrediction)
        .outerjoin(MlPrediction, MlPrediction.reading_id == SensorReading.id)
        .where(*_reading_where(f))
        .order_by(*_reading_order(f))
    )
    for reading, pred in db.execute(stmt).yield_per(500):
        yield reading, pred


def insert_reading_if_new(db: Session, values: dict[str, Any]) -> int | None:
    """Insert one reading; returns its id, or None if `(plant_id, timestamp)` already exists (no row written).
    Only that conflict is swallowed: every other constraint failure raises `IntegrityError`."""
    return db.execute(
        sqlite_insert(SensorReading)
        .values(**values)
        .on_conflict_do_nothing(index_elements=["plant_id", "timestamp"])
        .returning(SensorReading.id)
    ).scalar()


def existing_reading_keys(db: Session, plant_ids: Sequence[int], timestamps: Sequence[datetime]) -> set[tuple[int, datetime]]:
    """`(plant_id, tz-aware UTC timestamp)` pairs among the given ones that are already stored."""
    found: set[tuple[int, datetime]] = set()
    stamps = list(timestamps)
    for start in range(0, len(stamps), _INSERT_CHUNK):
        chunk = stamps[start : start + _INSERT_CHUNK]
        for pid, ts in db.execute(
            select(SensorReading.plant_id, SensorReading.timestamp).where(
                SensorReading.plant_id.in_(list(set(plant_ids))), SensorReading.timestamp.in_(chunk)
            )
        ):
            found.add((pid, ts))
    return found


def known_session_ids(db: Session) -> set[int]:
    return set(db.scalars(select(AcquisitionSession.session_id)))


def refresh_session_counts(db: Session, session_ids: Sequence[int]) -> None:
    """Recompute received/flagged/rejected counts and the end time of the given sessions from the readings."""
    stats = session_stats(db, session_ids)
    for session in db.scalars(select(AcquisitionSession).where(AcquisitionSession.session_id.in_(list(session_ids)))):
        s = stats.get(session.session_id)
        if s is None:
            continue
        session.received_records, session.flagged_records, session.rejected_records = s["received"], s["flagged"], s["rejected"]
        session.ended_at = s["last"]


def get_plant(db: Session, plant_id: int) -> Plant | None:
    return db.get(Plant, plant_id)


def latest_ok_prediction_per_plant(db: Session, plant_ids: Sequence[int] | None = None) -> dict[int, MlPrediction]:
    """Each plant's (or each of `plant_ids`'s) most recent `ok` prediction, by reading time."""
    where = [MlPrediction.status == "ok"]
    if plant_ids is not None:
        where.append(MlPrediction.plant_id.in_(list(plant_ids)))
    latest = (
        select(MlPrediction.plant_id, func.max(MlPrediction.reading_timestamp).label("ts"))
        .where(*where)
        .group_by(MlPrediction.plant_id)
        .subquery()
    )
    rows = db.scalars(
        select(MlPrediction)
        .join(latest, and_(MlPrediction.plant_id == latest.c.plant_id, MlPrediction.reading_timestamp == latest.c.ts))
        .where(MlPrediction.status == "ok")
    )
    return {p.plant_id: p for p in rows}


def latest_explained_prediction(db: Session, plant_id: int) -> MlPrediction | None:
    """The plant's newest `ok` prediction that carries a stored local explanation."""
    return db.scalars(
        select(MlPrediction)
        .where(MlPrediction.plant_id == plant_id, MlPrediction.status == "ok", MlPrediction.explanation_json.is_not(None))
        .order_by(MlPrediction.reading_timestamp.desc())
        .limit(1)
    ).first()


def explanation_rows(db: Session, limit: int = 500) -> list[tuple[int, dict[str, Any]]]:
    """`(plant_id, explanation)` of the newest stored explanations (at most `limit`)."""
    rows = db.execute(
        select(MlPrediction.plant_id, MlPrediction.explanation_json)
        .where(MlPrediction.status == "ok", MlPrediction.explanation_json.is_not(None))
        .order_by(MlPrediction.reading_timestamp.desc())
        .limit(limit)
    )
    return [(pid, expl) for pid, expl in rows if isinstance(expl, dict)]


def readings_series(db: Session, plant_id: int, limit: int) -> list[SensorReading]:
    """The plant's last `limit` readings, oldest first."""
    rows = list(
        db.scalars(
            select(SensorReading)
            .where(SensorReading.plant_id == plant_id)
            .order_by(SensorReading.timestamp.desc(), SensorReading.id.desc())
            .limit(limit)
        )
    )
    rows.reverse()
    return rows


def list_predictions(
    db: Session,
    *,
    plant_id: int | None = None,
    status: str | None = None,
    severity: str | None = None,
    is_anomaly: bool | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[MlPrediction], int]:
    where: list[Any] = []
    for column, value in (
        (MlPrediction.plant_id, plant_id),
        (MlPrediction.status, status),
        (MlPrediction.severity, severity),
        (MlPrediction.is_anomaly, is_anomaly),
    ):
        if value is not None:
            where.append(column == value)
    if start is not None:
        where.append(MlPrediction.reading_timestamp >= start)
    if end is not None:
        where.append(MlPrediction.reading_timestamp <= end)
    total = db.scalar(select(func.count()).select_from(MlPrediction).where(*where)) or 0
    rows = db.scalars(
        select(MlPrediction)
        .where(*where)
        .order_by(MlPrediction.reading_timestamp.desc(), MlPrediction.id.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(rows), total


def list_anomalies(
    db: Session,
    *,
    plant_id: int | None,
    limit: int,
    offset: int,
    sort: str = "recent",
    min_score: float | None = None,
) -> tuple[list[tuple[AnomalyRecord, Plant, SensorReading]], int]:
    where = [AnomalyRecord.is_anomaly.is_(True)]
    if plant_id is not None:
        where.append(AnomalyRecord.plant_id == plant_id)
    if min_score is not None:
        where.append(AnomalyRecord.score >= min_score)
    total = db.scalar(select(func.count()).select_from(AnomalyRecord).where(*where)) or 0
    order = (
        (AnomalyRecord.score.desc(), AnomalyRecord.reading_timestamp.desc(), AnomalyRecord.id.desc())
        if sort == "score"
        else (AnomalyRecord.reading_timestamp.desc(), AnomalyRecord.id.desc())
    )
    rows = db.execute(
        select(AnomalyRecord, Plant, SensorReading)
        .join(Plant, Plant.id == AnomalyRecord.plant_id)
        .join(SensorReading, SensorReading.id == AnomalyRecord.reading_id)
        .where(*where)
        .order_by(*order)
        .limit(limit)
        .offset(offset)
    )
    return [(a, p, r) for a, p, r in rows], total


def anomaly_summary_totals(db: Session) -> tuple[int, int, datetime | None]:
    """`(flagged records, plants with a flag, newest flagged reading timestamp)`."""
    flagged, plants, latest = db.execute(
        select(func.count(), func.count(func.distinct(AnomalyRecord.plant_id)), func.max(AnomalyRecord.reading_timestamp)).where(
            AnomalyRecord.is_anomaly.is_(True)
        )
    ).one()
    return int(flagged or 0), int(plants or 0), latest


def anomaly_counts_by_plant(db: Session) -> list[tuple[Plant, int, float, datetime]]:
    """One row per plant with at least one flag: `(plant, flags, max score, last flagged reading timestamp)`."""
    rows = db.execute(
        select(Plant, func.count(), func.max(AnomalyRecord.score), func.max(AnomalyRecord.reading_timestamp))
        .join(AnomalyRecord, AnomalyRecord.plant_id == Plant.id)
        .where(AnomalyRecord.is_anomaly.is_(True))
        .group_by(Plant.id)
        .order_by(func.count().desc(), Plant.id)
    )
    return [(p, int(n), float(score), ts) for p, n, score, ts in rows]


def count_anomalies(db: Session) -> int:
    return db.scalar(select(func.count()).select_from(AnomalyRecord).where(AnomalyRecord.is_anomaly.is_(True))) or 0


def prediction_status_counts(db: Session) -> dict[str, int]:
    return dict(db.execute(select(MlPrediction.status, func.count()).group_by(MlPrediction.status)).all())


def ok_prediction_counts(db: Session, column: Any) -> dict[Any, int]:
    """Counts of `ok` predictions grouped by one MlPrediction column (severity, stress_type_guess, cluster_id)."""
    return dict(db.execute(select(column, func.count()).where(MlPrediction.status == "ok").group_by(column)).all())


def plant_status_counts(db: Session) -> dict[str, int]:
    return dict(db.execute(select(Plant.status, func.count()).group_by(Plant.status)).all())


def session_node_counts(db: Session) -> dict[tuple[int, str], int]:
    rows = db.execute(
        select(SensorReading.session_id, SensorReading.node_id, func.count())
        .where(SensorReading.session_id.is_not(None))
        .group_by(SensorReading.session_id, SensorReading.node_id)
    )
    return {(sid, node): n for sid, node, n in rows}


def list_sessions(db: Session, limit: int = 500, offset: int = 0) -> list[AcquisitionSession]:
    return list(db.scalars(select(AcquisitionSession).order_by(AcquisitionSession.session_id.desc()).limit(limit).offset(offset)))


def list_devices(db: Session, limit: int = 500, offset: int = 0) -> list[Device]:
    return list(db.scalars(select(Device).order_by(Device.type, Device.id).limit(limit).offset(offset)))


def readings_per_node(db: Session) -> dict[str, int]:
    return dict(db.execute(select(SensorReading.node_id, func.count()).group_by(SensorReading.node_id)).all())


def plant_ids_per_node(db: Session) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for pid, node in db.execute(select(Plant.id, Plant.node_id).order_by(Plant.id)):
        out.setdefault(node, []).append(pid)
    return out


def get_settings_row(db: Session) -> AppSettings | None:
    return db.get(AppSettings, 1)


def state_token(db: Session) -> tuple:
    """A cheap fingerprint of everything the cached aggregates depend on (readings, predictions, anomalies,
    recommendations, plant statuses and the latest pipeline run). Changes whenever any of them changes."""
    readings = db.execute(
        select(func.count(), func.coalesce(func.max(SensorReading.id), 0), func.sum(func.iif(SensorReading.status == "flagged", 1, 0)))
    ).one()
    preds = db.execute(
        select(
            func.count(),
            func.coalesce(func.max(MlPrediction.id), 0),
            func.sum(func.iif(MlPrediction.status == "ok", 1, 0)),
            func.max(MlPrediction.created_at),
            func.max(MlPrediction.updated_at),
        )
    ).one()
    anomalies = db.execute(select(func.count(), func.coalesce(func.max(AnomalyRecord.id), 0))).one()
    recs = db.execute(
        select(func.count(), func.coalesce(func.max(Recommendation.id), 0), func.max(Recommendation.updated_at))
    ).one()
    run = latest_run(db)
    statuses = tuple(db.scalars(select(Plant.status).order_by(Plant.id)))
    return (
        tuple(readings), tuple(preds), tuple(anomalies), tuple(recs),
        (run.id, run.status, run.finished_at, run.processed) if run else None,
        statuses,
    )  # fmt: skip
