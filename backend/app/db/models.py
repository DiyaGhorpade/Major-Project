"""ORM models for all tables (master prompt section 5 + BACKEND_HANDOFF section 5).

Dataset labels (`label_*`) are kept apart from ML output (`ml_predictions`). Missing sensor
values stay NULL. All timestamps use `UTCDateTime`: stored as naive UTC (sortable as text in
SQLite) and returned tz-aware; format for the API with `mappings.to_ist_iso`.
`SensorReading.record_id` is derived (`R-<id>`), not stored.
"""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.core.mappings import ensure_aware
from app.db.database import Base


class UTCDateTime(TypeDecorator):
    """Timezone-aware datetime stored as naive UTC; naive inputs are assumed to be IST."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect) -> datetime | None:
        return None if value is None else ensure_aware(value).astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect) -> datetime | None:
        return None if value is None else value.replace(tzinfo=UTC)


def utcnow() -> datetime:
    return datetime.now(UTC)


class Plant(Base):
    __tablename__ = "plants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    name: Mapped[str] = mapped_column(String(120))
    species: Mapped[str] = mapped_column(String(80))
    variety: Mapped[str] = mapped_column(String(80))
    cluster: Mapped[str] = mapped_column(String(20))  # experimental group, "Cluster 1".."Cluster 4"
    cluster_label: Mapped[str] = mapped_column(String(80))
    stress_type: Mapped[str] = mapped_column(String(40))  # designed treatment of the group
    node_id: Mapped[str] = mapped_column(String(2))
    sampling_point: Mapped[int | None] = mapped_column(Integer)  # multiplexer channel of latest reading
    status: Mapped[str] = mapped_column(String(20), default="unmeasured")
    description: Mapped[str] = mapped_column(Text)
    simple_care_tip: Mapped[str] = mapped_column(Text)
    location: Mapped[str] = mapped_column(String(120))
    optimal_ranges: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)


class Device(Base):
    __tablename__ = "devices"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)  # jetson-01, esp32-node-1, ...
    name: Mapped[str] = mapped_column(String(80))
    type: Mapped[str] = mapped_column(String(10))  # jetson | esp32
    node_id: Mapped[str | None] = mapped_column(String(2))
    ip_address: Mapped[str | None] = mapped_column(String(45))
    mac_address: Mapped[str | None] = mapped_column(String(17))
    status: Mapped[str] = mapped_column(String(10), default="unknown")
    last_heartbeat: Mapped[datetime | None] = mapped_column(UTCDateTime)
    last_transmission: Mapped[datetime | None] = mapped_column(UTCDateTime)
    channels: Mapped[int | None] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String(160))
    simple_role: Mapped[str] = mapped_column(String(160))


class AcquisitionSession(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(Integer, unique=True)
    scheduled_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    ended_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    status: Mapped[str] = mapped_column(String(15), default="completed")
    expected_records: Mapped[int] = mapped_column(Integer, default=0)
    received_records: Mapped[int] = mapped_column(Integer, default=0)
    rejected_records: Mapped[int] = mapped_column(Integer, default=0)
    flagged_records: Mapped[int] = mapped_column(Integer, default=0)
    trigger_type: Mapped[str] = mapped_column(String(10), default="scheduled")
    node_expected: Mapped[dict[str, int]] = mapped_column(JSON, default=dict)  # {"1": 672, "2": 672}


class SensorReading(Base):
    __tablename__ = "sensor_readings"
    __table_args__ = (
        UniqueConstraint("plant_id", "timestamp", name="uq_reading_plant_timestamp"),
        Index("ix_reading_plant_ts", "plant_id", "timestamp"),
        Index("ix_reading_plant_node_ts", "plant_id", "node_id", "timestamp"),
        Index("ix_reading_session", "session_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(UTCDateTime)
    plant_id: Mapped[int] = mapped_column(ForeignKey("plants.id"))
    node_id: Mapped[str] = mapped_column(String(2))
    sampling_point: Mapped[int | None] = mapped_column(Integer)
    session_id: Mapped[int | None] = mapped_column(ForeignKey("sessions.session_id"))
    soil: Mapped[float | None] = mapped_column(Float)
    temperature: Mapped[float | None] = mapped_column(Float)
    humidity: Mapped[float | None] = mapped_column(Float)
    light: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(10), default="accepted")  # accepted|flagged|rejected
    validation_error: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(10), default="dataset")  # dataset|ingest
    received_at: Mapped[datetime | None] = mapped_column(UTCDateTime)  # NULL for dataset rows (receipt time unknown)
    accepted_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    # Dataset labels: ground truth from the CSV, never ML output.
    label_stress_level: Mapped[str | None] = mapped_column(String(15))
    label_stress_type: Mapped[str | None] = mapped_column(String(30))
    label_is_synthetic_anomaly: Mapped[bool | None] = mapped_column(Boolean)

    @property
    def record_id(self) -> str:
        return f"R-{self.id}"


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    status: Mapped[str] = mapped_column(String(10), default="queued")  # queued|running|completed|failed
    total: Mapped[int] = mapped_column(Integer, default=0)
    processed: Mapped[int] = mapped_column(Integer, default=0)
    failed: Mapped[int] = mapped_column(Integer, default=0)
    current_plant_id: Mapped[int | None] = mapped_column(Integer)
    message: Mapped[str | None] = mapped_column(Text)


class MlPrediction(Base):
    __tablename__ = "ml_predictions"
    __table_args__ = (Index("ix_pred_plant_ts", "plant_id", "reading_timestamp"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    reading_id: Mapped[int] = mapped_column(ForeignKey("sensor_readings.id"), unique=True)
    plant_id: Mapped[int] = mapped_column(ForeignKey("plants.id"))
    reading_timestamp: Mapped[datetime] = mapped_column(UTCDateTime)
    status: Mapped[str] = mapped_column(String(12))  # ok|unavailable|rejected
    error: Mapped[str | None] = mapped_column(Text)
    warnings: Mapped[list[str] | None] = mapped_column(JSON(none_as_null=True))  # None -> SQL NULL, not JSON 'null'
    model_version: Mapped[str | None] = mapped_column(String(40))
    run_id: Mapped[str | None] = mapped_column(String(60))  # ML artifact run id
    severity: Mapped[str | None] = mapped_column(String(15))  # healthy|low|moderate|high
    probabilities: Mapped[dict[str, float] | None] = mapped_column(JSON(none_as_null=True))  # None -> SQL NULL, not JSON 'null'
    confidence: Mapped[float | None] = mapped_column(Float)
    cluster_id: Mapped[int | None] = mapped_column(Integer)  # KMeans env state, not severity
    cluster_distance: Mapped[float | None] = mapped_column(Float)
    anomaly_score: Mapped[float | None] = mapped_column(Float)
    is_anomaly: Mapped[bool | None] = mapped_column(Boolean)
    stress_type_guess: Mapped[str | None] = mapped_column(String(40))
    explanation_json: Mapped[dict[str, Any] | None] = mapped_column(JSON(none_as_null=True))  # None -> SQL NULL, not JSON 'null'
    ml_recommendations_json: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON(none_as_null=True))  # None -> SQL NULL, not JSON 'null'
    history_rows_sent: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    pipeline_run_id: Mapped[int | None] = mapped_column(ForeignKey("pipeline_runs.id"))


class AnomalyRecord(Base):
    __tablename__ = "anomaly_records"
    __table_args__ = (Index("ix_anomaly_plant_ts", "plant_id", "reading_timestamp"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("ml_predictions.id"))
    reading_id: Mapped[int] = mapped_column(ForeignKey("sensor_readings.id"))
    plant_id: Mapped[int] = mapped_column(ForeignKey("plants.id"))
    reading_timestamp: Mapped[datetime] = mapped_column(UTCDateTime)
    score: Mapped[float] = mapped_column(Float)  # higher = more anomalous
    threshold: Mapped[float] = mapped_column(Float)
    is_anomaly: Mapped[bool] = mapped_column(Boolean)
    details: Mapped[str | None] = mapped_column(Text)


class Recommendation(Base):
    __tablename__ = "recommendations"
    __table_args__ = (
        UniqueConstraint("plant_id", "category", "source_reading_id", name="uq_rec_plant_cat_reading"),
        Index("ix_rec_plant_created", "plant_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(10))  # ml|heuristic
    prediction_id: Mapped[int | None] = mapped_column(ForeignKey("ml_predictions.id", ondelete="SET NULL"))
    plant_id: Mapped[int] = mapped_column(ForeignKey("plants.id"))
    category: Mapped[str] = mapped_column(String(20))
    ml_rec_id: Mapped[str | None] = mapped_column(String(40))
    title: Mapped[str] = mapped_column(String(160))
    simple_action: Mapped[str] = mapped_column(Text)
    reason: Mapped[str] = mapped_column(Text)
    scientific_context: Mapped[str] = mapped_column(Text)
    guidance_source: Mapped[str] = mapped_column(String(160))
    priority: Mapped[str] = mapped_column(String(10))  # high|medium|low
    confidence: Mapped[float | None] = mapped_column(Float)
    based_on: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(15), default="new")  # new|in_progress|completed|dismissed|superseded
    user_modified: Mapped[bool] = mapped_column(Boolean, default=False)
    source_reading_id: Mapped[int] = mapped_column(ForeignKey("sensor_readings.id"))  # NOT NULL so the unique key can dedupe
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)
    resolved_at: Mapped[datetime | None] = mapped_column(UTCDateTime)


class ModelInfo(Base):
    __tablename__ = "model_info"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(60), unique=True)
    model_version: Mapped[str] = mapped_column(String(40))
    trained_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    provenance: Mapped[str | None] = mapped_column(String(30))
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    confusion_matrix: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    labels: Mapped[list[str] | None] = mapped_column(JSON)
    feature_importance: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    cached_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class AppSettings(Base):
    """Single-row (id=1) system settings; the table is named `settings`."""

    __tablename__ = "settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    polling_interval_seconds: Mapped[int] = mapped_column(Integer)
    jetson_receiver_port: Mapped[int] = mapped_column(Integer)
    ph_sensor_installed: Mapped[bool] = mapped_column(Boolean)
    collection_times: Mapped[list[str]] = mapped_column(JSON)
    log_retention_days: Mapped[int] = mapped_column(Integer)
    storage_path: Mapped[str] = mapped_column(String(500))
    soil_adc_dry: Mapped[int] = mapped_column(Integer)
    soil_adc_wet: Mapped[int] = mapped_column(Integer)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)
