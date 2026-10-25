"""Server-side CSV export of the data logs (streamed, honours the same filters as the JSON view)."""

import csv
import io
from collections.abc import Iterator
from datetime import datetime

from sqlalchemy.orm import sessionmaker

from app.core.mappings import IST, to_ist_iso
from app.db import repositories as repo

COLUMNS = (
    "record_id", "timestamp", "session_id", "node_id", "sampling_point", "plant_id",
    "soil", "temperature", "humidity", "light", "status", "validation_error",
    "ml_status", "ml_severity", "ml_confidence", "ml_is_anomaly", "ml_anomaly_score",
    "dataset_label_stress_level", "dataset_label_stress_type", "dataset_label_is_synthetic_anomaly",
)  # fmt: skip
BATCH_ROWS = 500


def filename(now: datetime | None = None) -> str:
    stamp = (now or datetime.now(IST)).astimezone(IST).strftime("%Y%m%dT%H%M%S")
    return f"plant_sensor_logs_{stamp}.csv"


def _cell(value: object) -> object:
    return "" if value is None else value


def _flag(value: bool | None) -> object:
    return "" if value is None else int(value)


def stream_csv(session_factory: sessionmaker, filters: repo.ReadingFilters) -> Iterator[str]:
    """Yield the CSV text in chunks. Opens its own DB session (the request's session is closed by then).
    `ml_*` columns are the model's output; `dataset_label_*` columns are the dataset's labels: never mixed."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerow(COLUMNS)
    yield buffer.getvalue()
    with session_factory() as db:
        pending = 0
        buffer.seek(0), buffer.truncate()
        for reading, pred in repo.iter_readings_with_predictions(db, filters):
            writer.writerow(
                [
                    reading.record_id, to_ist_iso(reading.timestamp), _cell(reading.session_id), reading.node_id,
                    _cell(reading.sampling_point), reading.plant_id,
                    _cell(reading.soil), _cell(reading.temperature), _cell(reading.humidity), _cell(reading.light),
                    reading.status, _cell(reading.validation_error),
                    pred.status if pred else "not_run",
                    _cell(pred.severity) if pred else "",
                    _cell(pred.confidence) if pred else "",
                    _flag(pred.is_anomaly) if pred else "",
                    _cell(pred.anomaly_score) if pred else "",
                    _cell(reading.label_stress_level), _cell(reading.label_stress_type),
                    _flag(reading.label_is_synthetic_anomaly),
                ]
            )  # fmt: skip
            pending += 1
            if pending >= BATCH_ROWS:
                yield buffer.getvalue()
                buffer.seek(0), buffer.truncate()
                pending = 0
        if pending:
            yield buffer.getvalue()
