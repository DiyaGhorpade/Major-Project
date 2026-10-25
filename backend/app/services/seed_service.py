"""Idempotent startup seeding from the bundled CSV (master prompt section 6.1).

Reads the dataset with pandas (empty cells stay NULL, nothing is filled), validates each row
locally, and upserts plants, devices, sessions, readings and the default settings row. Rows
with missing/invalid/out-of-range values are kept with `status='flagged'` and a
`validation_error`; rows that cannot be keyed (bad timestamp, bad/unknown plant id) are
dropped and counted. Running it again inserts nothing new.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from app.core.config import BACKEND_DIR, Settings, display_path
from app.core.mappings import IST, node_for_plant, normalise_node_id, parse_timestamp
from app.core.plant_catalog import PLANT_IDS, catalog_entry
from app.db import repositories as repo

logger = logging.getLogger("app.seed")

SENSOR_RANGES: dict[str, tuple[float, float]] = {
    "soil": (0, 100),
    "temperature": (15, 40),
    "humidity": (20, 100),
    "light": (0, 100_000),
}
REQUIRED_COLUMNS = (
    "timestamp", "session_id", "node_id", "sampling_point", "plant_id",
    "soil", "temperature", "humidity", "light",
    "stress_level", "stress_type", "is_synthetic_anomaly",
)  # fmt: skip
# User-editable configuration defaults (placeholders), NOT measured values or sensor calibration.
DEFAULT_SETTINGS: dict[str, Any] = {
    "polling_interval_seconds": 5,
    "jetson_receiver_port": 5000,
    "ph_sensor_installed": False,
    "log_retention_days": 30,
    "soil_adc_dry": 3200,
    "soil_adc_wet": 1450,
}
DEVICE_COPY = {
    "jetson": ("Edge gateway (receives node data)",
               "Collects the data sent by the sensor nodes"),
    "esp32": ("Sensor node: samples plant sensors through a multiplexer",
              "Reads soil, temperature, humidity and light for its plants"),
}  # fmt: skip


@dataclass
class SeedResult:
    rows_in_file: int = 0
    dropped: int = 0
    duplicates_in_file: int = 0
    inserted: int = 0
    skipped_existing: int = 0
    flagged_in_file: int = 0
    plants: int = 0
    devices: int = 0
    sessions: int = 0
    settings_created: bool = False
    seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)


def validation_issues(row: dict[str, Any], unparseable: set[str]) -> str | None:
    """Describe problems of one row, e.g. `missing: soil, light; out_of_range: temperature=45.0`."""
    missing = [c for c in SENSOR_RANGES if row[c] is None and c not in unparseable]
    invalid = [c for c in SENSOR_RANGES if c in unparseable]
    out_of_range = [
        f"{c}={row[c]}"
        for c, (lo, hi) in SENSOR_RANGES.items()
        if row[c] is not None and not lo <= row[c] <= hi
    ]
    parts = [
        f"{label}: {', '.join(items)}"
        for label, items in (("missing", missing), ("invalid", invalid), ("out_of_range", out_of_range))
        if items
    ]
    return "; ".join(parts) or None


def _finite_numeric(series: pd.Series) -> pd.Series:
    """Parse to float; text, `inf` and `nan` all become NaN (callers flag non-empty originals as invalid)."""
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(np.isfinite(numeric))


def _nullable(series: pd.Series) -> list[Any]:
    return [None if pd.isna(v) else v for v in series.tolist()]


def _log_row_issue(result: SeedResult, action: str, index: int, reason: str) -> None:
    """Log one skipped CSV row by its 1-based row number (header excluded)."""
    message = f"{action} CSV row {index + 1}: {reason}"
    result.warnings.append(message)
    logger.warning(message)


def load_dataset(path: Path) -> tuple[pd.DataFrame, SeedResult]:
    """Parse and validate the CSV into a clean, sorted frame of insert-ready rows."""
    result = SeedResult()
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"dataset {path} lacks required columns: {missing_cols}")
    result.rows_in_file = len(raw)

    df = pd.DataFrame(index=raw.index)
    df["timestamp"] = pd.to_datetime(
        [parse_timestamp(v) if isinstance(v, str) else None for v in raw["timestamp"]], utc=True
    )
    df["plant_id"] = _finite_numeric(raw["plant_id"])
    unparseable: dict[str, pd.Series] = {}
    for col in SENSOR_RANGES:
        numeric = _finite_numeric(raw[col])
        unparseable[col] = raw[col].notna() & numeric.isna()
        df[col] = numeric
    for col in ("session_id", "sampling_point"):
        df[col] = _finite_numeric(raw[col])
    df["node_raw"] = raw["node_id"]
    df["label_stress_level"] = raw["stress_level"]
    df["label_stress_type"] = raw["stress_type"]
    df["label_is_synthetic_anomaly"] = raw["is_synthetic_anomaly"].map(
        {"1": True, "0": False, "true": True, "false": False, "True": True, "False": False}
    )

    keyed = df["timestamp"].notna() & df["plant_id"].isin(PLANT_IDS)
    for index in df.index[~keyed]:
        reason = "unparseable timestamp" if pd.isna(df.at[index, "timestamp"]) else f"unknown plant_id {raw.at[index, 'plant_id']!r}"
        _log_row_issue(result, "dropped", index, reason)
    result.dropped = int((~keyed).sum())
    df, raw_flags = df[keyed], pd.DataFrame(unparseable)[keyed]

    dup = df.duplicated(["plant_id", "timestamp"], keep="first")
    for index in df.index[dup]:
        _log_row_issue(result, "ignored", index, "duplicate (plant_id, timestamp), first occurrence kept")
    result.duplicates_in_file = int(dup.sum())
    df, raw_flags = df[~dup], raw_flags[~dup]

    df = df.assign(plant_id=df["plant_id"].astype(int)).sort_values(["timestamp", "plant_id"], kind="stable")
    raw_flags = raw_flags.loc[df.index]

    errors: list[str | None] = []
    for row, flags in zip(df.to_dict("records"), raw_flags.to_dict("records")):
        clean = {c: (None if pd.isna(row[c]) else float(row[c])) for c in SENSOR_RANGES}
        errors.append(validation_issues(clean, {c for c, bad in flags.items() if bad}))
    df["validation_error"] = errors
    result.flagged_in_file = int(df["validation_error"].notna().sum())
    return df.reset_index(drop=True), result


def _cadence(df: pd.DataFrame) -> pd.Timedelta | None:
    """Median gap between consecutive readings of the same plant."""
    gaps = df.groupby("plant_id")["timestamp"].diff().dropna()
    return gaps.median() if len(gaps) else None


def _plant_nodes(df: pd.DataFrame) -> dict[int, str]:
    """Node per plant: the most frequent node in the data, parity fallback when absent."""
    nodes: dict[int, str] = {}
    for plant_id, group in df.groupby("plant_id"):
        observed = group["node_raw"].map(normalise_node_id).dropna()
        if observed.nunique() > 1:
            logger.warning(
                "plant %s has readings on several nodes %s; using the most frequent",
                plant_id, sorted(observed.unique()),
            )  # fmt: skip
        nodes[int(plant_id)] = node_for_plant(int(plant_id), observed.mode().iat[0] if len(observed) else None)
    return nodes


def _reading_rows(df: pd.DataFrame, plant_nodes: dict[int, str]) -> list[dict[str, Any]]:
    cols = ["timestamp", "plant_id", "sampling_point", "session_id", *SENSOR_RANGES,
            "label_stress_level", "label_stress_type", "label_is_synthetic_anomaly", "validation_error"]  # fmt: skip
    data = {c: _nullable(df[c]) for c in cols}
    node_raw = df["node_raw"].tolist()
    timestamps = [t.to_pydatetime() for t in df["timestamp"]]
    rows = []
    for i in range(len(df)):
        row = {c: data[c][i] for c in cols}
        row["plant_id"] = int(row["plant_id"])
        for c in ("sampling_point", "session_id"):
            row[c] = None if row[c] is None else int(row[c])
        row["timestamp"] = timestamps[i]
        row["node_id"] = normalise_node_id(node_raw[i]) or plant_nodes[row["plant_id"]]
        row["status"] = "flagged" if row["validation_error"] else "accepted"
        row["source"] = "dataset"
        rows.append(row)
    return rows


def _plant_rows(df: pd.DataFrame, plant_nodes: dict[int, str], now: datetime) -> list[dict[str, Any]]:
    latest = df.drop_duplicates("plant_id", keep="last").set_index("plant_id")["sampling_point"]
    rows = []
    for plant_id in sorted(plant_nodes):
        entry = catalog_entry(plant_id)
        point = latest.get(plant_id)
        rows.append({
            "id": plant_id,
            "name": entry.name,
            "species": entry.species,
            "variety": entry.variety,
            "cluster": entry.group.cluster,
            "cluster_label": entry.group.label,
            "stress_type": entry.group.stress_type,
            "node_id": plant_nodes[plant_id],
            "sampling_point": None if pd.isna(point) else int(point),
            "description": entry.description,
            "simple_care_tip": entry.simple_care_tip,
            "location": entry.location,
            "optimal_ranges": entry.optimal_ranges,
            "updated_at": now,
        })  # fmt: skip
    return rows


def _session_rows(db: Session, df: pd.DataFrame, plant_nodes: dict[int, str]) -> list[dict[str, Any]]:
    """Session rows; expected counts come from the data's cadence, received counts from the DB."""
    cadence = _cadence(df)
    stats = repo.session_stats(db)
    rows = []
    for session_id, group in df.dropna(subset=["session_id"]).groupby("session_id"):
        sid = int(session_id)
        per_plant = 1 + round((group["timestamp"].max() - group["timestamp"].min()) / cadence) if cadence else 0
        plants = group["plant_id"].unique()
        node_expected = {n: sum(1 for p in plants if plant_nodes[int(p)] == n) * per_plant for n in ("1", "2")}
        s = stats.get(sid, {})
        rows.append({
            "session_id": sid,
            "scheduled_at": s.get("first"),
            "started_at": s.get("first"),
            "ended_at": s.get("last"),
            "status": "completed",
            "expected_records": sum(node_expected.values()),
            "received_records": s.get("received", 0),
            "rejected_records": s.get("rejected", 0),
            "flagged_records": s.get("flagged", 0),
            "node_expected": node_expected,
        })  # fmt: skip
    return rows


def _device_rows(db: Session) -> list[dict[str, Any]]:
    nodes = repo.node_stats(db)
    lasts = [n["last"] for n in nodes.values() if n["last"]]
    rows = [{
        "id": "jetson-01", "name": "Jetson Gateway", "type": "jetson", "node_id": None,
        "role": DEVICE_COPY["jetson"][0], "simple_role": DEVICE_COPY["jetson"][1],
        "last_transmission": max(lasts) if lasts else None, "channels": None,
    }]  # fmt: skip
    for node in ("1", "2"):
        info = nodes.get(node, {})
        rows.append({
            "id": f"esp32-node-{node}", "name": f"ESP32 Node {node}", "type": "esp32", "node_id": node,
            "role": DEVICE_COPY["esp32"][0], "simple_role": DEVICE_COPY["esp32"][1],
            "last_transmission": info.get("last"), "channels": info.get("channels"),
        })  # fmt: skip
    return rows


def _collection_times(df: pd.DataFrame) -> list[str]:
    """HH:MM slots (IST) on the first day of the lowest plant id: the dataset's actual schedule."""
    local = df.loc[df["plant_id"] == df["plant_id"].min(), "timestamp"].dt.tz_convert(IST)
    if local.empty:
        return []
    return [t.strftime("%H:%M") for t in local if t.date() == local.iloc[0].date()]


def ensure_seeded(db: Session, settings: Settings) -> SeedResult:
    """Load the bundled CSV into the DB. Safe to call repeatedly (second call inserts 0 rows)."""
    started = time.perf_counter()
    # The advisory ML `/validate` report runs after startup, never here: see
    # `dataset_validation.validate_with_ml`. The local checks in `load_dataset` stay authoritative.
    df, result = load_dataset(settings.DATASET_PATH)
    now = pd.Timestamp.now(tz="UTC").to_pydatetime()
    plant_nodes = _plant_nodes(df)

    repo.upsert_plants(db, _plant_rows(df, plant_nodes, now))
    # Sessions must exist before readings (FK); counts are refreshed once readings are in.
    repo.upsert_sessions(db, _session_rows(db, df, plant_nodes))
    result.inserted = repo.bulk_insert_readings(db, _reading_rows(df, plant_nodes))
    result.skipped_existing = len(df) - result.inserted
    repo.upsert_sessions(db, _session_rows(db, df, plant_nodes))
    repo.upsert_devices(db, _device_rows(db))

    path = make_url(settings.DATABASE_URL).database
    database = display_path(Path(path), BACKEND_DIR) if path and path != ":memory:" else ":memory:"  # never an absolute path
    result.settings_created = repo.insert_default_settings(
        db, {**DEFAULT_SETTINGS, "collection_times": _collection_times(df), "storage_path": database}
    )
    db.commit()

    result.plants, result.devices, result.sessions = repo.count_plants(db), repo.count_devices(db), repo.count_sessions(db)
    result.seconds = round(time.perf_counter() - started, 3)
    logger.info(
        "seed done: file_rows=%d inserted=%d skipped_existing=%d flagged_in_file=%d plants=%d devices=%d sessions=%d in %.2fs",
        result.rows_in_file, result.inserted, result.skipped_existing, result.flagged_in_file,
        result.plants, result.devices, result.sessions, result.seconds,
    )  # fmt: skip
    return result
