"""Seeding: idempotency, mapping, label storage, bad-row handling, and /api/health."""

import pandas as pd
from sqlalchemy import func, select

from app.core.mappings import to_ist_iso
from app.db import repositories as repo
from app.db.models import AcquisitionSession, Device, Plant, SensorReading
from app.services.seed_service import ensure_seeded

SENSORS = ["soil", "temperature", "humidity", "light"]


def test_seed_twice_is_idempotent(real_settings, db_for):
    db = db_for(real_settings)
    first = ensure_seeded(db, real_settings)
    second = ensure_seeded(db, real_settings)

    assert first.inserted == 5376 and second.inserted == 0
    assert second.skipped_existing == 5376
    assert repo.count_readings(db) == 5376
    assert (repo.count_plants(db), repo.count_sessions(db), repo.count_devices(db)) == (16, 4, 3)
    distinct = db.scalar(select(func.count()).select_from(select(SensorReading.plant_id, SensorReading.timestamp).distinct().subquery()))
    assert distinct == 5376


def test_flagged_equals_rows_with_missing_values(real_settings, real_csv, db_for):
    db = db_for(real_settings)
    ensure_seeded(db, real_settings)
    expected = int(pd.read_csv(real_csv)[SENSORS].isna().any(axis=1).sum())

    assert expected == 233
    assert repo.count_flagged_readings(db) == expected
    flagged = db.scalars(select(SensorReading).where(SensorReading.status == "flagged")).all()
    assert len(flagged) == 233
    for reading in flagged:
        assert reading.validation_error.startswith("missing: ")
        missing = [c for c in SENSORS if getattr(reading, c) is None]
        assert reading.validation_error == f"missing: {', '.join(missing)}"


def test_nodes_plants_sessions_devices(real_settings, db_for):
    db = db_for(real_settings)
    ensure_seeded(db, real_settings)

    plants = db.scalars(select(Plant)).all()
    assert len(plants) == 16
    for plant in plants:
        assert plant.node_id == ("1" if plant.id % 2 else "2")
    assert {r.node_id for r in db.scalars(select(SensorReading).where(SensorReading.plant_id.in_([1, 3, 5])))} == {"1"}
    assert {r.node_id for r in db.scalars(select(SensorReading).where(SensorReading.plant_id.in_([2, 4, 6])))} == {"2"}

    clusters = {p.id: (p.cluster, p.stress_type) for p in db.scalars(select(Plant))}
    assert clusters[1] == ("Cluster 1", "Healthy") and clusters[8] == ("Cluster 2", "Water Stress")
    assert clusters[12] == ("Cluster 3", "Light Stress")
    assert clusters[16] == ("Cluster 4", "Temperature/Humidity Stress")

    sessions = db.scalars(select(AcquisitionSession)).all()
    assert len(sessions) == 4
    for session in sessions:
        assert session.expected_records == session.received_records == 1344
        assert session.node_expected == {"1": 672, "2": 672}
        assert session.rejected_records == 0
    devices = {d.id: d for d in db.scalars(select(Device))}
    assert set(devices) == {"jetson-01", "esp32-node-1", "esp32-node-2"}
    assert devices["esp32-node-1"].channels == 12 and devices["esp32-node-1"].status == "unknown"
    assert devices["esp32-node-1"].ip_address is None and devices["esp32-node-1"].last_heartbeat is None


def test_timestamps_round_trip_as_ist(real_settings, db_for):
    db = db_for(real_settings)
    ensure_seeded(db, real_settings)

    readings = repo.readings_for_plant_ordered(db, 1)
    assert len(readings) == 336
    assert to_ist_iso(readings[0].timestamp) == "2026-09-21T00:00:00+05:30"
    assert to_ist_iso(readings[1].timestamp) == "2026-09-21T02:00:00+05:30"
    assert readings[0].timestamp.utcoffset().total_seconds() == 0
    assert all(a.timestamp < b.timestamp for a, b in zip(readings, readings[1:]))
    latest = repo.latest_reading_per_plant(db)
    assert [r.plant_id for r in latest] == list(range(1, 17))
    assert to_ist_iso(latest[0].timestamp) == to_ist_iso(readings[-1].timestamp)


def test_dataset_labels_kept_in_label_columns(real_settings, real_csv, db_for):
    db = db_for(real_settings)
    ensure_seeded(db, real_settings)
    csv = pd.read_csv(real_csv)

    levels = dict(db.execute(select(SensorReading.label_stress_level, func.count()).group_by(SensorReading.label_stress_level)).all())
    assert levels == csv["stress_level"].value_counts().to_dict()
    types = dict(db.execute(select(SensorReading.label_stress_type, func.count()).group_by(SensorReading.label_stress_type)).all())
    assert types == csv["stress_type"].value_counts().to_dict()
    anomalies = db.scalar(select(func.count()).where(SensorReading.label_is_synthetic_anomaly.is_(True)))
    assert anomalies == int(csv["is_synthetic_anomaly"].sum()) == 65


def test_missing_values_stay_null(real_settings, real_csv, db_for):
    db = db_for(real_settings)
    ensure_seeded(db, real_settings)
    csv = pd.read_csv(real_csv)
    for col in SENSORS:
        nulls = db.scalar(select(func.count()).where(getattr(SensorReading, col).is_(None)))
        assert nulls == int(csv[col].isna().sum())


def test_default_settings_row(real_settings, db_for):
    from app.db.models import AppSettings

    db = db_for(real_settings)
    ensure_seeded(db, real_settings)
    row = db.get(AppSettings, 1)
    assert (row.polling_interval_seconds, row.jetson_receiver_port, row.ph_sensor_installed) == (5, 5000, False)
    assert (row.soil_adc_dry, row.soil_adc_wet, row.log_retention_days) == (3200, 1450, 30)
    assert len(row.collection_times) == 12 and row.collection_times[0] == "00:00"


def test_bad_rows_flagged_or_dropped(bad_settings, db_for, caplog):
    db = db_for(bad_settings)
    result = ensure_seeded(db, bad_settings)
    again = ensure_seeded(db, bad_settings)

    assert result.rows_in_file == 11
    assert result.dropped == 2  # bad timestamp, unknown plant 99
    assert result.duplicates_in_file == 1
    assert result.inserted == 8 and again.inserted == 0

    plant1 = repo.readings_for_plant_ordered(db, 1)
    assert len(plant1) == 2
    ok, hot = plant1
    assert ok.status == "accepted" and ok.validation_error is None and ok.soil == 50.0  # first duplicate wins
    assert hot.status == "flagged" and hot.validation_error == "out_of_range: temperature=45.0"

    plant2 = repo.readings_for_plant_ordered(db, 2)
    assert len(plant2) == 2
    missing, naive = plant2
    assert to_ist_iso(naive.timestamp) == "2026-09-21T04:00:00+05:30"  # naive assumed IST
    assert naive.status == "accepted"
    assert missing.status == "flagged" and missing.validation_error == "missing: soil, light"
    assert missing.soil is None and missing.light is None  # never filled

    no_node, wide, non_finite = repo.readings_for_plant_ordered(db, 3)
    assert no_node.node_id == "1"  # fallback: odd plant -> node 1
    assert wide.validation_error == "out_of_range: soil=120.0, humidity=10.0"
    assert non_finite.soil is None and non_finite.validation_error == "invalid: soil"  # inf is not stored
    assert non_finite.node_id == "1" and non_finite.sampling_point is None

    (bad_soil,) = repo.readings_for_plant_ordered(db, 4)
    assert bad_soil.soil is None and bad_soil.validation_error == "invalid: soil"
    assert repo.count_flagged_readings(db) == 5

    logged = caplog.text
    assert "dropped CSV row 6: unparseable timestamp" in logged
    assert "dropped CSV row 7: unknown plant_id '99'" in logged
    assert "ignored CSV row 5: duplicate" in logged
