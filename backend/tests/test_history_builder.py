"""History slices: same plant only, strictly earlier timestamps, capped, cold-start safe."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from app.services.history_builder import history_bounds, history_rows_for, history_slice, read_ml_limits

T0 = datetime(2026, 9, 21, tzinfo=UTC)


def series(plant_id=1, n=10, step_h=2):
    return [SimpleNamespace(plant_id=plant_id, timestamp=T0 + timedelta(hours=step_h * i), idx=i) for i in range(n)]


def test_cold_start_first_reading_has_empty_history():
    assert history_slice(series(), 0, 60) == []


def test_history_is_strictly_earlier_and_oldest_first():
    rows = series()
    hist = history_slice(rows, 5, 60)
    assert [r.idx for r in hist] == [0, 1, 2, 3, 4]
    assert all(r.timestamp < rows[5].timestamp for r in hist)


def test_cap_is_respected():
    rows = series(n=100)
    hist = history_slice(rows, 90, 60)
    assert len(hist) == 60 and hist[0].idx == 30 and hist[-1].idx == 89
    assert history_slice(rows, 10, 3) == rows[7:10]
    assert history_slice(rows, 10, 0) == []


def test_short_history_when_fewer_rows_than_cap():
    assert len(history_slice(series(), 3, 60)) == 3


def test_equal_timestamp_rows_are_never_history():
    rows = series(n=5)
    rows.insert(3, SimpleNamespace(plant_id=1, timestamp=rows[2].timestamp, idx=99))  # duplicate of row 2's time
    hist = history_slice(rows, 3, 60)
    assert [r.idx for r in hist] == [0, 1]  # neither the equal-timestamp twin nor the reading itself
    assert all(r.timestamp < rows[3].timestamp for r in hist)


def test_mixed_plants_rejected():
    rows = series()
    rows[2] = SimpleNamespace(plant_id=2, timestamp=rows[2].timestamp, idx=2)
    with pytest.raises(ValueError, match="mixes plants"):
        history_slice(rows, 5, 60)


def test_per_plant_slices_never_cross_plants():
    a, b = series(plant_id=1), series(plant_id=2)
    for plant in (a, b):
        for i, row in enumerate(plant):
            hist = history_slice(plant, i, 60)
            assert all(h.plant_id == row.plant_id for h in hist)


def test_bad_index_rejected_and_bounds_match_slice():
    rows = series()
    with pytest.raises(ValueError):
        history_slice(rows, 10, 5)
    assert history_bounds(rows, 6, 4) == (2, 6) and history_bounds(rows, 0, 4) == (0, 0)


def test_history_rows_formula():
    assert history_rows_for(500, 49, 60) == 60  # DEV-1
    assert history_rows_for(500, 80, 60) == 80
    assert history_rows_for(40, 49, 60) == 40  # never above the ML cap
    assert history_rows_for(500, None, 60) == 60


def test_ml_limits_from_yaml_and_fallback(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("api:\n  max_history_rows: 123\n  max_batch_items: 7\n")
    assert read_ml_limits(cfg) == (123, 7)
    assert read_ml_limits(tmp_path / "missing.yaml") == (500, 64)
    cfg.write_text("api: {}\n")
    assert read_ml_limits(cfg) == (500, 64)


def test_non_positive_limits_fall_back_to_defaults(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("api:\n  max_history_rows: 0\n  max_batch_items: 64\n")
    assert read_ml_limits(cfg) == (500, 64)
    cfg.write_text("api:\n  max_history_rows: 500\n  max_batch_items: -3\n")
    assert read_ml_limits(cfg) == (500, 64)
    assert history_rows_for(0, 49, 60) == 49  # never zero history rows


def test_yaml_is_cached_per_file_version_and_failures_are_logged(tmp_path, caplog):
    import logging

    from app.services import history_builder as hb

    cfg = tmp_path / "c.yaml"
    cfg.write_text("api:\n  max_history_rows: 200\n  max_batch_items: 10\nrecommendations:\n  daytime_hours: [8, 17]\n")
    hb._parse_yaml.cache_clear()
    assert read_ml_limits(cfg) == (200, 10) and hb.read_daytime_hours(cfg) == (8, 17)
    before = hb._parse_yaml.cache_info()
    for _ in range(5):
        read_ml_limits(cfg), hb.read_daytime_hours(cfg)
    assert hb._parse_yaml.cache_info().misses == before.misses  # no re-parse while the mtime is unchanged
    cfg.write_text("api:\n  max_history_rows: 300\n  max_batch_items: 10\n")
    assert read_ml_limits(cfg) == (300, 10)  # picked up the change
    with caplog.at_level(logging.WARNING, logger="app.history"):
        cfg.write_text("api: [unclosed")
        assert read_ml_limits(cfg) == (500, 64)
        assert hb.read_daytime_hours(cfg) == (9, 16)
    text = caplog.text
    assert "cannot read ML limits" in text and "cannot read daytime hours" in text
