"""Standalone storage round-trips the current column order and UTF-8 values."""
import csv
from pathlib import Path
import tempfile
from unittest.mock import patch
from hypothesis import given, settings, strategies as st
import config
import storage


@given(st.lists(st.text(), min_size=9, max_size=9))
@settings(max_examples=100, deadline=None)
def test_save_valid_record_round_trip_preserves_all_fields(fields):
    # This utility receives already-validated fields; exercise CSV escaping too.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "readings.csv"
        with patch.object(config, "SENSOR_DATA_FILE", str(path)):
            storage.initialize_files()
            storage.save_valid_record(fields)
        with path.open(newline="", encoding="utf-8") as stream:
            assert list(csv.reader(stream)) == [config.CSV_HEADER, fields]
