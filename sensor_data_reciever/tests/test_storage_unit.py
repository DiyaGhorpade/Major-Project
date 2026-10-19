"""Standalone storage utility keeps its separate nine-field format."""
from concurrent.futures import ThreadPoolExecutor
import csv
import config
import storage


def test_initialization_append_and_restart(tmp_path, monkeypatch):
    path = tmp_path / "utility.csv"
    monkeypatch.setattr(config, "SENSOR_DATA_FILE", str(path))
    storage.initialize_files()
    fields = ["0", "7", "2", "4", "4", "50", "23", "60", "512"]
    storage.save_valid_record(fields)
    storage.initialize_files()
    storage.save_valid_record(fields)
    with path.open(newline="", encoding="utf-8") as stream:
        assert list(csv.reader(stream)) == [config.CSV_HEADER, fields, fields]


def test_concurrent_utility_writes(tmp_path, monkeypatch):
    path = tmp_path / "utility.csv"
    monkeypatch.setattr(config, "SENSOR_DATA_FILE", str(path))
    storage.initialize_files()
    records = [[str(i), "7", "1", "1", "1", "50", "23", "60", "512"] for i in range(32)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(storage.save_valid_record, records))
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == config.CSV_HEADER
    assert sorted(rows[1:]) == sorted(records)
