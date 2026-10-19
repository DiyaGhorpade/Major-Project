"""Storage failures never return an acceptance response."""
from unittest.mock import patch
import pytest
import receiver
from .helpers import read_rows


@pytest.mark.parametrize("operation", ["open", "write", "flush", "fsync"])
def test_disk_failure_returns_500_and_receiver_recovers(client, output, valid_fields, operation):
    real_open = open

    class BrokenFile:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            self.stream.__enter__()
            return self
        def __exit__(self, *args):
            return self.stream.__exit__(*args)
        def write(self, value):
            if operation == "write":
                raise OSError("disk full")
            return self.stream.write(value)
        def flush(self):
            if operation == "flush":
                raise OSError("flush failed")
            self.stream.flush()
        def fileno(self):
            return self.stream.fileno()

    def broken_open(path, mode="r", **kwargs):
        if mode == "a":
            if operation == "open":
                raise PermissionError("read-only destination")
            return BrokenFile(real_open(path, mode, **kwargs))
        return real_open(path, mode, **kwargs)

    with patch.object(receiver, "open", side_effect=broken_open, create=True):
        if operation == "fsync":
            with patch.object(receiver.os, "fsync", side_effect=OSError("sync failed")):
                response = client.post("/upload", data=",".join(valid_fields))
        else:
            response = client.post("/upload", data=",".join(valid_fields))
    assert response.status_code == 500
    # A write/flush/sync failure may leave bytes on disk: delivery is uncertain.
    assert client.post("/upload", data=",".join(valid_fields)).status_code == 200
    assert read_rows(output)[-1][3:] == valid_fields


@pytest.mark.parametrize("header", [
    "", "timestamp,session_id,node_id,sampling_point,plant_id,soil,temperature,humidity,light\n",
    "record_id,timestamp,plant_id,soil_moisture_percent,temperature_c,humidity_percent,light_lux\n",
    ",".join(reversed(receiver.CSV_HEADER)) + "\n",
])
def test_incompatible_header_fails_startup_and_upload_without_changing_data(
        client, output, valid_fields, header):
    output.write_text(header, encoding="utf-8")
    before = output.read_bytes()
    with pytest.raises(ValueError, match="Incompatible CSV header"):
        receiver._init_csv()
    assert client.post("/upload", data=",".join(valid_fields)).status_code == 500
    assert output.read_bytes() == before


def test_cli_output_option_leaves_legacy_default_untouched(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    script = Path(receiver.__file__).resolve()
    legacy = tmp_path / "sensor_data.csv"
    legacy.write_text("legacy header\n", encoding="utf-8")
    selected = tmp_path / "selected.csv"
    selected.write_text("incompatible header\n", encoding="utf-8")
    result = subprocess.run([sys.executable, str(script), "--output", str(selected)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=5)
    assert result.returncode != 0
    assert "Incompatible CSV header" in result.stderr
    assert str(selected) in result.stderr
    assert legacy.read_text(encoding="utf-8") == "legacy header\n"
    assert selected.read_text(encoding="utf-8") == "incompatible header\n"
