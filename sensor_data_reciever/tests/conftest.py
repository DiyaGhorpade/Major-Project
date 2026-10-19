"""Add the project root to sys.path so that validator, config, etc. are importable."""
import sys
import os

# Insert the project root (one level up from this tests/ directory)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import importlib.util
from pathlib import Path
import shutil
import threading

import pytest
import receiver
from werkzeug.serving import make_server


@pytest.fixture
def output(tmp_path, monkeypatch):
    """Every HTTP test gets a separate real output file."""
    path = tmp_path / "readings.csv"
    monkeypatch.setattr(receiver, "CSV_FILE", str(path))
    receiver._init_csv()
    return path


@pytest.fixture
def client(output):
    return receiver.app.test_client()


@pytest.fixture
def valid_fields():
    return ["123456", "7", "2", "4", "4", "45.2", "23.1", "60.5", "512.0"]


@pytest.fixture
def live_receiver(output):
    """Real HTTP sockets with the production Flask application on a free port."""
    server = make_server("127.0.0.1", 0, receiver.app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.fixture(scope="session")
def generated_records():
    """Compile actual sketches; capture the exact bodies handed to HTTPClient.

    The host mocks replace sensors and the Arduino network stack only.
    No local Wi-Fi credentials are copied into the compilation directory.
    """
    if not shutil.which("g++"):
        pytest.skip("g++ is required to generate records from real firmware")
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "firmware_http_mocks", root / "hardware-code/tests/test_http_transport.py")
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    main = r'''
int main() {
  setup(); WiFi.state=WL_CONNECTED;
  collectSample(7);
  for(const auto& row : bodies) std::printf("%s\n", row.c_str());
  bodies.clear(); badReading=true;
  collectSample(8);
  for(const auto& row : bodies) std::printf("%s\n", row.c_str());
}
'''
    records = {}
    for simulation in (True, False):
        scans = {}
        for sketch in sorted((root / "hardware-code").glob("data_log_producer*/*.ino")):
            node = "1" if "12-remaining" in sketch.name else "2"
            source = sketch.read_text(encoding="utf-8")
            if not simulation:
                source = source.replace("SIMULATION_MODE = true", "SIMULATION_MODE = false")
            defines = '#define SENSOR_WIFI_SSID "test"\n#define SENSOR_JETSON_HOST "192.168.1.100"\n'
            stdout = harness.TransportTests().compile_run(source + main, defines)
            scans[node] = stdout.splitlines()
        records[simulation] = scans
    return records
