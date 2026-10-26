"""Shape drift guard.

Every endpoint's JSON is (1) validated with the Pydantic response model and (2) compared, recursively, with the
interface in `frontend/src/types/index.ts`: each non-optional TS key (also nested object / array-element keys) must
be present. Keys that the README lists as removed from the frontend types are excluded. The TS parser itself is
tested so that an interface that parses to nothing fails loudly instead of passing vacuously.
"""

import re
from pathlib import Path
from typing import Any

import pytest

from app.schemas.ai import AnomalyOut, AnomalySummaryOut, InsightsOut
from app.schemas.dashboard import OverviewOut, PlantMonitoringOut
from app.schemas.device import DeviceOut
from app.schemas.model import ModelMetricsOut
from app.schemas.plant import PlantOut
from app.schemas.prediction import PredictionListItem
from app.schemas.recommendation import RecommendationOut
from app.schemas.sensor import SensorRecordOut
from app.schemas.session import SessionOut
from app.schemas.settings import SystemSettingsOut
from app.schemas.shap import GlobalShapOut, LocalShapOut

ROOT = Path(__file__).resolve().parents[2]
TYPES_TS = ROOT / "frontend" / "src" / "types" / "index.ts"
README = ROOT / "backend" / "README.md"

# --- a small TypeScript interface parser -----------------------------------------------------------
Member = dict[str, Any]  # {"optional": bool, "children": dict[str, Member] | None, "index": bool}


def _split_top_level(src: str) -> list[str]:
    """Split on `;` / newlines that are outside any (), [], {}, <> nesting."""
    parts, depth, current = [], 0, []
    for ch in src:
        if ch in "({[<":
            depth += 1
        elif ch in ")}]>":
            depth -= 1
        if ch in ";\n" and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return [p.strip() for p in parts if p.strip()]


def _balanced_block(text: str) -> str | None:
    """Contents of the first `{...}` block in `text` (balanced), or None."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        depth += text[i] == "{"
        depth -= text[i] == "}"
        if depth == 0:
            return text[start + 1 : i]
    return None


def parse_members(body: str) -> dict[str, Member]:
    members: dict[str, Member] = {}
    # a member may continue on the next line; rejoin lines of an unfinished `{`
    for item in _split_top_level(body):
        m = re.match(r"^(\[[^\]]*\]|[A-Za-z_]\w*)(\?)?\s*:\s*(.*)$", item, re.S)
        if not m:
            continue
        name, optional, type_text = m.group(1), bool(m.group(2)), m.group(3)
        block = _balanced_block(type_text)
        members[name] = {
            "optional": optional,
            "children": parse_members(block) if block is not None else None,
            "index": name.startswith("["),
            "type": type_text.strip(),
        }
    return members


def parse_interfaces(text: str) -> dict[str, dict[str, Member]]:
    text = re.sub(r"//[^\n]*", "", text)
    out: dict[str, dict[str, Member]] = {}
    for m in re.finditer(r"export interface (\w+)\s*\{", text):
        out[m.group(1)] = parse_members(_balanced_block(text[m.start() :]) or "")
    return out


def removed_keys() -> dict[str, set[str]]:
    """From the README section "Frontend type changes required": bullets like ``- `SystemSettings`: remove `a`, `b`.``"""
    section = README.read_text().split("## Frontend type changes required", 1)[1]
    out: dict[str, set[str]] = {}
    for line in section.splitlines():
        m = re.match(r"^- `(\w+)`: remove (.+)$", line.strip())
        if m:
            out.setdefault(m.group(1), set()).update(re.findall(r"`(\w+)`", m.group(2)))
    return out


INTERFACES = parse_interfaces(TYPES_TS.read_text())
REMOVED = removed_keys()


def _ref(type_text: str) -> str | None:
    """`SensorRecord`, `SensorRecord[]`, `Array<ShapFeatureContribution>` -> the interface name, else None."""
    m = re.match(r"^(?:Array<)?(\w+)(?:\[\])?>?$", type_text.strip())
    return m.group(1) if m and m.group(1) in INTERFACES else None


def check(payload: Any, members: dict[str, Member], where: str, interface: str = "") -> int:
    """Assert every non-optional member is present (recursively). Returns the number of keys checked."""
    assert isinstance(payload, dict), f"{where}: expected an object, got {type(payload).__name__}"
    checked = 0
    for name, member in members.items():
        if member["index"]:  # `[key: string]: {...}`: every value must match the nested shape
            for key, value in payload.items():
                checked += check(value, member["children"] or {}, f"{where}.{key}", interface)
            continue
        if member["optional"] or name in REMOVED.get(interface, set()):
            continue
        assert name in payload, f"{where}: JSON lacks TS key '{name}'"
        checked += 1
        value = payload[name]
        nested = member["children"]
        if nested is None and (ref := _ref(member["type"])):
            nested, sub_interface = INTERFACES[ref], ref
        else:
            sub_interface = interface
        if nested is None or value is None:
            continue
        if isinstance(value, list):
            assert value or "[]" in member["type"] or "Array" in member["type"], f"{where}.{name}: expected an object"
            for i, element in enumerate(value[:5]):
                checked += check(element, nested, f"{where}.{name}[{i}]", sub_interface)
        else:
            checked += check(value, nested, f"{where}.{name}", sub_interface)
    return checked


def assert_shape(payload: dict, interface: str, where: str = "", model: Any = None, minimum: int = 3) -> None:
    if model is not None:
        model.model_validate(payload)  # the response model accepts exactly this JSON
    n = check(payload, INTERFACES[interface], where or interface, interface)
    assert n >= minimum, f"{where or interface}: only {n} TS keys were checked (parser or payload empty?)"


# --- parser self-tests: nothing may pass vacuously ----------------------------------------------------------
TARGETS = {
    "Plant": 14, "PlantPrediction": 3, "PlantOptimalRange": 4, "SensorRecord": 9, "Session": 10, "Device": 9, "HardwareDetails": 1,
    "LocalShapExplanation": 9, "ShapFeatureContribution": 8, "GlobalShapData": 4, "GlobalShapFeature": 6, "BeeswarmPoint": 5,
    "ModelMetrics": 9, "Recommendation": 16, "SystemSettings": 8, "Anomaly": 10, "AnomalySummary": 7, "AnomalyPlantSummary": 7,
}  # fmt: skip


@pytest.mark.parametrize("interface,minimum", TARGETS.items())
def test_parser_found_the_interface_fields(interface, minimum):
    assert interface in INTERFACES, f"{interface} not found in types/index.ts"
    assert len(INTERFACES[interface]) >= minimum, f"{interface}: parsed {len(INTERFACES[interface])} members, expected >= {minimum}"


def test_parser_handles_nesting():
    session = INTERFACES["Session"]
    assert session["node_progress"]["children"]["[key: string]"]["children"].keys() == {"expected", "received", "status"}
    assert set(INTERFACES["PlantOptimalRange"]["soil"]["children"]) >= {"min", "max", "unit", "lowLabel", "idealLabel", "highLabel"}
    assert set(INTERFACES["Recommendation"]["supporting_sensor_values"]["children"]) == {"metric", "value", "target", "status"}
    assert set(INTERFACES["ModelMetrics"]["per_class"]["children"]) >= {"class_name", "precision", "recall", "f1_score", "support", "laymanStatus"}
    assert set(INTERFACES["ModelMetrics"]["confusion_matrix"]["children"]) == {"labels", "matrix"}
    assert INTERFACES["BeeswarmPoint"]["predicted_class"]["optional"] is True and INTERFACES["SensorRecord"]["plant_id"]["optional"] is False
    assert "expected" not in INTERFACES["Session"]  # nested keys do not leak to the top level


def test_check_fails_on_missing_keys_instead_of_passing():
    with pytest.raises(AssertionError, match="lacks TS key 'record_id'"):
        assert_shape({"timestamp": "x"}, "SensorRecord", minimum=0)
    top = {k: 0 for k in INTERFACES["Session"] if k != "node_progress"}
    with pytest.raises(AssertionError, match=r"\.1: JSON lacks TS key 'status'"):  # nested, inside the index signature
        check({**top, "node_progress": {"1": {"expected": 1, "received": 1}}}, INTERFACES["Session"], "s", "Session")
    complete = {**top, "node_progress": {"1": {"expected": 1, "received": 1, "status": "completed"}}}
    assert check(complete, INTERFACES["Session"], "s", "Session") >= 13
    with pytest.raises(AssertionError, match="only .* TS keys"):  # a payload that checks far fewer keys than expected
        assert_shape(complete, "Session", minimum=100)


def test_removed_keys_are_not_required():
    assert {"api_base_url", "demo_mode"} <= REMOVED["SystemSettings"]
    settings = {k: 1 for k in INTERFACES["SystemSettings"] if k not in ("api_base_url", "demo_mode")}
    assert check(settings, INTERFACES["SystemSettings"], "settings", "SystemSettings") == len(settings)


# --- endpoints -----------------------------------------------------------------------------------------------
def test_plants(ran_client):
    plants = ran_client.get("/api/plants").json()
    assert len(plants) == 16
    for plant in (plants[0], plants[7], ran_client.get("/api/plants/5").json()):
        assert_shape(plant, "Plant", model=PlantOut, minimum=40)
        assert plant["latestReading"]["plant_id"] == plant["id"]
        assert plant["conditionPrediction"]["modelVersion"] == "1.0.0"
        assert set(plant["optimalRanges"]) == {"soil", "temperature", "humidity", "light"}
    assert_shape(plants[0]["conditionPrediction"], "PlantPrediction")


def test_sensor_records(ran_client):
    for path in ("/api/sensors/readings?limit=3", "/api/data-logs?limit=3"):
        rows = ran_client.get(path).json()
        assert len(rows) == 3
        for row in rows:
            assert_shape(row, "SensorRecord", path, model=SensorRecordOut)
            assert row["timestamp"].endswith("+05:30") and 1 <= row["plant_id"] <= 16


def test_sessions_and_devices(ran_client):
    sessions = ran_client.get("/api/sessions").json()
    assert [s["session_id"] for s in sessions] == [4, 3, 2, 1]
    for s in sessions:
        assert_shape(s, "Session", model=SessionOut, minimum=15)
        assert set(s["node_progress"]) == {"1", "2"}
    devices = ran_client.get("/api/devices").json()
    assert {d["id"] for d in devices} == {"jetson-01", "esp32-node-1", "esp32-node-2"}
    for d in devices:
        assert_shape(d, "Device", model=DeviceOut, minimum=9)


def test_shap(ran_client):
    local = ran_client.get("/api/ai/shap/1").json()
    assert_shape(local, "LocalShapExplanation", model=LocalShapOut, minimum=15)
    assert local["features"] and all(f["direction"] in ("increases_prediction", "decreases_prediction", "neutral") for f in local["features"])
    global_ = ran_client.get("/api/ai/shap/global").json()
    assert_shape(global_, "GlobalShapData", model=GlobalShapOut, minimum=40)
    assert len(global_["features"]) == 15 and global_["beeswarm_points"]
    assert all(0 <= p["feature_value_norm"] <= 1 for p in global_["beeswarm_points"] if p["feature_value_norm"] is not None)


def test_model_metrics(ran_client):
    m = ran_client.get("/api/models/performance").json()
    assert_shape(m, "ModelMetrics", model=ModelMetricsOut, minimum=30)
    assert [p["class_name"] for p in m["per_class"]] == ["healthy", "low", "moderate", "high"]
    assert len(m["confusion_matrix"]["matrix"]) == 4


def test_recommendations_everywhere(ran_client):
    recs = ran_client.get("/api/recommendations").json()
    assert recs and all(r["priority"] in ("high", "medium", "low") and r["supporting_sensor_values"] for r in recs)
    for rec in recs[:5]:
        assert_shape(rec, "Recommendation", model=RecommendationOut, minimum=25)
    overview = ran_client.get("/api/monitoring/overview").json()
    OverviewOut.model_validate(overview)
    assert overview["topRecommendations"]
    for rec in overview["topRecommendations"]:
        assert_shape(rec, "Recommendation", "overview.topRecommendations")
    assert_shape(overview["latestSession"], "Session", "overview.latestSession")
    detail = ran_client.get("/api/monitoring/plants/1").json()
    PlantMonitoringOut.model_validate(detail)
    assert_shape(detail["plant"], "Plant", "monitoring.plant", minimum=40)
    for rec in detail["recommendations"]:
        assert_shape(rec, "Recommendation", "monitoring.recommendations")


def test_settings(ran_client):
    settings = ran_client.get("/api/settings").json()
    assert_shape(settings, "SystemSettings", model=SystemSettingsOut, minimum=8)
    assert "api_base_url" not in settings and "demo_mode" not in settings


def test_endpoints_without_a_ts_interface_match_their_models(ran_client):
    InsightsOut.model_validate(ran_client.get("/api/ai/insights").json())
    for item in ran_client.get("/api/ai/anomalies").json()[:5]:
        AnomalyOut.model_validate(item)
        assert_shape(item, "Anomaly", model=AnomalyOut, minimum=10)
    summary = ran_client.get("/api/ai/anomalies/summary").json()
    assert_shape(summary, "AnomalySummary", model=AnomalySummaryOut, minimum=14)
    assert_shape(summary["plants"][0], "AnomalyPlantSummary", "anomalies.summary.plants")
    items = ran_client.get("/api/predictions?limit=5").json()
    assert len(items) == 5
    for item in items:
        PredictionListItem.model_validate(item)
    post = ran_client.post("/api/predictions", json={"plantId": 3, "explain": False})
    assert post.status_code == 200
    PredictionListItem.model_validate(post.json())
    assert set(post.json()) == set(items[0])  # POST and GET share one shape
