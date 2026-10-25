"""Pure mapping functions (PHASE0_FINDINGS section 4)."""

import pytest

from app.core import mappings as m
from app.core.plant_catalog import PLANT_IDS, catalog_entry


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(1, "1"), (2, "2"), ("1", "1"), (" 2 ", "2"), ("1.0", "1"), (2.0, "2"),
     (3, None), (0, None), (None, None), ("", None), ("abc", None),
     ("inf", None), ("nan", None), (float("inf"), None), (float("nan"), None)],
)  # fmt: skip
def test_normalise_node_id(raw, expected):
    assert m.normalise_node_id(raw) == expected


def test_node_for_plant_uses_data_then_parity():
    assert [m.node_for_plant(p) for p in (1, 2, 15, 16)] == ["1", "2", "1", "2"]
    assert m.node_for_plant(2, 1) == "1"  # observed node wins over parity
    assert m.node_for_plant(2, "garbage") == "2"


def test_experimental_groups_cover_each_plant():
    expected = {
        range(1, 5): ("Cluster 1", "Cluster 1: Healthy Control", "Healthy"),
        range(5, 9): ("Cluster 2", "Cluster 2: Water Stress", "Water Stress"),
        range(9, 13): ("Cluster 3", "Cluster 3: Light Stress", "Light Stress"),
        range(13, 17): ("Cluster 4", "Cluster 4: Temperature/Humidity Stress", "Temperature/Humidity Stress"),
    }
    assert len(PLANT_IDS) == 16
    for plant_ids, (cluster, label, stress_type) in expected.items():
        for plant_id in plant_ids:
            group = m.group_for_plant(plant_id)
            assert (group.cluster, group.label, group.stress_type) == (cluster, label, stress_type)
    for bad in (0, 17):
        with pytest.raises(ValueError):
            m.group_for_plant(bad)


@pytest.mark.parametrize(
    ("severity", "status"),
    [("healthy", "healthy"), ("low", "attention"), ("moderate", "attention"), ("high", "critical"),
     (None, "unmeasured"), ("bogus", "unmeasured")],
)  # fmt: skip
def test_severity_to_status(severity, status):
    assert m.severity_to_status(severity) == status


def test_status_signature_has_no_anomaly_input():
    # Anomalies must never influence health status: the mapping only accepts severity.
    assert list(m.severity_to_status.__annotations__) == ["severity", "return"]


def test_stress_type_guess():
    assert m.stress_type_guess(None, ["soil"]) is None
    assert m.stress_type_guess("healthy", ["soil", "light"]) == "Healthy"
    assert m.stress_type_guess("moderate", ["soil"]) == "Water Stress"
    assert m.stress_type_guess("high", ["light", "light"]) == "Light Stress"
    assert m.stress_type_guess("low", ["temperature_humidity"]) == "Temperature/Humidity Stress"
    assert m.stress_type_guess("high", ["soil", "light"]) == "Mixed Stress"
    assert m.stress_type_guess("high", []) == "Unclassified Stress"
    assert m.stress_type_guess("high", ["data_quality", "general"]) == "Unclassified Stress"


def test_ml_category_to_frontend_category():
    assert {c: m.map_ml_category(c) for c in m.ML_REC_CATEGORY_BY_ID.values()} == {
        "data_quality": "sensor_check",
        "model_confidence": "monitoring",
        "soil": "irrigation",
        "light": "lighting",
        "temperature_humidity": "climate",
        "general": "monitoring",
    }
    assert m.map_ml_category("unknown") == "monitoring"


def test_rec_id_to_category_and_title():
    assert len(m.ML_REC_CATEGORY_BY_ID) == 9 and set(m.ML_REC_TITLES) == set(m.ML_REC_CATEGORY_BY_ID)
    assert m.ML_REC_CATEGORY_BY_ID["verify_sensor"] == "data_quality"
    assert m.ML_REC_CATEGORY_BY_ID["check_soil_drying"] == "soil"
    assert m.ML_REC_CATEGORY_BY_ID["check_climate"] == "temperature_humidity"
    assert m.ml_rec_title("check_watering") == "Check watering"
    assert m.ml_rec_title("brand_new_id") == "Brand new id"


def test_env_state_label_is_not_a_stress_level():
    assert m.env_state_label(2) == "Environmental state group 2"


def test_nutrient_stress_is_never_produced():
    produced = {g.stress_type for g in m.EXPERIMENTAL_GROUPS} | {catalog_entry(p).group.stress_type for p in PLANT_IDS}
    for severity in ("healthy", "low", "moderate", "high", None):
        for categories in ([], ["soil"], ["light"], ["temperature_humidity"], ["soil", "light"], ["general"]):
            produced.add(m.stress_type_guess(severity, categories))
    produced.discard(None)
    assert "Nutrient Stress" not in produced
    assert not any("nutrient" in v.lower() for v in produced)
    for p in PLANT_IDS:
        entry = catalog_entry(p)
        assert "nutrient" not in f"{entry.description} {entry.simple_care_tip} {entry.group.label}".lower()
