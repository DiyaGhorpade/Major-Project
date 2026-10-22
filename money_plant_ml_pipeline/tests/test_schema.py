import pytest

from plant_ml import schema
from plant_ml.synthetic import generate
from plant_ml.validation import check_columns


def test_required_column_order_is_exact():
    assert schema.REQUIRED_TRAIN_COLUMNS == ["timestamp", "session_id", "node_id", "sampling_point", "plant_id", "soil",
                                              "temperature", "humidity", "light", "stress_level", "stress_type"]
    assert schema.OPTIONAL_COLUMNS == ["experimental_group", "is_synthetic_anomaly"]


def test_check_columns_accepts_required_plus_optional_in_order():
    assert check_columns(schema.REQUIRED_TRAIN_COLUMNS) == []
    assert check_columns(schema.REQUIRED_TRAIN_COLUMNS + ["is_synthetic_anomaly"]) == []
    assert check_columns(schema.REQUIRED_TRAIN_COLUMNS + schema.OPTIONAL_COLUMNS) == []


def test_check_columns_rejects_wrong_order_and_unknown_extras():
    cols = list(schema.REQUIRED_TRAIN_COLUMNS)
    cols[0], cols[1] = cols[1], cols[0]
    assert check_columns(cols)
    assert check_columns(schema.REQUIRED_TRAIN_COLUMNS + ["mystery"])
    assert check_columns(schema.REQUIRED_TRAIN_COLUMNS + ["is_synthetic_anomaly", "experimental_group"])


def test_inference_columns_do_not_need_labels():
    assert check_columns(["timestamp", "plant_id", "soil", "temperature", "humidity", "light"], require_labels=False) == []
    assert check_columns(["timestamp", "plant_id", "soil"], require_labels=False)


def test_group_mapping_is_fixed_4x4():
    assert {g: len(p) for g, p in schema.EXPERIMENTAL_GROUPS.items()} == {
        "healthy_control": 4, "water_stress": 4, "light_stress": 4, "temperature_humidity_stress": 4}
    assert schema.plant_to_group(1) == "healthy_control" and schema.plant_to_group(8) == "water_stress"
    assert schema.plant_to_group(9) == "light_stress" and schema.plant_to_group(16) == "temperature_humidity_stress"
    with pytest.raises(ValueError):
        schema.plant_to_group(17)


def test_generator_emits_schema_order_and_is_deterministic():
    a, b = generate(n_days=3, seed=1), generate(n_days=3, seed=1)
    assert list(a.columns) == schema.REQUIRED_TRAIN_COLUMNS + schema.OPTIONAL_COLUMNS
    assert a.equals(b)
    assert not a.equals(generate(n_days=3, seed=2))


def test_full_size_generator_assertions():
    from plant_ml.synthetic import check_generated, n_days_for_rows
    df = generate(n_days_for_rows(8000), seed=42)
    check_generated(df, 8000)
    assert len(df) >= 8000
