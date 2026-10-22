"""Single source of truth for the CSV schema, units, ranges and label vocabularies."""
from __future__ import annotations

# --- Column contracts ---------------------------------------------------------------
REQUIRED_TRAIN_COLUMNS = [
    "timestamp", "session_id", "node_id", "sampling_point", "plant_id",
    "soil", "temperature", "humidity", "light", "stress_level", "stress_type",
]
OPTIONAL_COLUMNS = ["experimental_group", "is_synthetic_anomaly"]  # must come AFTER required, in this order

# Inference-time rows never need labels. session/node/sampling_point are optional context.
INFERENCE_REQUIRED_COLUMNS = ["timestamp", "plant_id", "soil", "temperature", "humidity", "light"]
INFERENCE_CONTEXT_COLUMNS = ["session_id", "node_id", "sampling_point"]

SENSOR_COLUMNS = ["soil", "temperature", "humidity", "light"]
SENSOR_UNITS = {"soil": "% (sensor-calibrated)", "temperature": "deg C", "humidity": "% RH", "light": "lux"}

# Hard physical / contract ranges (inclusive).
SENSOR_RANGES = {
    "soil": (0.0, 100.0),
    "temperature": (15.0, 40.0),
    "humidity": (20.0, 100.0),
    "light": (0.0, 100000.0),
}

# --- Label vocabularies -------------------------------------------------------------
STRESS_LEVELS = ["healthy", "low", "moderate", "high"]          # order == class index
LABEL_TO_INDEX = {name: i for i, name in enumerate(STRESS_LEVELS)}
INDEX_TO_LABEL = {i: name for name, i in LABEL_TO_INDEX.items()}
STRESS_TYPES = ["none", "water", "light", "temperature_humidity", "mixed"]

PLANT_ID_RANGE = (1, 16)
SESSION_MIN = 1
NODE_IDS = ["1", "2"]
SAMPLING_POINT_RANGE = (1, 12)

EXPERIMENTAL_GROUPS = {
    "healthy_control": list(range(1, 5)),
    "water_stress": list(range(5, 9)),
    "light_stress": list(range(9, 13)),
    "temperature_humidity_stress": list(range(13, 17)),
}


def plant_to_group(plant_id: int) -> str:
    """Fixed experimental group for a plant (used for reports only, never as a model input)."""
    for group, plants in EXPERIMENTAL_GROUPS.items():
        if int(plant_id) in plants:
            return group
    raise ValueError(f"plant_id {plant_id} is outside the experimental design (1-16)")


# --- Leakage guard ------------------------------------------------------------------
# These must never reach any model feature matrix (see feature_config.yaml + tests).
FORBIDDEN_MODEL_COLUMNS = [
    "stress_level", "stress_type", "experimental_group", "is_synthetic_anomaly",
    "plant_id", "session_id", "node_id", "sampling_point", "timestamp",
]
