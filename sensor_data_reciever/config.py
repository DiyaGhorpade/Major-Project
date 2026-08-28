# ============================================================
# SERIAL COMMUNICATION
# ============================================================

SERIAL_PORT: str = "COM5"
BAUD_RATE: int = 115200


# ============================================================
# FILE PATHS
# ============================================================

SENSOR_DATA_FILE: str = "sensor_data.csv"
ERROR_LOG_FILE: str = "error_log.csv"


# ============================================================
# EXPECTED INPUT STRUCTURE
# ============================================================

# NOTE: This project currently defines the sensor record as
# timestamp, session_id, sampling_point, plant_id, node_id, soil,
# temperature, humidity, light.
# The exact node-ID naming convention is configured centrally below.

EXPECTED_FIELDS: int = 9

INPUT_HEADER: list = [
    "timestamp",
    "session_id",
    "sampling_point",
    "plant_id",
    "node_id",
    "soil",
    "temperature",
    "humidity",
    "light",
]


# ============================================================
# NODE ID CONFIGURATION
# ============================================================

NUM_NODES: int = 4
NODE_ID_PREFIX: str = "NODE_"
NODE_ID_WIDTH: int = 2


def validate_config() -> bool:
    """Validate the startup configuration for the node deployment."""
    if not isinstance(NUM_NODES, int):
        return False
    if NUM_NODES <= 0:
        return False
    if not isinstance(NODE_ID_PREFIX, str) or not NODE_ID_PREFIX:
        return False
    if not isinstance(NODE_ID_WIDTH, int) or NODE_ID_WIDTH <= 0:
        return False
    return True


def is_valid_node_id(node_id: str) -> bool:
    """Return True if node_id matches the configured convention and range."""
    if not isinstance(node_id, str):
        return False

    value = node_id.strip()
    if not value:
        return False
    if not value.startswith(NODE_ID_PREFIX):
        return False

    suffix = value[len(NODE_ID_PREFIX):]
    if not suffix or not suffix.isdigit():
        return False
    if len(suffix) != NODE_ID_WIDTH:
        return False

    try:
        node_number = int(suffix)
    except ValueError:
        return False

    return 1 <= node_number <= NUM_NODES


# ============================================================
# OUTPUT STRUCTURES
# ============================================================

OUTPUT_HEADER: list = [
    "record_id",
    "timestamp",
    "session_id",
    "sampling_point",
    "plant_id",
    "node_id",
    "soil",
    "temperature",
    "humidity",
    "light",
]

ERROR_HEADER: list = [
    "record_id",
    "plant_id",
    "sensor",
    "bad_value",
    "reason",
    "node_id",
]


# ============================================================
# VALIDATION BOUNDS
# ============================================================

VALID_SESSION_ID_MIN: int = 1

VALID_SAMPLING_POINT_MIN: int = 1
VALID_SAMPLING_POINT_MAX: int = 6

VALID_PLANT_ID_MIN: int = 1
VALID_PLANT_ID_MAX: int = 16

VALID_SOIL_MIN: float = 0.0
VALID_SOIL_MAX: float = 100.0

VALID_TEMPERATURE_MIN: float = -40.0
VALID_TEMPERATURE_MAX: float = 80.0

VALID_HUMIDITY_MIN: float = 0.0
VALID_HUMIDITY_MAX: float = 100.0

VALID_LIGHT_MIN: float = 0.0
VALID_LIGHT_MAX: float = 1000.0
