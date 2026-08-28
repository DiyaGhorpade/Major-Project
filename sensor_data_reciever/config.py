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

EXPECTED_FIELDS: int = 8 

INPUT_HEADER: list = [
    "timestamp",
    "session_id",
    "sampling_point",
    "plant_id",
    "soil",
    "temperature",
    "humidity",
    "light",
]


# ============================================================
# OUTPUT STRUCTURES
# ============================================================

OUTPUT_HEADER: list = [
    "record_id",
    "timestamp",
    "session_id",
    "sampling_point",
    "plant_id",
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
