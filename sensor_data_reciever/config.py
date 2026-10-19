# ============================================================
# SERVER CONFIGURATION
# ============================================================

HOST: str = "0.0.0.0"   # bind on all interfaces so the ESP32 can reach us
PORT: int = 5000         # change to any available port if needed


# ============================================================
# FILE PATHS
# ============================================================

SENSOR_DATA_FILE: str = "sensor_data.csv"
ERROR_LOG_FILE: str = "error_log.csv"


# ============================================================
# CSV SCHEMA  (9 fields, order matters)
# ============================================================

EXPECTED_FIELDS: int = 9

CSV_HEADER: list = [
    "timestamp",
    "session_id",
    "node_id",
    "sampling_point",
    "plant_id",
    "soil",
    "temperature",
    "humidity",
    "light",
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
VALID_LIGHT_MAX: float = 100000.0   # lux; sunlight can exceed 1000
