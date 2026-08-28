from dataclasses import dataclass
from typing import Union

import config


# ============================================================
# VALIDATION ERROR
# ============================================================

@dataclass
class ValidationError:
    sensor: str     # field name that failed, or "record" for structural errors
    bad_value: str  # raw string value that failed (always str for logging)
    reason: str     # human-readable explanation
    plant_id: str   # raw plant_id string (may be empty for structural errors)
    node_id: str = ""  # raw node_id string if available, else empty


# ============================================================
# VALIDATE RECORD
# ============================================================

def validate_record(fields: list[str]) -> Union[dict, list[ValidationError]]:
    """
    Validate a raw CSV fields list.

    Returns a parsed dict on success:
        {
            "timestamp": str,
            "session_id": int,
            "sampling_point": int,
            "plant_id": int,
            "node_id": str,
            "soil": float,
            "temperature": float,
            "humidity": float,
            "light": float,
        }

    Returns a non-empty list[ValidationError] on any failure.
    The first error encountered causes an immediate return (fail-fast).
    """

    # --------------------------------------------------------
    # Structural validation — wrong field count
    # --------------------------------------------------------

    if len(fields) != config.EXPECTED_FIELDS:
        return [
            ValidationError(
                sensor="record",
                bad_value=",".join(fields),
                reason=f"Expected {config.EXPECTED_FIELDS} fields, received {len(fields)}",
                plant_id="",
            )
        ]

    if not config.validate_config():
        return [
            ValidationError(
                sensor="record",
                bad_value=",".join(fields),
                reason="Invalid node configuration", 
                plant_id="",
            )
        ]

    # --------------------------------------------------------
    # Extract raw fields
    # --------------------------------------------------------

    timestamp        = fields[0]
    session_id_raw   = fields[1]
    sampling_point_raw = fields[2]
    plant_id_raw     = fields[3]
    node_id_raw      = fields[4]
    soil_raw         = fields[5]
    temperature_raw  = fields[6]
    humidity_raw     = fields[7]
    light_raw        = fields[8]

    # --------------------------------------------------------
    # Node ID validation
    # --------------------------------------------------------

    if node_id_raw is None or node_id_raw.strip() == "":
        return [
            ValidationError(
                sensor="node_id",
                bad_value="" if node_id_raw is None else node_id_raw,
                reason="Missing node_id",
                plant_id=plant_id_raw,
                node_id="MISSING",
            )
        ]

    if not config.is_valid_node_id(node_id_raw):
        if node_id_raw.strip().startswith(config.NODE_ID_PREFIX):
            reason = "Node ID out of range"
        else:
            reason = "Expected node_id in NODE_XX format"
        return [
            ValidationError(
                sensor="node_id",
                bad_value=node_id_raw,
                reason=reason,
                plant_id=plant_id_raw,
                node_id=node_id_raw,
            )
        ]

    # --------------------------------------------------------
    # Parse integer fields
    # --------------------------------------------------------

    try:
        session_id = int(session_id_raw)
    except ValueError:
        return [
            ValidationError(
                sensor="session_id",
                bad_value=session_id_raw,
                reason="Expected an integer",
                plant_id=plant_id_raw,
                node_id=node_id_raw,
            )
        ]

    try:
        sampling_point = int(sampling_point_raw)
    except ValueError:
        return [
            ValidationError(
                sensor="sampling_point",
                bad_value=sampling_point_raw,
                reason="Expected an integer from 1 to 6",
                plant_id=plant_id_raw,
                node_id=node_id_raw,
            )
        ]

    try:
        plant_id = int(plant_id_raw)
    except ValueError:
        return [
            ValidationError(
                sensor="plant_id",
                bad_value=plant_id_raw,
                reason="Expected an integer from 1 to 16",
                plant_id=plant_id_raw,
                node_id=node_id_raw,
            )
        ]

    # --------------------------------------------------------
    # Parse float fields
    # (plant_id is now parsed; use str(plant_id) in errors)
    # --------------------------------------------------------

    try:
        soil = float(soil_raw)
    except ValueError:
        return [
            ValidationError(
                sensor="soil",
                bad_value=soil_raw,
                reason="Expected a numeric value",
                plant_id=str(plant_id),
                node_id=node_id_raw,
            )
        ]

    try:
        temperature = float(temperature_raw)
    except ValueError:
        return [
            ValidationError(
                sensor="temperature",
                bad_value=temperature_raw,
                reason="Expected a numeric value",
                plant_id=str(plant_id),
                node_id=node_id_raw,
            )
        ]

    try:
        humidity = float(humidity_raw)
    except ValueError:
        return [
            ValidationError(
                sensor="humidity",
                bad_value=humidity_raw,
                reason="Expected a numeric value",
                plant_id=str(plant_id),
                node_id=node_id_raw,
            )
        ]

    try:
        light = float(light_raw)
    except ValueError:
        return [
            ValidationError(
                sensor="light",
                bad_value=light_raw,
                reason="Expected a numeric value",
                plant_id=str(plant_id),
                node_id=node_id_raw,
            )
        ]

    # --------------------------------------------------------
    # Range checks (fail-fast; bad_value = str of parsed value)
    # --------------------------------------------------------

    if session_id < config.VALID_SESSION_ID_MIN:
        return [
            ValidationError(
                sensor="session_id",
                bad_value=str(session_id),
                reason="Expected session_id >= 1",
                plant_id=str(plant_id),
                node_id=node_id_raw,
            )
        ]

    if (
        sampling_point < config.VALID_SAMPLING_POINT_MIN
        or sampling_point > config.VALID_SAMPLING_POINT_MAX
    ):
        return [
            ValidationError(
                sensor="sampling_point",
                bad_value=str(sampling_point),
                reason="Expected sampling point 1-6",
                plant_id=str(plant_id),
                node_id=node_id_raw,
            )
        ]

    if plant_id < config.VALID_PLANT_ID_MIN or plant_id > config.VALID_PLANT_ID_MAX:
        return [
            ValidationError(
                sensor="plant_id",
                bad_value=str(plant_id),
                reason="Expected plant_id 1-16",
                plant_id=str(plant_id),
                node_id=node_id_raw,
            )
        ]

    if soil < config.VALID_SOIL_MIN or soil > config.VALID_SOIL_MAX:
        return [
            ValidationError(
                sensor="soil",
                bad_value=str(soil),
                reason="Expected 0-100%",
                plant_id=str(plant_id),
                node_id=node_id_raw,
            )
        ]

    if (
        temperature < config.VALID_TEMPERATURE_MIN
        or temperature > config.VALID_TEMPERATURE_MAX
    ):
        return [
            ValidationError(
                sensor="temperature",
                bad_value=str(temperature),
                reason="Expected -40 to 80 C",
                plant_id=str(plant_id),
                node_id=node_id_raw,
            )
        ]

    if humidity < config.VALID_HUMIDITY_MIN or humidity > config.VALID_HUMIDITY_MAX:
        return [
            ValidationError(
                sensor="humidity",
                bad_value=str(humidity),
                reason="Expected 0-100%",
                plant_id=str(plant_id),
                node_id=node_id_raw,
            )
        ]

    if light < config.VALID_LIGHT_MIN or light > config.VALID_LIGHT_MAX:
        return [
            ValidationError(
                sensor="light",
                bad_value=str(light),
                reason="Expected 0-1000 lux",
                plant_id=str(plant_id),
                node_id=node_id_raw,
            )
        ]

    # --------------------------------------------------------
    # All checks passed — return parsed dict
    # --------------------------------------------------------

    return {
        "timestamp":      timestamp,
        "session_id":     session_id,
        "sampling_point": sampling_point,
        "plant_id":       plant_id,
        "node_id":        node_id_raw,
        "soil":           soil,
        "temperature":    temperature,
        "humidity":       humidity,
        "light":          light,
    }
