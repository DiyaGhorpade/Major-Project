"""Single home for every dataset <-> ML <-> frontend mapping decision (PHASE0_FINDINGS section 4).

Decisions documented here:

* Node ids: the CSV stores ints, the API/DB/ML use the strings "1" and "2"
  (`normalise_node_id`). Node follows the CSV: odd plants -> "1", even plants -> "2". The
  parity rule in `node_for_plant` is only a fallback when the data carries no node (DEV-3).
* "Cluster" (`Plant.cluster`) is the fixed *experimental group* of the study design
  (plants 1-4 / 5-8 / 9-12 / 13-16), never a KMeans id. KMeans output is an "environmental
  state group" (`env_state_label`) and must not be used for health or status.
* Stress has two axes. Severity (healthy/low/moderate/high) comes from the ML classifier.
  Type is the group's designed treatment (`GroupInfo.stress_type`) for the plant, and a
  labelled heuristic (`stress_type_guess`) for a prediction. The dataset has no nutrient
  stress; the frontend's "Nutrient Stress" is replaced by "Temperature/Humidity Stress" (DEV-5).
* `Plant.status` derives from severity only (`severity_to_status`); anomalies never change it.
* ML recommendation categories/ids map to the frontend's `RecommendationCategory` and titles;
  `monitoring` is a new frontend category (DEV-7). `nutrition` is never produced.
* Timestamps are stored as UTC and emitted as ISO-8601 in Asia/Kolkata (+05:30); naive input
  is assumed to already be IST.
"""

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

NodeId = Literal["1", "2"]
PlantHealthStatus = Literal["healthy", "attention", "critical", "stale", "unmeasured"]
StressType = Literal[
    "Healthy",
    "Water Stress",
    "Light Stress",
    "Temperature/Humidity Stress",
    "Mixed Stress",
    "Unclassified Stress",
]
RecommendationCategory = Literal[
    "irrigation", "lighting", "nutrition", "sensor_check", "climate", "monitoring"
]


# --- time -----------------------------------------------------------------------------------
def ensure_aware(value: datetime) -> datetime:
    """Return `value` tz-aware; naive datetimes are assumed to be IST."""
    return value.replace(tzinfo=IST) if value.tzinfo is None else value


def parse_timestamp(value: object) -> datetime | None:
    """Parse an ISO-8601 string to a tz-aware datetime (naive -> IST); unparseable -> None."""
    try:
        return ensure_aware(datetime.fromisoformat(str(value).strip()))
    except ValueError:
        return None


def to_ist_iso(value: datetime | None) -> str | None:
    """Format as ISO-8601 in Asia/Kolkata, e.g. `2026-09-21T00:00:00+05:30`."""
    return None if value is None else ensure_aware(value).astimezone(IST).isoformat()


# --- identifiers ----------------------------------------------------------------------------
def normalise_node_id(value: object) -> NodeId | None:
    """Map 1, 2, "1", "1.0", 2.0 ... to "1"/"2"; anything else (incl. missing) -> None."""
    if value is None:
        return None
    try:
        number = float(str(value).strip())
        if not math.isfinite(number):
            return None
        as_int = int(number)
    except (ValueError, OverflowError):
        return None
    return "1" if as_int == 1 else "2" if as_int == 2 else None


def node_for_plant(plant_id: int, observed: object = None) -> NodeId:
    """Node a plant belongs to: the node observed in the data wins, parity is the fallback."""
    return normalise_node_id(observed) or ("1" if plant_id % 2 == 1 else "2")


# --- experimental groups ("Cluster 1..4") ---------------------------------------------------
@dataclass(frozen=True)
class GroupInfo:
    cluster: str  # frontend ClusterId
    label: str  # e.g. "Cluster 1: Healthy Control"
    stress_type: StressType  # designed treatment of the group
    plant_ids: range


EXPERIMENTAL_GROUPS: tuple[GroupInfo, ...] = (
    GroupInfo("Cluster 1", "Cluster 1: Healthy Control", "Healthy", range(1, 5)),
    GroupInfo("Cluster 2", "Cluster 2: Water Stress", "Water Stress", range(5, 9)),
    GroupInfo("Cluster 3", "Cluster 3: Light Stress", "Light Stress", range(9, 13)),
    GroupInfo(
        "Cluster 4", "Cluster 4: Temperature/Humidity Stress", "Temperature/Humidity Stress", range(13, 17)
    ),
)


def group_for_plant(plant_id: int) -> GroupInfo:
    for group in EXPERIMENTAL_GROUPS:
        if plant_id in group.plant_ids:
            return group
    raise ValueError(f"plant_id {plant_id} is outside the 16-plant experimental design")


def env_state_label(cluster_id: int) -> str:
    """Label for a KMeans id, used verbatim (no +1 shift). Not a stress level."""
    return f"Environmental state group {cluster_id}"


# --- severity / type ------------------------------------------------------------------------
_STATUS_BY_SEVERITY: dict[str, PlantHealthStatus] = {
    "healthy": "healthy",
    "low": "attention",
    "moderate": "attention",
    "high": "critical",
}


def severity_to_status(severity: str | None) -> PlantHealthStatus:
    """ML severity -> Plant.status. No (or unknown) prediction -> 'unmeasured'."""
    return _STATUS_BY_SEVERITY.get(severity or "", "unmeasured")


_TYPE_BY_ML_CATEGORY: dict[str, StressType] = {
    "soil": "Water Stress",
    "light": "Light Stress",
    "temperature_humidity": "Temperature/Humidity Stress",
}


def stress_type_guess(severity: str | None, ml_rec_categories: list[str] | tuple[str, ...]) -> StressType | None:
    """Heuristic stress type for a prediction (never a diagnosis).

    healthy -> 'Healthy'; otherwise derived from the condition categories of the ML
    recommendations (soil/light/temperature_humidity): one -> that type, two or more ->
    'Mixed Stress', none -> 'Unclassified Stress'. No severity -> None.
    """
    if severity is None:
        return None
    if severity == "healthy":
        return "Healthy"
    types = {_TYPE_BY_ML_CATEGORY[c] for c in ml_rec_categories if c in _TYPE_BY_ML_CATEGORY}
    if not types:
        return "Unclassified Stress"
    return next(iter(types)) if len(types) == 1 else "Mixed Stress"


# --- ML recommendations ---------------------------------------------------------------------
_CATEGORY_BY_ML_CATEGORY: dict[str, RecommendationCategory] = {
    "soil": "irrigation",
    "light": "lighting",
    "temperature_humidity": "climate",
    "data_quality": "sensor_check",
    "model_confidence": "monitoring",
    "general": "monitoring",
}

ML_REC_CATEGORY_BY_ID: dict[str, str] = {
    "verify_sensor": "data_quality",
    "retake_reading": "data_quality",
    "low_confidence": "model_confidence",
    "check_watering": "soil",
    "check_soil_drying": "soil",
    "check_light_low": "light",
    "check_light_high": "light",
    "check_climate": "temperature_humidity",
    "monitor_and_inspect": "general",
}

ML_REC_TITLES: dict[str, str] = {
    "verify_sensor": "Check the sensor",
    "retake_reading": "Retake the reading",
    "low_confidence": "Low model confidence",
    "check_watering": "Check watering",
    "check_soil_drying": "Soil is drying out",
    "check_light_low": "Light is too low",
    "check_light_high": "Light is too high",
    "check_climate": "Check temperature and humidity",
    "monitor_and_inspect": "Monitor and inspect",
}


def map_ml_category(ml_category: str) -> RecommendationCategory:
    """ML recommendation category -> frontend RecommendationCategory (unknown -> 'monitoring')."""
    return _CATEGORY_BY_ML_CATEGORY.get(ml_category, "monitoring")


def ml_rec_title(ml_rec_id: str) -> str:
    """Backend-owned title for an ML recommendation id (the ML service sends none)."""
    return ML_REC_TITLES.get(ml_rec_id, ml_rec_id.replace("_", " ").capitalize())
