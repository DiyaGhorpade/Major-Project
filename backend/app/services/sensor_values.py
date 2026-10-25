"""The one definition of a usable sensor value: not missing, and not flagged out of range.

A reading's `validation_error` lists problems as `missing: soil; out_of_range: temperature=45.0` (written by the
seed and by ingestion). Averages in recommendations and in the dashboard summaries must skip both kinds.
"""

import re
from collections.abc import Iterable

from app.db.models import SensorReading


def flagged_out_of_range(reading: SensorReading, attr: str) -> bool:
    """True when the reading's `validation_error` lists `attr` under `out_of_range`."""
    return any(
        part.strip().startswith("out_of_range:") and re.search(rf"(?:^|[\s:,]){attr}=", part)
        for part in (reading.validation_error or "").split(";")
    )


def valid_value(reading: SensorReading, attr: str) -> float | None:
    """The reading's value for `attr`, or None if it is missing or flagged out of range."""
    value = getattr(reading, attr)
    return None if value is None or flagged_out_of_range(reading, attr) else value


def valid_values(readings: Iterable[SensorReading], attr: str) -> list[float]:
    return [v for r in readings if (v := valid_value(r, attr)) is not None]
