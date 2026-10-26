"""Reference-plant readings for the ML service's cross-plant features (ML issue #5).

The ML classifier compares each plant with the healthy-control ("reference") plants measured at the same time. The
service is stateless, so the caller sends those plants' readings next to the plant's own history. `/metadata` says
which plants are references (`cross_plant.reference_plants`); an empty list means the model needs none and nothing
is sent. The ML service only ever uses, per row, a reference reading at or before that row's timestamp, so sending a
slightly wider window than needed is harmless.
"""

from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta
from typing import Any

from app.db.models import SensorReading
from app.integrations.ml_types import MLReading

# Reference readings slightly older than the first history row are still matched by the ML as-of join (its tolerance
# is 30 minutes); one hour of margin covers that without sending meaningfully more data.
MARGIN = timedelta(hours=1)


def reference_plants(metadata: Any) -> list[int]:
    """Reference plant ids from `/metadata` (`[]` when the model has no cross-plant features or the field is absent)."""
    block = getattr(metadata, "cross_plant", None)
    if block is None and isinstance(getattr(metadata, "model_extra", None), dict):
        block = metadata.model_extra.get("cross_plant")
    if not isinstance(block, dict) or not block.get("enabled"):
        return []
    try:
        return sorted({int(p) for p in block.get("reference_plants") or []})
    except (TypeError, ValueError):
        return []


class ReferenceIndex:
    """Time-ordered readings of the reference plants, sliced by time span."""

    def __init__(self, readings: Iterable[SensorReading], to_ml):
        rows = sorted(readings, key=lambda r: (r.timestamp, r.plant_id, r.id))
        self._times: list[datetime] = [r.timestamp for r in rows]
        self._ml: list[MLReading] = [to_ml(r) for r in rows]

    def __len__(self) -> int:
        return len(self._ml)

    def for_series(self, spans: Sequence[tuple[datetime, datetime]]) -> list[MLReading]:
        """Readings covering the union of `(first history timestamp, reading timestamp)` spans (deduplicated)."""
        merged: list[list[datetime]] = []
        for start, end in sorted((s - MARGIN, e) for s, e in spans):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        out: list[MLReading] = []
        for start, end in merged:  # disjoint, increasing intervals -> no duplicates
            out += self._ml[bisect_left(self._times, start) : bisect_right(self._times, end)]
        return out
