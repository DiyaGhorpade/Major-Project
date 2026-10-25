"""Per-plant history slices for stateless ML calls (DEV-1).

For the reading at index `i` of ONE plant's time-ordered series, the history is the (up to) `n` rows
immediately before it: `[max(0, i - n), i)`. Rows with an equal or later timestamp are never included
(the ML service rejects them), so the first reading of a plant has an empty (cold-start) history.
Readings with null sensor values are kept: the ML service accepts nulls and handles them itself.
"""

import logging
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy.orm import Session

from app.db import repositories as repo
from app.db.models import SensorReading

logger = logging.getLogger("app.history")

DEFAULT_MAX_HISTORY_ROWS = 500
DEFAULT_MAX_BATCH_ITEMS = 64
DEFAULT_REQUIRED_ROWS = 49


@lru_cache(maxsize=16)
def _parse_yaml(path: str, mtime_ns: int) -> dict[str, Any]:
    """Parse the ML YAML once per (path, modification time); errors propagate to the (cached) callers below."""
    return yaml.safe_load(Path(path).read_text()) or {}


@lru_cache(maxsize=16)
def _limits(path: str, mtime_ns: int | None) -> tuple[int, int]:
    """`(max_history_rows, max_batch_items)`; the warning for a bad file is logged once per file version."""
    try:
        if mtime_ns is None:
            raise FileNotFoundError(path)
        api = _parse_yaml(path, mtime_ns).get("api") or {}
        max_history, max_batch = int(api["max_history_rows"]), int(api["max_batch_items"])
        if max_history < 1 or max_batch < 1:
            raise ValueError(f"limits must be positive (got {max_history}/{max_batch})")
        return max_history, max_batch
    except Exception as exc:  # unreadable file, bad YAML, missing or non-positive values: never block a run
        logger.warning(
            "cannot read ML limits from %s (%s: %s); using defaults %d/%d",
            path, type(exc).__name__, exc, DEFAULT_MAX_HISTORY_ROWS, DEFAULT_MAX_BATCH_ITEMS,
        )  # fmt: skip
        return DEFAULT_MAX_HISTORY_ROWS, DEFAULT_MAX_BATCH_ITEMS


def _mtime(config_path: Path) -> int | None:
    try:
        return config_path.stat().st_mtime_ns
    except OSError:
        return None


def read_ml_limits(config_path: Path) -> tuple[int, int]:
    """`(max_history_rows, max_batch_items)` from the ML YAML (a file read, not an import).

    Cached per file version (re-read only when the mtime changes). Falls back to the documented defaults
    (500 / 64) with a logged warning if the file is unreadable.
    """
    return _limits(str(config_path), _mtime(config_path))


DEFAULT_DAYTIME_HOURS = (9, 16)


@lru_cache(maxsize=16)
def _daytime(path: str, mtime_ns: int | None) -> tuple[int, int]:
    try:
        if mtime_ns is None:
            raise FileNotFoundError(path)
        start, end = (int(x) for x in _parse_yaml(path, mtime_ns)["recommendations"]["daytime_hours"])
        if not 0 <= start <= end <= 23:
            raise ValueError(f"invalid hours {start}-{end}")
        return start, end
    except Exception as exc:  # missing file/key or bad values: the documented default applies
        logger.warning("cannot read daytime hours from %s (%s: %s); using %s", path, type(exc).__name__, exc, DEFAULT_DAYTIME_HOURS)
        return DEFAULT_DAYTIME_HOURS


def read_daytime_hours(config_path: Path) -> tuple[int, int]:
    """Inclusive local daytime hours from the ML YAML (`recommendations.daytime_hours`), default 9-16.
    Cached per file version; a missing/invalid value logs a warning once and uses the default."""
    return _daytime(str(config_path), _mtime(config_path))


def history_rows_for(max_history_rows: int, required_rows: int | None, default_rows: int) -> int:
    """`min(max_history_rows, max(required_rows, default_rows))` (DEV-1); always at least 1."""
    rows = min(max_history_rows, max(required_rows or DEFAULT_REQUIRED_ROWS, default_rows))
    if rows < 1:
        logger.warning("computed history_rows=%d is not positive; using %d", rows, DEFAULT_REQUIRED_ROWS)
        return DEFAULT_REQUIRED_ROWS
    return rows


def history_bounds(series: Sequence[Any], i: int, n: int) -> tuple[int, int]:
    """`(start, end)` such that `series[start:end]` is the history of `series[i]` (see `history_slice`)."""
    if not 0 <= i < len(series):
        raise ValueError(f"index {i} outside series of length {len(series)}")
    end = i
    while end > 0 and series[end - 1].timestamp >= series[i].timestamp:  # equal-timestamp guard
        end -= 1
    return (max(0, end - n) if n > 0 else end), end


def history_slice(series: Sequence[Any], i: int, n: int) -> list[Any]:
    """History for `series[i]`: same plant, strictly earlier, at most `n` rows, oldest first.

    `series` must be one plant's readings sorted by timestamp (ties allowed in the input but
    never returned as history). Raises ValueError for a mixed-plant series or a bad index.
    """
    start, end = history_bounds(series, i, n)
    out = list(series[start:end])
    current = series[i]
    for row in out:
        if row.plant_id != current.plant_id:
            raise ValueError(f"history mixes plants {row.plant_id} and {current.plant_id}")
        if not row.timestamp < current.timestamp:
            raise ValueError("history row is not strictly earlier than the reading")
    return out


def history_for_reading(db: Session, reading: SensorReading, n: int) -> list[SensorReading]:
    """DB helper: the `n` latest readings of the same plant strictly before `reading`, oldest first."""
    rows = repo.readings_before(db, reading.plant_id, reading.timestamp, n)
    history_slice(rows + [reading], len(rows), n)  # validates plant/ordering invariants
    return rows
