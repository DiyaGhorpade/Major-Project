"""CSV loading and consistent timestamp / dtype normalisation."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .schema import INFERENCE_REQUIRED_COLUMNS, REQUIRED_TRAIN_COLUMNS, SENSOR_COLUMNS


class DataLoadError(ValueError):
    """Raised with a human-readable message when a CSV cannot be loaded."""


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_timestamps(values: pd.Series, timezone: str = "Asia/Kolkata") -> pd.Series:
    """Parse ISO-8601 strings into tz-aware datetimes in ``timezone``.

    * values with an explicit offset are converted to ``timezone``;
    * naive values are ASSUMED to already be in ``timezone`` (documented behaviour);
    * unparseable values become NaT (validation reports them; they are never dropped silently).
    """
    s = pd.Series(values).astype("object")
    parsed = pd.to_datetime(s, errors="coerce", utc=True, format="ISO8601")
    # pd.to_datetime(utc=True) treats naive values as UTC; undo that for naive strings.
    naive_mask = s.map(lambda v: isinstance(v, str) and len(v) >= 10 and _is_naive_iso(v))
    if naive_mask.any():
        naive_parsed = pd.to_datetime(s[naive_mask], errors="coerce", format="ISO8601")
        try:
            naive_parsed = naive_parsed.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")
            parsed = parsed.copy()
            parsed[naive_mask] = naive_parsed.dt.tz_convert("UTC")
        except (TypeError, ValueError):
            pass
    return parsed.dt.tz_convert(timezone)


def _is_naive_iso(text: str) -> bool:
    t = text.strip()
    if t.endswith("Z") or t.endswith("z"):
        return False
    tail = t[10:]
    return ("+" not in tail) and ("-" not in tail)


def normalise_frame(df: pd.DataFrame, timezone: str = "Asia/Kolkata") -> pd.DataFrame:
    """Coerce dtypes. Invalid values become NaN/NaT so validation can report them."""
    out = df.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = parse_timestamps(out["timestamp"], timezone)
    for col in SENSOR_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    for col in ("plant_id", "session_id", "sampling_point", "is_synthetic_anomaly"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "node_id" in out.columns:
        out["node_id"] = out["node_id"].map(
            lambda v: None if pd.isna(v) else (str(int(v)) if isinstance(v, (int, float)) and float(v).is_integer() else str(v).strip())
        )
    for col in ("stress_level", "stress_type", "experimental_group"):
        if col in out.columns:
            out[col] = out[col].map(lambda v: None if pd.isna(v) else str(v).strip().lower())
    return out


def load_csv(path: str | Path, timezone: str = "Asia/Kolkata", require_labels: bool = True) -> pd.DataFrame:
    """Load a sensor CSV with clear errors. Does NOT validate ranges (see validation.py)."""
    p = Path(path)
    if not p.exists():
        raise DataLoadError(f"Input file not found: {p}")
    if p.stat().st_size == 0:
        raise DataLoadError(f"Input file is empty: {p}")
    try:
        raw = pd.read_csv(p)
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise DataLoadError(f"Could not parse CSV {p}: {exc}") from exc
    needed = REQUIRED_TRAIN_COLUMNS if require_labels else INFERENCE_REQUIRED_COLUMNS
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise DataLoadError(
            f"{p.name} is missing required column(s): {missing}. "
            f"Expected {'training' if require_labels else 'inference'} columns: {needed}"
        )
    return normalise_frame(raw, timezone)
