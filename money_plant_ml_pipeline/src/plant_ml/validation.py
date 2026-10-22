"""Data validation with configurable behaviour: fail fast, warn (mask), or quarantine.

Rows are NEVER silently dropped. Rows that cannot be used for time-series features
(bad timestamp / plant id / duplicate key) are returned in ``quarantined`` with the reason.
Missing sensor values (NaN) are legal and are reported, not "fixed".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .labels import add_group_column_if_missing, class_counts
from .schema import (
    INFERENCE_REQUIRED_COLUMNS, NODE_IDS, OPTIONAL_COLUMNS, PLANT_ID_RANGE, REQUIRED_TRAIN_COLUMNS,
    SAMPLING_POINT_RANGE, SENSOR_COLUMNS, SENSOR_RANGES, SESSION_MIN, STRESS_LEVELS, STRESS_TYPES,
)

MODES = ("fail", "warn", "quarantine")
STRUCTURAL_ISSUES = {"bad_timestamp", "bad_plant_id", "duplicate_plant_timestamp"}


class ValidationError(ValueError):
    """Raised in ``fail`` mode (or for unusable schemas) with a readable message."""

    def __init__(self, message: str, report: dict | None = None):
        super().__init__(message)
        self.report = report or {}


@dataclass
class ValidationResult:
    clean: pd.DataFrame
    quarantined: pd.DataFrame
    report: dict
    row_issues: pd.Series = field(default=None)  # index-aligned issue strings for the INPUT frame


def check_columns(columns: list[str], require_labels: bool = True) -> list[str]:
    """Return schema errors (empty list == OK). Training mode enforces the exact column order."""
    errors: list[str] = []
    cols = list(columns)
    if require_labels:
        head = cols[: len(REQUIRED_TRAIN_COLUMNS)]
        missing = [c for c in REQUIRED_TRAIN_COLUMNS if c not in cols]
        if missing:
            errors.append(f"missing required columns: {missing}")
        elif head != REQUIRED_TRAIN_COLUMNS:
            errors.append(f"required columns must come first in this exact order {REQUIRED_TRAIN_COLUMNS}; got {head}")
        extras = cols[len(REQUIRED_TRAIN_COLUMNS):]
        unknown = [c for c in extras if c not in OPTIONAL_COLUMNS]
        if unknown:
            errors.append(f"unknown extra columns after required block: {unknown} (allowed: {OPTIONAL_COLUMNS})")
        ordered = [c for c in OPTIONAL_COLUMNS if c in extras]
        if not unknown and [c for c in extras] != ordered:
            errors.append(f"optional columns must appear in order {OPTIONAL_COLUMNS}; got {extras}")
    else:
        missing = [c for c in INFERENCE_REQUIRED_COLUMNS if c not in cols]
        if missing:
            errors.append(f"missing required inference columns: {missing}")
    return errors


def _row_issue_flags(df: pd.DataFrame) -> dict[str, pd.Series]:
    idx = df.index
    flags: dict[str, pd.Series] = {}
    flags["bad_timestamp"] = df["timestamp"].isna()
    pid = df["plant_id"]
    flags["bad_plant_id"] = pid.isna() | (pid < PLANT_ID_RANGE[0]) | (pid > PLANT_ID_RANGE[1]) | (pid % 1 != 0)
    for col in SENSOR_COLUMNS:
        lo, hi = SENSOR_RANGES[col]
        v = df[col]
        flags[f"out_of_range_{col}"] = v.notna() & ((v < lo) | (v > hi))
    if "session_id" in df.columns:
        s = df["session_id"]
        flags["bad_session_id"] = s.notna() & ((s < SESSION_MIN) | (s % 1 != 0))
    if "node_id" in df.columns:
        flags["bad_node_id"] = df["node_id"].notna() & ~df["node_id"].isin(NODE_IDS)
    if "sampling_point" in df.columns:
        sp = df["sampling_point"]
        flags["bad_sampling_point"] = sp.notna() & ((sp < SAMPLING_POINT_RANGE[0]) | (sp > SAMPLING_POINT_RANGE[1]))
    if "stress_level" in df.columns:
        flags["bad_stress_level"] = df["stress_level"].notna() & ~df["stress_level"].isin(STRESS_LEVELS)
    if "stress_type" in df.columns:
        flags["bad_stress_type"] = df["stress_type"].notna() & ~df["stress_type"].isin(STRESS_TYPES)
    if "is_synthetic_anomaly" in df.columns:
        a = df["is_synthetic_anomaly"]
        flags["bad_anomaly_label"] = a.notna() & ~a.isin([0, 1])
    key_ok = ~(flags["bad_timestamp"] | flags["bad_plant_id"])
    dup = pd.Series(False, index=idx, dtype=bool)
    dup.loc[key_ok] = df.loc[key_ok].duplicated(subset=["plant_id", "timestamp"], keep="first").to_numpy(dtype=bool)
    flags["duplicate_plant_timestamp"] = dup
    return flags


def build_report(df: pd.DataFrame, flags: dict[str, pd.Series], require_labels: bool, mode: str,
                 schema_errors: list[str], n_quarantined: int) -> dict:
    n = len(df)
    report: dict = {
        "mode": mode,
        "rows": int(n),
        "plants": int(df["plant_id"].nunique()) if n else 0,
        "plant_ids": sorted(int(p) for p in df["plant_id"].dropna().unique()) if n else [],
        "time_range": {
            "start": df["timestamp"].min().isoformat() if n and df["timestamp"].notna().any() else None,
            "end": df["timestamp"].max().isoformat() if n and df["timestamp"].notna().any() else None,
        },
        "schema_errors": schema_errors,
        "missingness": {c: {"count": int(df[c].isna().sum()), "pct": round(float(df[c].isna().mean() * 100), 3)}
                        for c in df.columns} if n else {},
        "duplicate_plant_timestamp": int(flags["duplicate_plant_timestamp"].sum()),
        "sensor_range_violations": {c: int(flags[f"out_of_range_{c}"].sum()) for c in SENSOR_COLUMNS},
        "other_issue_counts": {k: int(v.sum()) for k, v in flags.items()
                               if not k.startswith("out_of_range_") and k != "duplicate_plant_timestamp"},
        "rows_quarantined": int(n_quarantined),
        "rows_per_plant": {int(k): int(v) for k, v in df["plant_id"].value_counts().sort_index().items()} if n else {},
        "sensor_ranges_used": {k: list(v) for k, v in SENSOR_RANGES.items()},
    }
    if "stress_level" in df.columns:
        report["stress_level_counts"] = class_counts(df["stress_level"])
    if "stress_type" in df.columns:
        report["stress_type_counts"] = {str(k): int(v) for k, v in df["stress_type"].value_counts(dropna=False).items()}
    if require_labels and "plant_id" in df.columns and n:
        g = add_group_column_if_missing(df)
        report["experimental_group_counts"] = {str(k): int(v) for k, v in g["experimental_group"].value_counts().items()}
        report["experimental_group_source"] = "column" if "experimental_group" in df.columns else "derived_from_plant_id (report only)"
    if "is_synthetic_anomaly" in df.columns:
        report["helper_anomaly_label_counts"] = {str(int(k)): int(v) for k, v in
                                                  df["is_synthetic_anomaly"].dropna().value_counts().items()}
    if n and df["timestamp"].notna().any():
        d = (df.dropna(subset=["timestamp", "plant_id"]).sort_values(["plant_id", "timestamp"])
               .groupby("plant_id")["timestamp"].diff().dt.total_seconds() / 3600.0).dropna()
        report["cadence_hours"] = ({"median": round(float(d.median()), 4), "min": round(float(d.min()), 4),
                                    "max": round(float(d.max()), 4)} if len(d) else None)
    return report


def validate_frame(df: pd.DataFrame, mode: str = "warn", require_labels: bool = True) -> ValidationResult:
    """Validate a normalised frame (see data_loader.normalise_frame)."""
    if mode not in MODES:
        raise ValueError(f"validation mode must be one of {MODES}, got {mode!r}")
    schema_errors = check_columns(list(df.columns), require_labels)
    hard_missing = [c for c in (REQUIRED_TRAIN_COLUMNS if require_labels else INFERENCE_REQUIRED_COLUMNS)
                    if c not in df.columns]
    if hard_missing:
        raise ValidationError(f"Cannot validate: missing required columns {hard_missing}", {"schema_errors": schema_errors})
    if mode == "fail" and schema_errors:
        raise ValidationError("Schema errors: " + "; ".join(schema_errors), {"schema_errors": schema_errors})

    flags = _row_issue_flags(df)
    names = list(flags.keys())
    issue_matrix = pd.DataFrame(flags)
    row_issues = issue_matrix.apply(lambda r: ",".join(n for n in names if r[n]), axis=1) if len(df) else pd.Series(dtype=str)
    any_issue = issue_matrix.any(axis=1) if len(df) else pd.Series(dtype=bool)
    structural = issue_matrix[[c for c in names if c in STRUCTURAL_ISSUES]].any(axis=1) if len(df) else any_issue

    if mode == "fail":
        quarantine_mask = pd.Series(False, index=df.index)
    elif mode == "quarantine":
        quarantine_mask = any_issue
    else:  # warn
        quarantine_mask = structural

    clean = df.loc[~quarantine_mask].copy()
    quarantined = df.loc[quarantine_mask].copy()
    if len(quarantined):
        quarantined["_issues"] = row_issues[quarantine_mask]

    if mode == "warn":
        # Mask (do not delete) invalid values so they cannot reach a model; counts are in the report.
        for col in SENSOR_COLUMNS:
            bad = flags[f"out_of_range_{col}"].loc[clean.index]
            clean.loc[bad, col] = np.nan
        for col, flag in (("stress_level", "bad_stress_level"), ("stress_type", "bad_stress_type")):
            if flag in flags:
                clean.loc[flags[flag].loc[clean.index], col] = np.nan

    report = build_report(df, flags, require_labels, mode, schema_errors, int(quarantine_mask.sum()))
    if mode == "fail" and any_issue.any():
        summary = {k: int(v.sum()) for k, v in flags.items() if v.any()}
        raise ValidationError(f"Validation failed (mode=fail): {summary}", report)
    return ValidationResult(clean=clean, quarantined=quarantined, report=report, row_issues=row_issues)


def validation_messages(report: dict) -> list[str]:
    """Human-readable warnings extracted from a report."""
    msgs: list[str] = []
    msgs += [f"schema: {e}" for e in report.get("schema_errors", [])]
    for col, n in report.get("sensor_range_violations", {}).items():
        if n:
            msgs.append(f"{n} out-of-range value(s) in '{col}'")
    if report.get("duplicate_plant_timestamp"):
        msgs.append(f"{report['duplicate_plant_timestamp']} duplicate (plant_id, timestamp) row(s)")
    for k, n in report.get("other_issue_counts", {}).items():
        if n:
            msgs.append(f"{n} row(s) flagged: {k}")
    if report.get("rows_quarantined"):
        msgs.append(f"{report['rows_quarantined']} row(s) quarantined")
    return msgs
