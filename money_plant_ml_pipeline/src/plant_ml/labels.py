"""Label encoding helpers. Labels are only ever used as targets / evaluation helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import INDEX_TO_LABEL, LABEL_TO_INDEX, STRESS_LEVELS, plant_to_group


def encode_stress_level(series: pd.Series) -> pd.Series:
    """Map stress_level strings to integer classes; unknown/missing -> NaN (never guessed)."""
    return series.map(LABEL_TO_INDEX)


def decode_stress_level(indices) -> list[str]:
    return [INDEX_TO_LABEL[int(i)] for i in indices]


def class_counts(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts(dropna=False)
    return {str(lvl): int(counts.get(lvl, 0)) for lvl in STRESS_LEVELS}


def add_group_column_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Derive experimental_group from plant_id for REPORTING only when the column is absent."""
    if "experimental_group" in df.columns:
        return df
    out = df.copy()
    def _group(p):
        try:
            return plant_to_group(p) if pd.notna(p) else np.nan
        except ValueError:          # invalid plant ids are reported by validation, not crashed on here
            return np.nan

    out["experimental_group"] = [_group(p) for p in out["plant_id"]]
    out.attrs["experimental_group_derived"] = True
    return out
