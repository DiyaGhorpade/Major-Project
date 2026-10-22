"""Leakage-aware train/test splitting. Random row-level splits are intentionally NOT offered."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import Config


@dataclass
class SplitResult:
    train_mask: pd.Series
    test_mask: pd.Series
    info: dict


def split_frame(feat_df: pd.DataFrame, cfg: Config, strategy: str | None = None,
                cutoff: str | None = None) -> SplitResult:
    """Return boolean masks aligned to ``feat_df``.

    chronological:  train = rows with timestamp <= cutoff, test = rows after it (cutoff is the
                    (1 - test_fraction) quantile of unique timestamps, unless supplied explicitly).
    group_holdout:  test = rows from configured holdout plants (unseen plants); train = other plants.
    """
    strategy = strategy or cfg.get("split.strategy", "chronological")
    if strategy == "chronological":
        ts = feat_df["timestamp"]
        if cutoff is None:
            frac = float(cfg.get("split.test_fraction", 0.25))
            if not 0.0 < frac < 1.0:
                raise ValueError("split.test_fraction must be in (0, 1)")
            uniq = ts.drop_duplicates().sort_values().reset_index(drop=True)
            cut = uniq.iloc[int(round(len(uniq) * (1.0 - frac))) - 1]
        else:
            cut = pd.Timestamp(cutoff)
        train = ts <= cut
        test = ts > cut
        info = {"strategy": strategy, "cutoff_timestamp": cut.isoformat(),
                "test_fraction": float(cfg.get("split.test_fraction", 0.25)) if cutoff is None else None,
                "note": "Train rows are strictly earlier than test rows; features use past rows only."}
    elif strategy == "group_holdout":
        holdout = [int(p) for p in cfg.get("split.holdout_plants", [4, 8, 12, 16])]
        test = feat_df["plant_id"].isin(holdout)
        train = ~test
        info = {"strategy": strategy, "holdout_plants": holdout,
                "note": "Tests generalisation to unseen plants (one per experimental group by default)."}
    else:
        raise ValueError(f"Unknown split strategy {strategy!r}; use 'chronological' or 'group_holdout'")
    if train.sum() == 0 or test.sum() == 0:
        raise ValueError(f"Split produced an empty partition (train={int(train.sum())}, test={int(test.sum())})")
    info.update({"train_rows": int(train.sum()), "test_rows": int(test.sum())})
    return SplitResult(train_mask=train, test_mask=test, info=info)
