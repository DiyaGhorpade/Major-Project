import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plant_ml.config import load_config  # noqa: E402
from plant_ml.data_loader import normalise_frame  # noqa: E402
from plant_ml.synthetic import generate  # noqa: E402
from plant_ml.training import run_training  # noqa: E402


@pytest.fixture(scope="session")
def base_cfg():
    return load_config()


@pytest.fixture(scope="session")
def small_df():
    """Deterministic 14-day SYNTHETIC dataset (16 plants, 2688 rows)."""
    return generate(n_days=14, seed=7)


@pytest.fixture(scope="session")
def small_csv(tmp_path_factory, small_df):
    p = tmp_path_factory.mktemp("data") / "small.csv"
    small_df.to_csv(p, index=False)
    return p


@pytest.fixture(scope="session")
def cfg_small(tmp_path_factory, base_cfg):
    cfg = base_cfg.copy()
    root = tmp_path_factory.mktemp("run")
    cfg.set("paths.artifacts_dir", str(root / "artifacts"))
    cfg.set("paths.reports_dir", str(root / "reports"))
    cfg.set("paths.processed_dir", str(root / "processed"))
    cfg.set("classifier.num_boost_round", 40)
    cfg.set("clustering.k_max", 4)
    cfg.set("clustering.n_init", 3)
    cfg.set("anomaly.n_estimators", 60)
    return cfg


@pytest.fixture(scope="session")
def trained(cfg_small, small_csv):
    return run_training(cfg_small, small_csv, with_shap=True)


@pytest.fixture(scope="session")
def predictor(cfg_small, trained):
    from plant_ml.inference import PlantPredictor
    return PlantPredictor.from_config(cfg_small)


def make_plant_frame(n=30, plant_id=1, start="2026-10-01T00:00:00+05:30", step_h=2, seed=0, **overrides):
    """Normalised frame (tz-aware timestamps) for one plant."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=n, freq=f"{step_h}h")
    df = pd.DataFrame({
        "timestamp": [t.isoformat() for t in ts], "plant_id": plant_id,
        "soil": 50 + rng.normal(0, 3, n), "temperature": 25 + rng.normal(0, 1, n),
        "humidity": 65 + rng.normal(0, 3, n), "light": np.abs(rng.normal(2000, 800, n)).round(),
    })
    for k, v in overrides.items():
        df[k] = v
    return normalise_frame(df)


def rows_to_readings(df: pd.DataFrame) -> list[dict]:
    out = []
    for _, r in df.iterrows():
        d = {"timestamp": r["timestamp"], "plant_id": int(r["plant_id"])}
        for c in ("soil", "temperature", "humidity", "light"):
            d[c] = None if pd.isna(r[c]) else float(r[c])
        out.append(d)
    return out
