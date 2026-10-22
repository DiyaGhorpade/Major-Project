"""YAML configuration loading. All paths are resolved relative to the project root."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # <root>/src/plant_ml/config.py -> <root>
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_FEATURE_CONFIG = PROJECT_ROOT / "config" / "feature_config.yaml"


class Config:
    """Thin wrapper over a nested dict with dotted-key access."""

    def __init__(self, data: dict, feature_data: dict, root: Path = PROJECT_ROOT):
        self.data = data
        self.features = feature_data
        self.root = Path(root)

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        value = self.get(dotted, None)
        if value is None:
            raise KeyError(f"Missing required config key: {dotted}")
        return value

    def path(self, dotted: str) -> Path:
        p = Path(str(self.require(dotted)))
        return p if p.is_absolute() else self.root / p

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self.data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def copy(self) -> "Config":
        return Config(copy.deepcopy(self.data), copy.deepcopy(self.features), self.root)

    def snapshot(self) -> dict:
        """JSON-serialisable snapshot stored in artifact metadata."""
        return {"config": copy.deepcopy(self.data), "feature_config": copy.deepcopy(self.features)}


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping")
    return loaded


def load_config(config_path: str | os.PathLike | None = None,
                feature_config_path: str | os.PathLike | None = None) -> Config:
    """Load config.yaml + feature_config.yaml. PLANT_ML_CONFIG env var overrides the default path."""
    cfg_path = Path(config_path or os.environ.get("PLANT_ML_CONFIG") or DEFAULT_CONFIG)
    feat_path = Path(feature_config_path or os.environ.get("PLANT_ML_FEATURE_CONFIG") or DEFAULT_FEATURE_CONFIG)
    root = PROJECT_ROOT
    cfg = Config(_read_yaml(cfg_path), _read_yaml(feat_path), root)
    # Optional environment overrides (documented in .env.example)
    if os.environ.get("PLANT_ML_ARTIFACTS_DIR"):
        cfg.set("paths.artifacts_dir", os.environ["PLANT_ML_ARTIFACTS_DIR"])
    if os.environ.get("PLANT_ML_API_HOST"):
        cfg.set("api.host", os.environ["PLANT_ML_API_HOST"])
    if os.environ.get("PLANT_ML_API_PORT"):
        cfg.set("api.port", int(os.environ["PLANT_ML_API_PORT"]))
    if os.environ.get("PLANT_ML_ALLOW_VERSION_MISMATCH"):
        cfg.set("artifacts.allow_version_mismatch",
                os.environ["PLANT_ML_ALLOW_VERSION_MISMATCH"].lower() in {"1", "true", "yes"})
    return cfg
