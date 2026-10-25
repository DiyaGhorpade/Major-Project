"""SHAP models: local (TS `LocalShapExplanation`, snake_case + camelCase text fields) and global (`GlobalShapData`)."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class ShapFeatureOut(BaseModel):
    name: str
    displayName: str
    observed_value: float | None
    unit: str
    shap_value: float
    direction: Literal["increases_prediction", "decreases_prediction", "neutral"]
    description: str
    laymanExplanation: str


class LocalShapOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    plant_id: int
    record_id: str
    model_version: str | None
    target_class: str
    predicted_probability: float | None
    base_value: float | None = None  # the ML service gives no base value (DEV-2)
    model_output: float | None = None
    features: list[ShapFeatureOut]
    laymanSummary: str
    explanation_note: str
    timestamp: str


class GlobalFeatureOut(BaseModel):
    name: str
    displayName: str
    mean_abs_shap: float
    importance_rank: int
    interpretation: str
    laymanTip: str


class BeeswarmPointOut(BaseModel):
    id: str
    feature: str
    feature_value_norm: float | None  # per-feature min-max of the observed value; null if missing/constant
    shap_value: float
    plant_id: int | None = None  # the report does not say which plant a row came from
    predicted_class: str | None = None


class GlobalShapOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_version: str | None
    sample_count: int
    features: list[GlobalFeatureOut]
    beeswarm_points: list[BeeswarmPointOut]
    perClassTop15: dict[str, list[dict[str, Any]]]
    source: dict[str, Any]
    approximate: dict[str, str]
    units: str
    note: str | None = None
