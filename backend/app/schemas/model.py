"""Model performance (TS `ModelMetrics`, snake_case) plus provenance fields (camelCase, DEV-35)."""

from typing import Any

from pydantic import BaseModel, ConfigDict


class Overall(BaseModel):
    accuracy: float
    precision: float  # macro mean over classes
    recall: float  # macro mean over classes
    f1_score: float  # macro F1


class PerClass(BaseModel):
    class_name: str
    precision: float
    recall: float
    f1_score: float
    support: int
    laymanStatus: str


class ConfusionMatrix(BaseModel):
    labels: list[str]
    matrix: list[list[int]]  # rows = true class, columns = predicted class


class BaselineMetrics(BaseModel):
    accuracy: float
    macro_f1: float


class ModelMetricsOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

    model_name: str
    model_version: str
    evaluated_at: str
    dataset_name: str
    total_samples: int
    split: str
    overall: Overall
    per_class: list[PerClass]
    confusion_matrix: ConfusionMatrix
    provenance: str = "synthetic"
    validatedOnRealPlants: bool = False
    runId: str
    balancedAccuracy: float | None
    weightedF1: float | None
    baselines: dict[str, BaselineMetrics]
    anomalyModel: dict[str, Any] | None
    clusteringModel: dict[str, Any] | None
    note: str = "Metrics measured on synthetic labels."
