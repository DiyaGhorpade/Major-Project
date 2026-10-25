"""Pydantic models mirroring the ML service's request/response contract (PHASE0_FINDINGS section 2).

Parsing is deliberately tolerant: loosely-typed nested dicts keep `extra="allow"` so a new field in the
ML service never breaks the backend. The backend sends only the six core reading fields.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Tolerant(BaseModel):
    model_config = ConfigDict(extra="allow")


# --- requests -------------------------------------------------------------------------------
class MLReading(BaseModel):
    """One sensor reading as the ML service expects it. Missing sensor values stay `None`."""

    timestamp: str  # ISO-8601 with +05:30
    plant_id: int
    soil: float | None = None
    temperature: float | None = None
    humidity: float | None = None
    light: float | None = None


class PredictRequest(BaseModel):
    reading: MLReading
    history: list[MLReading] = Field(default_factory=list)  # same plant, strictly earlier
    explain: bool | None = None


class BatchPredictRequest(BaseModel):
    items: list[PredictRequest]


class ValidateRequest(BaseModel):
    readings: list[MLReading]


# --- responses ------------------------------------------------------------------------------
class StressPrediction(_Tolerant):
    label: str
    probabilities: dict[str, float]


class Cluster(_Tolerant):
    id: int
    distance: float


class Anomaly(_Tolerant):
    is_anomaly: bool
    score: float
    threshold: float = 0.0
    interpretation: str = ""


class TopFeature(_Tolerant):
    feature: str
    value: float | None = None
    contribution: float
    direction: str = ""


class Explanation(_Tolerant):
    available: bool = False
    reason: str | None = None
    target_class: str | None = None
    top_features: list[TopFeature] = Field(default_factory=list)
    units: str | None = None
    note: str | None = None


class MLRecommendation(_Tolerant):
    id: str
    category: str
    text: str = ""
    reason: str = ""
    priority: str = "low"
    supporting_features: dict[str, Any] = Field(default_factory=dict)
    type: str = "heuristic_suggestion"


class HistoryInfo(_Tolerant):
    rows_available: int = 0
    rows_recommended: int = 0
    sufficient: bool = False


class ModelMetadata(_Tolerant):
    trained_on: str | None = None
    validated_on_real_plants: bool | None = None
    model_version: str | None = None
    run_id: str | None = None
    disclaimer: str | None = None


class PredictResponse(_Tolerant):
    plant_id: int
    timestamp: str
    stress_prediction: StressPrediction
    cluster: Cluster
    anomaly: Anomaly
    explanation: Explanation = Field(default_factory=Explanation)
    recommendations: list[MLRecommendation] = Field(default_factory=list)
    recommendations_disclaimer: str = ""
    history: HistoryInfo = Field(default_factory=HistoryInfo)
    warnings: list[str] = Field(default_factory=list)
    model_metadata: ModelMetadata = Field(default_factory=ModelMetadata)


class HealthResponse(_Tolerant):
    status: str
    models_loaded: bool = False
    artifact_error: str | None = None
    run_id: str | None = None
    trained_on: str | None = None
    explainability_available: bool | None = None
    warnings: list[str] = Field(default_factory=list)


class MetadataResponse(_Tolerant):
    run_id: str
    model_version: str
    trained_at: str | None = None
    trained_on: str | None = None
    validated_on_real_plants: bool | None = None
    label_mapping: Any = None
    explainability: dict[str, Any] = Field(default_factory=dict)
    required_history_rows_for_full_features: int | None = None
    disclaimer: str | None = None


class ValidateResponse(_Tolerant):
    valid: bool
    n_readings: int
    issues: list[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


# 422 codes that are about one reading's own data (or the history built for it). Anything else the ML service
# refuses (`bad_batch_size`, `history_too_long`, a malformed request, ...) is a client/config mismatch: it says
# nothing about the reading, so it must never be stored as a permanent `rejected` prediction (DEV-51).
DATA_REJECTION_CODES = frozenset({"invalid_readings", "mixed_plants", "history_not_before_reading"})


def is_data_rejection(code: str | None) -> bool:
    return code in DATA_REJECTION_CODES


def rejection_failure_text(code: str | None, message: str | None) -> str:
    """The run-failure message for a request-level / config-mismatch rejection."""
    detail = f": {message}" if message else ""
    return (
        f"ML service rejected the batch ({code or 'rejected'}){detail}. Check that api.max_batch_items and "
        "api.max_history_rows in the ML config match the running ML service, then restart it and run again."
    )


def format_rejection(code: str | None, message: str | None, issues: list[Any] | None = None) -> str:
    """Human-readable one-liner for a rejection, e.g. `invalid_readings: ... [row 5: soil_out_of_range]`."""
    text = f"{code}: {message}" if code and message else (code or message or "rejected")
    parts = []
    for issue in (issues or [])[:5]:
        if isinstance(issue, dict) and "issues" in issue:
            parts.append(f"row {issue.get('index')}: {','.join(map(str, issue['issues']))}")
    return f"{text} [{'; '.join(parts)}]" if parts else text


class BatchResult(BaseModel):
    """Normalised outcome of one batch item: ok, rejected (ML said 422) or unavailable (503/other)."""

    index: int
    status: Literal["ok", "rejected", "unavailable"]
    result: PredictResponse | None = None
    code: str | None = None
    message: str | None = None
    issues: list[Any] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def error_text(self) -> str | None:
        if self.status == "ok":
            return None
        return format_rejection(self.code, self.message or self.status, self.issues)
