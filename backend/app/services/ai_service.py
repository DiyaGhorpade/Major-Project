"""AI views: local/global SHAP, anomaly list and aggregated insights. Everything is built from stored predictions
and from the ML pipeline's report files; text is generated from those numbers and never invents claims."""

import csv
import json
import logging
import math
from collections import Counter
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from app.core import feature_dictionary as fd
from app.core.cache import ResponseCache
from app.core.config import Settings, display_path
from app.core.mappings import env_state_label, to_ist_iso, to_ist_text
from app.core.provenance import DISCLAIMER, PROVENANCE
from app.core.units import format_value
from app.db import repositories as repo
from app.db.models import MlPrediction
from app.schemas.ai import AnomalyOut, EnvStateCount, InsightsModelStatus, InsightsOut
from app.schemas.shap import BeeswarmPointOut, GlobalFeatureOut, GlobalShapOut, LocalShapOut, ShapFeatureOut
from app.services import model_service

logger = logging.getLogger("app.ai")

SEVERITIES = ("healthy", "low", "moderate", "high")
ANOMALY_TEXT = "Unusual reading detected"
UNITS_NOTE = "Units: log-odds contribution; base value not provided by the model service."
SHAP_SAMPLE_ROWS = 500  # scripts/generate_explanations.py explains min(500, test rows) rows (config explain.background_rows)
_DIRECTION = {
    "towards_predicted_class": "increases_prediction",
    "away_from_predicted_class": "decreases_prediction",
}


def _finite(value: object) -> float | None:
    """A float if `value` is a finite number, else None (NaN / inf must never reach a JSON response)."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class ExplanationUnavailable(LookupError):
    pass


class PlantNotFound(LookupError):
    pass


class GlobalShapUnavailable(LookupError):
    def __init__(self, run_id: str | None, reason: str):
        super().__init__(reason)
        run = run_id or "<run_id from artifacts/latest.json>"
        self.hint = (
            "cd money_plant_ml_pipeline && .venv/bin/python scripts/generate_explanations.py "
            f"--input data/raw/plant_sensor_data.csv --run-id {run}"
        )


# --- local SHAP -------------------------------------------------------------------------------
def direction_of(contribution: float, raw: str | None) -> str:
    if contribution == 0:
        return "neutral"
    return _DIRECTION.get(raw or "", "increases_prediction" if contribution > 0 else "decreases_prediction")


def local_shap(db: Session, plant_id: int) -> LocalShapOut:
    """Local explanation of the plant's latest explained prediction (stored `top_features`). Raises
    `PlantNotFound`, or `ExplanationUnavailable` if the plant has none yet."""
    if repo.get_plant(db, plant_id) is None:
        raise PlantNotFound(f"plant {plant_id} not found")
    pred = repo.latest_explained_prediction(db, plant_id)
    expl = pred.explanation_json if pred else None
    if pred is None or not expl or not expl.get("available", True) or not expl.get("top_features"):
        raise ExplanationUnavailable(f"no stored explanation for plant {plant_id}")
    target = expl.get("target_class") or pred.severity or ""
    features: list[ShapFeatureOut] = []
    for tf in expl["top_features"]:
        contribution = _finite(tf.get("contribution"))
        if contribution is None:
            continue  # a non-finite contribution cannot be shown
        info = fd.resolve(tf["feature"])
        direction = direction_of(contribution, tf.get("direction"))
        value = _finite(tf.get("value"))
        features.append(
            ShapFeatureOut(
                name=info.name,
                displayName=info.display_name,
                observed_value=value,
                unit=info.unit,
                shap_value=contribution,
                direction=direction,  # type: ignore[arg-type]
                description=info.description,
                laymanExplanation=fd.local_explanation(info, direction, target, value),
            )
        )
    if not features:
        raise ExplanationUnavailable(f"stored explanation for plant {plant_id} has no usable features")
    probability = _finite((pred.probabilities or {}).get(target))
    when = to_ist_iso(pred.reading_timestamp) or ""
    p_text = f" (P={probability:.2f})" if probability is not None else ""
    ranked = sorted(features, key=lambda f: abs(f.shap_value), reverse=True)[:3]
    named = ", ".join(
        f"{f.displayName} ({format_value(f.observed_value, f.unit)}, "
        f"{'towards' if f.direction == 'increases_prediction' else 'away from' if f.direction == 'decreases_prediction' else 'neutral for'} '{target}')"
        for f in ranked
    )
    summary = (
        f"For the reading at {to_ist_text(pred.reading_timestamp)}, the model rated stress '{target}'{p_text}. "
        f"The inputs that moved this most: {named}. This describes the model's behaviour, not a proven cause."
    )
    note = (expl.get("note") or "").strip()
    return LocalShapOut(
        plant_id=plant_id,
        record_id=f"R-{pred.reading_id}",
        model_version=pred.model_version,
        target_class=target,
        predicted_probability=probability,
        features=features,
        laymanSummary=summary,
        explanation_note=f"{note} {UNITS_NOTE}".strip(),
        timestamp=when,
    )


# --- global SHAP ------------------------------------------------------------------------------
def find_global_report(reports_dir: Path, run_id: str) -> Path | None:
    """Newest `reports/<run>/evaluation_*/shap_global_importance.json`, else `reports/<run>/shap_global_importance.json`."""
    base = reports_dir / run_id
    nested = sorted(base.glob("evaluation_*/shap_global_importance.json"), key=lambda p: p.parent.name)
    if nested:
        return nested[-1]
    flat = base / "shap_global_importance.json"
    return flat if flat.is_file() else None


def _beeswarm(csv_path: Path) -> tuple[list[BeeswarmPointOut], int]:
    try:
        with csv_path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return [], 0
    values: dict[str, list[float]] = {}
    parsed: list[tuple[dict[str, str], float | None, float]] = []
    for row in rows:
        shap = _finite(row.get("contribution"))
        if shap is None:
            continue  # missing or non-finite contribution: drop the point
        value = _finite(row.get("value")) if row.get("value") not in (None, "") else None
        if value is not None:
            values.setdefault(row["feature"], []).append(value)
        parsed.append((row, value, shap))
    points = []
    for i, (row, value, shap) in enumerate(parsed):
        norm = None
        if value is not None:
            lo, hi = min(values[row["feature"]]), max(values[row["feature"]])
            norm = (value - lo) / (hi - lo) if hi > lo else None
        points.append(
            BeeswarmPointOut(
                id=f"{row.get('row', i)}:{row['feature']}",
                feature=row["feature"],
                feature_value_norm=norm,
                shap_value=shap,
                predicted_class=row.get("predicted") or None,
            )
        )
    return points, len({r.get("row") for r, _, _ in parsed})


@lru_cache(maxsize=4)
def _load_global(report: str, report_mtime: int, csv_mtime: int, eval_mtime: int, model_version: str | None, reports_dir: str) -> GlobalShapOut:
    path = Path(report)
    data = json.loads(path.read_text(encoding="utf-8"))
    mean_abs: dict[str, float] = {k: v for k, raw in data["mean_abs_shap"].items() if (v := _finite(raw)) is not None}
    evaluation = model_service.read_optional_json(path.parent / "evaluation.json") or {}
    sample_count = next(
        (int(evaluation[k]) for k in ("shap_sample_count", "shap_rows", "n_explained") if isinstance(evaluation.get(k), int)),
        SHAP_SAMPLE_ROWS,
    )
    ranked = sorted(mean_abs.items(), key=lambda kv: kv[1], reverse=True)[:15]
    features = []
    for rank, (name, value) in enumerate(ranked, start=1):
        info = fd.resolve(name)
        features.append(
            GlobalFeatureOut(
                name=name,
                displayName=info.display_name,
                mean_abs_shap=value,
                importance_rank=rank,
                interpretation=fd.global_interpretation(info, value, rank, sample_count),
                laymanTip=fd.global_tip(info),
            )
        )
    per_class = {
        cls: [
            {"name": n, "displayName": fd.resolve(n).display_name, "mean_abs_shap": v}
            for n, v in sorted(((k, _finite(raw)) for k, raw in vals.items() if _finite(raw) is not None), key=lambda kv: kv[1], reverse=True)[:15]
        ]
        for cls, vals in (data.get("per_class_top15") or {}).items()
    }
    points, sampled_rows = _beeswarm(path.parent / "shap_sample_explanations.csv")
    generated = datetime.fromtimestamp(report_mtime / 1e9, UTC)
    return GlobalShapOut(
        model_version=model_version,
        sample_count=sample_count,
        features=features,
        beeswarm_points=points,
        perClassTop15=per_class,
        source={"path": display_path(path, Path(reports_dir)), "generatedAt": to_ist_iso(generated)},
        approximate={"beeswarm": f"sample of {sampled_rows} test rows, top-5 features each (not all rows or features)"},
        units="log-odds (mean absolute SHAP over the explained rows and all four classes)",
        note=data.get("note"),
    )


def report_signature(settings: Settings) -> tuple[str, int] | None:
    """`(path, mtime)` of the current global SHAP report, or None; part of the insights cache key."""
    try:
        report = find_global_report(settings.ML_REPORTS_DIR, model_service.current_run_id(settings.ML_ARTIFACTS_DIR))
        return (str(report), report.stat().st_mtime_ns) if report else None
    except (model_service.ArtifactsUnavailable, OSError):
        return None


def global_shap(settings: Settings) -> GlobalShapOut:
    try:
        run_id = model_service.current_run_id(settings.ML_ARTIFACTS_DIR)
    except model_service.ArtifactsUnavailable as exc:
        raise GlobalShapUnavailable(None, f"current model run unknown ({exc})") from exc
    report = find_global_report(settings.ML_REPORTS_DIR, run_id)
    if report is None:
        raise GlobalShapUnavailable(run_id, f"no global SHAP report for run {run_id}")
    meta = model_service.read_optional_json(settings.ML_ARTIFACTS_DIR / run_id / "classifier.meta.json") or {}
    csv_path, eval_path = report.parent / "shap_sample_explanations.csv", report.parent / "evaluation.json"

    def mtime(p: Path) -> int:
        return p.stat().st_mtime_ns if p.exists() else 0

    try:
        return _load_global(str(report), mtime(report), mtime(csv_path), mtime(eval_path), meta.get("model_version"), str(settings.ML_REPORTS_DIR))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise GlobalShapUnavailable(run_id, f"global SHAP report {report.name} is unreadable ({type(exc).__name__})") from exc


# --- anomalies --------------------------------------------------------------------------------
def list_anomalies(db: Session, plant_id: int | None, limit: int, offset: int) -> tuple[list[AnomalyOut], int]:
    rows, total = repo.list_anomalies(db, plant_id=plant_id, limit=limit, offset=offset)
    items = [
        AnomalyOut(
            id=f"anom-{a.id}",
            plant_id=a.plant_id,
            plant_name=p.name,
            record_id=r.record_id,
            timestamp=to_ist_iso(a.reading_timestamp) or "",
            score=a.score,
            threshold=a.threshold,
            interpretation=ANOMALY_TEXT,
            readings={"soil": r.soil, "temperature": r.temperature, "humidity": r.humidity, "light": r.light},
            datasetLabelIsSyntheticAnomaly=r.label_is_synthetic_anomaly,
        )
        for a, p, r in rows
    ]
    return items, total


# --- insights ---------------------------------------------------------------------------------
def _severity_counts(counts: dict[Any, int]) -> dict[str, int]:
    return {s: int(counts.get(s, 0)) for s in SEVERITIES}


def insights(db: Session, settings: Settings, *, ml_ok: bool, explain_available: bool | None) -> InsightsOut:
    status_counts = repo.prediction_status_counts(db)
    ok_total = status_counts.get("ok", 0)
    latest = repo.latest_ok_prediction_per_plant(db)
    latest_sev = Counter(p.severity for p in latest.values())
    latest_guess = Counter(p.stress_type_guess for p in latest.values() if p.stress_type_guess)
    guesses = {k: v for k, v in repo.ok_prediction_counts(db, MlPrediction.stress_type_guess).items() if k}
    env = repo.ok_prediction_counts(db, MlPrediction.cluster_id)
    anomalies = repo.count_anomalies(db)

    local = Counter()
    explained_plants: set[int] = set()
    for pid, expl in repo.explanation_rows(db):
        if expl.get("available", True):
            explained_plants.add(pid)
            for tf in expl.get("top_features") or []:
                local[tf["feature"]] += 1
    local_freq = [
        {"name": n, "displayName": fd.resolve(n).display_name, "count": c, "unit": fd.resolve(n).unit}
        for n, c in local.most_common(10)
    ]
    global_top = None
    global_source = None
    try:
        g = global_shap(settings)
        global_top = [{"name": f.name, "displayName": f.displayName, "meanAbsShap": f.mean_abs_shap} for f in g.features[:10]]
        global_source = g.source
    except GlobalShapUnavailable:
        pass

    model = repo.latest_model_info(db)
    model_version, model_run_id = model_service.display_model(db, latest.values())
    run = repo.latest_run(db)
    model_status = InsightsModelStatus(
        ml_service_ok=ml_ok,
        model_version=model_version,
        run_id=model_run_id,
        explainability_available=explain_available,
        trained_at=to_ist_iso(model.trained_at) if model else None,
    )
    return InsightsOut(
        has_predictions=ok_total > 0,
        predictions={k: int(status_counts.get(k, 0)) for k in ("ok", "unavailable", "rejected")},
        severity_counts=_severity_counts(repo.ok_prediction_counts(db, MlPrediction.severity)),
        latest_severity_counts=_severity_counts(latest_sev),
        condition_guess_counts={k: int(v) for k, v in guesses.items()},
        latest_condition_guess_counts=dict(latest_guess),
        env_state_counts=[
            EnvStateCount(id=int(cid), label=env_state_label(int(cid)), count=int(n))
            for cid, n in sorted((k, v) for k, v in env.items() if k is not None)
        ],
        anomalies={
            "count": anomalies,
            "rate": round(anomalies / ok_total, 6) if ok_total else None,
            "rateBasis": "anomaly records / ok predictions",
            "interpretation": ANOMALY_TEXT,
        },
        explanation_coverage={
            "plantsWithLocalShap": len(explained_plants),
            "totalPlants": repo.count_plants(db),
            "plantIds": sorted(explained_plants),
        },
        top_features={"global": global_top, "globalSource": global_source, "localFrequency": local_freq},
        model=model_status,
        last_pipeline_run=(
            {
                "runId": run.id, "status": run.status, "total": run.total, "processed": run.processed, "failed": run.failed,
                "startedAt": to_ist_iso(run.started_at), "finishedAt": to_ist_iso(run.finished_at), "message": run.message,
            }  # fmt: skip
            if run
            else None
        ),
        provenance=PROVENANCE,
        disclaimer=DISCLAIMER,
    )


def insights_cached(
    session_factory: sessionmaker, cache: ResponseCache, settings: Settings, *, ml_ok: bool, explain_available: bool | None
) -> InsightsOut:
    """Insights, cached until the data, the ML flags or the SHAP report change (`repositories.state_token`)."""
    with session_factory() as db:
        token = (repo.state_token(db), ml_ok, explain_available, report_signature(settings))
        return cache.get_or_compute(
            "insights", token, lambda: insights(db, settings, ml_ok=ml_ok, explain_available=explain_available)
        )
