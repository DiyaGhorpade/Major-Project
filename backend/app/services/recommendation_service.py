"""ML-derived and rule-based recommendations: dedupe, lifecycle, preserving user status (master prompt section 7).

Per plant, the *latest reading that has an `ok` prediction* is the current state. Candidates come from
two sources:

* `ml`: every recommendation the ML service returned for that prediction (text kept verbatim);
* `heuristic`: backend rules on the ML *outputs* plus the recent trend, used only to fill gaps / set
  priority (sustained stress escalation, anomaly -> "Sensor check").

Lifecycle (DEV-27): one active (`new`/`in_progress`) recommendation per `(plant, category)`. A re-run updates
it in place and never touches its status or `user_modified`; a recommendation for the same
`(plant, category, source reading)` is never resurrected after the user closed it; active ones whose
condition disappeared become `superseded` unless the user touched them. If a duplicate active row ever
exists, the user-modified (then newest) one is kept and untouched duplicates are removed or superseded.
Nothing is invented: every number quoted comes from a stored reading or prediction.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.mappings import (
    CATEGORY_BY_STRESS_TYPE,
    CONDITION_CATEGORIES,
    IST,
    PRIORITY_RANK,
    ensure_aware,
    map_ml_category,
    ml_rec_title,
    severity_to_status,
    to_ist_iso,
)
from app.core.plant_catalog import METRICS
from app.db import repositories as repo
from app.db.models import MlPrediction, Plant, Recommendation, SensorReading, utcnow
from app.schemas.recommendation import BasedOn, RecommendationOut, SensorValue
from app.services.history_builder import DEFAULT_REQUIRED_ROWS, read_daytime_hours

logger = logging.getLogger("app.recommendations")

ACTIVE = ("new", "in_progress")
CLOSED = ("completed", "dismissed")
SUSTAINED_ROWS = 3  # consecutive readings that make "moderate" a sustained condition
RECENT_ROWS = 12  # readings fetched per plant (trend + averages)
AVERAGE_ROWS = 6  # readings in the "recent average"
COLD_START_HISTORY_ROWS = DEFAULT_REQUIRED_ROWS - 1  # the ML service warns below 48 prior rows
COLD_START_NOTE = "Lower reliability: limited history for this plant."
# ML ids that mean "do not infer plant stress from this reading".
SUPPRESSING_ML_IDS = ("verify_sensor", "retake_reading", "low_confidence")

SCIENTIFIC_CONTEXT = {
    "irrigation": (
        "Tulsi (Ocimum) generally prefers evenly moist, well-drained soil; prolonged dryness and waterlogging both "
        "stress the roots. A soil-moisture sensor gives a relative reading that depends on the soil, pot and probe "
        "placement, so confirm by checking the soil by hand."
    ),
    "lighting": (
        "Tulsi generally does best in bright light. Too little light slows growth and gives weak, stretched stems, "
        "while very intense direct sun can scorch leaves. Light readings depend heavily on sensor placement and "
        "shading, and night-time values near zero are normal."
    ),
    "climate": (
        "Tulsi is a warm-season herb. Sustained cold, heat, very dry air or very damp air can slow growth and "
        "encourage leaf damage or fungal problems. Conditions at one sensor can differ from those at the leaves."
    ),
    "sensor_check": (
        "A flagged reading is statistically unusual compared with this plant's earlier readings. By itself that says "
        "nothing about the plant; sensor faults, loose wiring, node connection or placement are the usual causes to "
        "rule out first."
    ),
    "monitoring": (
        "The model gives a statistical estimate learned from a synthetic dataset and has not been validated on real "
        "plants. When it is unsure, or no single sensor explains a stress rating, inspecting the plant and sensors in "
        "person is the most reliable check."
    ),
    "nutrition": (
        "Nutrient problems cannot be measured with the sensors in this system; this category is not produced from "
        "sensor data."
    ),
}

HEURISTIC_COPY = {  # category -> (title, simple action) for heuristic stress-escalation candidates
    "irrigation": ("Sustained stress: check watering", "Check the soil moisture and your watering schedule."),
    "lighting": ("Sustained stress: check light", "Check the plant's placement and access to light."),
    "climate": ("Sustained stress: check climate", "Check temperature, airflow and humidity around the plant."),
    "monitoring": ("Sustained stress: inspect plant", "Inspect the plant and its sensors in person and keep monitoring."),
}
SENSOR_CHECK_TITLE = "Sensor check"
SENSOR_CHECK_ACTION = "Verify the sensor, its placement and this reading"

NIGHT_TARGET = "night (no target)"
CATEGORY_SENSORS = {  # sensors cited in a recommendation's reason
    "irrigation": ("soil",),
    "lighting": ("light",),
    "climate": ("temperature", "humidity"),
    "sensor_check": tuple(attr for attr, _ in METRICS),
}


# --- formatting -----------------------------------------------------------------------------
def format_value(metric: str, value: float | None, unit: str) -> str:
    """`21.3 %`, `28.4 °C`, `1,234 lux`; `n/a` when the sensor value is missing."""
    if value is None:
        return "n/a"
    return f"{value:,.0f} {unit}" if metric == "light" else f"{value:.1f} {unit}"


def is_daytime(reading: SensorReading, daytime: tuple[int, int]) -> bool:
    return daytime[0] <= ensure_aware(reading.timestamp).astimezone(IST).hour <= daytime[1]


def flagged_out_of_range(reading: SensorReading, attr: str) -> bool:
    """True when the reading's `validation_error` lists `attr` under `out_of_range` (set by the seed/ingest check)."""
    return any(
        part.strip().startswith("out_of_range:") and re.search(rf"(?:^|[\s:,]){attr}=", part)
        for part in (reading.validation_error or "").split(";")
    )


def supporting_sensor_values(
    reading: SensorReading, optimal_ranges: dict[str, Any], daytime: tuple[int, int] = (9, 16)
) -> list[dict[str, str]]:
    """The reading's four sensor values against the plant's optimal ranges. Missing -> `n/a` / `attention`.
    Light outside the ML service's daytime hours has no target: `good` / "night (no target)"."""
    out = []
    for attr, label in METRICS:
        value = getattr(reading, attr)
        rng = optimal_ranges[attr]
        lo, hi, unit = rng["min"], rng["max"], rng["unit"]
        target = f"{lo:,.0f}–{hi:,.0f} {unit}"
        if attr == "light" and not is_daytime(reading, daytime):
            target, status = NIGHT_TARGET, "good"
            if value is None:
                status = "attention"
        elif value is None:
            status = "attention"
        elif value < lo:
            status = "low"
        elif value > hi:
            status = "high"
        else:
            status = "good"
        out.append({"metric": label, "value": format_value(attr, value, unit), "target": target, "status": status})
    return out


# --- per-plant state ------------------------------------------------------------------------
@dataclass
class PlantState:
    reading: SensorReading
    pred: MlPrediction
    chain: list[MlPrediction]  # consecutive `ok` predictions ending at the latest, newest first
    recent: list[SensorReading]  # newest first, starting at the latest ok reading
    units: dict[str, str]  # sensor attribute -> unit, from the plant's optimal ranges

    @property
    def severity(self) -> str:
        return self.pred.severity or "healthy"

    @property
    def probability(self) -> float | None:
        probs = self.pred.probabilities or {}
        value = probs.get(self.pred.severity or "")
        return value if value is not None else self.pred.confidence

    @property
    def probability_text(self) -> str | None:
        """`P(high)=0.64`; when the class probability is absent, `confidence=0.64`; else None."""
        probs = self.pred.probabilities or {}
        if self.pred.severity in probs:
            return f"P({self.pred.severity})={probs[self.pred.severity]:.2f}"
        return f"confidence={self.pred.confidence:.2f}" if self.pred.confidence is not None else None

    @property
    def is_cold_start(self) -> bool:
        if any("cold start" in w.lower() for w in (self.pred.warnings or [])):
            return True
        return self.pred.history_rows_sent is not None and self.pred.history_rows_sent < COLD_START_HISTORY_ROWS

    @property
    def sustained_moderate(self) -> bool:
        window = self.chain[:SUSTAINED_ROWS]
        return (
            self.severity == "moderate"
            and len(window) == SUSTAINED_ROWS
            and all(p.severity in ("moderate", "high") for p in window)
        )

    def average(self, attr: str) -> float | None:
        """Mean over the last `AVERAGE_ROWS` readings, skipping nulls and values flagged out of range; needs >= 3."""
        values = [
            getattr(r, attr)
            for r in self.recent[:AVERAGE_ROWS]
            if getattr(r, attr) is not None and not flagged_out_of_range(r, attr)
        ]
        return sum(values) / len(values) if len(values) >= 3 else None

    def clause(self, attrs: tuple[str, ...]) -> str:
        """`Soil moisture 21.3 % (recent average 38.0 %)` for the given sensors."""
        parts = []
        for attr, label in METRICS:
            if attr not in attrs:
                continue
            unit = self.units[attr]
            text = f"{label} {format_value(attr, getattr(self.reading, attr), unit)}"
            avg = self.average(attr)
            if avg is not None and getattr(self.reading, attr) is not None:
                text += f" (recent average {format_value(attr, avg, unit)})"
            parts.append(text)
        return ", ".join(parts)


@dataclass
class Candidate:
    category: str
    source: str  # ml | heuristic
    title: str
    action: str
    reason: str
    priority: str
    ml_rec_id: str | None = None


def _priority(value: str | None) -> str:
    """ML/heuristic priority -> the frontend's `high|medium|low` (`critical`, if ever sent, becomes `high`: DEV-28)."""
    value = (value or "").lower()
    if value == "critical":
        return "high"
    return value if value in PRIORITY_RANK else "low"


def _reason(state: PlantState, category: str, base: str) -> str:
    """`<base reason>. <cited sensor values>; model P(<class>)=0.64.` (no model probability for sensor checks,
    whose point is that the reading itself may be unreliable)."""
    sentences = [base.strip().rstrip(".")] if base.strip() else []
    clause = state.clause(CATEGORY_SENSORS.get(category, ()))
    pred = state.pred
    if category == "sensor_check":
        if pred.anomaly_score is not None:
            sentences.append(f"Anomaly score {pred.anomaly_score:.2f} (higher means more unusual)")
        if clause:
            sentences.append(f"Latest values: {clause}")
        return ". ".join(sentences) + "."
    prob = state.probability_text or ""
    tail = (f"model {prob}" if clause else f"Model {prob}") if prob else ""
    body = "; ".join(x for x in (clause, tail) if x)
    if body:
        sentences.append(body)
    return ". ".join(sentences) + "."


# --- candidate generation -------------------------------------------------------------------
def _add(cands: dict[str, Candidate], new: Candidate) -> None:
    """Dedupe per category: keep the higher priority (first wins ties) and merge the other's reason."""
    old = cands.get(new.category)
    if old is None:
        cands[new.category] = new
        return
    winner, loser = (new, old) if PRIORITY_RANK[new.priority] < PRIORITY_RANK[old.priority] else (old, new)
    if loser.reason and loser.reason not in winner.reason:
        winner.reason = f"{winner.reason} Also: {loser.reason}"
    cands[new.category] = winner


def _ml_candidates(state: PlantState) -> list[Candidate]:
    out = []
    for item in state.pred.ml_recommendations_json or []:
        ml_id = str(item.get("id", ""))
        category = map_ml_category(str(item.get("category", "")))
        out.append(
            Candidate(
                category=category,
                source="ml",
                title=ml_rec_title(ml_id),
                action=str(item.get("text", "")),
                reason=_reason(state, category, str(item.get("reason", ""))),
                priority=_priority(item.get("priority")),
                ml_rec_id=ml_id or None,
            )
        )
    return out


def _escalation(state: PlantState) -> str | None:
    """Why the priority must be high, or None. Not applied when the reading is suspect or the model is unsure."""
    if state.pred.is_anomaly:
        return None
    if any(str(r.get("id")) in SUPPRESSING_ML_IDS for r in state.pred.ml_recommendations_json or []):
        return None
    if state.severity == "high":
        return "the model rates the latest reading 'high'"
    if state.sustained_moderate:
        return f"the model has rated the last {SUSTAINED_ROWS} consecutive readings 'moderate' or higher"
    return None


def build_candidates(state: PlantState) -> list[Candidate]:
    cands: dict[str, Candidate] = {}
    for cand in _ml_candidates(state):
        _add(cands, cand)

    why = _escalation(state)
    if why is not None:
        guessed = state.pred.stress_type_guess
        if guessed in CATEGORY_BY_STRESS_TYPE:
            targets = [CATEGORY_BY_STRESS_TYPE[guessed]]
        elif guessed == "Mixed Stress":
            targets = [c for c in CONDITION_CATEGORIES if c in cands] or ["monitoring"]
        else:
            targets = ["monitoring"]
        for category in targets:
            existing = cands.get(category)
            if existing is not None:
                if existing.priority != "high":
                    existing.priority = "high"
                    existing.reason += f" Priority raised to high: {why}."
                continue
            title, action = HEURISTIC_COPY[category]
            _add(
                cands,
                Candidate(category, "heuristic", title, action, _reason(state, category, f"Escalated because {why}"), "high"),
            )

    if state.pred.is_anomaly and "sensor_check" not in cands:
        _add(
            cands,
            Candidate(
                "sensor_check", "heuristic", SENSOR_CHECK_TITLE, SENSOR_CHECK_ACTION,
                _reason(state, "sensor_check", "The anomaly detector flagged the latest reading as unusual for this sensor history"),
                "medium",
            ),
        )  # fmt: skip

    if state.is_cold_start:
        for cand in cands.values():
            cand.reason += f" {COLD_START_NOTE}"
    return list(cands.values())


def load_state(db: Session, plant: Plant) -> PlantState | None:
    """Latest reading with an `ok` prediction, the consecutive `ok` chain behind it, and recent readings."""
    latest = repo.latest_ok_reading(db, plant.id)
    if latest is None:
        return None
    rows = repo.readings_with_predictions_upto(db, plant.id, latest, RECENT_ROWS)
    chain: list[MlPrediction] = []
    for _, pred in rows:
        if pred is None or pred.status != "ok":
            break
        chain.append(pred)
    units = {attr: plant.optimal_ranges[attr]["unit"] for attr, _ in METRICS}
    return PlantState(latest, chain[0], chain, [r for r, _ in rows], units)


# --- persistence ----------------------------------------------------------------------------
def _content(state: PlantState, cand: Candidate) -> dict[str, Any]:
    pred = state.pred
    if cand.source == "ml":
        guidance = f"ML heuristic rules · model {pred.model_version} (run {pred.run_id})"
    else:
        guidance = f"Backend rules on ML output · model {pred.model_version} (run {pred.run_id})"
    return {
        "source": cand.source,
        "ml_rec_id": cand.ml_rec_id,
        "title": cand.title,
        "simple_action": cand.action,
        "reason": cand.reason,
        "scientific_context": SCIENTIFIC_CONTEXT.get(cand.category, ""),
        "guidance_source": guidance,
        "priority": cand.priority,
        "confidence": state.probability,
        "based_on": {
            "timestamp": to_ist_iso(state.reading.timestamp),
            "modelVersion": pred.model_version,
            "runId": pred.run_id,
        },
        "source_reading_id": state.reading.id,
        "prediction_id": pred.id,
    }


def _update(rec: Recommendation, values: dict[str, Any]) -> bool:
    """Set the changed fields only (so an unchanged re-run issues no UPDATE). Status is never touched."""
    changed = False
    for key, value in values.items():
        if getattr(rec, key) != value:
            setattr(rec, key, value)
            changed = True
    return changed


@dataclass
class _Tally:
    created: int = 0
    updated: int = 0
    superseded: int = 0
    removed: int = 0


def _sync_plant(db: Session, plant_id: int, state: PlantState, existing: list[Recommendation], tally: _Tally) -> None:
    cands = {c.category: c for c in build_candidates(state)}
    actives: dict[str, list[Recommendation]] = {}
    for rec in existing:
        if rec.status in ACTIVE:
            actives.setdefault(rec.category, []).append(rec)
    primaries: dict[str, Recommendation] = {}  # per category: the user-modified, then newest, active row
    for category, recs in actives.items():
        recs.sort(key=lambda r: (r.user_modified, r.id), reverse=True)
        primaries[category] = recs[0]
        if sum(r.user_modified for r in recs) > 1:
            logger.warning(
                "plant %d has %d user-modified active recommendations for %s (ids %s); leaving them",
                plant_id, sum(r.user_modified for r in recs), category, [r.id for r in recs if r.user_modified],
            )  # fmt: skip
    by_key = {(r.category, r.source_reading_id): r for r in existing}

    targets: set[int] = set()
    removed: set[int] = set()
    for category, cand in cands.items():
        values = _content(state, cand)
        same = by_key.get((category, state.reading.id))
        primary = primaries.get(category)
        target = primary if primary is not None else same
        if target is None:
            db.add(Recommendation(plant_id=plant_id, category=category, status="new", user_modified=False, **values))
            tally.created += 1
            continue
        if same is not None and same is not target:
            # Another row already owns (plant, category, this reading). The unique key allows one of them.
            if not same.user_modified and same.status in ("new", "superseded"):
                db.delete(same)  # an untouched generated duplicate: the target moves onto the reading instead
                db.flush()
                removed.add(same.id)
                tally.removed += 1
            elif same.status in CLOSED and not target.user_modified:
                target = same  # the user closed this very reading; the untouched older active row is superseded below
            else:
                logger.warning(
                    "plant %d %s: recommendation %d cannot move to reading %d (owned by recommendation %d); left as is",
                    plant_id, category, target.id, state.reading.id, same.id,
                )  # fmt: skip
                targets.add(target.id)
                continue
        targets.add(target.id)
        if target.status in CLOSED:  # the user closed this very recommendation: leave it closed
            _update(target, {"prediction_id": values["prediction_id"]})
            continue
        if target.status == "superseded":  # its condition is back, and nobody touched it
            target.status, target.resolved_at = "new", None
        if _update(target, values):
            tally.updated += 1
    for rec in existing:
        if rec.status in ACTIVE and rec.id not in targets and rec.id not in removed and not rec.user_modified:
            rec.status = "superseded"
            tally.superseded += 1
    db.flush()


def _refresh(db: Session, plant: Plant, existing: list[Recommendation], tally: _Tally) -> None:
    state = load_state(db, plant)
    plant.status = severity_to_status(state.pred.severity if state else None)
    if state is not None:
        _sync_plant(db, plant.id, state, existing, tally)


def refresh_plant(db: Session, plant_id: int) -> None:
    """Recompute one plant's recommendations and `plants.status` from its stored predictions. The caller commits."""
    plant = db.get(Plant, plant_id)
    if plant is not None:
        _refresh(db, plant, repo.recommendations_for_plant(db, plant_id), _Tally())


def build_for_run(db: Session, pipeline_run_id: int) -> int:
    """Rebuild the recommendations and plant statuses from the stored ML output. Returns how many were created.

    Idempotent: unchanged data changes nothing. A plant with no `ok` prediction keeps its recommendations and
    gets status `unmeasured`.
    """
    by_plant = repo.recommendations_by_plant(db)
    tally = _Tally()
    for plant in repo.list_plants(db):
        _refresh(db, plant, by_plant.get(plant.id, []), tally)
    db.commit()
    logger.info(
        "run %d recommendations: %d created, %d updated, %d superseded, %d duplicates removed (total %d)",
        pipeline_run_id, tally.created, tally.updated, tally.superseded, tally.removed, repo.count_recommendations(db),
    )  # fmt: skip
    return tally.created


def reset_generated(db: Session) -> None:
    """Before a `reset=true` recompute: delete untouched recommendations; keep user-modified ones (with their
    status) and detach them from the predictions that are about to be deleted. The caller commits."""
    deleted = repo.delete_generated_recommendations(db)
    detached = repo.detach_recommendations_from_predictions(db)
    logger.info("reset: deleted %d untouched recommendations, detached %d from their predictions", deleted, detached)


# --- API-facing operations ------------------------------------------------------------------
class RecommendationNotFound(LookupError):
    pass


class RecommendationConflict(Exception):
    def __init__(self, other_id: int):
        super().__init__(f"another recommendation (rec-{other_id}) is already active for this plant and category")
        self.other_id = other_id


_REC_ID = re.compile(r"^(?:rec-)?(\d{1,12})$")


def parse_rec_id(raw: str) -> int | None:
    match = _REC_ID.match(raw)
    return int(match.group(1)) if match else None


def to_out(rec: Recommendation, plant: Plant, reading: SensorReading, daytime: tuple[int, int]) -> RecommendationOut:
    based_on = rec.based_on
    return RecommendationOut(
        id=f"rec-{rec.id}",
        plant_id=rec.plant_id,
        plant_name=plant.name,
        cluster=plant.cluster,
        title=rec.title,
        simple_action=rec.simple_action,
        priority=rec.priority,
        status=rec.status,
        category=rec.category,
        supporting_sensor_values=[SensorValue(**v) for v in supporting_sensor_values(reading, plant.optimal_ranges, daytime)],
        reason=rec.reason,
        scientific_context=rec.scientific_context,
        guidance_source=rec.guidance_source,
        created_at=to_ist_iso(rec.created_at) or "",
        updated_at=to_ist_iso(rec.updated_at),
        resolved_at=to_ist_iso(rec.resolved_at),
        source=rec.source,
        confidence=rec.confidence,
        based_on=BasedOn(timestamp=based_on["timestamp"], model_version=based_on.get("modelVersion"), run_id=based_on.get("runId"))
        if based_on
        else None,
        ml_rec_id=rec.ml_rec_id,
    )


def list_out(db: Session, config_path: Path, **filters: Any) -> tuple[list[RecommendationOut], int]:
    rows, total = repo.list_recommendations(db, **filters)
    daytime = read_daytime_hours(config_path)
    return [to_out(rec, plant, reading, daytime) for rec, plant, reading in rows], total


def update_status(db: Session, raw_id: str, status: str, config_path: Path) -> RecommendationOut:
    """A user status change: marks the recommendation user-modified so re-runs preserve it.

    Raises `RecommendationNotFound`, or `RecommendationConflict` when reopening a closed/superseded
    recommendation while another one is already active for the same plant and category.
    """
    rec_id = parse_rec_id(raw_id)
    rec = repo.get_recommendation(db, rec_id) if rec_id is not None else None
    if rec is None:
        raise RecommendationNotFound(f"recommendation {raw_id} not found")
    if status in ACTIVE and rec.status not in ACTIVE:
        other = repo.other_active_recommendation(db, rec)
        if other is not None:
            raise RecommendationConflict(other.id)
    set_status(db, rec, status)
    row = repo.get_recommendation_row(db, rec.id)
    if row is None:
        raise RecommendationNotFound(f"recommendation {raw_id} not found")
    return to_out(*row, read_daytime_hours(config_path))


def set_status(db: Session, rec: Recommendation, status: str, now: datetime | None = None) -> Recommendation:
    now = now or utcnow()
    rec.status = status
    rec.user_modified = True
    rec.resolved_at = now if status in CLOSED else None
    rec.updated_at = now
    db.commit()
    return rec
