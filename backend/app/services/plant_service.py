"""Plant views: catalog fields + latest reading + latest ML prediction, all derived from stored data.

`conditionPrediction` comes from the plant's latest *ok* prediction (the one `plants.status` is derived
from). Its `simpleExplanation` is generated from the stored numbers only.
"""

from sqlalchemy.orm import Session

from app.core.mappings import to_ist_iso
from app.db import repositories as repo
from app.db.models import MlPrediction, Plant, SensorReading
from app.schemas.plant import PlantOut, PlantPredictionOut
from app.schemas.prediction import AnomalyInfo
from app.services import prediction_service
from app.services.sensor_service import to_record


def simple_explanation(pred: MlPrediction) -> str:
    """Plain sentence built from stored values; says what the numbers are and that the type is a guess."""
    when = to_ist_iso(pred.reading_timestamp)
    probability = (pred.probabilities or {}).get(pred.severity or "", pred.confidence)
    p_text = f" (P={probability:.2f})" if probability is not None else ""
    text = f"Model rates stress '{pred.severity}'{p_text} from the latest reading at {when}"
    if pred.stress_type_guess and pred.stress_type_guess not in ("Healthy",):
        text += f"; the stress type ('{pred.stress_type_guess}') is a heuristic guess, not a diagnosis."
    elif pred.severity == "healthy":
        text += "; no stress type is suggested."
    else:
        text += "."
    if pred.is_anomaly:
        text += " This reading also looks unusual (check the sensor); that is not a plant-health verdict."
    return text


def prediction_out(pred: MlPrediction) -> PlantPredictionOut:
    return PlantPredictionOut(
        condition=pred.stress_type_guess,
        severity=pred.severity,
        probabilities=pred.probabilities,
        confidence=pred.confidence,
        timestamp=to_ist_iso(pred.reading_timestamp) or "",
        model_version=pred.model_version,
        record_id=f"R-{pred.reading_id}",
        simple_explanation=simple_explanation(pred),
        env_state=prediction_service.env_state(pred),
        anomaly=AnomalyInfo(is_anomaly=pred.is_anomaly, score=pred.anomaly_score),
    )


def to_out(plant: Plant, reading: SensorReading | None, pred: MlPrediction | None) -> PlantOut:
    return PlantOut(
        id=plant.id,
        name=plant.name,
        species=plant.species,
        variety=plant.variety,
        cluster=plant.cluster,
        cluster_label=plant.cluster_label,
        stress_type=plant.stress_type,
        node_id=plant.node_id,
        sampling_point=reading.sampling_point if reading is not None else plant.sampling_point,
        optimal_ranges=plant.optimal_ranges,
        latest_reading=to_record(reading) if reading is not None else None,
        status=plant.status,
        condition_prediction=prediction_out(pred) if pred is not None else None,
        description=plant.description,
        simple_care_tip=plant.simple_care_tip,
        location=plant.location,
    )


def list_plants(db: Session, limit: int = 100, offset: int = 0) -> tuple[list[PlantOut], int]:
    """A page of plants (ordered by id) with their latest reading / prediction, plus the total plant count."""
    plants = repo.list_plants(db, limit, offset)
    ids = [p.id for p in plants]
    readings = {r.plant_id: r for r in repo.latest_reading_per_plant(db, ids)}
    preds = repo.latest_ok_prediction_per_plant(db, ids)
    return [to_out(p, readings.get(p.id), preds.get(p.id)) for p in plants], repo.count_plants(db)


def get_plant(db: Session, plant_id: int) -> PlantOut | None:
    plant = repo.get_plant(db, plant_id)
    if plant is None:
        return None
    pred = repo.latest_ok_prediction_per_plant(db, [plant_id]).get(plant_id)
    return to_out(plant, repo.latest_reading_for_plant(db, plant_id), pred)


def plant_exists(db: Session, plant_id: int) -> bool:
    return repo.get_plant(db, plant_id) is not None
