"""Test helpers: write a synthetic ML report folder (global SHAP) shaped like the ML pipeline's own output."""

import csv
import json
from pathlib import Path

from tests.conftest import REAL_ARTIFACTS

CLASSES = ("healthy", "low", "moderate", "high")


def feature_names() -> list[str]:
    run = json.loads((REAL_ARTIFACTS / "latest.json").read_text())["run_id"]
    return json.loads((REAL_ARTIFACTS / run / "classifier.meta.json").read_text())["feature_names"]


def write_shap_report(
    reports_dir: Path, run_id: str, *, stamp: str = "20261025T085725Z", nested: bool = True, scale: float = 1.0
) -> Path:
    """`reports/<run>/evaluation_<stamp>/{shap_global_importance.json, evaluation.json, shap_sample_explanations.csv}`
    (or the flat `reports/<run>/` layout with `nested=False`). Importance falls with the feature's position."""
    folder = reports_dir / run_id / (f"evaluation_{stamp}" if nested else "")
    folder.mkdir(parents=True, exist_ok=True)
    names = feature_names()
    mean_abs = {n: scale * (1.0 - i / (len(names) + 1)) for i, n in enumerate(names)}
    per_class = {c: dict(list(mean_abs.items())[:15]) for c in CLASSES}
    (folder / "shap_global_importance.json").write_text(
        json.dumps({"mean_abs_shap": mean_abs, "per_class_top15": per_class, "note": "test report"})
    )
    (folder / "evaluation.json").write_text(json.dumps({"run_id": run_id, "provenance": "synthetic"}))
    with (folder / "shap_sample_explanations.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row", "predicted", "feature", "value", "contribution", "direction"])
        # feature A varies (0, 5, 10), feature B is constant (7), feature C has a missing value
        writer.writerow([1, "low", names[0], 0.0, 0.2, "towards_predicted_class"])
        writer.writerow([2, "high", names[0], 5.0, -0.1, "away_from_predicted_class"])
        writer.writerow([3, "high", names[0], 10.0, 0.4, "towards_predicted_class"])
        writer.writerow([1, "low", names[1], 7.0, 0.3, "towards_predicted_class"])
        writer.writerow([2, "high", names[1], 7.0, 0.1, "towards_predicted_class"])
        writer.writerow([3, "high", names[2], "", 0.05, "towards_predicted_class"])
        writer.writerow([2, "high", names[2], 3.0, -0.05, "away_from_predicted_class"])
    return folder
