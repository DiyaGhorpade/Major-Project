"""Cluster-card aggregation: "Unclassified Stress" and missing guesses never win the top-condition vote."""

from types import SimpleNamespace

import pytest

from app.db import repositories as repo
from app.services import monitoring_service as ms


@pytest.mark.parametrize(
    ("guesses", "expected"),
    [
        (["Unclassified Stress"] * 3 + ["Temperature/Humidity Stress"], "Temperature/Humidity Stress"),  # real type beats a bigger unclassified pile
        (["Unclassified Stress", "Healthy", None, "Healthy"], "Healthy"),
        (["Water Stress", "Light Stress", "Water Stress"], "Water Stress"),
        (["Unclassified Stress", "Unclassified Stress"], None),  # nothing concrete -> "No dominant type" in the UI
        ([None, None], None),
        ([], None),
    ],
)  # fmt: skip
def test_top_stress_type_ignores_unclassified_and_missing(guesses, expected):
    assert ms.top_stress_type(guesses) == expected


def _pred(severity, guess):
    return SimpleNamespace(severity=severity, stress_type_guess=guess)


def _top_by_cluster(monkeypatch, preds):
    monkeypatch.setattr(repo, "latest_reading_per_plant", lambda db: [])
    monkeypatch.setattr(repo, "latest_ok_prediction_per_plant", lambda db: preds)
    return {s.cluster: s.top_condition for s in ms._cluster_summaries(None)}


def test_cluster_summary_top_condition_skips_unclassified(monkeypatch):
    preds = {
        13: _pred("moderate", "Unclassified Stress"),
        14: _pred("moderate", "Unclassified Stress"),
        15: _pred("high", "Temperature/Humidity Stress"),
        16: _pred("moderate", "Unclassified Stress"),
        5: _pred("high", "Unclassified Stress"),
        6: _pred("moderate", "Unclassified Stress"),
        1: _pred("healthy", "Healthy"),
    }
    top = _top_by_cluster(monkeypatch, preds)
    assert top["Cluster 4"] == "Temperature/Humidity Stress"
    assert top["Cluster 2"] is None  # only unclassified guesses -> no dominant type
    assert top["Cluster 1"] == "Healthy"
    assert top["Cluster 3"] is None  # no predictions at all
