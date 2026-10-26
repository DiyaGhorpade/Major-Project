"""Past-only temporal smoothing of per-reading class probabilities (issue #5).

Stress severity changes slowly, while the per-reading classifier output (and the labels) flicker between adjacent
classes. A hidden-Markov forward filter combines the classifier evidence of the trailing ``window`` readings of a plant
with a class-transition matrix:

    alpha_0 = prior * e_0;   alpha_t = (A^T alpha_{t-1}) * e_t;   e_t = (p_t / prior) ** temperature

``A`` (4x4) and ``prior`` are fitted on TRAINING labels only (consecutive readings of the same plant). The filter restarts
``window`` readings back, so it needs no stored state: inference re-predicts the trailing rows from the history the caller
already sends. Only readings at or before the current one are used.
"""
from __future__ import annotations

import numpy as np

N_CLASSES = 4
METHODS = ("none", "hmm_forward_filter")


def smoothing_params(cfg) -> dict:
    s = cfg.get("classifier.smoothing", None) or {}
    method = str(s.get("method", "none"))
    if method not in METHODS:
        raise ValueError(f"classifier.smoothing.method must be one of {METHODS}")
    window = int(s.get("window", 12))
    if window < 1:
        raise ValueError("classifier.smoothing.window must be >= 1")
    return {"method": method, "window": window, "temperature": float(s.get("temperature", 0.3)),
            "transition_pseudocount": float(s.get("transition_pseudocount", 1.0))}


def fit_transitions(y: np.ndarray, groups: np.ndarray, pseudocount: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Row-normalised class-transition matrix and class prior from labels in chronological order within each group."""
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    counts = np.full((N_CLASSES, N_CLASSES), float(pseudocount))
    for g in np.unique(groups):
        yy = y[groups == g]
        np.add.at(counts, (yy[:-1], yy[1:]), 1.0)
    A = counts / counts.sum(axis=1, keepdims=True)
    prior = (np.bincount(y, minlength=N_CLASSES) + float(pseudocount)) / (len(y) + N_CLASSES * float(pseudocount))
    return A, prior


def forward_filter(proba: np.ndarray, groups: np.ndarray, A: np.ndarray, prior: np.ndarray, temperature: float,
                   window: int) -> np.ndarray:
    """Filtered probabilities for every row. Rows must be chronological within each group (other groups may interleave);
    row i uses only rows of its own group at positions max(0, k - window + 1)..k (k = its position in the group)."""
    proba = np.asarray(proba, dtype=float)
    groups = np.asarray(groups)
    A = np.asarray(A, dtype=float)
    prior = np.asarray(prior, dtype=float)
    ev = (np.clip(proba, 1e-6, 1.0) / prior) ** float(temperature)
    out = np.empty_like(proba)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        e = ev[idx]
        for k in range(len(idx)):
            start = max(0, k - window + 1)
            a = prior * e[start]
            a /= a.sum()
            for j in range(start + 1, k + 1):
                a = (A.T @ a) * e[j]
                a /= a.sum()
            out[idx[k]] = a
    return out


def to_meta(params: dict, A: np.ndarray, prior: np.ndarray) -> dict:
    return {**params, "transition_matrix": np.round(A, 6).tolist(), "prior": np.round(prior, 6).tolist(),
            "fitted_on": "training labels only (consecutive readings of the same plant)"}


def apply_from_meta(meta: dict | None, proba: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Smoothed probabilities using a stored ``temporal_smoothing`` block; unchanged when absent or method 'none'."""
    if not meta or meta.get("method", "none") == "none":
        return proba
    return forward_filter(proba, groups, np.asarray(meta["transition_matrix"]), np.asarray(meta["prior"]),
                          float(meta["temperature"]), int(meta["window"]))
