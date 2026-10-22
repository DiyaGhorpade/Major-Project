#!/usr/bin/env python
"""Offline SHAP report (summary plot + tabular explanations). Requires the optional 'shap' + 'matplotlib' packages."""
import argparse
import sys

import _bootstrap  # noqa: F401
from plant_ml.artifact_manager import ArtifactError
from plant_ml.config import load_config
from plant_ml.evaluation_run import evaluate_run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input")
    ap.add_argument("--run-id")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    try:
        out = evaluate_run(cfg, a.input or cfg.path("paths.default_input"), a.run_id, with_shap=True)
    except (ArtifactError, ValueError, FileNotFoundError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    shap = out.get("shap", {})
    if not shap.get("available"):
        print("SHAP unavailable:", shap.get("reason"))
        return 2
    print("SHAP reports written to", out["reports_dir"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
