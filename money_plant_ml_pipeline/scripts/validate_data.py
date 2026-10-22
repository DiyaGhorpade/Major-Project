#!/usr/bin/env python
"""Validate a sensor CSV and print / save a data-quality report. Exit code 1 if validation fails (mode=fail)."""
import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from plant_ml.config import load_config
from plant_ml.data_loader import DataLoadError, load_csv
from plant_ml.reports import write_json
from plant_ml.validation import ValidationError, validate_frame, validation_messages


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", help="CSV path (default: paths.default_input in config.yaml)")
    ap.add_argument("--mode", choices=["fail", "warn", "quarantine"], help="override validation.mode")
    ap.add_argument("--inference", action="store_true", help="labels not required (inference-style CSV)")
    ap.add_argument("--output", help="optional path for the JSON report")
    ap.add_argument("--config", help="config.yaml path")
    a = ap.parse_args()
    cfg = load_config(a.config)
    path = Path(a.input) if a.input else cfg.path("paths.default_input")
    try:
        df = load_csv(path, cfg.get("timezone"), require_labels=not a.inference)
        res = validate_frame(df, mode=a.mode or cfg.get("validation.mode", "warn"), require_labels=not a.inference)
    except (DataLoadError, ValidationError) as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(res.report, indent=2, default=str))
    for m in validation_messages(res.report):
        print("WARNING:", m, file=sys.stderr)
    if a.output:
        write_json(Path(a.output), res.report)
        print(f"report written to {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
