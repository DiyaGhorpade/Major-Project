#!/usr/bin/env python
"""Bundle one artifact run (+ latest.json pointer) into a .tar.gz for copying to another machine (e.g. the Jetson)."""
import argparse
import json
import sys
import tarfile
from pathlib import Path

import _bootstrap  # noqa: F401
from plant_ml.artifact_manager import ArtifactError, ArtifactStore
from plant_ml.config import load_config


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", help="default: latest")
    ap.add_argument("--output", default="reports/exported_artifacts.tar.gz")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    store = ArtifactStore(cfg.path("paths.artifacts_dir"))
    try:
        run = store.resolve_run(a.run_id)
    except ArtifactError as exc:
        print(f"EXPORT FAILED: {exc}", file=sys.stderr)
        return 1
    out = Path(a.output)
    out = out if out.is_absolute() else cfg.root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    pointer = out.parent / "latest.json"
    pointer.write_text(json.dumps({"run_id": run.name, "previous_run_id": None}), encoding="utf-8")
    with tarfile.open(out, "w:gz") as tar:
        tar.add(run, arcname=f"artifacts/{run.name}")
        tar.add(pointer, arcname="artifacts/latest.json")
    pointer.unlink()
    print(f"Exported {run.name} -> {out}\nExtract inside the project root on the target machine: tar -xzf {out.name}")
    print("The target must run the same scikit-learn / xgboost major.minor versions (see manifest.json), or retrain there.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
