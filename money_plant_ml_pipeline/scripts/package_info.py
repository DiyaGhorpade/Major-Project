#!/usr/bin/env python
"""Print Python / platform / dependency versions and artifact status (useful for Jetson setup and bug reports)."""
import json
import platform
import sys

import _bootstrap  # noqa: F401
from plant_ml.artifact_manager import ArtifactError, ArtifactStore, dependency_versions
from plant_ml.config import load_config


def main() -> int:
    cfg = load_config()
    info = {"python": sys.version, "machine": platform.machine(), "system": platform.platform(),
            "dependencies": dependency_versions()}
    store = ArtifactStore(cfg.path("paths.artifacts_dir"))
    info["artifact_runs"] = store.list_runs()
    try:
        info["latest_run"] = store.resolve_run().name
    except ArtifactError as exc:
        info["latest_run"] = f"none ({exc})"
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
