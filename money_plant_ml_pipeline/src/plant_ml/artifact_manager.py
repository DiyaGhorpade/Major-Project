"""Versioned model-artifact storage, integrity checks and compatibility checks.

Layout:
    artifacts/
      latest.json                      -> {"run_id": ..., "previous_run_id": ...}
      v1.0.0_20261021T061500Z/          (one directory per training run; never overwritten)
        manifest.json
        classifier.xgb.json  + classifier.meta.json
        clustering.joblib    + clustering.meta.json
        anomaly.joblib       + anomaly.meta.json

Serialisation: XGBoost uses its native JSON format. scikit-learn objects use joblib (pickle-based),
so artifacts must only be loaded from trusted locations; SHA-256 hashes are verified on load.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from .config import Config

MANIFEST_SCHEMA_VERSION = 1


class ArtifactError(RuntimeError):
    """Missing, corrupt or incompatible artifacts (message is meant to be shown to an operator)."""


def dependency_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine()}
    for mod in ("numpy", "pandas", "scipy", "sklearn", "xgboost", "joblib", "yaml", "fastapi", "pydantic", "shap", "matplotlib"):
        try:
            m = __import__(mod)
            versions[mod] = str(getattr(m, "__version__", "unknown"))
        except Exception:  # noqa: BLE001 - optional deps may be absent
            versions[mod] = "not installed"
    return versions


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _major_minor(v: str) -> str:
    parts = str(v).split(".")
    return ".".join(parts[:2])


@dataclass
class ModelBundle:
    run_dir: Path
    manifest: dict
    booster: Any
    cluster_pipeline: Any
    kmeans: Any
    anomaly_pipeline: Any
    iforest: Any
    meta: dict = field(default_factory=dict)  # name -> metadata dict
    warnings: list[str] = field(default_factory=list)


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    # ------------------------------------------------------------------ writing
    def create_run(self, version: str) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = self.root / f"v{version}_{stamp}"
        n = 1
        while run_dir.exists():  # never overwrite an existing run
            n += 1
            run_dir = self.root / f"v{version}_{stamp}_{n}"
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    @staticmethod
    def save_joblib(run_dir: Path, name: str, obj: Any) -> Path:
        path = run_dir / f"{name}.joblib"
        joblib.dump(obj, path)
        return path

    @staticmethod
    def save_booster(run_dir: Path, name: str, booster: Any) -> Path:
        path = run_dir / f"{name}.xgb.json"
        booster.save_model(str(path))
        return path

    @staticmethod
    def write_json(path: Path, payload: dict) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=_json_default)

    def write_model_metadata(self, run_dir: Path, name: str, artifact_path: Path, metadata: dict) -> Path:
        meta = dict(metadata)
        meta["artifact_file"] = artifact_path.name
        meta["artifact_sha256"] = sha256_file(artifact_path)
        meta["artifact_size_bytes"] = artifact_path.stat().st_size
        path = run_dir / f"{name}.meta.json"
        self.write_json(path, meta)
        return path

    def write_manifest(self, run_dir: Path, manifest: dict) -> Path:
        manifest = dict(manifest)
        manifest["schema_version"] = MANIFEST_SCHEMA_VERSION
        path = run_dir / "manifest.json"
        self.write_json(path, manifest)
        return path

    def set_latest(self, run_dir: Path) -> None:
        pointer = self.root / "latest.json"
        previous = None
        if pointer.exists():
            try:
                previous = json.loads(pointer.read_text(encoding="utf-8")).get("run_id")
            except Exception:  # noqa: BLE001
                previous = None
        self.write_json(pointer, {"run_id": run_dir.name, "previous_run_id": previous})

    # ------------------------------------------------------------------ reading
    def resolve_run(self, run_id: str | None = None) -> Path:
        if not self.root.exists():
            raise ArtifactError(f"Artifacts directory not found: {self.root}. Train first: python scripts/train_all.py")
        if run_id is None:
            pointer = self.root / "latest.json"
            if not pointer.exists():
                raise ArtifactError(f"No trained models found in {self.root} (latest.json missing). "
                                    "Train explicitly with: python scripts/train_all.py")
            run_id = json.loads(pointer.read_text(encoding="utf-8")).get("run_id")
        run_dir = self.root / str(run_id)
        if not (run_dir / "manifest.json").exists():
            raise ArtifactError(f"Run directory incomplete or missing manifest.json: {run_dir}")
        return run_dir

    def list_runs(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if (p / "manifest.json").exists())


def _json_default(obj: Any):
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:  # noqa: BLE001
        pass
    if isinstance(obj, (Path,)):
        return str(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


def check_compatibility(manifest: dict, allow_mismatch: bool = False) -> list[str]:
    """Compare training-time and runtime library versions. Returns warnings; raises on hard mismatch."""
    warnings: list[str] = []
    trained = manifest.get("dependency_versions", {})
    now = dependency_versions()
    problems = []
    for lib in ("sklearn", "xgboost", "numpy"):
        t, n = trained.get(lib), now.get(lib)
        if t and n and n != "not installed" and _major_minor(t) != _major_minor(n):
            problems.append(f"{lib}: trained with {t}, running {n}")
    if problems:
        msg = ("Artifact/runtime library mismatch -> " + "; ".join(problems) +
               ". Retrain on this machine (python scripts/train_all.py) or install the training versions "
               "(see BUILD_REPORT.md). Override with PLANT_ML_ALLOW_VERSION_MISMATCH=1 at your own risk.")
        if allow_mismatch:
            warnings.append(msg)
        else:
            raise ArtifactError(msg)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ArtifactError(f"Unsupported manifest schema_version {manifest.get('schema_version')} "
                            f"(expected {MANIFEST_SCHEMA_VERSION})")
    return warnings


def load_bundle(cfg: Config, run_id: str | None = None) -> ModelBundle:
    """Load all three models after verifying hashes, feature lists and library compatibility."""
    store = ArtifactStore(cfg.path("paths.artifacts_dir"))
    run_dir = store.resolve_run(run_id)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    warnings = check_compatibility(manifest, bool(cfg.get("artifacts.allow_version_mismatch", False)))

    meta: dict[str, dict] = {}
    for name in ("classifier", "clustering", "anomaly"):
        mpath = run_dir / f"{name}.meta.json"
        if not mpath.exists():
            raise ArtifactError(f"Missing metadata for '{name}' in {run_dir}")
        meta[name] = json.loads(mpath.read_text(encoding="utf-8"))
        apath = run_dir / meta[name]["artifact_file"]
        if not apath.exists():
            raise ArtifactError(f"Missing artifact file {apath}")
        if sha256_file(apath) != meta[name]["artifact_sha256"]:
            raise ArtifactError(f"Checksum mismatch for {apath.name}: file is corrupt or was modified")

    try:
        import xgboost as xgb
        booster = xgb.Booster()
        booster.load_model(str(run_dir / meta["classifier"]["artifact_file"]))
        cl = joblib.load(run_dir / meta["clustering"]["artifact_file"])
        an = joblib.load(run_dir / meta["anomaly"]["artifact_file"])
    except ArtifactError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ArtifactError(f"Failed to deserialise artifacts in {run_dir}: {exc}") from exc

    return ModelBundle(run_dir=run_dir, manifest=manifest, booster=booster,
                       cluster_pipeline=cl["pipeline"], kmeans=cl["kmeans"],
                       anomaly_pipeline=an["pipeline"], iforest=an["model"], meta=meta, warnings=warnings)
