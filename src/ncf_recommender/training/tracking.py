"""Experiment tracking utilities."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    """UTC timestamp in ISO format."""
    return datetime.now(UTC).isoformat()


def write_run_manifest(run_dir: Path, payload: dict[str, Any]) -> Path:
    """Persist one run manifest as json."""
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = payload.get("run_id", "run")
    manifest_path = run_dir / f"{run_id}.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


class MlflowTracker:
    """Best-effort MLflow adapter (optional dependency)."""

    def __init__(self, enabled: bool, tracking_uri: str, experiment_name: str) -> None:
        self.enabled = enabled
        self._mlflow = None
        self.active = False
        if not enabled:
            return
        try:
            import mlflow  # type: ignore

            self._mlflow = mlflow
            if tracking_uri.strip():
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            mlflow.start_run()
            self.active = True
        except ImportError:
            print("MLflow enabled in config but mlflow package is not installed.")

    def log_params(self, params: dict[str, Any]) -> None:
        if self.active and self._mlflow is not None:
            self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self.active and self._mlflow is not None:
            self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: Path) -> None:
        if self.active and self._mlflow is not None and path.exists():
            self._mlflow.log_artifact(str(path))

    def close(self) -> None:
        if self.active and self._mlflow is not None:
            self._mlflow.end_run()
