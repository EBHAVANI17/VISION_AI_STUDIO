"""Experiment tracking abstraction for MLflow, Weights & Biases, and offline runs."""

from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class ExperimentTracker(Protocol):
    """Protocol for logging parameters, metrics and artifacts."""

    def log_params(self, params: dict[str, Any]) -> None: ...
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...
    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None: ...
    def finish(self) -> None: ...


class NullTracker:
    """Fallback tracker that logs locally when no tracking backend is active."""

    def log_params(self, params: dict[str, Any]) -> None:
        logger.info("Params: %s", params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        step_str = f" [step {step}]" if step is not None else ""
        logger.info("Metrics%s: %s", step_str, metrics)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        logger.info("Artifact saved: %s -> %s", local_path, artifact_path)

    def finish(self) -> None:
        logger.info("Run finished.")


class MLflowTracker:
    """MLflow experiment tracker."""

    def __init__(self, experiment_name: str, run_name: str | None = None) -> None:
        import mlflow

        self._mlflow = mlflow
        mlflow.set_experiment(experiment_name)
        self._run = mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict[str, Any]) -> None:
        self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        self._mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def finish(self) -> None:
        self._mlflow.end_run()


class WandbTracker:
    """Weights & Biases experiment tracker."""

    def __init__(self, project: str, run_name: str | None = None, config: dict[str, Any] | None = None) -> None:
        import wandb

        self._wandb = wandb
        self._run = wandb.init(project=project, name=run_name, config=config)

    def log_params(self, params: dict[str, Any]) -> None:
        if self._run:
            self._run.config.update(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self._wandb.log(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        artifact = self._wandb.Artifact(name="model-artifact", type="model")
        artifact.add_file(local_path)
        self._wandb.log_artifact(artifact)

    def finish(self) -> None:
        self._wandb.finish()


def get_tracker(system: str = "none", experiment_name: str = "vision-ai-studio", **kwargs: Any) -> ExperimentTracker:
    """Factory to instantiate the appropriate experiment tracker."""
    system = system.lower()
    if system == "mlflow":
        try:
            return MLflowTracker(experiment_name=experiment_name, run_name=kwargs.get("run_name"))
        except ImportError:
            logger.warning("MLflow not installed; falling back to NullTracker.")
            return NullTracker()
    elif system == "wandb":
        try:
            return WandbTracker(project=experiment_name, run_name=kwargs.get("run_name"), config=kwargs.get("config"))
        except ImportError:
            logger.warning("wandb not installed; falling back to NullTracker.")
            return NullTracker()
    return NullTracker()
