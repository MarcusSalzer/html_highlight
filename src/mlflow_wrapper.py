import logging
from collections.abc import Sequence

import mlflow
from pydantic import BaseModel


def init(
    exp_name: str,
    tracking_uri: str = "sqlite:///./data/mlflow.db",
    log_level: int = logging.WARNING,
    sys_metrics_interval: int | None = 5,
):
    """Start mlflow tracking."""
    # important set URI first to use desired DB
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(exp_name)

    print("[MLFLOW] DB:", mlflow.get_tracking_uri())

    # log level
    logging.getLogger("mlflow").setLevel(log_level)
    logging.getLogger("alembic.runtime.plugins").setLevel(log_level)

    if sys_metrics_interval:
        mlflow.enable_system_metrics_logging()
        mlflow.set_system_metrics_sampling_interval(sys_metrics_interval)


def log_params_pydantic(models: Sequence[BaseModel]):
    for m in models:
        mlflow.log_params(m.model_dump())
