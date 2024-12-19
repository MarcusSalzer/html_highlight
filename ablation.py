from typing import Callable


class Ablation:
    def __init__(
        self, name: str, metrics: dict[str, Callable], save_dir: str = "ablations"
    ):
        self.name = name
        self.metrics = metrics

    def evaluate(y_true, y_pred, meta: dict):
        pass
