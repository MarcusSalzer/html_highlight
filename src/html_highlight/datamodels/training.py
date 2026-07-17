from dataclasses import dataclass

import torch


@dataclass(slots=True)
class EpochSnapshot:
    """For logging during training."""

    epoch: int
    metrics: dict[str, float]
    model: torch.nn.Module
    best: dict
    time: float
