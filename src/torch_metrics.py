"""Some evaluation metrics, implemented for torch tensors."""

import torch
from torch import Tensor

# Direction to optimize
METRIC_DIR = {"loss": -1, "acc": 1, "balanced_acc": 1}


def balanced_acc(pred: Tensor, labels: Tensor):
    """Multiclass balanced accuracy.

    Inspired by https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score
    """
    # divide by class frequency
    wh = 1 / (labels.reshape(-1, 1) == labels).sum(0)
    return (((pred == labels) * wh).sum() / wh.sum()).item()


def acc(pred: Tensor, labels: Tensor):
    """Plain old accuracy."""
    return (pred == labels).mean(dtype=torch.float32).item()


def acc_logits(logits: Tensor, labels: Tensor):
    """Plain old accuracy, from logits."""

    pred = logits.argmax(dim=-1)
    return acc(pred, labels)


def balanced_acc_logits(logits: Tensor, labels: Tensor):
    pred = logits.argmax(dim=-1)
    return balanced_acc(pred, labels)
