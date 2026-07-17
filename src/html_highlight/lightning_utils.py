import logging
from pathlib import Path
from typing import NamedTuple

import lightning
from lightning.pytorch import callbacks
from lightning.pytorch.callbacks import Callback


class EpochPrintCallback(Callback):
    """Prints a one-liner every `print_every` epochs."""

    def __init__(self, print_every: int = 1, keys=("train_loss", "val_loss", "lr")):
        self.print_every = print_every
        self.keys = keys

    def on_validation_epoch_end(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ):
        epoch = trainer.current_epoch
        if epoch % self.print_every != 0:
            return
        metrics = trainer.callback_metrics
        parts = [f"epoch={epoch:4d}"]
        for k in self.keys:
            if k in metrics:
                parts.append(f"{k}={metrics[k]:.6f}")
        print(" | ".join(parts))


def simple_checkpoint(model_dir: str | Path, n_epochs: int = 5, monitor: str = "val_loss"):
    return callbacks.ModelCheckpoint(
        model_dir,
        "best_vl",
        monitor=monitor,
        save_top_k=1,
        every_n_epochs=n_epochs,
        enable_version_counter=False,
    )


class _SuppressMatching(logging.Filter):
    def __init__(self, *fragments: str):
        self.fragments = fragments

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(f in msg for f in self.fragments)


def stop_annoying_info_messages():
    logging.getLogger("lightning.pytorch.utilities.rank_zero").addFilter(
        _SuppressMatching("try installing")
    )


class LossPair(NamedTuple):
    epoch: int
    train_loss: float
    val_loss: float


class GeneralizationGapCallback(Callback):
    """Track the best validation loss, and the corresponding best training loss."""

    def __init__(self):
        self.best_val = LossPair(0, float("inf"), float("inf"))
        self.best_train = LossPair(0, float("inf"), float("inf"))

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        if "val_loss" not in metrics or "train_loss" not in metrics:
            return

        val_loss = metrics["val_loss"].item()
        train_loss = metrics["train_loss"].item()

        if val_loss < self.best_val.val_loss:
            self.best_val = LossPair(trainer.current_epoch, train_loss, val_loss)

        if train_loss < self.best_train.train_loss:
            self.best_train = LossPair(trainer.current_epoch, train_loss, val_loss)
