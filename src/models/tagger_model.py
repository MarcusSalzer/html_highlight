import time
from collections.abc import Callable
from typing import Any

import pydantic
import torch
from torch import nn
from torch.types import Tensor
from torch.utils.data import DataLoader

from src import torch_metrics as tm
from src.datamodels.training import EpochSnapshot


class TrainSettings(pydantic.BaseModel):
    """Config for training the RNNTagger and similar models!"""

    model_config = pydantic.ConfigDict(
        extra="forbid",  # dont allow extra trash
        arbitrary_types_allowed=True,  # allow torch tensor etc
    )

    # optimizer & loss
    start_lr: float = 1e-3
    weight_decay: float = 0.01
    label_smoothing: float = 0.0
    loss_weights: Tensor | None = None
    # data
    bs_train: int = 4
    # stopping etc
    plateau_lr_patience: int = 30
    plateau_lr_factor: float = 0.5
    min_lr: float = 1e-7
    stop_patience: int = 40
    max_epochs: int = 10_000  # Large number, typically stop earlier

    def model_post_init(self, context: Any) -> None:
        assert self.plateau_lr_patience < self.stop_patience, (
            "expects LR patience lower than stop patience"
        )


class TaggerModel(nn.Module):
    """Superclass for sequence tagger models"""

    def __init__(self, vocab_sz_token: int, vocab_sz_tag: int) -> None:
        super().__init__()
        self.vocab_sz_token = vocab_sz_token
        self.vocab_sz_tag = vocab_sz_tag
        self.metric_funs = {
            "acc": tm.acc_logits,
            "balanced_acc": tm.balanced_acc_logits,
        }

    @property
    def tot_weights(self):
        return sum(p.numel() for p in self.parameters())

    def _train_utils(self, settings: TrainSettings):
        """Prepare utilities for training model."""
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=settings.start_lr,
            weight_decay=settings.weight_decay,
        )
        lrs_plat = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            factor=settings.plateau_lr_factor,
            patience=settings.plateau_lr_patience,
            min_lr=settings.min_lr,
        )

        if settings.loss_weights is not None:
            # extra check
            assert len(settings.loss_weights) == self.vocab_size_token, (
                f"expected {self.vocab_size_token} weights"
            )

            assert settings.loss_weights.device == self.device, f"expects device: {self.device}"

        loss_fn = torch.nn.CrossEntropyLoss(
            weight=settings.loss_weights,
            label_smoothing=settings.label_smoothing,
        )

        return opt, lrs_plat, loss_fn

    @property
    def device(self):
        return next(self.parameters()).device

    def train_batch(self, tensors: dict[str, Tensor], loss_fn, opt: torch.optim.Optimizer):
        tensors = {k: v.to(self.device) for k, v in tensors.items()}
        # Forward pass
        logits = self._fw_batch(tensors)

        # Reshape for loss calculation
        loss = loss_fn(logits.view(-1, logits.size(-1)), tensors["tags"].view(-1))

        # Backward pass and optimization
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        bs = logits.shape[0]
        return loss.item() * bs, logits

    def _fw_batch(self, tensors: dict[str, Tensor]):
        return self(tokens=tensors["tokens"], tags_det=tensors["tags_det"])

    def _evaluate(self, dl_val: DataLoader):

        metric_agg: dict[str, float] = {k: 0.0 for k in self.metric_funs}
        n_samples = 0
        self.eval()
        with torch.no_grad():
            for b in dl_val:
                b = {k: v.to(self.device) for k, v in b.items()}
                logits = self._fw_batch(b)
                n_samples += len(logits)

                # compute all metrics
                for k, fun in self.metric_funs.items():
                    metric_agg[k] += float(
                        fun(logits.view(-1, logits.size(-1)), b["tags"].view(-1))
                    ) * len(logits)

        # normalize to average
        for k in metric_agg:
            metric_agg[k] /= n_samples

        return metric_agg

    def complete_train_loop(
        self,
        settings: TrainSettings,
        dl_train: DataLoader,
        dl_val: DataLoader,
        verbose: bool = False,
        epoch_cb: Callable[[EpochSnapshot], None] | None = None,
    ):
        """Train the tagger model."""

        # Training utilities
        opt, lrs_plat, loss_fn = self._train_utils(settings)
        # inlcude loss for validation
        self.metric_funs["loss"] = loss_fn

        # log all metrics during training
        # useful for learning curves
        metrics: dict[str, list[float]] = {
            k: [] for k in ["train_loss"] + [f"val_{k}" for k in self.metric_funs]
        }
        # track best epoch for each validation metric
        # useful for early stopping condition.
        best_val: dict[str, tuple[int, float]] = {
            k: (0, -tm.METRIC_DIR[k] * float("inf")) for k in self.metric_funs
        }

        for epoch in range(settings.max_epochs):
            t_ep_start = time.time()
            # --- Training ---
            self.train()
            loss_train: float = 0.0
            n_samples = 0  # to normalize loss
            for b in dl_train:
                loss_b, logits = self.train_batch(b, loss_fn, opt)
                loss_train += loss_b
                n_samples += len(logits)

            # normalize
            metrics["train_loss"].append(loss_train / n_samples)

            # --- Validation  ---
            val = self._evaluate(dl_val)
            for k, v in val.items():
                # Store results
                metrics[f"val_{k}"].append(v)
                # update best values
                d = tm.METRIC_DIR[k]
                if v * d > best_val[k][1] * d:
                    best_val[k] = (epoch, v)

            # --- Prints ---
            snap = EpochSnapshot(
                epoch=epoch,
                metrics={k: v[-1] for k, v in metrics.items()},
                model=self,
                best=best_val,
                time=time.time() - t_ep_start,
            )

            if epoch_cb is not None:
                epoch_cb(snap)

            if verbose:
                self._epoch_print(snap)

            # --- LR plateau & Early stopping ---
            lrs_plat.step(metrics["val_loss"][-1])
            # check all for early stopping
            since = [epoch - best_ep for (best_ep, _) in best_val.values()]
            if all(s > settings.stop_patience for s in since):
                print("Early stopping")
                break

        return metrics

    def _epoch_print(self, snap: EpochSnapshot):
        print(
            f"{snap.epoch=:4d}. train_loss={snap.metrics['train_loss']:.4f} | "
            + " | ".join(f"val_{k}= {snap.metrics[f'val_{k}']:.4f}" for k in self.metric_funs)
            + f" | t={snap.time:.2f} s"
        )
