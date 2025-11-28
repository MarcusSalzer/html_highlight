from collections.abc import Callable
from pathlib import Path
from typing import Any

import pydantic
import torch
from torch import nn
from torch.types import Tensor
from torch.utils.data import DataLoader

from src import torch_metrics as tm
from src.torch_util import SequenceDataset


class RNNTaggerConfig(pydantic.BaseModel):
    """All model specific parameters for the RNNTagger"""

    model_config = pydantic.ConfigDict(extra="forbid")  # dont allow extra trash

    # dimensions
    d_emb_token: int = 12
    d_emb_tag: int = 8
    d_hidden_rnn: int = 64
    # layers
    rnn_variant: str = "lstm"
    n_rnn_layers: int = 1
    mlp_sizes: list[int] | None = None
    bidi: bool = True
    dropout_rnn: float = 0.0
    dropout_between: float = 0.0
    dropout_mlp: float = 0.0

    def model_post_init(self, context: Any) -> None:
        assert self.rnn_variant in {"rnn", "gru", "lstm"}, "unexpected RNN variant"
        if self.n_rnn_layers == 1:
            assert self.dropout_rnn == 0, "Cannot apply dropout with single RNN layer"


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
        metric_funs: dict[str, Callable[[Tensor, Tensor], float]] = {
            "loss": loss_fn,
            "acc": tm.acc_logits,
            "balanced_acc": tm.balanced_acc_logits,
        }
        return opt, lrs_plat, loss_fn, metric_funs

    @property
    def device(self):
        return next(self.parameters()).device

    def complete_train_loop(
        self,
        settings: TrainSettings,
        dset_train: SequenceDataset,
        dset_val: SequenceDataset,
        verbose: bool = False,
        epoch_cb: Callable | None = None,
        save_dir: Path | None = None,
        save_wait: int = 10,  # never save before this epoch
    ):
        """Train the tagger model."""
        if save_dir is not None:
            save_dir.mkdir(exist_ok=True, parents=True)

        # Training utilities
        opt, lrs_plat, loss_fn, metric_funs = self._train_utils(settings)

        device_t = self.device.type
        assert device_t in ("cuda", "cpu"), "expects cuda or cpu"
        # Data
        if verbose:
            print(f"train: {dset_train}")
            print(f"val  : {dset_val}")
        dl_train = DataLoader(dset_train, settings.bs_train, shuffle=True)
        if dset_train.tokens.device.type != device_t:
            print(f"NOTE: model on {device_t}, data on {dset_train.tokens.device}")

        # put validation data on device "permanently"
        dset_val.to_device(device_t)

        # log all metrics during training
        # useful for learning curves
        metrics: dict[str, list[float]] = {
            k: [] for k in ["train_loss"] + [f"val_{k}" for k in metric_funs]
        }
        # track best epoch for each validation metric
        # useful for early stopping condition.
        best_val: dict[str, tuple[int, float]] = {
            k: (0, -tm.METRIC_DIR[k] * float("inf")) for k in metric_funs
        }

        for epoch in range(settings.max_epochs):
            # --- Training ---
            self.train()
            loss_train: float = 0.0
            n_samples = 0  # to normalize loss
            for inputs, labels in dl_train:
                inputs = {k: v.to(device_t) for k, v in inputs.items()}
                labels = labels.to(device_t)
                # Forward pass
                logits = self(**inputs)
                bs = logits.shape[0]
                # Reshape for loss calculation
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss_train += loss.item() * bs
                n_samples += bs
                # Backward pass and optimization
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
            # normalize
            loss_train /= n_samples
            metrics["train_loss"].append(loss_train)

            # --- Validation (all data at once) ---
            self.eval()
            with torch.no_grad():
                inputs, labels = dset_val.get_all()

                logits = self(**inputs)

                # compute all metrics
                for k in metric_funs:
                    value = float(metric_funs[k](logits.view(-1, logits.size(-1)), labels.view(-1)))
                    metrics[f"val_{k}"].append(value)

                    # --- Track best validation metrics ---

                    d = tm.METRIC_DIR[k]
                    if value * d > best_val[k][1] * d:
                        best_val[k] = (epoch, value)
                        # save model state if specified
                        if save_dir is not None and epoch > save_wait:
                            fp = save_dir / f"best_{k}_state.pth"
                            torch.save(self.state_dict(), fp)

            # --- Prints ---
            if verbose:
                print(
                    f"{epoch=:4d}. train_loss={metrics['train_loss'][-1]:.4f} | "
                    + " | ".join(f"val_{k}= {metrics[f'val_{k}'][-1]:.4f}" for k in metric_funs)
                    + f" | lr={opt.param_groups[0]['lr']:.1e}"
                )
            if epoch_cb is not None:
                epoch_cb(epoch, {k: v[-1] for k, v in metrics.items()})

            # --- LR plateau & Early stopping ---
            lrs_plat.step(metrics["val_loss"][-1])
            # check all for early stopping
            since = [epoch - best_ep for (best_ep, _) in best_val.values()]
            if all(s > settings.stop_patience for s in since):
                print("Early stopping")
                break

        return metrics


class RNNTagger(TaggerModel):
    """A recurrent network for sequence tagging"""

    # Lookup for available RNN variants
    rnn_variants = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}

    def __init__(
        self,
        conf: RNNTaggerConfig,
        vocab_sz_token: int,
        vocab_sz_tag: int,
    ):
        super().__init__(vocab_sz_token, vocab_sz_tag)

        self.embedding_tokens = nn.Embedding(vocab_sz_token, conf.d_emb_token, padding_idx=0)
        self.embedding_labels = nn.Embedding(vocab_sz_tag, conf.d_emb_tag, padding_idx=0)

        # choose layer type for recurrent layers
        self.rnn = self.rnn_variants[conf.rnn_variant](
            conf.d_emb_token + conf.d_emb_tag,  # LSTM will receive tokens, tags stacked
            conf.d_hidden_rnn,
            conf.n_rnn_layers,
            batch_first=True,
            dropout=conf.dropout_rnn,
            bidirectional=conf.bidi,
        )

        self.dropout_between = (
            nn.Dropout(conf.dropout_between) if conf.dropout_between > 0 else nn.Identity()
        )

        # what dimension will the hidden state have? double if bidirectional
        d_hidden = conf.d_hidden_rnn * (2 if conf.bidi else 1)

        # Build FF layers if sizes given
        if conf.mlp_sizes:
            self.mlp = nn.Sequential()
            for sz in conf.mlp_sizes:
                self.mlp.append(nn.Linear(d_hidden, sz))
                self.mlp.append(nn.ReLU(inplace=True))
                if conf.dropout_mlp > 0:
                    self.mlp.append(nn.Dropout(conf.dropout_mlp))
                d_hidden = sz  # input size for next layer
        else:
            self.mlp = nn.Identity()

        # output
        self.tag_clf = nn.Linear(d_hidden, vocab_sz_tag)

    def __str__(self):
        return f"RNNTagger_{type(self.rnn).__name__}"

    def forward(self, tokens: Tensor, labels_det: Tensor) -> Tensor:
        bs, seq_len = tokens.shape[:2]

        # embed tokens and inital labels
        embeds_tokens = self.embedding_tokens(tokens)
        embeds_labels = self.embedding_labels(labels_det)

        # Cat -> (bs, seq_len, emb_token + emb_tag)
        embeds = torch.cat([embeds_tokens, embeds_labels], dim=-1)

        lstm_out, _ = self.rnn(embeds)  #  -> (bs, seq_len, actual_hidden)
        lstm_out = self.dropout_between(lstm_out)  # Dropout or Identity
        last_hidden = self.mlp(lstm_out)  # MLP or Identity
        # final clf layer
        logits = self.tag_clf(last_hidden)  # -> (bs, seq_len, tagset_size)

        # Sanity check
        assert logits.shape == (bs, seq_len, self.vocab_sz_tag)

        return logits
