"""
Sequence tagger models refactored to use PyTorch Lightning.

Usage:
    model = RNNTagger(vocab_sz_token=..., vocab_sz_tag=...)
    module = TaggerLitModule(model, settings=TrainSettings())
    trainer = build_trainer(settings)
    trainer.fit(module, dl_train, dl_val)
"""

import lightning
import pydantic
import torch
import torchmetrics
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from torch import nn
from torch.types import Tensor
from torch.utils.data import DataLoader

from html_highlight.tensor_sequence_dataset import TensorSequenceDataset, collate_fn_pad

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class TrainSettings(pydantic.BaseModel):
    """Hyperparameters for training any TaggerModel variant."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # Optimization
    lr: float = 1e-3
    weight_decay: float = 0.01
    bs_train: int = 64
    # Loss
    label_smoothing: float = 0.0
    loss_weights: Tensor | None = None

    # LR scheduler (ReduceLROnPlateau)
    plateau_lr_patience: int = 5
    plateau_lr_factor: float = 0.5
    min_lr: float = 1e-7

    # Early stopping, monitored metric is "val_loss"
    stop_patience: int = 15
    max_epochs: int = -1  # default forever


# ---------------------------------------------------------------------------
# Lightning module  (training logic, independent of architecture)
# ---------------------------------------------------------------------------


class TaggerLitModule(lightning.LightningModule):
    """
    Lightning wrapper around any TaggerModel.

    Subclasses of TaggerModel only need to implement `_fw_batch`; everything
    else (optimizer, scheduler, logging, early stopping) is handled here or
    by the Trainer.
    """

    def __init__(self, model: "TaggerModel", settings: TrainSettings | None = None) -> None:
        super().__init__()
        self.model = model
        self.settings = settings or TrainSettings()
        self.save_hyperparameters(ignore=["model"])

        s = self.settings
        self.loss_fn = nn.CrossEntropyLoss(
            weight=s.loss_weights,
            label_smoothing=s.label_smoothing,
        )

        # torchmetrics handles device placement automatically
        num_classes = model.vocab_sz_tag
        self.val_acc = torchmetrics.Accuracy(
            num_classes=num_classes, average="macro", task="multiclass"
        )
        # macro -> balanced acc
        self.val_bal_acc = torchmetrics.Accuracy(
            num_classes=num_classes, average="macro", task="multiclass"
        )

    # --- forward ---------------------------------------------------------

    def forward(self, **kwargs):
        return self.model(**kwargs)

    # --- shared step -----------------------------------------------------

    def _step(self, batch: dict[str, Tensor]):
        logits = self.model._fw_batch(batch)
        flat_logits = logits.view(-1, logits.size(-1))
        flat_tags = batch["tags"].view(-1)
        loss = self.loss_fn(flat_logits, flat_tags)
        return loss, flat_logits, flat_tags

    # --- train -----------------------------------------------------------

    def training_step(self, batch: dict[str, Tensor], batch_idx: int):
        loss, _, _ = self._step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    # --- validation ------------------------------------------------------

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int):
        loss, flat_logits, flat_tags = self._step(batch)
        preds = flat_logits.argmax(dim=-1)

        self.val_acc.update(preds, flat_tags)
        self.val_bal_acc.update(preds, flat_tags)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        self.log("val_acc", self.val_acc.compute(), prog_bar=True)
        self.log("val_balanced_acc", self.val_bal_acc.compute(), prog_bar=True)
        self.val_acc.reset()
        self.val_bal_acc.reset()

    # --- optimizer + scheduler -------------------------------------------

    def configure_optimizers(self):
        s = self.settings
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=s.lr,
            weight_decay=s.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            factor=s.plateau_lr_factor,
            patience=s.plateau_lr_patience,
            min_lr=s.min_lr,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # watched metric
                "interval": "epoch",
                "frequency": 1,
            },
        }


class TaggerDataModule(lightning.LightningDataModule):
    def __init__(
        self, dsets: dict[str, TensorSequenceDataset], bs_train: int = 32, bs_val: int = 64
    ):
        super().__init__()
        self.dsets = dsets
        self.bs_train = bs_train
        self.bs_val = bs_val

    def train_dataloader(self):
        return DataLoader(
            self.dsets["train"],
            batch_size=self.bs_train,
            collate_fn=collate_fn_pad,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dsets["val"],
            batch_size=self.bs_val,
            collate_fn=collate_fn_pad,
            num_workers=4,
            persistent_workers=True,
        )


def build_trainer(
    stop_patience: int,
    max_epochs: int,
    *,
    log_every_n_steps: int = 1,
    **trainer_kwargs,
) -> lightning.Trainer:
    """
    Build a lightning.Trainer with early stopping and checkpointing baked in.

    Extra keyword arguments are forwarded to lightning.Trainer, so you can still
    pass accelerator="gpu", devices=1, logger=..., etc.
    """
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=stop_patience,
            mode="min",
            verbose=True,
        ),
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="best-{epoch:04d}",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    return lightning.Trainer(
        max_epochs=max_epochs,
        log_every_n_steps=log_every_n_steps,
        callbacks=callbacks,
        **trainer_kwargs,
    )


# ---------------------------------------------------------------------------
# Base architecture class  (pure nn.Module, no Lightning here)
# ---------------------------------------------------------------------------


class TaggerModel(nn.Module):
    """
    Abstract base for sequence taggers.

    Subclasses must implement `forward(tokens, **kwargs) -> Tensor`
    (logits of shape [B, T, vocab_sz_tag]).

    `_fw_batch` can be overridden if a variant needs extra inputs beyond
    `tokens` and `tags_det` (e.g. character embeddings).
    """

    def __init__(self, vocab_sz_token: int, vocab_sz_tag: int) -> None:
        super().__init__()
        self.vocab_sz_token = vocab_sz_token
        self.vocab_sz_tag = vocab_sz_tag

    @property
    def tot_weights(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _fw_batch(self, tensors: dict[str, Tensor]) -> Tensor:
        """
        Default batch forward: passes tokens + tags_det to self.forward.
        """
        return self(tokens=tensors["tokens"], tags_det=tensors["tags_det"])


class RNNTagger(TaggerModel):
    """Bidirectional GRU sequence tagger."""

    def __init__(
        self,
        vocab_sz_token: int,
        vocab_sz_tag: int,
        emb_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__(vocab_sz_token, vocab_sz_tag)
        self.emb = nn.Embedding(vocab_sz_token, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(
            emb_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim * 2, vocab_sz_tag)

    def forward(self, tokens: Tensor, **kwargs) -> Tensor:
        x = self.emb(tokens)  # -> (B, T, E)
        x, _ = self.rnn(x)  #   -> (B, T, 2H)
        return self.head(x)  #  -> (B, T, C)


class TrfEncoderTagger(TaggerModel):
    """Encoder-only Transformer sequence tagger."""

    def __init__(
        self,
        vocab_sz_token: int,
        vocab_sz_tag: int,
        emb_dim: int = 32,
        n_heads: int = 2,
        n_layers: int = 2,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        max_len: int = 512,
    ) -> None:
        super().__init__(vocab_sz_token, vocab_sz_tag)
        self.emb = nn.Embedding(vocab_sz_token, emb_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(emb_dim, vocab_sz_tag)

    def forward(self, tokens: Tensor, **kwargs) -> Tensor:
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, -1)
        x = self.emb(tokens) + self.pos_emb(pos)  # [B, T, E]
        pad_mask = tokens == 0  # True where padded
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.head(x)  # [B, T, C]
