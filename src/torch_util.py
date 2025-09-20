from dataclasses import dataclass
import os
from pathlib import Path
from timeit import default_timer
from typing import Callable, cast

import polars as pl
import torch
from colorama import Fore, Style
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from src import text_process, types


class LSTMTagger(nn.Module):
    def __init__(
        self,
        token_vocab_size: int,
        label_vocab_size: int,
        embedding_dim: int = 12,
        hidden_dim: int = 128,
        n_lstm_layers: int = 2,
        dropout_lstm: float = 0.3,
        bidi: bool = True,
        n_extra: int = 0,
    ):
        super(LSTMTagger, self).__init__()
        self.embedding_tokens = nn.Embedding(
            token_vocab_size, embedding_dim, padding_idx=0
        )
        self.embedding_labels = nn.Embedding(
            label_vocab_size, embedding_dim, padding_idx=0
        )

        # LSTM will recieve embedded tokens, tags and possibly extra_feats
        lstm_in_dim = (3 if n_extra > 0 else 2) * embedding_dim

        self.lstm = nn.LSTM(
            lstm_in_dim,
            hidden_dim,
            n_lstm_layers,
            batch_first=True,
            dropout=dropout_lstm,
            bidirectional=bidi,
        )
        actual_hidden = hidden_dim * (2 if bidi else 1)
        self.n_extra = n_extra
        if n_extra > 0:
            # project extra features to same dim as tokens and labels
            self.feature_proj = nn.Linear(n_extra, embedding_dim)

        # double size if bidirectional
        self.hidden2tag = nn.Linear(actual_hidden, label_vocab_size)

    def forward(
        self,
        tokens: torch.Tensor,
        labels_det: torch.Tensor,
        extra: torch.Tensor | None = None,
    ):
        embeds_tokens = self.embedding_tokens(tokens)
        embeds_labels = self.embedding_labels(labels_det)

        embeds = torch.cat([embeds_tokens, embeds_labels], dim=-1)
        if hasattr(self, "feature_proj"):
            assert extra is not None, "needs extra features"
            feats_emb = self.feature_proj(extra)
            embeds = torch.cat([embeds, feats_emb], dim=-1)
        #  (bs, seq_len, (2 or 3) * emb_dim)
        lstm_out, _ = self.lstm(embeds)
        #  (bs, seq_len, actual_hidden)
        logits = self.hidden2tag(lstm_out)
        # (bs, seq_len, tagset_size)
        return logits


class SequenceDataset(Dataset):
    """Dataset of sequences."""

    def __init__(
        self,
        tokens: types.ArrayLike[list[str]],
        labels_det: types.ArrayLike[list[str]],
        labels_true: types.ArrayLike[list[str]],
        token2idx: dict[str, int],
        label2idx: dict[str, int],
        device: str | torch.device | None = None,
        extra_feats: int = 0,
    ):
        if not len(tokens) == len(labels_det) == len(labels_true):
            raise ValueError("inconsistent lengths")

        token_idx = [[token2idx.get(t, 1) for t in seq] for seq in tokens]
        self.tokens = seqs2padded_tensor(token_idx, device=device, verbose=False)

        label_det_idx = [[label2idx.get(t, 1) for t in seq] for seq in labels_det]
        self.labels_det = seqs2padded_tensor(
            label_det_idx, device=device, verbose=False
        )

        label_true_idx = [[label2idx.get(t, 1) for t in seq] for seq in labels_true]
        self.labels_true = seqs2padded_tensor(
            label_true_idx, device=device, verbose=False
        )
        if extra_feats > 0:
            self.extra = torch.stack(
                [make_extra_feats(ts, padto=self.tokens.shape[1]) for ts in tokens],
                dim=0,
            ).to(device)
            assert self.extra.shape[-1] == extra_feats

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        inputs = {
            "tokens": self.tokens[idx],
            "labels_det": self.labels_det[idx],
        }
        if hasattr(self, "extra"):
            inputs["extra"] = self.extra[idx]

        return inputs, self.labels_true[idx]


def seqs2padded_tensor(
    sequences: list[list[int]],
    pad_value=0,
    verbose=True,
    device: torch.device | str | None = None,
):
    t = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(s) for s in sequences],
        batch_first=True,
        padding_value=pad_value,
    ).to(device)
    if verbose:
        print("padded tensor:", tuple(t.size()), t.device)
    return t


def class_weights(tag_counts: dict, tag_vocab: list[str], smoothing=1.0):
    """Compute class weights for unbalanced data"""
    tag_weights = torch.tensor(
        [1 / tag_counts.get(k, torch.inf) + smoothing for k in tag_vocab]
    )
    tag_weights /= sum(tag_weights)
    return tag_weights


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn,
    optimizer: optim.Optimizer | None = None,
):
    """General training/validation epoch"""
    if optimizer is not None:
        model.train()
    else:
        model.eval()

    loss_agg = 0
    n_elements = 0  # to normalize loss by sequence length
    for inputs, labels in loader:
        # Forward pass
        logits: torch.Tensor = model(**inputs)
        bs = logits.shape[0]
        # Reshape for loss calculation
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss_agg += loss.item() * bs
        n_elements += bs
        if optimizer is not None:
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return loss_agg / n_elements


@dataclass
class Trainer:
    model: torch.nn.Module
    train_dl: DataLoader
    val_dl: DataLoader
    optimizer: optim.Optimizer
    loss_function: Callable
    lr_s: optim.lr_scheduler.LRScheduler | None = None
    name: str = ""
    save_dir: Path | None = None
    save_wait: int = 5
    printerval: int | None = 1
    time_limit: int | None = None
    reduce_lr_on_plat: types.LrsPlatConfig | None = None
    stop_patience: int | None = None
    epoch_callback: Callable | None = None

    def train_loop(self, max_epochs: int = 500):
        """Train a tagger model

        ## returns
        - metrics: dict with keys "train_loss", "val_loss", "val_acc"
        """
        if self.save_dir is not None and not self.save_dir.exists():
            self.save_dir.mkdir(parents=True)

        if self.reduce_lr_on_plat:
            self.lrs_plat = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, **self.reduce_lr_on_plat.model_dump()
            )
        else:
            self.lrs_plat = None

        losses_train = []
        losses_val = []
        val_accs = []

        best_loss = float("inf")
        best_epoch = 0
        best_acc = 0
        tstart = default_timer()

        for epoch in range(max_epochs):
            if self.stop_patience is not None:
                if epoch > best_epoch + self.stop_patience:
                    print(f"[EARLY STOPPING at {epoch = }]")
                    break
            # TRAINING
            train_loss = run_epoch(
                self.model, self.train_dl, self.loss_function, self.optimizer
            )
            losses_train.append(train_loss)

            # VALIDATION
            with torch.no_grad():
                val_loss = run_epoch(self.model, self.val_dl, self.loss_function)
                losses_val.append(val_loss)
                val_acc_now = val_acc(
                    self.model, cast(SequenceDataset, self.val_dl.dataset)
                )
                val_accs.append(val_acc_now)

            if self.lr_s is not None:
                self.lr_s.step()
            if self.lrs_plat is not None:
                self.lrs_plat.step(val_loss)

            m_extra = " "
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                if self.save_dir is not None and epoch > self.save_wait:
                    fp = self.save_dir / f"{self.name}_state.pth"
                    torch.save(self.model.state_dict(), fp)
                    m_extra += f"{Fore.CYAN} Saved in {self.save_dir} (best VL) {Style.RESET_ALL}"
            if val_acc_now > best_acc:
                best_acc = val_acc_now
                if self.save_dir is not None and epoch > self.save_wait:
                    fp = os.path.join(self.save_dir, f"{self.name}_acc_state.pth")
                    torch.save(self.model.state_dict(), fp)
                    m_extra += f"{Fore.CYAN} Saved in {self.save_dir} (best Acc) {Style.RESET_ALL}"

            self.epoch_print(epoch, train_loss, val_loss, val_accs[-1], m_extra)

            if self.time_limit is not None:
                if default_timer() - tstart > self.time_limit:
                    break

            if self.epoch_callback is not None:
                self.epoch_callback({"val_acc": val_accs[-1], "epoch": epoch})

        return {
            "train_loss": losses_train,
            "val_loss": losses_val,
            "val_acc": val_accs,
        }

    def epoch_print(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_acc: float,
        m_extra: str,
    ):
        if self.printerval is not None and (epoch) % self.printerval == 0:
            msg = f"{epoch + 1:4d}, {Style.DIM} train_loss: {train_loss:.6f},{Style.RESET_ALL} _loss: {val_loss:.6f}, val_acc: {val_acc:.2%}"

            if self.lr_s is not None:
                msg += f"{Style.DIM} LR: {self.lr_s.get_last_lr()[0]:.6f} {Style.RESET_ALL}"
            if self.lrs_plat is not None:
                msg += f"{Style.DIM} LR: {self.lrs_plat.get_last_lr()[0]:.6f} {Style.RESET_ALL}"
            print(msg + m_extra)
            if epoch % (10 * self.printerval) == 0 and epoch > 0:
                print()


def data2torch(
    df: pl.DataFrame,
    bs: int,
    token2idx: dict[str, int],
    tag2idx: dict[str, int],
    device: str | torch.device = "cpu",
    extra_feats: int = 0,
):
    """Dataframe -> Dataloader"""

    df = df.with_columns(
        tags_det=pl.col("tokens").map_elements(
            lambda tks: text_process.process("".join(tks))[1],
            pl.List(pl.String),
        )
    )
    dset = SequenceDataset(
        cast(types.ArrayLike[list[str]], df["tokens"].to_list()),
        cast(types.ArrayLike[list[str]], df["tags_det"].to_list()),
        cast(types.ArrayLike[list[str]], df["tags"].to_list()),
        token2idx,
        tag2idx,
        device=device,
        extra_feats=extra_feats,
    )

    dl = DataLoader(
        dset,
        batch_size=bs,
        shuffle=True,
    )
    return dl


def make_extra_feats(tokens: list[str], padto: int = 0):
    """Prepare extra features for tagger

    Returns: features (len, Nextra)
    """
    assert isinstance(tokens[0], str), "should be strings"

    # paddding
    features = torch.zeros((max(len(tokens), padto), 3), dtype=torch.float32)
    laststart = tokens[0]
    for i, token in enumerate(tokens):
        is_capitalized = 1.0 if token[0].isupper() else -1.0
        word_length = min(len(token), 10) / 16  # normalized token length
        if i > 0 and tokens[i - 1] == "\n":
            laststart = token
        line_starts_with = (hash(laststart) % 10) / 10  # Bucket encoding

        features[i, :] = torch.tensor([is_capitalized, word_length, line_starts_with])
    return features


def val_acc(model, dset: SequenceDataset):
    """Compute mean accuracy for whole dataset"""
    # logits: (n_ex, max_len, n_class)
    if hasattr(dset, "extra"):
        logits = model(dset.tokens, dset.labels_det, dset.extra)
    else:
        logits = model(dset.tokens, dset.labels_det)

    # prediction: (n_ex, max_len)
    preds = logits.argmax(dim=-1)

    correct = preds == dset.labels_true
    return (correct.sum() / correct.numel()).item()


def get_dev():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
