import os
from timeit import default_timer
from typing import cast

import polars as pl
import torch
from colorama import Fore, Style
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from src import text_process
from src.util import ArrayLike


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
        tokens: ArrayLike[list[str]],
        labels_det: ArrayLike[list[str]],
        labels_true: ArrayLike[list[str]],
        token2idx: dict[str, int],
        label2idx: dict[str, int],
        device: str | None = None,
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
    device: str | None = None,
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


def train_loop(
    model: torch.nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader[SequenceDataset],
    epochs=100,
    optimizer: optim.Optimizer | None = None,
    loss_function=None,
    lr_s: optim.lr_scheduler.LRScheduler | None = None,
    name="",
    save_dir=None,
    save_wait: int = 5,
    printerval=1,
    time_limit: int | None = None,
    reduce_lr_on_plat: dict | None = None,
):
    """Train a tagger model

    ## returns
    - metrics: dict with keys "train_loss", "val_loss", "val_acc"
    """
    if save_dir is not None and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if optimizer is None:
        optimizer = optim.Adam(model.parameters())
    if loss_function is None:
        loss_function = nn.CrossEntropyLoss()

    if reduce_lr_on_plat:
        lrs_plat = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **reduce_lr_on_plat
        )
    else:
        lrs_plat = None

    losses_train = []
    losses_val = []
    val_accs = []

    prev_best_vl = 0.0
    best_loss = float("inf")
    prev_best_acc = 0
    best_acc = 0
    tstart = default_timer()

    for epoch in range(epochs):
        # TRAINING
        train_loss = run_epoch(model, train_dl, loss_function, optimizer)
        losses_train.append(train_loss)

        # VALIDATION
        with torch.no_grad():
            val_loss = run_epoch(model, val_dl, loss_function)
            losses_val.append(val_loss)
            val_acc_now = val_acc(model, cast(SequenceDataset, val_dl.dataset))
            val_accs.append(val_acc_now)

        if lr_s is not None:
            lr_s.step()
        if lrs_plat is not None:
            lrs_plat.step(val_loss)

        m_extra = " "
        if val_loss < best_loss:
            best_loss = val_loss
            if save_dir is not None and epoch > save_wait:
                fp = os.path.join(save_dir, f"{name}_state.pth")
                torch.save(model.state_dict(), fp)
                m_extra += (
                    f"{Fore.CYAN} Saved in {save_dir} (best VL) {Style.RESET_ALL}"
                )
        if val_acc_now > best_acc:
            best_acc = val_acc_now
            if save_dir is not None and epoch > save_wait:
                fp = os.path.join(save_dir, f"{name}_acc_state.pth")
                torch.save(model.state_dict(), fp)
                m_extra += (
                    f"{Fore.CYAN} Saved in {save_dir} (best Acc) {Style.RESET_ALL}"
                )

        if (epoch) % printerval == 0:
            msg = f"{epoch + 1:4d}/{epochs},{Style.DIM} train: {train_loss:.6f},{Style.RESET_ALL} val: {val_loss:.6f},"

            impr_vl, prev_best_vl = prev_best_vl - best_loss, best_loss
            msg += (
                Fore.GREEN + f" min-loss: {best_loss:.6f}, " + Style.RESET_ALL
                if impr_vl > 0
                else " " * 21
            )

            impr_acc, prev_best_acc = best_acc - prev_best_acc, best_acc

            msg += (
                Fore.GREEN * (impr_acc > 0)
                + f"acc: {val_accs[-1] * 100:.2f}%"
                + Style.RESET_ALL * (impr_acc > 0)
            )

            if lr_s is not None:
                msg += f"{Style.DIM} LR: {lr_s.get_last_lr()[0]:.6f} {Style.RESET_ALL}"
            if lrs_plat is not None:
                msg += (
                    f"{Style.DIM} LR: {lrs_plat.get_last_lr()[0]:.6f} {Style.RESET_ALL}"
                )
            print(msg + m_extra)

            if epoch % (10 * printerval) == 0 and epoch > 0:
                print()
        if time_limit is not None:
            if default_timer() - tstart > time_limit:
                break
    return {
        "train_loss": losses_train,
        "val_loss": losses_val,
        "val_acc": val_accs,
    }


def data2torch(
    df: pl.DataFrame,
    bs: int,
    token2idx: dict[str, int],
    tag2idx: dict[str, int],
    device="cpu",
    extra_feats: int = 0,
):
    """Dataframe -> Dataloader"""

    df = df.with_columns(
        tags_det=pl.col("tokens").map_elements(
            lambda tks: text_process.process("".join(tks))[1], pl.List(pl.String)
        )
    )
    dset = SequenceDataset(
        df["tokens"].to_list(),
        df["tags_det"].to_list(),
        df["tags"].to_list(),
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
        return "cuda"
    else:
        return "cpu"
