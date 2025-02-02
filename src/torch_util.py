import os
from timeit import default_timer
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import polars as pl
from src import text_process
from colorama import Fore, Style


class LSTMTagger(nn.Module):
    def __init__(
        self,
        token_vocab_size,
        label_vocab_size,
        embedding_dim=12,
        hidden_dim=128,
        n_lstm_layers=2,
        dropout_lstm=0.3,
        bidi=True,
    ):
        super(LSTMTagger, self).__init__()
        self.embedding_tokens = nn.Embedding(
            token_vocab_size, embedding_dim, padding_idx=0
        )
        self.embedding_labels = nn.Embedding(
            label_vocab_size, embedding_dim, padding_idx=0
        )
        self.lstm = nn.LSTM(
            2 * embedding_dim,
            hidden_dim,
            n_lstm_layers,
            batch_first=True,
            dropout=dropout_lstm,
            bidirectional=bidi,
        )
        # double size if bidirectional
        self.hidden2tag = nn.Linear(hidden_dim * (2 if bidi else 1), label_vocab_size)

    def forward(self, tokens, labels):
        embeds_tokens = self.embedding_tokens(tokens)
        embeds_labels = self.embedding_labels(labels)

        embeds = torch.cat([embeds_tokens, embeds_labels], dim=-1)
        #  (batch_size, seq_len, embedding_dim)
        lstm_out, _ = self.lstm(embeds)
        #  (batch_size, seq_len, hidden_dim)
        logits = self.hidden2tag(lstm_out)
        # (batch_size, seq_len, tagset_size)
        return logits


class SequenceDataset(Dataset):
    """Dataset of sequences."""

    def __init__(
        self,
        tokens: list[list[str]],
        labels_det: list[list[str]],
        labels_true: list[list[str]],
        token2idx: dict[str, int],
        label2idx: dict[str, int],
        device: str | None = None,
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

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx], self.labels_det[idx], self.labels_true[idx]


def seqs2padded_tensor(
    sequences: list[list[int | float]],
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
    for tokens, labels_det, labels_true in loader:
        # Forward pass
        logits: torch.Tensor = model(tokens, labels_det)
        # Reshape for loss calculation
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels_true.view(-1))
        loss_agg += loss.item()
        n_elements += labels_true.numel()
        if optimizer is not None:
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return loss_agg / n_elements


def train_loop(
    model: torch.nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    epochs=100,
    optimizer: optim.Optimizer | None = None,
    loss_function=None,
    lr_s: optim.lr_scheduler.LRScheduler | None = None,
    name="",
    save_dir=None,
    print_interval=5,
    time_limit: int | None = None,
):
    fp = os.path.join(save_dir, f"{name}_state.pth")
    if save_dir is not None and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if optimizer is None:
        optimizer = optim.Adam(model.parameters())
    if loss_function is None:
        loss_function = nn.CrossEntropyLoss()

    losses_train = []
    losses_val = []
    val_accs = []

    prev_best = 0.0
    best_loss = float("inf")
    tstart = default_timer()

    for epoch in range(epochs):
        # TRAINING
        train_loss = run_epoch(model, train_dl, loss_function, optimizer)
        losses_train.append(train_loss)

        # VALIDATION
        with torch.no_grad():
            val_loss = run_epoch(model, val_dl, loss_function)
            losses_val.append(val_loss)
            val_accs.append(val_acc(model, val_dl.dataset))

        if lr_s is not None:
            lr_s.step()

        if val_loss < best_loss:
            best_loss = val_loss
            if save_dir is not None:
                torch.save(model.state_dict(), fp)

        if (epoch) % print_interval == 0:
            msg = f"{epoch:4d}/{epochs},{Style.DIM} train: {train_loss:.6f},{Style.RESET_ALL} val: {val_loss:.6f},"

            impr = prev_best - best_loss
            prev_best = best_loss
            msg += (
                Fore.GREEN + f" min-loss: {best_loss:.6f}, " + Style.RESET_ALL
                if impr > 0
                else " " * 21
            )

            msg += f"acc: {val_accs[-1] * 100:.2f}%"
            if lr_s is not None:
                msg += f"{Style.dim} LR: {lr_s.get_last_lr()[0]:.6f} {Style.RESET_ALL}"
            print(msg)

            if epoch % (10 * print_interval) == 0 and epoch > 0:
                print()
        if time_limit is not None:
            if default_timer() - tstart > time_limit:
                break
    return losses_train, losses_val, val_accs


def data2torch(
    df: pl.DataFrame,
    bs: int,
    token2idx: dict[str, int],
    tag2idx: dict[str, int],
    device="cpu",
):
    """Dataframe -> Dataloader"""

    df = df.with_columns(
        tags_det=pl.col("tokens").map_elements(
            lambda tks: text_process.process("".join(tks))[1], pl.List(pl.String)
        )
    )
    dset = SequenceDataset(
        df["tokens"],
        df["tags_det"],
        df["tags"],
        token2idx,
        tag2idx,
        device=device,
    )

    dl = DataLoader(
        dset,
        batch_size=bs,
        shuffle=True,
    )
    return dl


def val_acc(model, dset: SequenceDataset):
    """Compute mean accuracy for whole dataset"""
    # logits: (n_ex, max_len, n_class)
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
