from collections.abc import Iterable, Sequence

import polars as pl
import torch
from torch import Tensor, optim
from torch.utils.data import DataLoader, Dataset

from . import text_process
from .vocab import VocabDuo


class SequenceDataset(Dataset):
    """Dataset of sequences. NOTE: in memory dataset. NOTE: Pads it all in the beginning"""

    def __init__(
        self,
        tokens: Sequence[list[str]],
        labels_det: Sequence[list[str]],
        labels_true: Sequence[list[str]],
        vocs: VocabDuo,
        device: str | torch.device | None = None,
        extra_feats: int = 0,
    ):
        if not len(tokens) == len(labels_det) == len(labels_true):
            raise ValueError("inconsistent lengths")

        # Encode each sequence -> (N, Maxlen)
        token_idx = [vocs.token.encode(seq) for seq in tokens]
        self.tokens = seqs2padded_tensor(token_idx, device=device, verbose=False)

        label_det_idx = [vocs.tag.encode(seq) for seq in labels_det]
        self.labels_det = seqs2padded_tensor(label_det_idx, device=device, verbose=False)

        label_true_idx = [vocs.tag.encode(seq) for seq in labels_true]
        self.labels_true = seqs2padded_tensor(label_true_idx, device=device, verbose=False)

        # Optionally add extra features
        if extra_feats > 0:
            self.extra = torch.stack(
                [make_extra_feats(ts, padto=self.tokens.shape[1]) for ts in tokens],
                dim=0,
            ).to(device)
            assert self.extra.shape[-1] == extra_feats

    def __str__(self) -> str:
        return f"SequenceDataset({len(self)} items, {self.tokens.device})"

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index: int):
        inputs = {
            "tokens": self.tokens[index],  # (maxlen,)
            "labels_det": self.labels_det[index],  # (maxlen,)
        }
        if hasattr(self, "extra"):
            inputs["extra"] = self.extra[index]

        return inputs, self.labels_true[index]

    def to_device(self, device: str | torch.device):
        self.tokens = self.tokens.to(device)
        self.labels_det = self.labels_det.to(device)
        self.labels_true = self.labels_true.to(device)

    def get_all(self):
        """Get all samples at once"""
        if hasattr(self, "extra"):
            raise NotImplementedError("get all not implemented with extra features")

        inputs = {
            "tokens": self.tokens,
            "labels_det": self.labels_det,
        }
        return inputs, self.labels_true

    @classmethod
    def from_dataframe(
        cls,
        df: pl.DataFrame,
        vocs: VocabDuo,
        device: str | torch.device | None = None,
    ):
        missing = {"tokens", "tags", "tags_det"}.difference(df.columns)
        assert not missing, f"got cols {df.columns} (missing {missing})"

        return cls(
            df["tokens"].to_list(),
            df["tags_det"].to_list(),
            df["tags"].to_list(),
            vocs,
            device,
        )


def add_tag_det_col(df: pl.DataFrame):
    return df.with_columns(
        tags_det=pl.col("tokens").map_elements(
            lambda tks: text_process.process("".join(tks))[1],
            pl.List(pl.String),
        )
    )


def df_to_tensorlists(
    df: pl.DataFrame,
    vocs: VocabDuo,
    device: torch.device | None = None,
) -> dict[str, list[Tensor]]:
    """Extract (tokens, tags, tags_det) from DF and encode as integer tensors."""

    missing = {"tokens", "tags", "tags_det"}.difference(df.columns)
    assert not missing, f"got cols {df.columns} (missing {missing})"

    def to_tens(x):
        return torch.tensor(x, dtype=torch.int64, device=device)

    return {
        "tokens": [to_tens(vocs.token.encode(s)) for s in df["tokens"]],
        "tags": [to_tens(vocs.tag.encode(s)) for s in df["tags"]],
        "tags_det": [to_tens(vocs.token.encode(s)) for s in df["tags_det"]],
    }


def seqs2padded_tensor(
    sequences: Iterable[list[int]],
    pad_value: int = 0,
    verbose: bool = True,
    device: torch.device | str | None = None,
):
    """DEPRECATED? Convert lists to tensors and pad.
    Returns
    -------
    padded: Tensor
        of shape (BS, Maxlen)
    """
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
    tag_weights = torch.tensor([1 / tag_counts.get(k, torch.inf) + smoothing for k in tag_vocab])
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


def data2torch(
    df: pl.DataFrame,
    bs: int,
    vocs: VocabDuo,
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
        df["tokens"].to_list(),
        df["tags_det"].to_list(),
        df["tags"].to_list(),
        vocs,
        device=device,
        extra_feats=extra_feats,
    )

    dl = DataLoader(
        dset,
        batch_size=bs,
        shuffle=True,
    )
    return dl


def make_extra_feats(tokens: list[str], padto: int = 0, t_len_max: int = 24):
    """Prepare extra features for tagger

    Returns: features (len, Nextra)
    """
    assert isinstance(tokens[0], str), "should be strings"

    n_cases = len(text_process.WordCase)
    N = max(len(tokens), padto)
    # padding
    wordcase_oh = torch.zeros((N, n_cases), dtype=torch.float32)
    for i, t in enumerate(tokens):
        wc = text_process.get_word_case(t)
        wordcase_oh[i, wc.value] = 1

    # normalized token length
    t_lens = torch.zeros((N, 1), dtype=torch.float32)
    t_lens[: len(tokens), 0] = torch.clamp(
        torch.tensor([len(t) for t in tokens], dtype=torch.float32) / t_len_max, min=0, max=1
    )

    return torch.cat((wordcase_oh, t_lens), dim=1)


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
