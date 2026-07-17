from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


@dataclass
class EncodedExample:
    """After encoding strings using the vocab we have this"""

    tokens: Tensor  # (L,) int tensor
    tags: Tensor  # (L,) int tensor
    tags_det: Tensor  # (L,) int tensor
    extra_feats: Tensor | None  # optional (L, nf) float32 tensor


class TensorSequenceDataset(Dataset):
    """Dataset that keeps list of individual tensors in memory"""

    def __init__(
        self,
        tokens: list[Tensor],
        tags: list[Tensor],
        tags_det: list[Tensor],
    ) -> None:

        # validate
        N = len(tokens)
        assert len(tags) == N
        assert len(tags_det) == N

        for seq in (tokens, tags, tags_det):
            assert all(t.ndim == 1 for t in seq), "expects 1d tensors"
            # supported by Embedding layer
            assert all(t.dtype == torch.int64 for t in seq), "expects int64 dtype"

        # data
        self.tokens = tokens
        self.tags = tags
        self.tags_det = tags_det

    def __len__(self):
        return len(self.tokens)

    def __str__(self) -> str:
        return f"TensorSequenceDataset({len(self)} samples : {self.total_tokens()} tokens)"

    def total_tokens(self):
        return sum(len(s) for s in self.tokens)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {
            "tokens": self.tokens[idx],
            "tags": self.tags[idx],
            "tags_det": self.tags_det[idx],
        }


def collate_fn_pad(batch):
    # Separate the different keys
    tokens = [item["tokens"] for item in batch]
    tags = [item["tags"] for item in batch]
    tags_det = [item["tags_det"] for item in batch]

    # Pad sequences
    padded_tokens = pad_sequence(tokens, batch_first=True)
    padded_tags = pad_sequence(tags, batch_first=True)
    padded_tags_det = pad_sequence(tags_det, batch_first=True)

    return {"tokens": padded_tokens, "tags": padded_tags, "tags_det": padded_tags_det}


def get_dl(dset: TensorSequenceDataset, bs: int = 16, shuffle: bool = True, n_workers: int = 0):
    """Get a dataloader"""
    return DataLoader(
        dataset=dset,
        batch_size=bs,
        shuffle=shuffle,
        collate_fn=collate_fn_pad,
        num_workers=n_workers,
        persistent_workers=n_workers > 0,
    )
