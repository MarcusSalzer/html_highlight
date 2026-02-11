from collections.abc import Iterable
from typing import NamedTuple

import polars as pl

from src.constants import VOCAB_TAGS


def vocab_candidates(
    examples: pl.DataFrame,
    vocab_allowed_tags: tuple[str, ...] | None = VOCAB_TAGS,
):
    """Heuristics for which tokens to include."""
    assert {"tokens", "tags"}.issubset(examples.columns)
    vocab_cands = examples.select(pl.col("tokens", "tags").explode())

    if vocab_allowed_tags is not None:
        vocab_cands = vocab_cands.filter(pl.col("tags").is_in(vocab_allowed_tags))

    include = (
        vocab_cands.group_by("tokens")
        .agg(pl.len())
        .sort("len", "tokens", descending=True)["tokens"]
        .to_list()
    )
    return include


class CodeVocab:
    """A vocab for encoding/decoding.

    Will contain, in order

    - 1 padding token (at index 0)
    - n_unknown "unknown" tokens
    - word tokens
    """

    def __init__(self, vocab_list: list[str]) -> None:

        self.vocab_list = vocab_list

        self.n_unknown = sum(t.startswith("<unk") for t in self.vocab_list)
        assert self.n_unknown >= 0, "expects 0+ unknown tokens"

        self.token2idx = {t: i for i, t in enumerate(self.vocab_list)}

    @classmethod
    def from_words(cls, word_tokens: set[str], n_unknown: int = 1):
        # build unknown-tokens
        unk_tokens = [f"<unk{k}>" for k in range(n_unknown)]
        return cls(["<pad>"] + unk_tokens + sorted(word_tokens))

    def __str__(self) -> str:
        return f"Vocab(n={len(self.vocab_list)}, n_unk={self.n_unknown})"

    def __len__(self) -> int:
        return len(self.vocab_list)

    def __iter__(self):
        return iter(self.vocab_list)

    def encode(self, tokens: Iterable[str]) -> list[int]:
        """Encode words as integers."""
        # keep track of unique unknown tokens, in order of appearance
        found_unk: dict[str, int] = {}
        idxs = []
        for t in tokens:
            if t in self.token2idx:
                idxs.append(self.token2idx[t])
            else:
                assert self.n_unknown > 0, "No unknown tokens!"

                if t not in found_unk:
                    # wrap around when running out of unknowns
                    # count +1 for padding token
                    found_unk[t] = (len(found_unk) % self.n_unknown) + 1
                idxs.append(found_unk[t])
        return idxs

    def decode(self, idxs: Iterable[int]) -> list[str]:
        """Reconstruct tokens."""
        return [self.vocab_list[i] for i in idxs]


class VocabDuo(NamedTuple):
    token: CodeVocab
    tag: CodeVocab


def both_vocabs(
    examples: pl.DataFrame,
    n_unknown_token: int,
    n_unknown_tag: int = 1,
    vocab_allowed_tags: tuple[str, ...] | None = VOCAB_TAGS,
) -> VocabDuo:
    """Get vocabs for tags and tokens"""

    # pick out tokens
    tokens = set(vocab_candidates(examples, vocab_allowed_tags))

    # consider all unique tags
    tags = set(examples["tags"].explode().unique())

    return VocabDuo(
        CodeVocab.from_words(tokens, n_unknown_token), CodeVocab.from_words(tags, n_unknown_tag)
    )
