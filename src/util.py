# For mapping to a smaller label space
import json
from glob import glob
from pathlib import Path
from typing import Iterable, Literal

import polars as pl
import numpy as np

from src.DatasetRecord import DatasetRecord
from src._constants import VOCAB_TAGS


def load_split_idx(filename: str = "split_index.json"):
    """Find and load the file."""

    fps = glob(f"../**/data/**/{filename}", recursive=True)
    if len(fps) > 1:
        raise ValueError(f"Found {len(fps)} matches")
    if not fps:
        raise ValueError(f"Couldn't find {filename}")
    with open(fps[0], "r") as f:
        split_index = json.load(f)

    assert isinstance(split_index, dict)
    return split_index["examples"], split_index["date"]


def load_dataset_parallel(
    path=Path("data/dataset.ndjson"),
    filter_lang: list[str] | None = None,
) -> list[DatasetRecord]:
    """Load the annoted data (Newline delimited JSON)"""
    with path.open("r", encoding="utf-8") as f:
        dataset = [
            d
            for d in (DatasetRecord(**json.loads(line)) for line in f)
            if (filter_lang is None or d.lang in filter_lang)
        ]

    return dataset


# def load_dataset_zip(
#     path=Path("data/dataset.ndjson"),
#     filter_lang: list[str] | None = None,
# ) -> list[DatasetRecord]:
#     """Load the annoted data (Newline delimited JSON)"""

#     with path.open("r", encoding="utf-8") as f:
#         dataset = []
#         for line in f:
#             record = json.loads(line)
#             if filter_lang is None or record["lang"] in filter_lang:
#                 tokens, tags = zip(*record["sequence"])
#                 d = DatasetRecord(
#                     record["name"],
#                     record["lang"],
#                     list(tokens),
#                     list(tags),
#                     record["difficulty"],
#                 )
#                 dataset.append(d)
#     return dataset


def load_dataset_splits(
    split_idx: dict[str, str],
    path=Path("data/dataset.ndjson"),
) -> dict[str, list[DatasetRecord]]:
    """Load the annoted data (Newline delimited JSON)"""
    splits: dict[str, list[DatasetRecord]] = {}
    n_skip = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            d = DatasetRecord(**json.loads(line))
            # where should this example go?
            sk = split_idx.get(d.id)
            if sk is None:
                n_skip += 1
            else:
                splits.setdefault(sk, []).append(d)

    if n_skip > 0:
        print(f"[NOTE] skipped {n_skip} examples")

    return splits


def dataset_to_df(data: Iterable[DatasetRecord]):
    df = pl.DataFrame([d.toDict(with_id=True) for d in data])
    return df


def split_to_chars(tokens: list[str], tags: list[str], only_starts=False):
    chars: list[str] = []
    char_tags = []
    for token, tag in zip(tokens, tags):
        chars.extend(token)
        if only_starts:
            char_tags.extend(["start"] + ["-"] * (len(token) - 1))
        else:
            char_tags.extend(["start-" + tag] + [tag] * (len(token) - 1))

    return chars, char_tags


def make_vocab(
    examples: pl.DataFrame,
    insert=["<pad>", "<unk>"],
    vocab_allowed_tags: list[str] | None = VOCAB_TAGS,
) -> tuple[list, dict[str, int], list, dict[str, int]]:
    """Make vocab, and inverse map"""
    vocab_cands = examples.select(pl.col("tokens", "tags").explode())
    if vocab_allowed_tags is not None:
        vocab_cands = vocab_cands.filter(pl.col("tags").is_in(vocab_allowed_tags))

    token_cands = (
        vocab_cands.group_by("tokens")
        .agg(pl.len().alias("count"))
        .sort("count", "tokens", descending=True)
    )
    tag_cands = (
        examples.select("tags")
        .explode("tags")
        .group_by("tags")
        .agg(pl.len().alias("count"))
        .sort("count", "tags", descending=True)
    )

    # token vocab
    vocab = insert + token_cands["tokens"].to_list()
    token2idx = {t: i for i, t in enumerate(vocab)}

    # tag vocab
    tag_vocab = insert + tag_cands["tags"].to_list()
    tag2idx = {t: i for i, t in enumerate(tag_vocab)}
    return vocab, token2idx, tag_vocab, tag2idx


def MAPE(y_true, y_pred, symmetric=False):
    """Mean absolute percentage error"""
    if not (isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray)):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
    if symmetric:
        return np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))
    else:
        return np.mean(np.abs((y_true - y_pred) / y_true))


def value_counts(
    series: pl.Series | list,
    verbose=False,
    sort_by: None | Literal["count", "value"] = "count",
) -> pl.DataFrame:
    """Count occurences of each unique value in a pl.Series or list

    ## returns
    - vc dataframe
    """
    if isinstance(series, list):
        series = pl.Series("value", series)
    cc_name = series.name + "_count"
    vc = series.value_counts(name=cc_name)
    if sort_by == "count":
        vc = vc.sort(cc_name, series.name, descending=True)
    elif sort_by == "value":
        vc = vc.sort(series.name, cc_name)

    if verbose:
        print(
            f"{len(vc)} unique ({series.name}): ",
            ", ".join([repr(k) for k in vc[series.name].head(5)]),
            ",...",
        )

    return vc


def value_counts_dict(
    series: pl.Series | list,
    verbose=False,
    sort_by: None | Literal["count", "value"] = "count",
) -> dict:
    """Count occurences of each unique value in a pl.Series or list

    ## returns
    - a dict of `value : count` pairs, sorted descending
    """

    return {r[0]: r[1] for r in value_counts(series, verbose, sort_by).rows()}
