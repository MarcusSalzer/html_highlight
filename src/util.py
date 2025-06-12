# For mapping to a smaller label space
import json
from glob import glob
from typing import Iterator, Protocol, TypeVar

import polars as pl
import numpy as np

MAP_TAGS = {
    "opun": "op",
    "opcm": "op",
    "opbi": "op",
    "opas": "op",
    "fnme": "fn",
    "fnst": "fn",
    "fnas": "fn",
    "fnfr": "fn",
    "kwfl": "kw",
    "kwop": "kw",
    "kwim": "kw",
    "kwva": "kw",
    "kwfn": "kw",
    "kwmo": "kw",
    "kwio": "kw",
    "kwde": "kw",
    "at": "va",
    "cofl": "co",
    "coil": "co",
    "coml": "co",
}

# allow these in vocab
VOCAB_TAGS = [
    "kwfl",
    "kwty",
    "kwop",
    "kwmo",
    "kwva",
    "kwde",
    "kwfn",
    "kwim",
    "kwio",
    "id",
    "ws",
    "nl",
    "brop",
    "brcl",
    "sy",
    "pu",
    "bo",
    "li",
    "opcm",
    "opbi",
    "opun",
    "opas",
    "an",
    "uk",
]

T = TypeVar("T", covariant=True)


class ArrayLike(Protocol[T]):
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[T]: ...
    def __getitem__(self, index: int) -> T: ...


def load_split_idx(split_idx_id: str):
    name = f"split_index_{split_idx_id}.json"
    fps = glob(f"../**/data/**/{name}", recursive=True)
    if len(fps) > 1:
        raise ValueError(f"Found {len(fps)} matches")
    if not fps:
        raise ValueError(f"Couldn't find {name}")
    with open(fps[0], "r") as f:
        split_index = json.load(f)
    return split_index


def load_examples_json(
    path: str | None = None,
    tag_map: dict | None = None,
    filter_lang: list[str] | None = None,
    split_idx_id: str | None = None,
    verbose=True,
):
    """Load all annotated examples
    ## parameters
    - path: optionally specify a file other than the default dataset.
    - tag_map (dict|None): optionally map tags.

    ## returns
    - examples (Dataframe): tokens, tags, lang, length."""

    if path is None:
        path = glob("../**/data/examples_annot.json", recursive=True)[0]

    with open(path, encoding="utf-8") as f:
        d = json.load(f)

    if split_idx_id is not None:
        split_index = load_split_idx(split_idx_id)

    rows = []
    for k, ex in d.items():
        splits = k.split("_")
        ex["name"] = "_".join(splits[:-1])
        ex["lang"] = splits[-1]
        ex["id"] = k
        if tag_map:
            ex["tags"] = [tag_map.get(t, t) for t in ex["tags"]]
        if split_idx_id:
            ex["split"] = split_index.get(k)
            # skip if missing from index
            if ex["split"] is None:
                continue

        rows.append(ex)

    data = pl.DataFrame(rows).with_columns(length=pl.col("tokens").list.len())

    if filter_lang is not None:
        data = data.filter(pl.col("lang").is_in(filter_lang))
    if verbose:
        print(f"Loaded {len(data)} examples")

    if split_idx_id:
        # separate dataframe per split
        data_splits = {
            g[0]: df for g, df in data.group_by("split", maintain_order=True)
        }
        if verbose:
            for k, df in data_splits.items():
                print(f"    {k}: {len(df)}")
        return data_splits
    # check for duplicates
    duplicates = (
        data.group_by("tokens")
        .agg("id", pl.len())
        .filter(pl.col("len") > 1)["id"]
        .explode()
        .to_list()
    )

    if len(duplicates) > 0:
        raise ValueError(
            f"found {len(duplicates)} duplicate examples: {', '.join(duplicates)}"
        )
    elif verbose:
        print("No duplicates found :)")

    return data


def save_examples_json(data: pl.DataFrame, path: str):
    """Format dataframe and save as JSON"""
    new_data_dict = {
        f"{d['name']}_{d['lang']}": {
            "difficulty": d["difficulty"],
            "tokens": d["tokens"],
            "tags": d["tags"],
        }
        for d in data.rows(named=True)
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(new_data_dict, f)


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
