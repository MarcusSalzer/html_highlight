# For mapping to a smaller label space
import json
from collections.abc import Iterable
from glob import glob
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from src.datamodels.dataset_record import DatasetRecord
from src.datamodels.split_index import SplitIndex


def load_split_idx(filename: str = "split_index.json") -> SplitIndex:
    """Find and load the file."""

    fps = glob(f"../**/data/**/{filename}", recursive=True)
    if len(fps) > 1:
        raise ValueError(f"Found {len(fps)} matches")
    if not fps:
        raise ValueError(f"Couldn't find {filename}")
    with open(fps[0]) as f:
        raw = json.load(f)

    return SplitIndex(**raw)


def load_dataset_parallel(
    path=Path("data/dataset.ndjson"),
    filter_lang: list[str] | None = None,
) -> list[DatasetRecord]:
    """Load the annoted data (Newline delimited JSON)

    - Tokens and tags stored as parallel lists.
    """
    with path.open("r", encoding="utf-8") as f:
        dataset = [
            d
            for d in (DatasetRecord(**json.loads(line)) for line in f)
            if (filter_lang is None or d.lang in filter_lang)
        ]

    return dataset


def load_dataset_df(path=Path("data/dataset.ndjson")):
    """Load the data directly to a dataframe."""
    schema = {
        "lang": pl.Utf8,
        "name": pl.Utf8,
        "tokens": pl.List(pl.Utf8),
        "tags": pl.List(pl.Utf8),
        "difficulty": pl.Utf8,
    }

    df = pl.read_ndjson(path, schema=schema).with_columns(id=pl.col("lang") + "_" + pl.col("name"))
    return df


def load_dataset_zip(
    path=Path("data/dataset.ndjson"),
    filter_lang: list[str] | None = None,
) -> list[DatasetRecord]:
    """Load the annoted data (Newline delimited JSON)"""

    raise DeprecationWarning("use parallel instead...")
    with path.open("r", encoding="utf-8") as f:
        dataset = []
        for line in f:
            record = json.loads(line)
            if filter_lang is None or record["lang"] in filter_lang:
                tokens, tags = zip(*record["sequence"], strict=True)
                d = DatasetRecord(
                    record["name"],
                    record["lang"],
                    list(tokens),
                    list(tags),
                    record["difficulty"],
                )
                dataset.append(d)
    return dataset


def load_dataset_splits(
    split_idx: dict[str, str],
    path=Path("data/dataset.ndjson"),
    limit: int | None = None,
    filter_lang: set[str] | None = None,
) -> dict[str, list[DatasetRecord]]:
    """Load the annoted data (Newline delimited JSON), and get a list for each split"""
    splits: dict[str, list[DatasetRecord]] = {}
    n_skip = 0
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            d = DatasetRecord(**json.loads(line))

            # optionally filter by lang
            if filter_lang and d.lang not in filter_lang:
                continue

            # where should this example go?
            sk = split_idx.get(d.id)
            if sk is None:
                n_skip += 1
            else:
                splits.setdefault(sk, []).append(d)

            if limit is not None and i > limit:
                break

    if n_skip > 0:
        print(f"[NOTE] skipped {n_skip} examples")

    # measure overlaps
    # n_ngram = 3
    # print(f"Measuring token overlap ({n_ngram}-grams)...")

    # results = datafun.overlap_splits(
    #     {k: [d.tokens for d in data] for k, data in splits.items()}, n_ngram
    # )
    # for k1, k2, ovr in results:
    #     print(f"  overlap({k1}, {k2}) = {ovr:.2%}")

    return splits


def remap_val_train(splits: dict[str, list[DatasetRecord]]):
    """Make a new train set that includes original validation data.

    Useful for re-training a final model, with maximum data.
    """
    assert set(splits.keys()) == {"train", "val", "test"}, f"unexpected {splits.keys()=}"
    return {"train": splits["train"] + splits["val"], "test": splits["test"]}


def dataset_to_df(data: Iterable[DatasetRecord]):
    """Convert DatasetRecords to a DF"""
    df = pl.DataFrame(data=[d.toDict(with_id=True) for d in data])
    return df


def split_to_chars(tokens: list[str], tags: list[str], only_starts=False):
    chars: list[str] = []
    char_tags = []
    for token, tag in zip(tokens, tags, strict=True):
        chars.extend(token)
        if only_starts:
            char_tags.extend(["start"] + ["-"] * (len(token) - 1))
        else:
            char_tags.extend(["start-" + tag] + [tag] * (len(token) - 1))

    return chars, char_tags


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
