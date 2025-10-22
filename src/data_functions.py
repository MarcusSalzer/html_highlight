import itertools
import random
from collections.abc import Sequence
from typing import Literal

import numpy as np
import polars as pl


def modify_name(name: str):
    """Make a new name, with similar structure"""

    newname = ""
    for c in name:
        if c.isalpha():
            c2 = chr(random.randint(97, 122))
            if c.isupper():
                c2 = c2.upper()
        elif c.isnumeric():
            c2 = str(random.randint(0, 9))
        else:
            c2 = c
        newname += c2
    return newname


def randomize_names(tokens: list[str], tags: list[str]):
    """Randomize names of tokens with arbitrary names.

    Affected classes: `pa`, `mo`, `fnme`, `fnas`, `fnsa`, `va`, `at`
    """

    renameable = ["pa", "mo", "fnme", "fnas", "fnsa", "va", "at"]

    renamed = [False] * len(tokens)
    tokens_new = tokens.copy()
    for i, (token, tag) in enumerate(zip(tokens, tags)):
        if tag in renameable and not renamed[i]:
            newname = modify_name(token)
            # print(token + "->" + newname)
            for j in range(i, len(tokens)):
                if tokens[j] == token:
                    renamed[j] = True
                    tokens_new[j] = newname

    return tokens_new


def make_example_groups(examples: pl.DataFrame, min_group_count: int = 3):
    """add a group column, grouping by:
    - approx length
    - lang
    """
    examples = examples.with_columns(
        length=pl.col("tokens").list.len(),
    )
    examples = examples.with_columns(
        group=(
            pl.when(pl.col("length") < pl.col("length").quantile(1 / 3))
            .then(pl.lit("short"))
            .when(pl.col("length") < pl.col("length").quantile(2 / 3))
            .then(pl.lit("medium"))
            .otherwise(pl.lit("long"))
            + "_"
            + pl.col("lang")
        ),
    )

    # keep all these in "other"
    rare_groups = (
        examples.group_by("group").agg(pl.len()).filter(pl.col("len") < min_group_count)
    )["group"]

    examples = examples.with_columns(
        group=pl.when(pl.col("group").is_in(rare_groups))
        .then(pl.lit("other"))
        .otherwise("group")
    )
    return examples


def data_split(
    data: pl.DataFrame,
    ratios: list[float] = [0.6, 0.2, 0.2],
    stratify_col: str | None = "group",
    shuffle: bool = True,
    seed: int | None = None,
) -> list[pl.DataFrame]:
    """Split dataframe"""

    def get_splits(n: int, splits: list[float]):
        """get split indices"""
        n_split = len(splits)
        if n < len(splits):
            raise ValueError(f"too few to split: {n} <  {len(splits)}")

        ends = [int(sum(splits[:k]) * n) for k in range(1, n_split + 1)]
        starts = [0] + ends[:-1]
        return starts, ends

    ssum = sum(ratios)
    ratios = [s / ssum for s in ratios]

    ## list of df:s for each split
    split_dfs: list[list[pl.DataFrame]] = [[] for _ in ratios]
    for _, group_df in data.group_by(stratify_col, maintain_order=True):
        n_group = len(group_df)
        if shuffle:
            group_df = group_df.sample(fraction=1.0, shuffle=True, seed=seed)

        # split one group
        for split_id, (s, e) in enumerate(zip(*get_splits(n_group, ratios))):
            split_dfs[split_id].append(group_df[s:e])

    if shuffle:
        return [
            pl.concat(dfs).sample(fraction=1.0, shuffle=True, seed=seed)
            for dfs in split_dfs
        ]
    else:
        return [pl.concat(dfs) for dfs in split_dfs]


def get_ngrams(tokens: list[str], n: int):
    """Get the set of unique n-grams in tokens"""

    # ngrams = []
    # for i in range(len(tokens) - n + 1):
    #     ngrams.append(tuple(tokens[i : i + n]))
    # return set(ngrams)

    ngrams: set[tuple[str, ...]] = set(zip(*[tokens[i:] for i in range(n)]))
    return ngrams


def get_overlap(a: set, b: set, norm: Literal["iou", "max"] = "iou"):
    if len(a) == 0 or len(b) == 0:
        return np.nan

    if norm == "iou":
        return len(a & b) / len(a | b)  # IoU
    elif norm == "max":
        return len(a & b) / max(len(a), len(b))
    else:
        raise ValueError(f"unknown normalization: {norm}")


def overlap_pairwise_simple(docs: Sequence[list[str]], n: int = 3, thr=0.5):
    """Compare n-gram overlap for all document pairs"""

    results = np.eye(len(docs))
    high = []
    for (i, d1), (j, d2) in itertools.combinations(enumerate(docs), 2):
        ng1 = get_ngrams(d1, n)
        ng2 = get_ngrams(d2, n)

        overlap = get_overlap(ng1, ng2)

        # fill matrix
        results[i, j] = overlap
        results[j, i] = overlap

        # keep track of highest
        if overlap > thr:
            high.append((i, j, overlap))

    # sort by descending overlap
    high.sort(key=lambda t: -t[-1])
    return results, high


def overlap_splits(splits: dict[str, list[list[str]]], n: int = 3):
    """Pairwise overlap between sets"""

    all_ngrams: dict[str, set[tuple[str, ...]]] = {}
    for k, spl in splits.items():
        all_ngrams[k] = set()
        for seq in spl:
            all_ngrams[k].update(get_ngrams(seq, n))

    results: list[tuple[str, str, float]] = []
    for k1, k2 in itertools.combinations(all_ngrams.keys(), 2):
        overlap = get_overlap(all_ngrams[k1], all_ngrams[k2])
        results.append((k1, k2, overlap))
    return results
