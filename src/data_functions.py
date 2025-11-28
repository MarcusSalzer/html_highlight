import itertools
from collections.abc import Sequence
from typing import Any, Literal, cast

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from sklearn.model_selection import KFold


def make_example_groups(df: pl.DataFrame, min_group_count: int = 3):
    """Add a group column to examples dataframe.

    Grouping by:
    - length quantile
    - lang
    """
    df = df.with_columns(
        length=pl.col("tokens").list.len(),
    )
    df = df.with_columns(
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
    rare_groups = (df.group_by("group").agg(pl.len()).filter(pl.col("len") < min_group_count))[
        "group"
    ]

    df_updated = df.with_columns(
        group=pl.when(pl.col("group").is_in(rare_groups)).then(pl.lit("other")).otherwise("group")
    )

    group_counts = dict(df_updated["group"].value_counts(sort=True).iter_rows())
    return df_updated, group_counts


def data_split(
    data: pl.DataFrame,
    ratios: Sequence[float] = (0.6, 0.2, 0.2),
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
        for split_id, (s, e) in enumerate(zip(*get_splits(n_group, ratios), strict=True)):
            split_dfs[split_id].append(group_df[s:e])

    if shuffle:
        return [pl.concat(dfs).sample(fraction=1.0, shuffle=True, seed=seed) for dfs in split_dfs]
    else:
        return [pl.concat(dfs) for dfs in split_dfs]


def get_ngrams(tokens: list[str], n: int):
    """Get the set of unique n-grams in tokens"""

    # ngrams = []
    # for i in range(len(tokens) - n + 1):
    #     ngrams.append(tuple(tokens[i : i + n]))
    # return set(ngrams)

    ngrams: set[tuple[str, ...]] = set(zip(*[tokens[i:] for i in range(n)], strict=False))
    return ngrams


def get_ngrams_all(docs: Sequence[list[str]], n: int, n_jobs: int = 1):
    """Get the ngrams from each doc

    NOTE: n_jobs>1 makes it slower for small/medium datasets
    """
    if n_jobs == 1:
        return [get_ngrams(d, n) for d in docs]

    ngram_sets = Parallel(n_jobs)(delayed(lambda x: get_ngrams(x, n))(s) for s in docs)
    assert isinstance(ngram_sets, list)
    ngram_sets = cast(list[set[tuple[str, ...]]], ngram_sets)
    return ngram_sets


def get_overlap(a: set[Any], b: set[Any], norm: Literal["iou", "max"] = "iou"):
    """Measure overlap between two sets

    parameters
    ----------
    a, b: set
      sets to compare
    norm: str
      how to normalize the result
    """
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

    # store all ngram sets ahead of time to avoid recomputing
    # shouldnt need too much memory
    ngram_sets = [get_ngrams(s, n) for s in docs]
    for i, j in itertools.combinations(range(len(docs)), 2):
        overlap = get_overlap(ngram_sets[i], ngram_sets[j])

        # fill matrix
        results[i, j] = overlap
        results[j, i] = overlap

        # keep track of highest
        if overlap > thr:
            high.append((i, j, overlap))

    # sort by descending overlap
    high.sort(key=lambda t: -t[-1])
    return results, high


def overlap_split_pair(
    a: list[list[str]],
    b: list[list[str]],
    n: int,
    norm: Literal["iou", "max"] = "iou",
):
    ngrams_a, ngrams_b = set(), set()
    ngrams_a.update(*[get_ngrams(seq, n) for seq in a])
    ngrams_b.update(*[get_ngrams(seq, n) for seq in b])
    return get_overlap(ngrams_a, ngrams_b, norm=norm)


def overlap_splits(splits: dict[str, list[list[str]]], n: int = 3):
    """Pairwise overlap between sets"""

    assert len(splits) > 1, "expects more than 1 split"
    # Collect all ngrams for each split
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


def simple_folds(df: pl.DataFrame, k: int, shuffle: bool, seed: int | None = None):
    """Split data in k folds with k-1 for training and 1 for test"""
    kf = KFold(k, shuffle=shuffle, random_state=seed)
    all_idxs = np.arange(len(df))
    splits = [(df[ix_train], df[ix_test]) for (ix_train, ix_test) in kf.split(all_idxs)]
    return splits


# =================================
# Ideas on how to get better splits


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12):
    """Compute Jensen Shannon Divergence.

    Idea
    ----
    Measure how similiar the feature distribution is between splits/folds.

    """
    # Normalize
    # KL-divergence (nested function?)
    # JS divergence (is symmetric?)
    pass


def fold_feature_js(folds: list[pl.DataFrame], feature: str):
    """Compute mean JSD between each fold and global distribution for a categorical feature."""
    # TODO: does not need whole dataframes?
    pass


def partition_distr_overlap_combo_score():
    """Compute score.

    Idea
    ----
    When splitting data for train/val, we want to:

    - Minimize distribution variance
    - Minimize overlap (leakage)

    Or concretely, we want to evenly split groups,
    but, there is more overlap within groups.

    """
    pass


def group_based_greedy_partition(
    groups_map: dict,  # maybe?
    n_folds=4,
    alpha=0.7,  # balance objectives
    features=("lang", "group"),
    ngram_n: int = 3,  # for overlap computation
    restarts: int = 5,
):
    pass
