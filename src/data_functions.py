# mypy: disable-error-code="misc"

import random
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
    - difficulty?
    """
    examples = examples.with_columns(
        group=(
            pl.when(pl.col("length") < pl.col("length").quantile(1 / 3))
            .then(pl.lit("short"))
            .when(pl.col("length") < pl.col("length").quantile(2 / 3))
            .then(pl.lit("medium"))
            .otherwise(pl.lit("long"))
            + "_"
            + pl.col("lang")
        )
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
