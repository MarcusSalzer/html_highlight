# For mapping to a smaller label space
import json
from glob import glob

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
    "kwty": "cl",
    "kwfl": "kw",
    "kwop": "kw",
    "kwim": "kw",
    "kwva": "kw",
    "kwfn": "kw",
    "kwmo": "kw",
    "kwio": "kw",
    "kwde": "kw",
    "at": "va",
    "mo": "va",
    "cofl": "co",
    "coil": "co",
    "coml": "co",
    "id": "ws",
}


def load_examples(
    tag_map: dict | None = None,
    filter_lang: list[str] | None = None,
    split_index_name: str | None = None,
    verbose=True,
) -> pl.DataFrame | dict[str, pl.DataFrame]:
    """Load all annotated examples
    ## parameters
    - tag_map (dict|None): optionally map tags.

    ## returns
    - examples (Dataframe): tokens, tags, lang, length."""

    fp = glob("../**/data/examples_annot.json", recursive=True)[0]
    with open(fp, encoding="utf-8") as f:
        d = json.load(f)

    if split_index_name is not None:
        fp = glob(f"../**/data/{split_index_name}.json", recursive=True)[0]
        with open(fp, "r") as f:
            split_index = json.load(f)

    rows = []
    for k, ex in d.items():
        ex["name"] = k
        ex["lang"] = k.split("_")[-1]
        if tag_map:
            ex["tags"] = [tag_map.get(t, t) for t in ex["tags"]]
        if split_index_name:
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

    if split_index_name:
        # separate dataframe per split
        data = {g[0]: df for g, df in data.group_by("split", maintain_order=True)}
        if verbose:
            for k, df in data.items():
                print(f"    {k}: {len(df)}")

    return data


def split_to_chars(tokens: list[str], tags: list[str], only_starts=False):
    chars = []
    char_tags = []
    for token, tag in zip(tokens, tags):
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
