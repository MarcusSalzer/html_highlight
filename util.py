# For mapping to a smaller label space
import json
from glob import glob

import polars as pl

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


def load_examples(tag_map: dict | None = None, filter_lang: list[str] | None = None):
    """Load all annotated examples
    ## parameters
    - tag_map (dict|None): optionally map tags.

    ## returns
    - examples (Dataframe): tokens, tags, lang, length."""

    datafile = glob("../**/data/examples_annot.json", recursive=True)[0]
    with open(datafile, encoding="utf-8") as f:
        d = json.load(f)

    if tag_map:
        tags = [[tag_map.get(t, t) for t in e["tags"]] for e in d.values()]
    else:
        tags = [e["tags"] for e in d.values()]

    examples = pl.DataFrame(
        {
            "name": ["_".join(k.split("_")) for k in d.keys()],
            "lang": [k.split("_")[-1] for k in d.keys()],
            "difficulty": [e["difficulty"] for e in d.values()],
            "tokens": [e["tokens"] for e in d.values()],
            "tags": tags,
        }
    )
    if filter_lang is not None:
        examples = examples.filter(pl.col("lang").is_in(filter_lang))

    return examples.with_columns(length=pl.col("tokens").list.len())


def data_split(df: pl.DataFrame, fraction=0.2, shuffle=True, verbose=True):
    """Typical train/val split of a polars dataframe."""
    n = int((1 - fraction) * len(df))
    df2 = df.sample(fraction=1, shuffle=shuffle)
    if verbose:
        print(f"splitted {n} & {len(df)-n}" + " (shuffled)" * shuffle)
    return df2.head(n), df2.tail(-n)


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
