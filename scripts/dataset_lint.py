# mypy: disable-error-code="import-untyped"
"""A few rules for checking the quality of the dataset."""

import os
import sys

import polars as pl
from datatools import tabular as dttab
from datatools.benchmark import SequentialTimer

sys.path.append(".")
import src.util as util

from src import text_process


REQUIRES_PRE = {"id": ("nl", "id"), "opas": ("ws", "va", "pa", "brcl", "shfl")}

# bad tag bigrams
ILLEGAL_BIGRAMS = [
    "nl ws",
    "ws nl",
    "va cl",
    "opbi opbi",
    "va va",
    "opcm opcm",
    "kw kw",
]


# TOKENS that are determined for a language
LANG_SPEC_TOKENS = {
    "python": {
        "=": "opas",
        "*=": "opas",
        "//": "opbi",
        "if": "kwfl",
        "else": "kwfl",
        "<": "opcm",
        ">": "opcm",
        "self": "pa",
    },
    "js": {"this": "pa"},
}


def main():
    interactive = "-i" in sys.argv
    if interactive:
        print("\n=== DATASET LINT (INTERACTIVE) ===\n")
    else:
        print("\n=== DATASET LINT ===\nrun interactive with `-i` flag\n")

    timer = SequentialTimer()

    data = util.load_examples_json()
    timer.add("load")
    print("loaded:", data.columns)

    # value_counts
    print("\n value counts:")
    dttab.value_counts(data["lang"], verbose=True)
    dttab.value_counts(data["tokens"].explode(), verbose=True)
    timer.add("value counts")

    # lint each example
    # list of dicts
    examples = list(data.iter_rows(named=True))
    err_count = 0
    delete_rows = [False] * len(data)
    for i, ex in enumerate(examples):
        try:
            lint(ex)
        except LintError as err:
            print(f"{ex['name']} ({ex['lang']})", err)
            print("".join(ex["tokens"]))
            err_count += 1
            if interactive:
                # try:
                #     tokens_fixed, tags_fixed = auto_fix(ex["tokens"], ex["tags"])
                #     doFix = input("auto-fix?")
                #     if doFix.lower() == "y":
                #         ex["tokens"] = tokens_fixed
                #         ex["tags"] = tags_fixed
                # except FixError:
                #     print("no auto-fix")
                resp = input("\naction? ")
                if resp == "delete" or resp == "del":
                    delete_rows[i] = True

            print("-" * 30 + "\n")
    timer.add("lint loop")

    data = (
        pl.DataFrame(examples)
        .with_columns(pl.Series("delete", delete_rows))
        .filter(pl.col("delete").not_())
    )
    if interactive:
        print(f"Keep {len(data)} examples")
        util.save_examples_json(data, "data/examples_annot_new.json")
    else:
        print(f"{err_count} errors ({err_count / len(data) * 100:0.1f}%)\n")

    print(timer)


def lint(ex: dict):
    tokens = ex["tokens"]
    tags = ex["tags"]
    lang = ex.get("lang")
    name = ex.get("name")

    reprocess_check(tokens, tags)
    bigram_check(tags)

    if lang is not None:
        lang_spec_check(tokens, tags, lang)
        # if name is not None:
        #     reload_check(tokens, tags, name, lang)
    if tokens[-1] == "\n" or tags[-1] == "nl":
        raise LintError("Trailing NL")


def reprocess_check(tokens: list[str], tags: list[str], text: str | None = None):
    """put tokens together and run deterministic process

    ## parameters
    - tokens
    - tags
    - text: optionally provide text to process. If none, use `tokens`.
    """

    if text is None:
        text = "".join(tokens)

    tk, ta = text_process.process(text)
    if len(tk) != len(tokens):
        raise LintError(f"wrong length: {len(tokens)} -> {len(tk)}")
    for tag, newtag in zip(tags, ta):
        if newtag == "uk":
            if tag in text_process.DET_TAGS:
                raise LintError(f"reprocess missed a `det_tag`: {tag}")
        elif tag != newtag:
            raise LintError(f"reprocess tag mismatch: {tag} -> {newtag}")


def reload_check(tokens: list[str], tags: list[str], name: str, lang: str):
    """Try to load original example and reprocess that"""
    path = os.path.join("data", "examples", lang, name + ".txt")
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = text_process.clean_text(f.read())

        if "".join(tokens) != text:
            raise LintError(f"Tokens do not match source example:\n==\n{text}\n==\n")
        reprocess_check(tokens, tags, text)
    except FileNotFoundError:
        print(f"could not find: {path}")


def bigram_check(tags: list[str]):
    """Check for required tags before current tag"""
    for tag_pre, tag in zip(tags, tags[1:]):
        if " ".join([tag_pre, tag]) in ILLEGAL_BIGRAMS:
            raise LintError(f"Illegal bigram: `{tag_pre}` `{tag}`")

        req_pre = REQUIRES_PRE.get(tag)
        if req_pre and tag_pre not in req_pre:
            raise LintError(f"`{tag}` requires {req_pre} before, got `{tag_pre}`")


def lang_spec_check(tokens: list[str], tags: list[str], lang: str):
    """Language_specific checks"""
    token_spec = LANG_SPEC_TOKENS.get(lang)
    for token, tag in zip(tokens, tags):
        # if tag_spec:
        #     req_token = tag_spec.get(tag)
        #     if req_token and token != req_token:
        #         raise LintError(
        #             f"({lang}) need {req_token}->`{tag}`, got  {token}->`{tag}`"
        #         )
        if token_spec:
            req_tag = token_spec.get(token)
            if req_tag and tag != req_tag:
                raise LintError(
                    f"({lang}) need `{token}`->`{req_tag}`, got  `{token}`->`{tag}`"
                )


def auto_fix(tokens: list[str], tags: list[str]):
    """Try to fix simple errors"""
    tokens_new = [tokens[0]]
    tags_new = [tags[0]]
    for i in range(1, len(tokens)):
        if tags[i - 1] == "id" and tags[i] in ["ws", "id"]:
            tokens[i - 1] += tokens[i]
        else:
            tokens_new.append(tokens[i])
            tags_new.append(tags[i])
    try:
        lint({"tokens": tokens_new, "tags": tags_new})
    except LintError:
        raise FixError("could not fix")

    return tokens_new, tags_new


class LintError(Exception):
    def __init__(self, *args):
        super().__init__(*args)


class FixError(Exception):
    def __init__(self, *args):
        super().__init__(*args)


if __name__ == "__main__":
    main()
