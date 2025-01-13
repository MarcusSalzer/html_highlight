"""A few rules for checking the quality of the dataset."""

import sys

import polars as pl
from datatools import tabular as dttab
from datatools.benchmark import SequentialTimer

sys.path.append(".")
import src.util as util

from src import text_process


REQUIRES_PRE = {"id": ("nl", "id"), "opas": ("ws", "va", "pa", "brcl")}

ILLEGAL_BIGRAMS = [
    "nl ws",
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
    },
}


def main():
    timer = SequentialTimer()

    data = util.load_examples()
    timer.add("load")
    print(data.columns)

    # value_counts
    print("\n value counts:")
    dttab.value_counts(data["lang"], verbose=True)
    dttab.value_counts(data["tokens"].explode(), verbose=True)
    timer.add("value counts")

    # trailing newlines?
    trailing_nl = data.filter(pl.col("tags").list[-1] == "nl")
    print(f"\n{len(trailing_nl)} examples with ending nl\n")
    timer.add("trailing nl")

    # reprocess and check length
    data = data.with_columns(
        repr_len=pl.col("tokens").map_elements(
            lambda tks: len(text_process.process("".join(tks))[1]), pl.Int32
        )
    )

    wrong_length = data.filter(pl.col("repr_len") != pl.col("length"))
    if not wrong_length.is_empty():
        print("\n wrong length")
        print(wrong_length)

    timer.add("reprocess")

    # lint each example
    err_count = 0
    for ex in data.iter_rows(named=True):
        try:
            lint(ex)
        except LintError as err:
            print(ex["name"], err)
            print("".join(ex["tokens"]))
            print("-" * 30 + "\n")
            err_count += 1
    timer.add("lint loop")

    print(f"{err_count} errors ({err_count / len(data) * 100:0.1f}%)\n")

    print(timer)


def lint(ex: dict):
    reprocess_check(ex["tokens"], ex["tags"])
    bigram_check(ex["tags"])
    lang_spec_check(ex["tokens"], ex["tags"], ex["lang"])


def reprocess_check(tokens: list[str], tags: list[str]):
    """put tokens together and run deterministic process"""
    tk, ta = text_process.process("".join(tokens))
    if len(tk) != len(tokens):
        raise LintError(f"wrong length: {len(tokens)} -> {len(tk)}")
    for tag, newtag in zip(tags, ta):
        if newtag == "uk":
            if tag in text_process.DET_TAGS:
                raise LintError(f"reprocess missed a `det_tag`: {tag}")
        elif tag != newtag:
            raise LintError(f"reprocess tag mismatch: {tag} -> {newtag}")


def bigram_check(tags: list[str]):
    """Check for required tags before current tag"""
    for tag_pre, tag in zip(tags, tags[1:]):
        if "".join([tag_pre, tag]) in ILLEGAL_BIGRAMS:
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
                    f"({lang}) need {token}->`{req_tag}`, got  {token}->`{tag}`"
                )


class LintError(Exception):
    def __init__(self, *args):
        super().__init__(*args)


if __name__ == "__main__":
    main()
