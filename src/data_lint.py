import pointblank as pb
import polars as pl

from src import text_process
from src._constants import (
    DET_TAGS,
    ILLEGAL_BIGRAMS,
    LANG_SPEC_TOKENS,
    LANGS,
    POSSIBLE_PER_TOKEN,
    REQUIRES_PRE,
)
from src.DatasetRecord import DatasetRecord
from src import data_functions as datafun


def lint_data_df(df: pl.DataFrame):
    return (
        pb.Validate(df)
        .col_schema_match(
            pb.Schema(
                {
                    "name": "String",
                    "lang": "String",
                    "tokens": "List(String)",
                    "tags": "List(String)",
                    "difficulty": "String",
                    "id": "String",
                }
            ),
        )
        .rows_distinct(["name", "lang"])
        .interrogate()
    )


def lint_single_record(rec: DatasetRecord, allowed_tags: list[str]):
    reprocess_check(rec.tokens, rec.tags)
    bigram_check(rec.tags)
    lang_spec_check(rec.tokens, rec.tags, rec.lang)

    for to, ta in zip(rec.tokens, rec.tags):
        if to in POSSIBLE_PER_TOKEN.keys() and ta not in POSSIBLE_PER_TOKEN[to]:
            raise LintError(f"'{to}' -> '{ta}' can only be: '{POSSIBLE_PER_TOKEN[to]}'")

    for to in rec.tags:
        if to not in allowed_tags:
            raise LintError(f"disallowed tag '{to}'")

    if rec.tokens[-1] == "\n" or rec.tags[-1] == "nl":
        raise LintError("Trailing NL")

    if rec.lang not in LANGS:
        raise LintError(f"unexpected lang: {rec.lang}")

    if (n_token := len(rec.tokens)) != (n_tag := len(rec.tags)):
        raise LintError(f"inconsistent length: {n_token} tokens | {n_tag} tags")


class LintError(Exception):
    def __init__(self, *args):
        super().__init__(*args)


def reprocess_check(tokens: list[str], tags: list[str]):
    """put tokens together and run deterministic process

    ## parameters
    - tokens
    - tags
    """

    text = "".join(tokens)

    tk, ta = text_process.process(text)
    if len(tk) != len(tokens):
        raise LintError(f"wrong length: {len(tokens)} -> {len(tk)}")
    for tag, newtag in zip(tags, ta):
        if newtag == "uk":
            if tag in DET_TAGS:
                raise LintError(f"reprocess missed a `det_tag`: {tag}")
        elif tag != newtag:
            raise LintError(f"reprocess tag mismatch: {tag} -> {newtag}")


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


def n_gram_overlap_check(records: list[DatasetRecord], n_ngram=3, thr=0.5):
    high = {}
    _, high["tag"] = datafun.overlap_pairwise_simple(
        [d.tags for d in records], n_ngram, thr=thr
    )
    _, high["token"] = datafun.overlap_pairwise_simple(
        [d.tokens for d in records], n_ngram, thr=thr
    )

    for k, res in high.items():
        highest = res[0]
        print(
            f"  overlap({k}) > {thr:.0%} : "
            + f"{len(res) / len(records):.1%} of records"
            + f" (max: {highest[-1]:.1%})"
        )

        if k == "token" and highest[-1] >= 1:
            names = records[highest[0]].name, records[highest[1]].name
            raise LintError(f"max token overlap {highest[-1]} (between {names})")
