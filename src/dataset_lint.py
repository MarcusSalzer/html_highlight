"""A few rules for checking the quality of the dataset.

- id must follow nl or id.

Some forbidden sequences?

- repeated bi/as/cm op
- word-like things without spaces between

"""

import sys

import polars as pl
from datatools import tabular as dttab
from datatools.benchmark import SequentialTimer

sys.path.append(".")
import util

from src import text_process


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

    # None tags
    print("\ncontains None:")
    for name, tag_list in zip(data["name"], data["tags"]):
        if None in tag_list:
            print("  " + name)

    # TODO also check that det prediction is not altered! (there is at least one)
    wrong_length = data.filter(pl.col("repr_len") != pl.col("length"))
    if not wrong_length.is_empty():
        print("\n wrong length")
        print(wrong_length)

    timer.add("reprocess")

    for ex in data.iter_rows(named=True):
        try:
            lint(ex)
        except LintError as err:
            print(ex["name"], err)
            print(ex["tokens"])
    timer.add("lint loop")

    print(timer)


def lint(ex: dict):
    check_brackets(ex["tags"])


def check_brackets(tags):
    counts = dttab.value_counts(tags, as_dict=True)
    if counts.get("brop", 0) != counts.get("brcl", 0):
        raise LintError("unmacthed brackets")


class LintError(Exception):
    def __init__(self, *args):
        super().__init__(*args)


if __name__ == "__main__":
    main()
