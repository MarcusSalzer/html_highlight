"""A few rules for checking the quality of the dataset.

- id must follow nl or id.

Some forbidden sequences?

- repeated bi/as/cm op
- word-like things without spaces between

"""

import sys
from timeit import default_timer

import polars as pl
from datatools import tabular as dttab

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
    dttab.value_counts(data["tags"].explode(), verbose=True)
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

    print(timer)


class SequentialTimer:
    """Measure runtime for each segment of a script."""

    def __init__(self):
        self.tt = [("init", default_timer())]

    def add(self, name: str):
        """Insert a timestamp."""
        self.tt.append((name, default_timer()))

    def get_diffs(self):
        """Compute runtime for each segment"""
        return [(k, t - tp) for (k, t), (_, tp) in zip(self.tt[1:], self.tt)]

    def __str__(self):
        lines = ["Timings: "] + [
            f"  -{k.ljust(10)}\t {t:.5f} s" for k, t in self.get_diffs()
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    main()
