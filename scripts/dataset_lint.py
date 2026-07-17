"""A few rules for checking the quality of the dataset."""

import json
import sys
from pathlib import Path

sys.path.append(".")
from html_highlight.data_lint import LintError
from html_highlight.datamodels.overlap_stat import OverlapStat
from src import cli_util, data_lint, util
from src import data_functions as data_functions

allowed_tags = list(json.loads(Path("data/class_aliases_str.json").read_text()).keys())


def print_overlaps(stats: list[OverlapStat], name_ids: list[str]):
    for st in stats:
        names = tuple(name_ids[i] for i in st.idxs)
        print(f"{names}: {st.overlap:.1%}")


def main():
    print("\n=== DATASET LINT ===\n")

    data = util.load_dataset_parallel()
    print(f"loaded: {len(data)} records.")

    # # value_counts

    # print("\n value counts:")
    # util.value_counts(data["lang"], verbose=True)
    # util.value_counts(data["tokens"].explode(), verbose=True)

    err_count = 0
    # lint each example
    for ex in data:
        try:
            data_lint.lint_single_record(ex, allowed_tags)
        except LintError as err:
            print(f"{ex.name} ({ex.lang})", err)
            cli_util.pretty_print_code(ex.tokens, ex.tags)
            err_count += 1
            print("-" * 30 + "\n")
    print(f"{err_count} errors ({err_count / len(data) * 100:0.1f}%)\n")

    N_TOKEN = 3  # overlap ngram length
    N_TAG = 8  # overlap ngram length
    THR_TOKEN = 0.5
    THR_TAG = 0.8

    print("Overlap check")
    _, high_token = data_functions.overlap_pairwise_simple(
        [d.tokens for d in data], n=N_TOKEN, thr=THR_TOKEN
    )
    _, high_tag = data_functions.overlap_pairwise_simple(
        [d.tags for d in data], n=N_TAG, thr=THR_TOKEN
    )

    print(f"\nToken overlaps (>{THR_TOKEN:.0%}, n={N_TOKEN}): {len(high_token)}")
    print_overlaps(high_token, name_ids=[d.id for d in data])

    print(f"\nTag overlaps (>{THR_TAG:.0%}, n={N_TAG}): {len(high_tag)}")
    print_overlaps(high_tag, name_ids=[d.id for d in data])


if __name__ == "__main__":
    main()
