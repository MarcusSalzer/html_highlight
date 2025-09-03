"""A few rules for checking the quality of the dataset."""

import json
from pathlib import Path
import sys


sys.path.append(".")
from src import data_lint
from src.data_lint import LintError

import src.util as util

from src import cli_util

allowed_tags = list(json.loads(Path("data/class_aliases_str.json").read_text()).keys())


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

    # print("DF validation...")
    # result = data_lint.lint_data_df(util.dataset_to_df(data))
    # Path("media").mkdir(exist_ok=True)
    # result.get_tabular_report().save("media/validation.png")


if __name__ == "__main__":
    main()
