"""Make a train/val/test split"""

from datetime import datetime
import json
from pathlib import Path
import sys

sys.path.append(".")
from src import data_functions, util

filepath = Path("data") / "split_index.json"

if filepath.exists():
    print("Already exists. delete/move old before making new")
    exit(1)


def make_split(
    ratios=[0.7, 0.2, 0.1],
    splitnames=["train", "val", "test"],
    min_group_count=4,
    seed: int | None = None,
):
    examples = util.dataset_to_df(util.load_dataset_parallel())

    examples = data_functions.make_example_groups(
        examples, min_group_count=min_group_count
    )

    print("\nGroups:")
    for g, c in examples["group"].value_counts(sort=True).iter_rows():
        print(f"  {g:.<20} {c}")

    splits = data_functions.data_split(
        examples, ratios, stratify_col="group", seed=seed
    )
    result_splits = [len(df) / len(examples) for df in splits]

    print(f"\nSplits:{','.join(f' {r * 100:.1f}%' for r in result_splits)}")
    sMape_splits = util.MAPE(result_splits, ratios, symmetric=True)
    print(f"sMAPE = {sMape_splits * 100:.1f}%")

    split_index = {}
    for split, splitname in zip(splits, splitnames):
        split_index.update(dict.fromkeys(split["id"].to_list(), splitname))

    now = datetime.now()
    with open(filepath, "w") as f:
        json.dump(
            {
                "date": f"{now:%Y-%m-%d}",
                "examples": split_index,
            },
            f,
        )


if __name__ == "__main__":
    make_split()
