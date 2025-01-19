"""Make a train/val/test split"""

from datetime import datetime
import json
from src import util, data_functions
import polars as pl


def make_split(
    ratios=[0.6, 0.25, 0.15],
    splitnames=["train", "val", "test"],
    min_group_count=4,
    seed: int | None = None,
):
    examples = util.load_examples_json()

    examples = data_functions.make_example_groups(
        examples, min_group_count=min_group_count
    )

    print("\nGroups:")
    for g, c in examples["group"].value_counts(sort=True).iter_rows():
        print(f"  {g}: {c}")

    splits = data_functions.data_split(examples, ratios, seed=seed)
    result_splits = [len(df) / len(examples) for df in splits]

    print(f"\nSplits:{','.join(f' {r * 100:.1f}%' for r in result_splits)}")
    sMape_splits = util.MAPE(result_splits, ratios, symmetric=True)
    print(f"sMAPE = {sMape_splits * 100:.1f}%")

    split_index = {}
    for split, splitname in zip(splits, splitnames):
        split_index.update(dict.fromkeys(split["id"].to_list(), splitname))

    now = datetime.now()
    name = f"split_index_{now.month:02d}{now.day:02d}"

    with open(f"./data/{name}.json", "w") as f:
        json.dump(split_index, f)


if __name__ == "__main__":
    make_split()
