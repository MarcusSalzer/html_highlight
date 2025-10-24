"""Make a train/val/test split"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(".")
from src import data_functions as datafun
from src import util

filepath = Path("data") / "split_index.json"
logpath = Path("data") / "split_stats.txt"

if filepath.exists():
    print("Already exists. delete/move old before making new")
    exit(1)

log: list[str] = []


def add_log(m: str):
    print(m)
    log.append(m)


def make_split(
    ratios=[0.7, 0.2, 0.1],
    splitnames=["train", "val", "test"],
    min_group_count=4,
    seed: int | None = None,
):
    examples = util.dataset_to_df(util.load_dataset_parallel())

    examples = datafun.make_example_groups(examples, min_group_count=min_group_count)

    add_log("Groups:")
    for g, c in examples["group"].value_counts(sort=True).iter_rows():
        add_log(f"  {g:.<20} {c}")

    splits = datafun.data_split(examples, ratios, stratify_col="group", seed=seed)

    # fraction of data in ech split
    result_splits = [len(df) / len(examples) for df in splits]
    sMape_splits = util.MAPE(result_splits, ratios, symmetric=True)

    add_log(f"\nSplits:{','.join(f' {r * 100:.1f}%' for r in result_splits)}")
    add_log(f"sMAPE = {sMape_splits * 100:.1f}%")

    # measure overlaps
    n_ngram = 3
    add_log(f"Measuring token overlap ({n_ngram}-grams)...")

    results = datafun.overlap_splits(
        {k: df["tokens"].to_list() for k, df in zip(splitnames, splits)}, n_ngram
    )
    for k1, k2, ovr in results:
        add_log(f"  overlap({k1}, {k2}) = {ovr:.2%}")

    # compute index instead of saving copies of data
    split_index = {}
    for split, splitname in zip(splits, splitnames):
        split_index.update(dict.fromkeys(split["id"].to_list(), splitname))

    now = datetime.now()
    alldata = {"date": f"{now:%Y-%m-%d}", "examples": split_index}
    filepath.write_text(json.dumps(alldata))

    add_log(f"date: {alldata['date']}")
    logpath.write_text("\n".join(log))


if __name__ == "__main__":
    make_split()
