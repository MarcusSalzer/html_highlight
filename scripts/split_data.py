"""Make a train/val/test split"""

import json
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(".")
from src import data_functions as datafun
from src import util


# TODO instead of add_log, save meta in the file
def make_split(
    ratios=[0.7, 0.2, 0.1],
    splitnames=["train", "val", "test"],
    min_group_count=4,
    seed: int | None = None,
    max_data: int | None = None,
):
    examples = util.load_dataset_parallel()
    random.shuffle(examples)

    if max_data:
        examples = examples[:max_data]

    filepath = Path("data") / (f"split_index_{max_data}.json" if max_data else "split_index.json")

    if filepath.exists():
        print("Already exists. delete/move old before making new")
        exit(1)

    data = util.dataset_to_df(examples)
    # add group column
    data, group_counts = datafun.make_example_groups(data, min_group_count=min_group_count)
    splits = datafun.data_split(data, ratios, stratify_col="group", seed=seed)
    # fraction of data in ech split
    result_split_ratios = [len(df) / len(data) for df in splits]
    sMape_splits = util.MAPE(result_split_ratios, ratios, symmetric=True)

    meta = {
        "group_counts": group_counts,
        "split_ratios": result_split_ratios,
        "split_ratio_sMAPE": sMape_splits,
    }

    print(f"\nSplits:{','.join(f' {r * 100:.1f}%' for r in result_split_ratios)}")
    print(f"sMAPE = {sMape_splits * 100:.1f}%")

    # measure overlaps
    n_ngram = 3
    print(f"Measuring token overlap ({n_ngram}-grams)...")

    meta["overlap"] = datafun.overlap_splits(
        {k: df["tokens"].to_list() for k, df in zip(splitnames, splits)}, n_ngram
    )
    for k1, k2, ovr in meta["overlap"]:
        print(f"  overlap({k1}, {k2}) = {ovr:.2%}")

    # compute index instead of saving copies of data
    split_index = {}
    for split, splitname in zip(splits, splitnames):
        split_index.update(dict.fromkeys(split["id"].to_list(), splitname))

    now = datetime.now()
    alldata = {"date": f"{now:%Y-%m-%d}", "meta": meta, "examples": split_index}
    filepath.write_text(json.dumps(alldata))

    print(f"date: {alldata['date']}")


if __name__ == "__main__":
    max_data = int(sys.argv[1]) if len(sys.argv) == 2 else None
    if max_data is None:
        print("[note] using all data!")
    make_split(max_data=max_data)
