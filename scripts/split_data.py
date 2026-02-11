"""Make a train/val/test split"""

import json
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(".")

from src import data_functions as datafun
from src import util
from src.datamodels.split_index import SplitIndex


def make_split(
    ratios=(0.7, 0.2, 0.1),
    splitnames=("train", "val", "test"),
    min_group_count=4,
    seed: int | None = None,
    max_data: int | None = None,
):
    examples = util.load_dataset_parallel()
    random.shuffle(examples)

    if max_data:
        examples = examples[:max_data]

    filepath = Path("data") / (f"split_index_{max_data}.json" if max_data else "split_index.json")

    # if filepath.exists():
    #     print("Already exists. delete/move old before making new")
    #     exit(1)

    data = util.dataset_to_df(examples)
    # add group column
    data, group_counts = datafun.make_example_groups(data, min_group_count=min_group_count)
    splits = datafun.data_split(data, ratios, stratify_col="group", seed=seed)
    # fraction of data in ech split
    result_split_ratios = [len(df) / len(data) for df in splits]

    meta = {
        "group_counts": group_counts,
        "split_ratios": result_split_ratios,
    }

    print(f"\nSplits:{','.join(f' {r * 100:.1f}%' for r in result_split_ratios)}")
    print(f"sMAPE = {util.MAPE(result_split_ratios, ratios, symmetric=True):.1%}")

    # measure overlaps
    n_ngram = 3
    print(f"Measuring token overlap ({n_ngram}-grams)...")

    # compute index instead of saving copies of data
    to_group = {}
    for split, splitname in zip(splits, splitnames, strict=True):
        to_group.update(dict.fromkeys(split["id"].to_list(), splitname))

    now = datetime.now()

    alldata = {"date": f"{now:%Y-%m-%d}", "meta": meta, "examples": to_group}

    model = SplitIndex(
        date=now.date(),
        group_counts=group_counts,
        split_ratios=dict(zip(splitnames, result_split_ratios, strict=True)),
        overlap=datafun.overlap_splits(
            {k: df["tokens"].to_list() for k, df in zip(splitnames, splits, strict=True)}, n_ngram
        ),
        overlap_ngram=n_ngram,
        id_to_group=to_group,
    )
    filepath.write_text(json.dumps(model.model_dump(mode="json"), indent=2))

    print(f"date: {alldata['date']}")


if __name__ == "__main__":
    max_data = int(sys.argv[1]) if len(sys.argv) == 2 else None
    if max_data is None:
        print("[note] using all data!")
    make_split(max_data=max_data)
