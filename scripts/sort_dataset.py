"""
Simply sort the dataset file by id.
"""

import sys
import time

sys.path.append(".")
from src import util

OUT_FILE = "data/dataset_sorted.ndjson"


def main_records():
    data = util.load_dataset_parallel()
    print(f"loaded: {len(data)} records.")

    data.sort(key=lambda x: x.id)
    print(f"sorted: {data[0].id} ... {data[-1].id}")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for rec in data:
            f.write(rec.model_dump_json() + "\n")


def main_df():
    data = util.load_dataset_df()
    print(f"loaded: {len(data)} records.")

    data = data.sort("id")

    print(f"sorted: {data['id'][0]} ... {data['id'][-1]}")
    data.select(["lang", "name", "difficulty", "tokens", "tags"]).write_ndjson(OUT_FILE)
    print(f"saved {OUT_FILE}")


if __name__ == "__main__":
    print("\n=== DATASET SORT ===\n")

    t0 = time.time()
    # main_df()
    main_records()
    print(f"time: {(time.time() - t0) * 1000:.1f}ms")
