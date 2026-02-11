"""CLI for rendering examples"""

import argparse
import os
import sys
from glob import glob
from pathlib import Path

sys.path.append(".")

from src import html_process, util


def render_data(data, title, correct=None, names=False):
    html_process.render_preview(data, "./_style.css", title, correct=correct, show_names=names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Renders examples to a HTML file")
    parser.add_argument("data", choices=["train", "val", "test"])
    parser.add_argument("-l", "--lang", type=str)
    parser.add_argument("-n", "--names", action="store_true")
    parser.add_argument("-c", "--clear", action="store_true")

    args = parser.parse_args()
    dataset = args.data
    lang_filter = str(args.lang)
    include_names = args.names
    clear = args.clear
    if clear:
        for f in glob("previews/*.html"):
            os.remove(f)
        print("cleared old HTML!")

    if not isinstance(lang_filter, list):
        lang_filter = [lang_filter]

    # RENDER DATASET
    split_idx = util.load_split_idx()
    data_true = util.load_dataset_splits(
        split_idx=split_idx.id_to_group, filter_lang=set(lang_filter)
    )

    data_true = data_true[dataset]
    print(f"Loaded {len(data_true)} examples")

    # preview-document title
    title = dataset
    if lang_filter is not None:
        title += "_" + "".join(lang_filter)

    render_data(data_true, title=title + "_GT", names=include_names)

    # RENDER PREDICTIONS
    # find all predictions
    fps = sorted(glob("./output/*.json"))
    all_data = {}
    for fp in fps:
        data = util.load_dataset_splits(
            path=Path(fp), split_idx=split_idx.id_to_group, filter_lang=set(lang_filter)
        )

        if dataset is not None and dataset != "all":
            data = data[dataset]
        all_data[fp.split("/")[-1].split(".")[0]] = data
    lens = [str(len(df)) for df in all_data.values()]
    d = f"{len(all_data)} * {lens[0]}" if len(set(lens)) == 1 else " + ".join(lens)
    print(f"Loaded {d} predictions")
    for k, df in all_data.items():
        render_data(df, title=title + "_" + k, correct=data_true, names=include_names)

    # for easier access
    html_process.make_previews_index()

    # start local server.
    os.system("php -S localhost:1337 -t previews/")
