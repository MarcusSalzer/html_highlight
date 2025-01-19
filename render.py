"""CLI for rendering examples"""

import argparse
from glob import glob
import os
from src import html_process, util


def render_data(data, title, correct=None, names=False):
    html_process.render_preview(
        data, "./_style.css", title, correct=correct, show_names=names
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Renders examples to a HTML file")
    parser.add_argument("data", choices=["all", "train", "val", "test"])
    parser.add_argument("-l", "--lang")
    parser.add_argument("-s", "--splits")
    parser.add_argument("-n", "--names", action="store_true")
    parser.add_argument("-c", "--clear", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    verbose = args.verbose
    dataset = args.data
    lang_filter = args.lang
    splits = args.splits
    include_names = args.names
    clear = args.clear
    if clear:
        for f in glob("previews/*.html"):
            os.remove(f)

    if isinstance(lang_filter, str):
        lang_filter = [lang_filter]

    # RENDER DATASET
    data_true = util.load_examples_json(
        filter_lang=lang_filter, split_idx_id=splits, verbose=False
    )
    if dataset in ["train", "val", "test"]:
        if splits is None:
            print("Error: requires `splits` if data is not 'all'.")
            files = glob("split_index_*.json", root_dir="data/")
            print("try: -s ", [n[12:].split(".")[0] for n in files])
            exit(1)

        data_true = data_true[dataset]
    if verbose:
        print(f"Loaded {len(data_true)} examples")

    # preview-document title
    title = dataset
    if lang_filter is not None:
        title += "_" + "".join(lang_filter)

    render_data(data_true, title=title, names=include_names)

    # RENDER PREDICTIONS
    # find all predictions
    fps = sorted(glob("./output/*.json"))
    all_data = {}
    for fp in fps:
        data = util.load_examples_json(
            path=fp, split_idx_id=splits, filter_lang=lang_filter, verbose=False
        )
        if dataset is not None and dataset != "all":
            data = data[dataset]
        all_data[fp.split("/")[-1].split(".")[0]] = data
    if verbose:
        lens = [str(len(df)) for df in all_data.values()]
        if len(set(lens)) == 1:
            d = f"{len(all_data)} * {lens[0]}"
        else:
            d = " + ".join(lens)
        print(f"Loaded {d} predictions")
        for k, df in all_data.items():
            render_data(
                df, title=title + "_" + k, correct=data_true, names=include_names
            )

    # for easier access
    html_process.make_previews_index()

    # start local server.
    os.system("php -S localhost:1337 -t previews/")
