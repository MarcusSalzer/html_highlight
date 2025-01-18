"""CLI for rendering examples"""

import argparse
from glob import glob
import os
from src import html_process, util


def render_data(data, title):
    html_process.render_preview(data, "./_style.css", title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Renders examples to a HTML file")
    parser.add_argument("data", choices=["all", "train", "val", "test"])
    parser.add_argument("-l", "--lang")
    parser.add_argument("-s", "--splits")
    parser.add_argument(
        "-p", "--php", action="store_true", help="start PHP local server to show result"
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    verbose = args.verbose
    dataset = args.data
    lang_filter = args.lang
    splits = args.splits
    php = args.php

    if isinstance(lang_filter, str):
        lang_filter = [lang_filter]

    # RENDER DATASET
    all_data = util.load_examples_json(
        filter_lang=lang_filter, split_idx_id=splits, verbose=False
    )
    if dataset in ["train", "val", "test"]:
        if splits is None:
            print("Error: requires `splits` if data is not 'all'.")
            files = glob("split_index_*.json", root_dir="data/")
            print("try: -s ", [n[12:].split(".")[0] for n in files])
            exit(1)

        all_data = all_data[dataset]
    if verbose:
        print(f"Loaded {len(all_data)} examples")

    # preview-document title
    title = dataset + "_" + "".join(lang_filter) * (lang_filter is not None)
    render_data(all_data, title=title)

    # RENDER PREDICTIONS
    # find all predictions
    fps = sorted(glob("./output/*.json"))
    all_data = {}
    for fp in fps:
        data = util.load_examples_json(
            path=fp, split_idx_id=splits, filter_lang=lang_filter, verbose=False
        )
        if dataset is not None:
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
            render_data(df, title=title + "_" + k)

    if php:
        paths = sorted(glob("*.html", root_dir="previews/"))
        paths.remove("index.html")
        links = [f'<li><a href = "{p}">{p.split(".")[0]}</a></li>' for p in paths]
        content = "<ul>" + "\n".join(links) + "</ul>"
        document = '<!DOCTYPE html><html><head><meta charset="utf-8" />'
        document += f"</head><body>{content}</body></html>"
        with open("previews/index.html", "w", encoding="utf-8") as f:
            f.write(document)

        os.system("php -S localhost:1337 -t previews/")
