"""Run inference on all examples"""

import argparse
import json
import sys
from glob import glob

from tqdm import tqdm

sys.path.append(".")
from src import util
from src.inference import Inference


def find_models():
    fps = glob("**/*", root_dir="models", recursive=True)
    names = [f.split(".")[0].replace("_meta", "").replace("_state", "") for f in fps]
    return sorted(set(names))


if __name__ == "__main__":
    models = find_models()
    parser = argparse.ArgumentParser(description="Run inference on all examples")
    parser.add_argument("-m", "--model", choices=models)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    verbose = args.verbose
    model_name = args.model
    if isinstance(model_name, str):
        models = [model_name]

    # TODO Tagmap?
    data = util.load_examples_json(verbose=False)
    if verbose:
        print(f"Loaded {len(data)} examples")

    print(f"Running {len(models)} model{'s' * (len(models) != 1)}")
    if len(models) > 1 and verbose:
        models = tqdm(models)

    for mn in models:
        infer = Inference(mn, model_dir="models")
        outputs = {}
        for ex in data.iter_rows(named=True):
            tags_pred = infer.run(ex["tokens"], ex["tags"])
            outputs[ex["name"] + "_" + ex["lang"]] = {
                "tokens": ex["tokens"],
                "tags": tags_pred,
            }
        with open(f"./output/pred_{mn}.json", "w", encoding="utf-8") as f:
            json.dump(outputs, f)
