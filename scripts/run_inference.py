"""Run inference on all examples"""

import json
import sys
from glob import glob
from pathlib import Path

from tqdm import tqdm

sys.path.append(".")
from src import util
from src.inference import Inference


def find_models():
    model_paths = glob("**/*_meta.json", root_dir="models_trained", recursive=True)
    names = [f.removesuffix("_meta.json") for f in model_paths]
    return sorted(set(names))


if __name__ == "__main__":
    model_ids = find_models()
    if not model_ids:
        print("No models found")
        exit(1)

    data = util.load_dataset_zip()
    print(f"Loaded {len(data)} examples")

    print(f"Running {len(model_ids)} model{'s' * (len(model_ids) != 1)}")
    if len(model_ids) > 1:
        model_ids = tqdm(model_ids)

    for mn in model_ids:
        infer = Inference(mn, model_dir=Path("models_trained"))
        outputs = {}
        for ex in data:
            tags_pred = infer.run(ex.tokens, ex.tags)
            outputs[ex.name + "_" + ex.lang] = {
                "tokens": ex.tokens,
                "tags": tags_pred,
            }
        with open(f"./output/pred_{mn}.json", "w", encoding="utf-8") as f:
            json.dump(outputs, f)
