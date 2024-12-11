"""A few rules for checking the quality of the dataset.

- id must follow nl or id.

Some forbidden sequences?

- repeated bi/as/cm op
- word-like things without spaces between

"""

import json


if __name__ == "__main__":
    with open("data/examples_annot.json", encoding="utf-8") as f:
        dataset: dict = json.load(f)

    print(dataset)

    for name, ex in dataset.items():
        print(len(ex["tokens"]), len(ex["tags"]))
