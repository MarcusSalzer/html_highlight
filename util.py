# For mapping to a smaller label space
import json
import os

import polars as pl

ANNOTATED_EXAMPLES_DIR = os.path.join("data", "annotated")


MAP_TAGS = {
    "op2": "op",
    "op3": "op",
    "opma": "op",
    "opun": "op",
    "opcm": "op",
    "fnme": "fn",
    "fnst": "fn",
    "kwty": "cl",
    "kwfl": "kw",
    "pa": "va",
    "at": "va",
    "se": "pu",
}


def load_examples(tag_map: dict | None = None):
    """Load all annotated examples
    ## parameters
    - tag_map (dict|None): optionally map tags.


    ## returns
    - examples (Dataframe): tokens, tags, lang, length."""
    raise NotImplementedError("UPDATE")
    data_files = os.listdir(ANNOTATED_EXAMPLES_DIR)
    print(f"found {len(data_files)} examples")

    examples = []

    for filename in data_files:
        lang = filename.split("_")[-1].split(".")[0]
        with open(os.path.join(ANNOTATED_EXAMPLES_DIR, filename)) as f:
            d = json.load(f)
            if len(d["tokens"]) != len(d["tags"]):
                raise ValueError("mismatched sequence length")
            d.update({"lang": lang, "length": len(d["tokens"])})

            if tag_map:
                d["tags"] = [MAP_TAGS.get(t, t) for t in d["tags"]]
            examples.append(d)

    return pl.DataFrame(examples)
