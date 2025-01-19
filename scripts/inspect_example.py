import os
import sys
import polars as pl

sys.path.append(".")
from src import text_process, util

name = sys.argv[1]
print(name)

data = util.load_examples_json(verbose=False)
ex = data.row(by_predicate=pl.col("name") == name, named=True)
tokens = ex["tokens"]
tags = ex["tags"]
lang = ex.get("lang")


path = os.path.join("data", "examples", lang, name + ".txt")
try:
    with open(path, "r", encoding="utf-8") as f:
        original = text_process.clean_text(f.read())
    print("\n= original =\n" + original + "\n==\n")
except FileNotFoundError:
    print("no original found\n")

joined = "".join(tokens)
print("\n= joined =\n" + joined + "\n==\n")

print("character lengths:")
print(f"  original: {len(original)}")
print(f"  original (cleaned): {len(text_process.clean_text(original))}")
print(f"  joined: {len(joined)}")
print(f"  joined (cleaned): {len(text_process.clean_text(joined))}")

tokens_repr, tags_repr = text_process.process(joined)

indent_count = sum([t == "id" for t in tags])
indent_count_repr = sum([t == "id" for t in tags_repr])

print("\nindentations")
print(f"  in data: {indent_count}")
print(f"  join+repr: {indent_count_repr}")
