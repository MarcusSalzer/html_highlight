"""Clean up old examples.

After an example has been included in `dataset.ndjson`, we no longer need to keep its input file.
"""

import sys
from pathlib import Path

sys.path.append(".")
from src import util

examples = util.load_dataset_parallel()

print(f"has {len(examples)} examples")

existing = []
for ex in examples:
    og_file = Path("data/examples") / ex.lang / f"{ex.name}.txt"
    if og_file.exists():
        existing.append(og_file)

print(f"still has {len(existing)} orignals")

matching: list[Path] = []
for ex in examples:
    og_file = Path("data/examples") / ex.lang / f"{ex.name}.txt"
    if og_file.exists() and og_file.read_text("utf-8") == "".join(ex.tokens):
        matching.append(og_file)


print(f"Out of those, {len(matching)} match the tokens in the dataset exactly")

for p in matching:
    targ = f"data/trash/{p.name}"
    print("->", targ)
    p.rename(targ)
