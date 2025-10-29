"""Convert parallel dataset to zip-version (MAYBE NOPE)"""

import json
import sys
from pathlib import Path

sys.path.append(".")

from src import util

IN_FILE = Path("data/dataset_para.ndjson")
OUT_FILE = Path("data/dataset_zip.ndjson")

data = util.load_dataset_parallel(IN_FILE)

lines = []
for d in data:
    record = {
        "name": d.name,
        "lang": d.lang,
        "difficulty": d.difficulty,
        "sequence": list(zip(d.tokens, d.tags, strict=True)),
    }
    lines.append(json.dumps(record, ensure_ascii=False))

OUT_FILE.write_text("\n".join(lines))


data_2 = util.load_dataset_zip(OUT_FILE)

assert data == data_2
