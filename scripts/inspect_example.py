import os
import sys

sys.path.append(".")
from src import text_process, util

name = sys.argv[1]
print(name)

data = util.load_dataset_zip()
try:
    ex = [d for d in data if d.name == name][0]
except IndexError:
    print("Not found")
    exit(1)

path = os.path.join("data", "examples", ex.lang, name + ".txt")
try:
    with open(path, encoding="utf-8") as f:
        original = text_process.clean_text(f.read())
    print("\n= original =\n" + original + "\n==\n")
except FileNotFoundError:
    print("no original found\n")
    original = None

joined = "".join(ex.tokens)
print("\n= joined =\n" + joined + "\n==\n")


print("character lengths:")
if original is not None:
    print(f"  original: {len(original)}")
    print(f"  original (cleaned): {len(text_process.clean_text(original))}")
print(f"  joined: {len(joined)}")
print(f"  joined (cleaned): {len(text_process.clean_text(joined))}")

tokens_repr, tags_repr = text_process.process(joined)

indent_count = sum([t == "id" for t in ex.tags])
indent_count_repr = sum([t == "id" for t in tags_repr])

print("\nindentations")
print(f"  in data: {indent_count}")
print(f"  join+repr: {indent_count_repr}")
