from highlighting_functions import Highlighter
import json
import pyperclip as pc


try:
    in_text = pc.paste()
except Exception:
    in_text = "?"


print("-" * 30)
pattern_path = "patterns/p_python.json"
with open(pattern_path) as f:
    p_python = json.load(f)

highlighter = Highlighter(p_python, None)

out_text = highlighter.process(in_text)
print(out_text)

with open("output.html", "w") as f:
    f.write(out_text)

pc.copy(out_text)