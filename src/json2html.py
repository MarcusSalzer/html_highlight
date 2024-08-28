import os
import json
from src.text_functions import tokens_to_html, make_legend_html, bracket_levels


"""
Convert annotation to html to visually check"""

DIR = "data/annotated_codes"

examples = os.listdir(DIR)

ex = examples[5]

ex_path = os.path.join(DIR, ex)
with open(ex_path) as f:
    data = json.load(f)

typ: str = data["type"]
tokens: list[str] = data["tokens"]
tags: list[str] = data["tags"]
changed: list[bool] = data["changed"]

print(f"Loaded {ex}")

tags, _ = bracket_levels(tags)  # TODO?

title = f"<h2>{ex.split('.')[0]}</h2>\n"
html_text = tokens_to_html(tokens, tags)

css_path = "_style.css"
css_link = f'<link rel="stylesheet" type="text/css" href="{css_path}">'
final_html = f"""
<head>
    {css_link}
</head>
<body>
{title}
{html_text}
{make_legend_html()}
</body>
"""

with open("output.html", "w") as f:
    f.write(final_html)
