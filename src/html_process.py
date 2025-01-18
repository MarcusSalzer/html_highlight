import json

import regex as re
from src import text_process


import polars as pl

from src.text_process import (
    bracket_levels,
)


def make_head_html(css_path: str, title: str | None = None):
    """Make document-head with css link"""
    content = ""
    if title:
        content += f"<title>{title}</title>\n"
    content += f'<link rel="stylesheet" type="text/css" href="{css_path}">\n'

    return f"<head>\n{content}\n</head>"


def make_legend_html() -> str:
    """Load class names and display in a list"""
    try:
        with open("data/class_aliases_str.json") as f:
            classes: dict = json.load(f)

        classnames = classes.keys()
    except FileNotFoundError:
        return "<section><p>Could not find classnames.</p></section>"
    except AttributeError:
        return "<section><p>Invalid class alias JSON.</p></section>"

    lines = []
    for c in classnames:
        lines.append(f"""<li><span class="{c}" title="{c}">{c}</span></li>""")
    lines = "\n".join(lines)
    return f"""<section><h2>Legend</h2><ul>{lines}</ul></section>"""


def html_specials(text: str) -> str:
    """Replace html reserved characters"""

    text = re.sub(r"&", r"&amp;", text)
    text = re.sub(r"<", r"&lt;", text)
    text = re.sub(r">", r"&gt;", text)
    text = re.sub(r'"', r"&quot;", text)
    text = re.sub(r"'", r"&apos;", text)
    return text


def format_html(
    tokens: list[str],
    tags: list[str],
    override_elements: list[str] | None = None,
    exclude_tags: list[str] = ["ws", "uk"],
    level_brackets: bool = True,
    css_path: str | None = None,
    legend: bool = False,
    tooltips: bool = False,
) -> str:
    """Format HTML document of tagged text."""
    tokens_with_tags = []

    # If no overrides
    if override_elements is None:
        override_elements = [None] * len(tokens)

    if level_brackets:
        tags, _ = bracket_levels(tags)
    for token, tag, el in zip(tokens, tags, override_elements, strict=True):
        # fix html specials
        token_text = html_specials(token)
        if tag in exclude_tags:
            tokens_with_tags.append(token_text)
        else:
            if el is None:
                el = "span"

            tokens_with_tags.append(
                f'<{el} class="{tag}" {f'title="{tag}"' * tooltips}>{token_text}</{el}>'
            )

    text = "".join(tokens_with_tags)
    text = f'<pre><code class="code-snippet">{text} </code></pre>'

    if legend:
        text += "\n" + make_legend_html()

    if css_path:
        head = make_head_html(css_path)
        return head + f"\n<body>\n{text}\n</body>"
    else:
        return text


def render_preview(data: pl.DataFrame, css_path: str, title: str | None = None):
    """Write a complete HTML document"""
    document = make_head_html(css_path, title) + "<body>\n"

    if title:
        document += f"<h1>{title}</h1>"

    for ex in data.iter_rows(named=True):
        _, tags_det = text_process.process("".join(ex["tokens"]))
        mark = ["mark" if (t == "uk") else None for t in tags_det]
        ex_text = format_html(
            ex["tokens"], ex["tags"], override_elements=mark, tooltips=True
        )
        ex_title = f"<p>{ex['name']} ({ex['lang']})</p>"
        document += ex_title + "\n" + ex_text + "<br>"
    document += "</body>\n"

    if title:
        filename = re.sub(r"[\\\/\.]", "_", title)
    else:
        filename = "output"

    with open(f"./previews/{filename}.html", "w", encoding="utf-8") as f:
        f.write(document)
    print(f"wrote {len(document.splitlines())} lines")
