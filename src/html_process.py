import json
from collections.abc import Sequence
from glob import glob

import polars as pl
import regex as re

from src import text_process
from src.text_process import bracket_levels


def make_head_html(css_path: str, title: str | None = None):
    """Make document-head with css link."""
    content = ""
    if title:
        content += f"<title>{title}</title>\n"
    content += f'<link rel="stylesheet" type="text/css" href="{css_path}">\n'

    return f"<head>\n{content}\n</head>"


def make_legend_html() -> str:
    """Load class names and display in a list."""
    try:
        with open("data/class_aliases_str.json") as f:
            classes: dict = json.load(f)
    except FileNotFoundError:
        return "<section><p>Could not find classnames.</p></section>"
    except AttributeError:
        return "<section><p>Invalid class alias JSON.</p></section>"

    classnames = classes.keys()
    lines = []
    for c in classnames:
        lines.append(f"""<li><span class="{c}" title="{c}">{c}</span></li>""")

    return f"""<section><h2>Legend</h2><ul>{"\n".join(lines)}</ul></section>"""


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
    override_elements: Sequence[str | None] | None = None,
    exclude_tags: list[str] = ["ws", "uk", "id", "<unk>"],
    level_brackets: bool = True,
    css_path: str | None = None,
    legend: bool = False,
    tooltips: bool = False,
    errors: list[bool] | None = None,
) -> str:
    """Format HTML document of tagged text.

    ## parameters
    - override_elements: optionally use something else than `<span>`.
    - exclude_tags: these will be left as plain text
    - level_brackets: if true, replace brop/brcl with br{n}
    - css_path: if not None, make document with `<head>` and `<body>`
    """
    tagged_elements = []

    # If no overrides
    if override_elements is None:
        override_elements = [None] * len(tokens)

    # If no error markings
    if errors is None:
        errors = [False] * len(tokens)

    if level_brackets:
        tags, _ = bracket_levels(tags)
    for token, tag, el, is_err in zip(
        tokens, tags, override_elements, errors, strict=True
    ):
        # fix html specials
        token_text = html_specials(token)
        if tag in exclude_tags:
            tagged_elements.append(token_text)
        else:
            # if no overwritten element
            if el is None:
                el = "span"

            # mark errors
            tag += " error" * is_err

            tagged_elements.append(
                f'<{el} class="{tag}" {f'title="{tag}"' * tooltips}>{token_text}</{el}>'
            )

    text = "".join(tagged_elements)
    text = f'\n<pre><code class="code-snippet">{text} </code></pre>\n'

    if legend:
        text += "\n" + make_legend_html()

    if css_path:
        head = make_head_html(css_path)
        return head + f"\n<body>\n{text}\n</body>"
    else:
        return text


def format_html_simple(
    tokens: list[str],
    tags: list[str],
) -> str:
    """Format HTML document of tagged text.

    ## parameters
    - override_elements: optionally use something else than `<span>`.
    - exclude_tags: these will be left as plain text
    - level_brackets: if true, replace brop/brcl with br{n}
    - css_path: if not None, make document with `<head>` and `<body>`
    """

    # no need to tag whitespace/unknown tokens
    exclude_tags = ["ws", "uk"]

    tagged_elements: list[str] = []
    for token, tag in zip(tokens, tags):
        # fix html specials
        token_text = html_specials(token)
        if tag in exclude_tags:
            tagged_elements.append(token_text)
        else:
            s = f'<span class="{tag}">{token_text}</span>'
            tagged_elements.append(s)

    text = "".join(tagged_elements)
    final_html = f'\n<pre><code class="code-snippet">{text}</code></pre>\n'

    return final_html


def render_preview(
    data: pl.DataFrame,
    css_path: str,
    title: str | None = None,
    correct: pl.DataFrame | None = None,
    show_names: bool = False,
    mark_nondet: bool = False,
):
    """Write a complete HTML document"""

    if correct is not None:
        if len(data) != len(correct):
            raise ValueError("# examples mismatch")
        data = data.join(
            correct.select("name", "lang", "tags"),
            on=["name", "lang"],
            how="left",
            suffix="_correct",
        )

        if len(data) != len(correct):
            raise ValueError("join mismatch")

    # to compute mean accuracy (if `correct` provided)
    accs = []
    document_examples = ""  # append all markup to this
    for ex in data.iter_rows(named=True):
        _, tags_det = text_process.process("".join(ex["tokens"]))
        if correct is not None:
            errors = [
                t != tc for t, tc in zip(ex["tags"], ex["tags_correct"], strict=True)
            ]
            acc = 1 - (sum(errors) / len(errors))
            accs.append(acc)

        else:
            errors = None
            acc = None

        if mark_nondet:
            mark = ["mark" if (t == "uk") else None for t in tags_det]
        else:
            mark = None
        ex_text = format_html(
            ex["tokens"],
            ex["tags"],
            override_elements=mark,
            tooltips=True,
            errors=errors,
        )
        ex_title = f"{(ex['name'] + ' ') * show_names} ({ex['lang']})"
        if correct is not None and acc is not None:
            ex_title += f" {acc * 100:.1f}% acc"
        ex_title = f"\n<p>{ex_title}</p>"
        document_examples += ex_title + "\n" + ex_text + "\n<br>\n"

    document_full = make_head_html(css_path, title) + "<body>\n"
    if title:
        document_full += f"<h1>{title}</h1>"
    if correct is not None:
        mean_acc = sum(accs) / len(accs)
        document_full += f"\n<p>Mean accuracy: {mean_acc * 100:.1f}%</p>\n"

    document_full += document_examples + "</body>\n"

    if title:
        filename = re.sub(r"[\\\/\.]", "_", title)
    else:
        filename = "output"

    with open(f"./previews/{filename}.html", "w", encoding="utf-8") as f:
        f.write(document_full)
    print(f"wrote {len(document_full.splitlines())} lines")


def make_previews_index():
    """Create a index with links to previews"""

    paths = sorted(glob("*.html", root_dir="previews/"))
    try:
        paths.remove("index.html")
    except ValueError:
        pass

    links = [f'<li><a href = "{p}">{p.split(".")[0]}</a></li>' for p in paths]
    content = "<ul>" + "\n".join(links) + "</ul>"
    document = '<!DOCTYPE html><html><head><meta charset="utf-8" />'
    document += f"</head><body>{content}</body></html>"
    with open("previews/index.html", "w", encoding="utf-8") as f:
        f.write(document)
