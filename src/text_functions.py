"""OLD THINGS"""

import regex as re
import json


def tokenize(text) -> tuple[list[str], list[str]]:
    print("WARNING: deprecated")
    """Tokenize a code snippet. Also tag string literals and numbers.

    ## Parameters
    - text (str) a piece of source code

    ## Returns
    - tokens (list) tokenized text
    - tags (list) tags for string literals and numbers"""
    # Possible string delimiters
    string_delims = {'"', "'", '"""', "'''"}

    # split everything
    tokens: list[str] = re.findall(r"(\w+|[^\w\s]+|\s+|\n)", text)

    ## split tokens without letters or numbers
    tmp: list[str] = []
    for t in tokens:
        if bool(re.search(r"\p{L}|\p{digit}", t)):
            tmp.append(t)
        # elif len(t) > 1 and all((c == t[0] for c in t)):
        #     tmp.append(t)
        else:
            tmp.extend(list(t))

    tokens = tmp

    ## reunite string literals
    tmp = []
    tags = []
    current_delim = ""
    current_string = ""
    for i, t in enumerate(tokens):
        if current_delim == "":
            if t in string_delims:
                # start a string
                current_delim = t
                current_string += t
            else:
                tmp.append(t)
                tags.append("unk")
        else:
            if t == current_delim:
                # break current string
                current_delim = ""
                current_string += t
                tmp.append(current_string)
                tags.append("str")
                current_string = ""
            else:
                # add to current string
                current_string += t

    tokens = tmp

    ## reunite decimal numbers
    tmp = []
    tmp_tags = []
    current_num = ""
    for i, (token, tag) in enumerate(zip(tokens, tags)):
        token_next = None
        if i < len(tokens) - 1:
            token_next = tokens[i + 1]

        num_char = token.isdigit() or (token == "." and token_next.isdigit())

        if current_num == "":
            if num_char:
                current_num = token
            else:
                tmp.append(token)
                tmp_tags.append(tag)
        else:
            if num_char:
                current_num += token
            else:
                # end number
                tmp.append(current_num)
                tmp_tags.append("num")
                current_num = ""
                tmp.append(token)
                tmp_tags.append(tag)
    if current_num != "":  # if end with number
        tmp.append(current_num)
        tmp_tags.append("num")

    tokens = tmp
    tags = tmp_tags

    return tokens, tags


def tag_individuals(
    tokens: list[str], tags: list[str], known_str: dict = None
) -> list[str]:
    """Tag individual tokens based on known symbols/strings

    ## Parameters
    - tokens
    - tags
    - known_str: a dict of individual tokens that can be tagged without context.
        - if None: only tag `num` and `wsp`.

    ## Returns
    - tokens: updated tokens"""

    for i, token in enumerate(tokens):
        if token.isdigit():
            tags[i] = "num"
            continue
        elif not bool(re.search("[^\s]", token)):
            tags[i] = "wsp"

        if tags[i] == "unk":
            for key in known_str.keys():
                if token in known_str[key]:
                    tags[i] = key
                    break

    return tags


def tag_functions(tokens, tags):
    """Find and tag function names, based on ``(``."""
    functions = []
    for i, token in enumerate(tokens[:-2]):
        if tags[i] == "unk" and tokens[i + 1] == "(":
            functions.append(token)

    functions = set(functions)
    for i, token in enumerate(tokens):
        if token in functions:
            tags[i] = "func"

    return tags


def tag_variables(tokens, tags):
    """Find and tag variable names, based on 'assign' and 'function' tags."""
    variables = []
    for i in range(len(tokens)):
        if tags[i] == "assign":
            if i >= 1 and tags[i - 1] == "unk":
                # unk, assign,
                variables.append(tokens[i - 1])
            elif i >= 2 and tags[i - 1] == "wsp" and tags[i - 2] == "unk":
                # unk, wsp, assign
                variables.append(tokens[i - 2])
        elif "brac" in tags[i] or tags[i] in ("punct", "op", "wsp"):
            if (
                (i >= 3)
                and (tags[i - 1] == "unk")
                and ("brac" in tags[i - 2])
                and (tags[i - 3] == "func")
            ):
                # func, brac, unk, brac/punct
                variables.append(tokens[i - 1])
            elif tokens[i] != "(" and (
                tags[i - 1] == "unk" and tags[i - 2] in ("punct")
            ):
                # punct, unk, non-func-brac
                variables.append(tokens[i - 1])

    variables = set(variables)

    for i, token in enumerate(tokens):
        if token in variables:
            tags[i] = "var"

    return tags


def merge_adjacent(tokens, tags, known_str=None):
    """Merge adjacent tokens if they have the same tag.
    NOTE: OLD version
    ## Parameters
    - tokens
    - tags"""

    if not known_str:
        known_str = {}
    tmp = []
    tmp_tags = []

    current_seq = []
    for i, (token, tag) in enumerate(zip(tokens, tags)):
        # next tag, None if at end
        tag_next = tags[i + 1] if i < len(tags) - 1 else None

        if not tag == tag_next:
            if len(current_seq) > 0:
                current_seq.append(token)

                new_token = "".join(current_seq)
                new_tag = tag_individuals([new_token], ["unk"], known_str)[0]

                tmp.append(new_token)
                tmp_tags.append(new_tag)
                current_seq = []
            else:
                tmp.append(token)
                tmp_tags.append(tag)
        elif "brac" not in tag:  # avoid merging brackets
            current_seq.append(token)
        else:
            tmp.append(token)
            tmp_tags.append(tag)

    return tmp, tmp_tags


def html_specials(text: str) -> str:
    """Replace html reserved characters, preserve indentation and line-breaks"""
    text = re.sub(r"&", r"&amp;", text)
    text = re.sub(r"<", r"&lt;", text)
    text = re.sub(r">", r"&gt;", text)
    text = re.sub(r'"', r"&quot;", text)
    text = re.sub(r"'", r"&apos;", text)
    return text


def make_legend_html():
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
