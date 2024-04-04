import regex as re


def tokenize(text) -> list[str]:
    """Tokenize a code snippet. Also tag string literals and numbers."""
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


def tag_individuals(tokens, tags, known_str: dict = None) -> list[str]:
    """Tag individual tokens based on known symbols/strings"""
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
    """Find and tag function names, based on '('."""
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


def bracket_levels(tags):
    """Rename bracket tags from brac_op/cl to brac_num.

    ## Returns
    - tags (list[str]): modified tags
    - brac_level (list[int]): bracket depth for all tokens."""
    brac_level = []
    current_level = 0
    for i in range(len(tags)):
        if tags[i] == "brac_op":
            brac_level.append(current_level)
            current_level += 1
            tags[i] = f"brac{current_level}"
        elif tags[i] == "brac_cl":
            tags[i] = f"brac{current_level}"
            current_level -= 1
            brac_level.append(current_level)
        else:
            brac_level.append(current_level)

    return tags, brac_level


def merge_adjacent(tokens, tags, known_str=None):
    """Merge adjacent tokens if they have the same tag.
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
    text = re.sub(r"  ", r"&nbsp; ", text)
    text = re.sub(r"\n", r"<br>\n", text)
    return text


def tokens_to_html(tokens, tags, exclude_tags=("wsp", "unk")):
    """Format text with html spans."""

    tokens_with_tags = []
    for token, tag in zip(tokens, tags):
        # fix html specials
        token_text = html_specials(token)
        if tag in exclude_tags:
            tokens_with_tags.append(token_text)
        else:
            tokens_with_tags.append(f"""<span class="{tag}">{token_text}</span>""")

    text = "".join(tokens_with_tags)
    return f"""<div class="code-snippet">{text}</div>"""


def highlight_code(text: str, css_path="_style.css"):
    """Format a code snippet with classes.

    ## Parameters
    - text(str): source code to highlight
    - css_path(str): path to css file to link

    ## Returns
    """

    known_default = {
        "assign": ["=", "<-"],
        "punct": [",", ";", "."],
        "op": r"!%&/+-*:<>^",
        "brac_op": r"([{",
        "brac_cl": r")]}",
        "keyword": [
            "for",
            "while",
            "foreach",
            "as",
            "in",
            "if",
            "else",
            "elif",
            "and",
            "or",
            "not",
            "return",
        ],
    }
    tokens, tags = tokenize(text)
    # first: tag individual tokens
    tags = tag_individuals(tokens, tags, known_default)

    # merge and tag again to catch multiple character assignment etc.
    tokens, tags = merge_adjacent(tokens, tags, known_default)

    # rename brackets
    tags, brac_level = bracket_levels(tags)

    # second: context
    tags = tag_functions(tokens, tags)
    tags = tag_variables(tokens, tags)

    tokens, tags = merge_adjacent(tokens, tags, known_default)
    EXCLUDE_TAGS = ("unk", "wsp")
    html_text = tokens_to_html(tokens, tags, EXCLUDE_TAGS)
    classes = tuple(sorted(set(tags)))

    css_path = "_style.css"
    css_link = f'<link rel="stylesheet" type="text/css" href="{css_path}">'
    final_html = f"""
<head>
    {css_link}
</head>
<body>
{html_text}
</body>
"""

    return final_html, classes
