import regex as re


class Highlighter:
    patterns_find = {}
    patterns_repl = {}

    def __init__(self, patterns_find, patterns_repl):
        self.patterns_find = patterns_find
        self.patterns_repl = patterns_repl

    def pre_process(self, text: str):
        # convert html reserved characters
        text = re.sub(r"&", r"&amp;", text)
        text = re.sub(r"<", r"&lt;", text)
        text = re.sub(r">", r"&gt;", text)
        text = re.sub(r'"', r"&quot;", text)
        text = re.sub(r"'", r"&apos;", text)
        # preserve line-breaks
        text = re.sub(r"\n", r"<br>\n", text)
        # preserve indentation
        text = re.sub(r"  ", r"&nbsp; ", text)
        return text

    def find_variables(self, text: str):
        """Finds variables in code.

        ## Returns
        variables: list[str]
            List of variable names

        pattern_variable: str
            regex pattern for finding variables"""

        variables = re.findall(r"\b[^\s]+ ?=", text)
        variables = set(map(lambda s: re.sub(" ?=", "", s), variables))
        pattern_variable = r"\b(" + "|".join(variables) + r")\b"

        return variables, pattern_variable

    def syntax_highlight(self, text: str):
        patterns = self.patterns_find

        tagged = {}
        # tag with placeholders
        for tag in patterns.keys():
            tagged[tag] = []
            while True:
                m = re.search(patterns[tag], text)
                if m:
                    tagged[tag].append(f"""<span class="{tag}">{m.group()}</span>""")
                    text = text[: m.start()] + f"£££{tag}£££" + text[m.end() :]
                else:
                    break

            # fix parantheses
            for i, t in enumerate(tagged[tag]):
                tagged[tag][i] = re.sub(r"\(</span>", r"</span>(", t)

        # replace placeholders
        for tag in tagged.keys():
            for i, t in enumerate(tagged[tag]):
                m = re.search(f"£££{tag}£££", text)
                text = text[: m.start()] + t + text[m.end() :]

        css_link = '<link rel="stylesheet" type="text/css" href="_highlight_style.css">'
        html_text = f"""<div class="code-snippet">{text}</div>"""
        final_html = f"""
        <head>
            {css_link}
        </head>
        <body>
            {html_text}
        </body>
        """
        return final_html

    def process(self, text: str):
        """Preprocesses, finds variables and highlights."""

        text = self.pre_process(text)
        variables, p_variable = self.find_variables(text)
        if len(variables) > 0:
            self.patterns_find["variable"] = p_variable

        text = self.syntax_highlight(text)
        return text


## new functions
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
        elif "brac" in tags[i] or tags[i] == "punct":
            if (
                (i >= 3)
                and (tags[i - 1] == "unk")
                and ("brac" in tags[i - 2])
                and (tags[i - 3] == "func")
            ):
                # func, brac, unk, brac/punct
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
    """Merge adjacent tokens if they have the same tag."""
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
        else:
            current_seq.append(token)

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
