import math
import regex as re


# which tags are completely deterministic?
DET_TAGS = ["cofl", "brop", "brcl", "id", "nu", "ws", "nl", "pu"]

# which tokens have a few possible tags?
# just a few examples...
POSSIBLE_PER_TOKEN = {
    "<": ["sy", "opcm"],
    "+": ["opbi", "opun"],
    "-": ["opbi", "opun"],
    "  ": ["ws", "id"],
}


# patterns for some basic tokens
# in order
basic_pats = [
    ("cofl", r"^(?:\/{2,3}|#|%).+$"),  # one full line comment
    ("st", r"\"[^\"]*\""),
    ("st", r"'[^']*'"),
    ("brop", r"[\(\[\{]"),
    ("brcl", r"[\)\]\}]"),
    # catch some syntax features before numbers
    ("sy", r"\.{2,3}"),
    # numbers: scientific
    ("nu", r"(?<!\w)\d+(?:\.\d+)?+e-\d+"),
    # numbers: hex, bin,
    ("nu", r"(?<!\w)0x[0-9a-fA-F]+|0b[01]+"),
    # numbers: integer, decimal
    ("nu", r"(?<!\w)\d[\d_]*(?:\.\d+)?\w*"),
    ("ws", r"[\r\t\f\v ]+"),
    ("nl", r"\n+"),
    ("sy", r">>>"),
    # comparison operators
    ("opcm", r"===|!==|<=>|<=|>=|==|!="),
    ("opbi", r"<<|>>|\*\*|\/\/|\.\^|\|\||&&"),
    ("opun", r"\+\+|--"),
    ("opas", r"=|<-|\+=|-=|\*=|\/="),
    ("sy", r"->|=>|::|:|(?<=[^\s])\.(?=[^\s])"),
    ("pu", r",|;"),
    ("uk", r"\$[_\p{L}][_\p{L}\d]*"),
    ("uk", r"\w+|[^\w\s]+?"),
]

# NOTE: needed priorites:
# comment < string < everything
# bigger compound operators <= smaller operators


# named groups to capture some initial tags


def process_regex(text: str, patterns: list[tuple[str, str]] = basic_pats):
    """Tokenize and find some basic tags"""

    regex_token = re.compile("|".join(f"(?P<{t}>{p})" for t, p in patterns), re.M)
    tokens = []
    tags = []

    # Iterate over all matches
    for m in regex_token.finditer(text):
        tokens.append(m.group())  # matched token
        tags.append(m.lastgroup)  # corresponding tag

    if "".join(tokens) != text:
        raise ValueError("missed something")

    return tokens, tags


def merge_adjacent(
    tokens,
    tags,
    merge_only: list[str] | None = None,
    dont_merge: list[str] = [],
    interactive: bool = False,
):
    """Merge adjacent tokens if they have the same tag.
    ## Parameters
    - tokens
    - tags
    - merge_only: if provided, merge only these tags, otherwise merge all.
    - interactive: ask for confirmation before merging

    ## returns
    - tokens_merged
    - tags_merged
    """

    if len(tokens) != len(tags):
        raise ValueError("Inconsistent sequence length")

    tokens_merged = []
    tags_merged = []
    # keep track of where merges where made
    merge_idx = []

    current_seq = []
    for i, (token, tag) in enumerate(zip(tokens, tags)):
        # next tag, None if at end
        tag_next = tags[i + 1] if i < len(tags) - 1 else None

        if tag == tag_next:
            if (merge_only is None or tag in merge_only) and tag not in dont_merge:
                if interactive:
                    if (
                        input(f"merge: `{token}` + `{tokens[i+1]}` ({tag}) ? ").lower()
                        != "y"
                    ):
                        continue

                current_seq.append(token)
                continue

        else:
            if current_seq:
                # break current merging
                merge_idx.append(len(tokens_merged))
                current_seq.append(token)
                tokens_merged.append("".join(current_seq))
                tags_merged.append(tag)

                current_seq = []
                continue

        tokens_merged.append(token)
        tags_merged.append(tag)

    return tokens_merged, tags_merged, merge_idx


def infer_indent(text: str, max_symbols=4) -> str | None:
    """Consider the beginning of each line to infer the indentation token"""
    counts = []
    symbols = set()
    for m in re.finditer(r"^([\t ])+", text, re.MULTILINE):
        counts.append(len(m.group(0)))
        symbols.add(m.group(1))

    if not counts or len(symbols) != 1:
        return None

    id_token = symbols.pop()
    if id_token == "\t":
        c = 1
    else:
        c = min(math.gcd(*counts), max_symbols)

    return id_token * c


def process(text: str):
    """Run complete process"""
    # remove trailing newline
    text = re.sub(r"\n+$", "", text)

    id_token = infer_indent(text)
    # make sure indentation is matched before whitspace
    if id_token:
        basic_pats.insert(0, ("id", id_token))

    tokens, tags = process_regex(text, basic_pats)

    return tokens, tags


def format_html(
    tokens: list[str],
    tags: list[str],
    exclude_tags: list[str] = ["ws", "uk"],
    level_brackets: bool = True,
    css_path: str | None = None,
) -> str:
    """Format HTML document of tagged text."""
    tokens_with_tags = []
    if level_brackets:
        tags, _ = bracket_levels(tags)
    for token, tag in zip(tokens, tags):
        # fix html specials
        token_text = html_specials(token)
        if tag in exclude_tags:
            tokens_with_tags.append(token_text)
        else:
            tokens_with_tags.append(f'<span class="{tag}">{token_text}</span>')

    text = "".join(tokens_with_tags)
    text = f'<pre><code class="code-snippet">{text} </code></pre>'
    if css_path:
        css_link = f'<link rel="stylesheet" type="text/css" href="{css_path}">'
        return f"<head>\n{css_link}\n</head>\n<body>\n{text}\n</body>"
    else:
        return text


def bracket_levels(tags: list[str]) -> tuple[list[str], list[int]]:
    """Rename bracket tags from br_op/cl to br_{n}.

    ## Returns
    - tags_new (list[str]): modified tags
    - brac_level (list[int]): bracket depth for all tokens."""
    brac_level = []
    current_level = 0
    tags_new = tags.copy()
    for i in range(len(tags_new)):
        if tags_new[i] == "brop":
            brac_level.append(current_level)
            tags_new[i] = f"br{current_level}"
            current_level += 1
        elif tags_new[i] == "brcl":
            current_level -= 1
            tags_new[i] = f"br{current_level}"
            brac_level.append(current_level)
        else:
            brac_level.append(current_level)

    return tags_new, brac_level


def html_specials(text: str) -> str:
    """Replace html reserved characters"""

    text = re.sub(r"&", r"&amp;", text)
    text = re.sub(r"<", r"&lt;", text)
    text = re.sub(r">", r"&gt;", text)
    text = re.sub(r'"', r"&quot;", text)
    text = re.sub(r"'", r"&apos;", text)
    return text
