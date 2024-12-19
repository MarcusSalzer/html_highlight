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
    ("st", r"[\"'][^\"']*[\"']"),
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
