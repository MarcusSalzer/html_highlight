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
    ("cofl", r"(?<=^\s+)(?:\/{2,3}|#|%).+$"),  # comment after indentation
    # html-comment
    ("cofl", r"<!--.+-->\s*?$"),
    # php/jsdoc multiline comments
    ("coml", r"\/\*{1,2}[\s\S]+?\*\/"),
    # inline comments (NOTE: quite sketchy)
    ("coil", r"(?<=\s+)\/{2} ?(?:[^+-=]+[\s:]+[^+-=].*[^\n]+)"),
    ("coil", r"(?<=\s+)# ?.*"),  # shell/py style, less confusion with op
    # triple-quote string
    ("st", r"\"{3}[\s\S\n]*\"{3}"),
    ("st", r"'{3}[\s\S\n]*'{3}"),
    # basic double-quote string
    ("st", r"\"[^\"\n]*\""),
    # special quote
    ("st", r"“[^\"\n]*”"),
    # rust lifetime annotations (NOTE: before single-strings)
    ("an", r"(?<=&|<|< |\+ |, )'\p{L}+(?=[>,])"),  # in angle brackets
    ("an", r"(?<=&)'\p{L}+"),  # in references
    # single quote string
    ("st", r"'[^'\n]*'"),
    ("brop", r"[\(\[\{]"),
    ("brcl", r"[\)\]\}]"),
    # catch some syntax features before numbers
    ("sy", r"\.{3}|\.{2}[=?]?"),
    # numbers: scientific
    ("nu", r"(?<!\w)\d+(?:\.\d+)?+e-\d+"),
    # numbers: hex, bin,
    ("nu", r"(?<!\w)0x[0-9a-fA-F]+|0b[01]+"),
    # numbers: integer, decimal, percent
    ("nu", r"(?<!\w)\d[\d_]*(?:\.\d+)?[\w%]*"),
    ("id", r"^[\t ]+"),
    ("ws", r"[\r\t\f\v ]+"),
    ("nl", r"\n+"),
    ("sy", r">>>"),
    # CSS classes (classes are identifiers, elements are type kws)
    ("va", r"^\.\p{L}\S*(?=.*{)"),
    # bash flags
    ("shfl", r"(?<!\S)--\p{L}+(?=\s|=|$)"),
    # bash flag or op or css attr
    ("uk", r"\p{L}*-?\p{L}+(?=\s|=|$|:)"),
    # rust macros
    ("uk", r"\S+!(?=\()"),
    # comparison operators
    ("opcm", r"===|!==|<=>|<=|>=|==|!="),
    ("opbi", r"<<|>>|\*\*|\/\/|\.\^|\|\||&&|~\/"),
    ("opun", r"\+\+|--"),
    ("sy", r"->|=>|\|>|::|:|(?<=[^\s])\.(?=[^\s])"),
    ("opas", r"<-|\+=|-=|\*=|\/="),
    ("uk", r"="),
    ("pu", r",|;"),
    # php/bash variable/parameter
    ("uk", r"\$[_\p{L}\d]+"),
    # bash special parameters
    ("uk", r"\$[*@?-]"),
    ("uk", r"\$#"),
    ("uk", r"\$\$"),
    # annotations
    ("an", r"^@\S+"),
    # everything else
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

    rec = "".join(tokens)
    if rec != text:
        diffc = len(text) - len(rec)
        raise ProcessError(f"Missed {diffc} characters")

    return tokens, tags


def merge_adjacent(
    tokens: list[str],
    tags: list[str],
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
        raise ProcessError("Inconsistent sequence length")

    tokens_merged: list[str] = []
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
                        input(
                            f"merge: `{token}` + `{tokens[i + 1]}` ({tag}) ? "
                        ).lower()
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
    """DEPRECATED?

    Consider the beginning of each line to infer the indentation token"""
    counts = []
    symbols = set()
    # hopefully avoids multiline comments
    for m in re.finditer(r"^([\t ])+(?!\*)", text, re.MULTILINE):
        counts.append(len(m.group(0)))
        symbols.add(m.group(1))

    if not counts or len(symbols) != 1:
        return None

    id_token = symbols.pop()
    if id_token == "\t":
        c = 1
    else:
        if {1, 4}.issubset(counts):
            # Avoid rare short indentation
            c = 4
        else:
            c = min(math.gcd(*counts), max_symbols)

    return id_token * c


def clean_text(text: str):
    """Basic cleanup"""
    # remove trailing newline
    text_clean = re.sub(r"\n+$", "", text)
    # remove trailing whitespace on lines
    text_clean = re.sub(r"[\t ]+$", "", text_clean, flags=re.M)

    return text_clean


def process(text: str):
    """Run complete process

    ## returns
    - tokens
    - tags
    """

    pats = basic_pats.copy()

    text = clean_text(text)

    tokens, tags = process_regex(text, pats)

    return tokens, tags


def process_with_inferindent(text: str, verbose: bool = False):
    """DEPRECATED?
    Run complete process"""
    # remove trailing newline
    text = re.sub(r"\n+$", "", text)
    # remove trailing whitespace on lines
    text = re.sub(r"\s+$", "", text, flags=re.M)

    id_token = infer_indent(text)

    pats = basic_pats.copy()

    # make sure indentation is matched before whitespace
    if id_token:
        pats.insert(0, ("id", r"^(?<=(?:" + id_token + ")*)" + id_token))
        if verbose:
            print(f"indentation: {repr(id_token)}, length {len(id_token)}")
    elif verbose:
        print("No indentation.")
    tokens, tags = process_regex(text, pats)

    return tokens, tags


def bracket_levels(tags: list[str]) -> tuple[list[str], list[int]]:
    """Rename bracket tags from br_op/cl to br{n}.

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


class ProcessError(Exception):
    "Something went wrong when processing text."

    def __init__(self, *args):
        super().__init__(*args)
