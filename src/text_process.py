import regex as re

# patterns for some basic tokens
# in order
basic_pats = {
    "cofl": r"^(?:\/{2,3}|#|%).+$",  # one full line comment
    "st": r"[\"'][^\s]*[\"']",
    "br": r"[\(\)\[\]\{\}]",
    # numbers: hex, bin, decimal/scientific
    "nu": r"(?<!\w)(?:0x[0-9a-fA-F]+|0b[01]+|\d[\d_]*\.?\d*(?:e\d+|e-\d+|\w+)?)",
    "ws": r"[\r\t\f\v ]+",
    "nl": r"\n+",
    "op3": r"===|!==|<=>",
    "op2": r"(?:<<|>>|<=|>=|==|!=|\+=|-=|--|\*\*|\/\/|\+\+|\*=|\/=|.\^)|\|\||&&",
    "opas": r"(?:=|<-)",
    "op": r"[\+\-\*\/%\^!\|]",
    "pu": r"[\.,;:]",
    "uk": r"\w+|[^\w\s]+?",
}

# NOTE: needed priorites:
# comment < string < everything
# bigger compound operators <= smaller operators


# named groups to capture some initial tags
regex_token = re.compile("|".join(f"(?P<{k}>{v})" for k, v in basic_pats.items()), re.M)


def process_regex(text: str):
    """Tokenize and find some basic tags"""
    tokens = []
    tags = []

    # Iterate over all matches
    for m in regex_token.finditer(text):
        tokens.append(m.group())  # matched token
        tags.append(m.lastgroup)  # corresponding tag

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


class TextProcess:
    def __init__(self, text: str):
        self.tokens, self.tags = process_regex(text)
