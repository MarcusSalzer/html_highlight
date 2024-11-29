import regex as re


# patterns for some basic tokens
basic_pats = {
    "nu": r"(?<!\w)(?:0x[0-9a-fA-F]+|0b[01]+|[\d_]+\.?\d*(?:e\d+|e-\d+|\w+)?)",
    "ws": r"[\r\t\f\v ]+",
    "nl": r"\n+",
    "pu": r"[\.,;:]+",
    "uk": r"\w+|[^\w\s]+?",
}

# todo: include "f" for float in number?

# named groups to capture some initial tags
regex_token = re.compile("|".join(f"(?P<{k}>{v})" for k, v in basic_pats.items()), re.I)


def tokenize_plus(text: str):
    """Tokenize and find some basic tags"""
    tokens = []
    tags = []

    # Iterate over all matches
    for m in regex_token.finditer(text):
        tokens.append(m.group())  # matched token
        tags.append(m.lastgroup)  # corresponding tag

    return tokens, tags


class TextProcess:
    def __init__(self, text: str):
        self.tokens, self.tags = tokenize_plus(text)
