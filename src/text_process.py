import regex as re


# patterns for some basic tokens
# in order
basic_pats = {
    "cofl": r"^(?:\/\/|#).+$",  # one full line comment
    "st": r"[\"'][^\s]*[\"']",
    "br": r"[\(\)\[\]\{\}]",
    "nu": r"(?<!\w)(?:0x[0-9a-fA-F]+|0b[01]+|\d[\d_]*\.?\d*(?:e\d+|e-\d+|\w+)?)",
    "ws": r"[\r\t\f\v ]+",
    "nl": r"\n+",
    "op3": r"===|!==|<=>",
    "op2": r"(?:<=|>=|==|!=|\+=|-=|--|\*\*|\/\/|\+\+|\*=|\/=|.\^)|\|\||&&",
    "opas": r"(?:=|<-)",
    "op": r"[\+\-\*\/%\^!]",
    "pu": r"[\.,;:]",
    "uk": r"\w+|[^\w\s]+?",
}

# NOTE: needed priorites:
# comment < string < everything
# bigger compound operators <= smaller operators

# todo: include "f" for float in number?

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


class TextProcess:
    def __init__(self, text: str):
        self.tokens, self.tags = process_regex(text)
