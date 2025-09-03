# allowed languages in dataset
LANGS = [
    "bash",
    "c",
    "cpp",
    "csharp",
    "css",
    "dart",
    "go",
    "html",
    "java",
    "js",
    "json",
    "kotlin",
    "latex",
    "lua",
    "matlab",
    "php",
    "pseudo",
    "python",
    "r",
    "ruby",
    "rust",
    "sql",
    "ts",
]


REQUIRES_PRE = {"id": ("nl", "id"), "opas": ("ws", "va", "pa", "brcl", "shfl")}

# bad tag bigrams
ILLEGAL_BIGRAMS = [
    "nl ws",
    "ws nl",
    "va cl",
    "opbi opbi",
    "va va",
    "kw kw",
]


# TOKENS that are determined for a language
LANG_SPEC_TOKENS = {
    "python": {
        "=": "opas",
        "*=": "opas",
        "//": "opbi",
        "if": "kwfl",
        "else": "kwfl",
        "<": "opbi",
        ">": "opbi",
    },
    "js": {
        "=": "opas",
        "this": "va",
        "extends": "kwmo",
    },
    "php": {
        "=": "opas",
        "implements": "kwmo",
    },
    "dart": {
        "?": "opmo",
    },
    "rust": {
        "usize": "ty",
        # "&": "opmo OR opun IDK??",
    },
}

# which tags are completely deterministic?
DET_TAGS = ["brop", "brcl", "id", "nu", "ws", "nl", "pu"]

# which tokens have a few possible tags?
# just a few examples...
POSSIBLE_PER_TOKEN = {
    "<": ["sy", "opbi"],
    ">": ["sy", "opbi"],
    "+": ["opbi", "opun"],
    "-": ["opbi", "opun"],
    "  ": ["ws", "id"],
    "=": ["opas", "opbi", "sy"],
    "&": ["opmo", "opun", "opbi"],
    "*": ["opmo", "opun", "opbi"],
    "float": ["ty", "tyco", "pa"],
    "int": ["ty", "tyco"],
    "str": ["ty", "tyco"],
    "dict": ["ty", "tyco"],
    "set": ["ty", "tyco"],
}
# allow these in vocab
VOCAB_TAGS = [
    "kwfl",
    "kwty",
    "kwop",
    "kwmo",
    "kwde",
    "kwim",
    "id",
    "ws",
    "nl",
    "brop",
    "brcl",
    "sy",
    "pu",
    "bo",
    "li",
    "opbi",
    "opun",
    "opas",
    "an",
    "uk",
]
MAP_TAGS_SIMPLE = {
    "opun": "op",
    "opbi": "op",
    "opas": "op",
    "fnme": "fn",
    "fnst": "fn",
    "fnas": "fn",
    "fnfr": "fn",
    "kwop": "kw",
    "kwim": "kw",
    "kwva": "kw",
    "kwfn": "kw",
    "kwmo": "kw",
    "at": "va",
    "clco": "cl",
}
