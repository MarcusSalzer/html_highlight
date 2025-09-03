import os
from typing import Any, Iterable

from scripts.annotate_util import console
from src import _constants

from rich.markup import escape


def clearCLI():
    os.system("cls" if os.name == "nt" else "clear")


def pretty_print_code(tokens: list[str], tags: list[str]):
    text = ""
    styles = {
        "fn": "cyan",
        "brop": "yellow",
        "brcl": "yellow",
        "sy": "blue",
        "pu": "deep_pink4",
        "va": "red italic",
        "pa": "light_salmon3 italic",
        "st": "green",
        "nu": "blue_violet",
        "bo": "magenta",
        "kw": "gold3 bold",
        "cl": "light_slate_blue",
        "co": "grey23",
        "mo": "light_sky_blue1",
        "op": "yellow3",
        "ty": "light_slate_grey underline",
        "kwde": "bright_yellow",
        "kwfl": "dark_orange",
        "li": "rosy_brown",
    }
    for token, tag in zip(tokens, tags, strict=True):
        t_simple = _constants.MAP_TAGS_SIMPLE.get(tag, tag)
        st = styles.get(t_simple)
        if st is not None:
            text += f"[{st}]{escape(token)}[/{st}]"
        else:
            text += escape(token)
    console.print(text, highlight=False)


def pick_option[T](options: list[T], prompt: str) -> T | None:
    for i, m in enumerate(options):
        print(f"{i}. {m}")

    res = None
    while res is None:
        try:
            return options[int(input(prompt))]
        except ValueError:
            pass
