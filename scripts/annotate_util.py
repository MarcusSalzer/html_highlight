"""Interactive annotation CLI."""

import json
import os
from pathlib import Path
import sys
from glob import glob
from typing import Literal
import polars as pl
import regex as re
from rich.console import Console


sys.path.append(".")
from src import cli_util, text_process, util
from src._constants import LANGS

console = Console()

EXAMPLE_DIR = Path("data/examples")
DATASET_FILE = Path("data/dataset.ndjson")


IGNORE_PRINT = ["ws", "id", "nl", "brop", "brcl"]


def main():
    cli_util.clearCLI()
    # get example files
    name, lang = get_example()

    # load tag names and aliases
    aliases = load_aliases("data/class_aliases_str.json")
    console.print("-" * 30 + "\n", style="dim")
    lines = [f"[bold]{t:<6}[/bold][dim]:[/dim] {a[-1]}" for t, a in aliases.items()]
    width = max(len(li) for li in lines)

    for idx, li in enumerate(lines):
        console.print(li.ljust(width + 2), end="")
        if idx % 2 == 1:
            print()

    ex_file = EXAMPLE_DIR / lang / f"{name}.txt"
    text = ex_file.read_text("utf-8")

    # basic initial tagging
    tokens, tags = text_process.process(text)

    print("\n\ntype `ignore` to save example for later")
    print("type class +`!` to mark all\n")

    pad = "[green]" + "-" * 15 + "[/green]"
    console.print(f"{pad} {name} ({lang}) {pad}")

    tags = [canonicalize_tag(t, aliases) for t in tags]

    cli_util.pretty_print_code(tokens, tags)

    # annotate unknowns
    tags_new = annotate_loop(tokens, tags, aliases)

    if tags_new is None:
        ignore(name, lang)
        exit()

    console.print("-" * 30 + "\n", style="dim")
    cli_util.pretty_print_code(tokens, tags_new)
    console.print("-" * 30 + "\n", style="dim")

    # print tokens that might have changed
    max_t_len = max(len(t) for t in tokens)
    for token, tag_new, tag_old in zip(tokens, tags_new, tags, strict=True):
        if tag_old == "uk":
            # print last alias (most verbose)
            print(f"{token.ljust(max_t_len + 2)} -> {aliases[tag_new][-1]}")
        if tags_new == "uk":
            print(f"NOTICE: {token:<30} -> UNKNOWN")

    console.print("-" * 30 + "\n", style="dim")
    diff = read_difficulty()
    if diff == "ignore":
        ignore(name, lang)
    else:
        save(name, lang, tokens, tags_new, diff)
        ex_file.rename(Path("data") / "trash" / f"{name}_{lang}.txt")


def read_difficulty():
    while True:
        diff_response = input(
            "difficulty: (e)asy, (n)ormal, (a)mbiguous, (u)nknown ?\n"
        ).lower()

        diff = None
        if diff_response == "ignore":
            return "ignore"
        elif diff_response in ["easy", "e"]:
            diff = "easy"
        elif diff_response in ["normal", "n"]:
            diff = "normal"
        elif diff_response in ["ambiguous", "a"]:
            diff = "ambiguous"
        elif diff_response in ["unknown", "u"]:
            diff = "unknown"

        if diff:
            return diff
        else:
            print("enter difficulty or ignore, or ctrl+C to cancel")


def save(
    name: str,
    lang: str,
    tokens: list[str],
    tags: list[str],
    difficulty: Literal["easy", "normal", "ambiguous", "unknown"],
):
    record = {
        "name": name,
        "lang": lang,
        "difficulty": difficulty,
        "tokens": tokens,
        "tags": tags,
    }

    # Append as one line of JSON
    with open(DATASET_FILE, "a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")

    console.print("Done! Saved annotation", style="bold green")


def ignore(name, lang):
    print(f"Ignoring example: {name} ({lang})!")
    with open(os.path.join("data", "ignore.txt"), "a") as f:
        f.write(f"{name}_{lang}\n")


def get_example():
    files = glob("**/*txt", root_dir=EXAMPLE_DIR)

    if not DATASET_FILE.exists():
        console.print("No dataset file, creates", style="yellow")
        DATASET_FILE.touch()

    dataset = util.load_dataset_parallel(DATASET_FILE)

    done_examples = [d.id for d in dataset]

    try:
        with open(os.path.join("data", "ignore.txt")) as f:
            ignore_files = f.readlines()
    except FileNotFoundError:
        ignore_files = []

    console.print(f"found {len(files)} files")
    console.print(f"already done {len(done_examples)} files")
    console.print(f"ignoring {len(ignore_files)} files")

    file_data = []
    for f in files:
        lang, name = re.split(r"[\.\/\\]", f)[:2]
        if lang not in LANGS:
            raise ValueError(f"incorrect language: {repr(lang)}")
        size = os.path.getsize(os.path.join(EXAMPLE_DIR, f))
        file_data.append([name.strip(), lang.strip(), size])

    done_ignore_data = []
    for f in done_examples:
        splts = f.split("_")
        lang = splts[-1].split(".")[0].strip()
        if lang not in LANGS:
            raise ValueError(f"incorrect language: {repr(lang)} (in done)")
        name = "_".join(splts[:-1])
        done_ignore_data.append([name.strip(), lang.strip()])
    for f in ignore_files:
        splts = f.split("_")
        lang = splts[-1].strip()
        if lang not in LANGS:
            raise ValueError(f"incorrect language: {repr(lang)} (in ignore)")
        name = "_".join(splts[:-1])
        done_ignore_data.append([name.strip(), lang.strip()])

    done_df = pl.DataFrame(done_ignore_data, schema=["name", "lang"], orient="row")

    files_df = pl.DataFrame(
        file_data,
        schema={"name": pl.String, "lang": pl.String, "size": pl.Int32},
        orient="row",
    )

    if len(done_df) > 0:
        files_df = files_df.join(done_df, ["name", "lang"], how="anti")

    console.print(f"{len(files_df)} files left")

    if files_df.is_empty():
        print("No examples left")
        exit(0)
    name, lang, _ = files_df.sort("size").row(0)
    assert isinstance(name, str)
    assert isinstance(lang, str)

    return name, lang


def annotate_loop(tokens: list[str], tags: list[str], aliases: dict[str, list[str]]):
    """Successively ask for input to annotate unknown tokens.

    ## Parameters
    - tokens
    - tags
    - fill_copies (bool): Tag all occurrences of a token at once.

    ## Returns
    - tags_new (list[str]): New list of tags
    """
    tags_new = tags.copy()
    max_t_len = max(len(t) for t in tokens)

    print("\nAnnotating:\n")
    for i, token in enumerate(tokens):
        # for aligned print
        if tags_new[i] == "uk":
            t = input(f"{repr(token).ljust(max_t_len + 2)}: ").lower().strip()

            # optionally mark all copies of the same token
            if t and t[-1] == "!":
                fill_copies = True
                t = t[:-1]
            else:
                fill_copies = False

            if t == "ignore":
                return None

            t_new = canonicalize_tag(t, aliases)
            if t_new == "ignore":
                return None
            tags_new[i] = t_new

            if fill_copies:
                for j, t2 in enumerate(tokens):
                    if t2 == token:
                        tags_new[j] = tags_new[i]

        elif tags_new[i] not in IGNORE_PRINT:
            # if already set, print verbose tag name
            console.print(
                f"{repr(token).ljust(max_t_len + 2):}: " + aliases[tags_new[i]][-1],
                style="dim",
            )

    return tags_new


def canonicalize_tag(tag: str, aliases: dict[str, list[str]], verbose=False):
    """Replace aliases to convention"""
    tag_new = None
    if tag in aliases.keys():
        # use normal form if match
        tag_new = tag
    else:
        for k in aliases.keys():
            if tag in aliases[k]:
                tag_new = k

    while tag_new is None:
        print(f"unsupported tag: {tag}. Replace?")
        tag_new = canonicalize_tag(input().strip(), aliases)

    if verbose:
        print(f"{tag} -> {tag_new}")

    return tag_new


def load_aliases(path: str) -> dict[str, list[str]]:
    """Load alias dictionary for tags and check uniqueness."""
    with open(path) as f:
        class_aliases: dict[str, list[str]] = json.load(f)

    alias_list = [a for als in class_aliases.values() for a in als]
    counts = {}
    for a in alias_list:
        if a in counts:
            counts[a] += 1
        else:
            counts[a] = 1

    if len(alias_list) != len(set(alias_list)):
        raise ValueError(
            f"Duplicate aliases: {', '.join(k for k in counts if counts[k] > 1)}"
        )

    return class_aliases


if __name__ == "__main__":
    main()
