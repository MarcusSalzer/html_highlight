import json
import os
import sys
from glob import glob
import polars as pl
import regex as re

sys.path.append(".")
from src import text_process


EXAMPLE_DIR = os.path.join("data", "examples")
OUTPUT_DIR = os.path.join("data", "annotated_codes")


def main():
    os.system("cls" if os.name == "nt" else "clear")

    # load tag names and aliases
    aliases = load_aliases("data/class_aliases_str.json")
    print("possible tags:")
    print("\n".join([f"{t:<6}: {a[-1]}" for t, a in aliases.items()]) + "\n")

    # get example files
    files = get_example_files()
    name, lang, _ = files.sort("size").row(0)
    with open(os.path.join(EXAMPLE_DIR, lang, name + ".txt")) as f:
        text = f.read()

    print("type `ignore` to save example for later\n")

    print("-" * 15 + f" {name} ({lang}) " + "-" * 15 + "\n")

    print(text)

    # basic initial tagging
    tokens, tags = text_process.process_regex(text)

    tags = [simplify_tag(t, aliases) for t in tags]

    # annotate unknowns
    tags_new = annotate_loop(tokens, tags, aliases, fill_copies=False)

    if tags_new is None:
        print(f"Ignoring example: {name} ({lang})!")
        with open(os.path.join("data", "ignore.txt"), "a") as f:
            f.write(f"{name}_{lang}\n")
        exit(0)

    print("-" * 30 + "\n")
    print(text)
    print("-" * 30 + "\n")

    # print tokens that might have changed
    for token, tag_new, tag_old in zip(tokens, tags_new, tags, strict=True):
        if tag_new == "uk" or tag_old == "uk":
            # print last alias (most verbose)
            print(f"{token:<30} -> {aliases[tag_new][-1]}")
    print("-" * 30 + "\n")
    accept = input("accept annotation?(y/n)").lower()

    if accept == "y":
        save_path = os.path.join(OUTPUT_DIR, f"{name}_{lang}.json")
        with open(save_path, "w") as f:
            json.dump(dict(tokens=tokens, tags=tags), f)
        print("Done! Saved annotations")


def get_example_files():
    files = glob("**/*txt", root_dir=EXAMPLE_DIR)
    done_files = os.listdir(OUTPUT_DIR)
    langs = os.listdir(EXAMPLE_DIR)

    try:
        with open(os.path.join("data", "ignore.txt")) as f:
            ignore_files = f.readlines()
    except FileNotFoundError:
        ignore_files = []

    print(f"found {len(files)} files")
    print(f"already done {len(done_files)} files")
    print(f"ignoring {len(ignore_files)} files")

    file_data = []
    for f in files:
        lang, name = re.split(r"[\.\/\\]", f)[:2]
        if lang not in langs:
            raise ValueError(f"incorrect language: {repr(lang)}")
        size = os.path.getsize(os.path.join(EXAMPLE_DIR, f))
        file_data.append([name.strip(), lang.strip(), size])

    done_ignore_data = []
    for f in done_files:
        name, lang = re.split(r"[\._]", f)[:2]
        if lang not in langs:
            raise ValueError(f"incorrect language: {repr(lang)}")
        done_ignore_data.append([name.strip(), lang.strip()])
    for f in ignore_files:
        splts = f.split("_")
        lang = splts[-1].strip()
        if lang not in langs:
            raise ValueError(f"incorrect language: {repr(lang)}")
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
    print(f"{len(files_df)} files left")

    return files_df


def annotate_loop(
    tokens: list[str], tags: list[str], aliases: dict[str, list[str]], fill_copies=True
):
    """Successively ask for input to annotate unknown tokens.

    ## Parameters
    - tokens
    - tags
    - fill_copies (bool): Tag all occurrences of a token at once.

    ## Returns
    - tags_new (list[str]): New list of tags
    """
    tags_new = tags.copy()

    print("\nAnnotating:\n")
    for i, token in enumerate(tokens):
        # for aligned print
        if tags_new[i] == "uk":
            t = input(f"{repr(token):<30}: ").lower().strip()

            if t == "ignore":
                return None

            tags_new[i] = simplify_tag(t, aliases)

            if fill_copies:
                for j, t2 in enumerate(tokens):
                    if t2 == token:
                        tags_new[j] = tags_new[i]

        else:
            # in already set, print verbose tag name
            print(f"{repr(token):<30}: " + aliases[tags_new[i]][-1])

    return tags_new


def simplify_tag(tag: str, aliases: dict[str, list[str]], verbose=False):
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
        tag_new = simplify_tag(input().strip(), aliases)

    if verbose:
        print(f"{tag} -> {tag_new}")

    return tag_new


def load_aliases(path: str) -> dict:
    """Load alias dictionary for tags and check uniqueness."""
    with open(path) as f:
        class_aliases: dict = json.load(f)

    alias_list = [a for als in class_aliases.values() for a in als]

    assert len(alias_list) == len(set(alias_list)), "Duplicate aliases"

    return class_aliases


if __name__ == "__main__":
    main()
