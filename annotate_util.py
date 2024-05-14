import sys
import json
import os
import highlighting_functions as hf

example_file = "examples/ex01.py"


def main():
    # load example to annotate
    with open(example_file) as f:
        text = f.read()

    print(text)

    # load known (basic) individual tokens
    with open("data/known_minimal.json", "r") as f:
        known_minimal = json.load(f)

    # load tag names and aliases
    aliases = load_aliases("data/class_aliases.json")

    # do basic tagging
    tokens, tags = hf.tokenize(text)
    tags = hf.tag_individuals(tokens, tags, known_minimal)
    print(tokens)
    print(tags)

    # annotate unknowns
    tags_new = annotate_loop(tokens, tags)

    print(tags_new)
    # TODO: translating tag name utility
    # TODO: save results


def annotate_loop(tokens, tags):
    """Successively ask for input to annotate unknown tokens."""
    max_length = 20
    tags_new = tags.copy()

    print("\nAnnotating:\n")
    for i, token in enumerate(tokens):
        # for aligned print
        padding = " " * (max_length - len(token))
        if tags[i] == "unk":
            tags_new[i] = input(token + padding + "? ")
        else:
            print(token + padding + ": " + tags[i])

    print("-" * 30)
    accept = input("accept annotation?(y/n)").lower()

    if accept == "y":
        return tags_new


def simplify_tags(tags: str | list[str], aliases: dict[str, list[str]]):
    """Replace aliases to convention"""
    tags_out = []
    for tag in tags:
        if tag in aliases.keys():
            tags_out.append(tag)
        else:
            print("TODO")  # TODO


def load_aliases(path: str):
    """Load alias dictionary for tags and check uniqueness."""
    with open(path) as f:
        class_aliases: dict = json.load(f)

    alias_list = [a for als in class_aliases.values() for a in als]

    assert len(alias_list) == len(set(alias_list)), "Duplicate aliases"

    return class_aliases


if __name__ == "__main__":
    main()
