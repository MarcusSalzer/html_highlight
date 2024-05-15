import sys
import json
import os
import highlighting_functions as hf

example_file = "data/examples/ex01.py"


def main():
    # load example to annotate
    with open(example_file) as f:
        text = f.read()

    print(text)

    # load known (basic) individual tokens
    with open("data/known_minimal.json", "r") as f:
        known_minimal = json.load(f)

    # load tag names and aliases
    aliases = load_aliases("data/class_aliases_str.json")

    # do basic tagging
    tokens, tags = hf.tokenize(text)
    tags = hf.tag_individuals(tokens, tags, known_minimal)
    print(tokens)
    print(tags)

    # annotate unknowns
    tags_new = annotate_loop(tokens, tags)
    tags_new = simplify_tags(tags_new, aliases)

    print("-" * 30 + "\n")
    for x in zip(tokens, tags_new, tags, strict=True):
        if "unk" in x:
            print(x[0], ":", x[1])
    print("-" * 30 + "\n")
    accept = input("accept annotation?(y/n)").lower()

    # if accept == "y":

    print(tags_new)
    # TODO: save results


def annotate_loop(tokens: list[str], tags: list[str], fill_copies=True):
    """Successively ask for input to annotate unknown tokens.

    ## Parameters
    - tokens
    - tags
    - fill_copies (bool): Tag all occurrences of a token at once.

    ## Returns
    - tags_new (list[str]): New list of tags
    """
    max_length = 20
    tags_new = tags.copy()

    print("\nAnnotating:\n")
    for i, token in enumerate(tokens):
        # for aligned print
        padding = " " * (max_length - len(token))
        if tags_new[i] == "unk":
            tags_new[i] = input(repr(token) + padding + "? ")

            if fill_copies:
                for j, t2 in enumerate(tokens):
                    if t2 == token:
                        tags_new[j] = tags_new[i]

        else:
            print(repr(token) + padding + ": " + tags_new[i])

    return tags_new


def simplify_tags(tags: str | list[str], aliases: dict[str, list[str]], verbose=False):
    """Replace aliases to convention"""
    tags_out = []
    for tag in tags:
        name = "unk"
        if tag in aliases.keys():
            name = tag
        else:
            for k in aliases.keys():
                if tag in aliases[k]:
                    name = k
        if verbose:
            print(tag, name)
        tags_out.append(name)
    return tags_out


def load_aliases(path: str) -> dict:
    """Load alias dictionary for tags and check uniqueness."""
    with open(path) as f:
        class_aliases: dict = json.load(f)

    alias_list = [a for als in class_aliases.values() for a in als]

    assert len(alias_list) == len(set(alias_list)), "Duplicate aliases"

    return class_aliases


if __name__ == "__main__":
    main()
