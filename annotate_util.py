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

    print("-"*30)
    accept = input("accept annotation?(y/n)").lower()

    if accept == "y":
        return tags_new


if __name__ == "__main__":
    main()
