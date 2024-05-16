import json
import os
import text_functions as tf

EXAMPLE_DIR = "data/examples"
OUTPUT_DIR = "data/annotated_codes"


def main():
    text, example_name = load_example()

    print("-" * 30 + "\n")

    print(text)

    # load known (basic) individual tokens
    with open("data/known_minimal.json", "r") as f:
        known_minimal = json.load(f)

    # load tag names and aliases
    aliases = load_aliases("data/class_aliases_str.json")

    # do basic tagging
    tokens, tags = tf.tokenize(text)
    tags = tf.tag_individuals(tokens, tags, known_minimal)
    tokens, tags = tf.merge_adjacent(tokens, tags, known_minimal)
    tags = tf.tag_functions(tokens, tags)
    tags = tf.tag_variables(tokens, tags)
    tags = simplify_tags(tags, aliases)

    # annotate unknowns
    tags_new = annotate_loop(tokens, tags)
    tags_new = simplify_tags(tags_new, aliases)

    changed = [False] * len(tags)
    for i, (old, new) in enumerate(zip(tags, tags_new, strict=True)):
        if old != new:
            changed[i] = True

    print("-" * 30 + "\n")
    for x in zip(tokens, tags_new, tags, strict=True):
        if "unk" in x:
            print(x[0], ":", x[1])
    print("-" * 30 + "\n")
    accept = input("accept annotation?(y/n)").lower()

    if accept == "y":
        save_annotated(tokens, tags_new, changed, example_name)


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


def load_example() -> tuple[str, str]:
    example_files = os.listdir(EXAMPLE_DIR)
    out_files = os.listdir(OUTPUT_DIR)

    todo_examples = []
    for ex_f in example_files:
        nam, typ = ex_f.split(".")
        name = f"{nam}_{typ}.json"
        if name not in out_files:
            todo_examples.append(ex_f)

    assert len(todo_examples) > 0, "Out of examples"

    print("examples to do:", len(todo_examples))

    example_name = todo_examples[0]

    example_path = os.path.join(EXAMPLE_DIR, example_name)

    # load example to annotate
    with open(example_path) as f:
        text = f.read()
    return text, example_name


def save_annotated(
    tokens: list[str], tags: list[str], changed: list[bool], example_name: str
):
    nam, typ = example_name.split(".")
    save_path = os.path.join(OUTPUT_DIR, f"{nam}_{typ}.json")
    with open(save_path, "w") as f:
        json.dump(dict(type=typ, tokens=tokens, tags=tags, changed=changed), f)


if __name__ == "__main__":
    main()
