import random


def modify_name(name: str):
    """Make a new name, with similar structure"""

    newname = ""
    for c in name:
        if c.isalpha():
            c2 = chr(random.randint(97, 122))
            if c.isupper():
                c2 = c2.upper()
        elif c.isnumeric():
            c2 = str(random.randint(0, 9))
        else:
            c2 = c
        newname += c2
    return newname


def randomize_names(tokens: list[str], tags: list[str]):
    """Randomize names of tokens with arbitrary names.

    Affected classes: `pa`, `mo`, `fnme`, `fnas`, `fnsa`, `va`, `at`
    """

    renameable = ["pa", "mo", "fnme", "fnas", "fnsa", "va", "at"]

    renamed = [False] * len(tokens)
    tokens_new = tokens.copy()
    for i, (token, tag) in enumerate(zip(tokens, tags, strict=True)):
        if tag in renameable and not renamed[i]:
            newname = modify_name(token)
            # print(token + "->" + newname)
            for j in range(i, len(tokens)):
                if tokens[j] == token:
                    renamed[j] = True
                    tokens_new[j] = newname

    return tokens_new
