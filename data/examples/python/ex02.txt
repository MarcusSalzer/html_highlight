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