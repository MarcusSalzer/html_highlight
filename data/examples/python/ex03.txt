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