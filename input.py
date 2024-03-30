print("# tokens", len(tokens))

known_default = {
    "assign": r"=",
    "punct": r",;",
    "op": r"!%&/+-*:",
    "brac_op": r"([{",
    "brac_cl": r")]}",
    "keyword": r"for|while|foreach|as|in|if|else|elif|and|or|not",
}


# first: tag individual tokens
tags = hf.tag_individuals(tokens, tags, known_default)

# second: context

tags = hf.tag_variables(tokens, tags)
tags = hf.tag_functions(tokens, tags)