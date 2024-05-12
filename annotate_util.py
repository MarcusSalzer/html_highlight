import sys
import os
import highlighting_functions as hf

example_file = "examples/ex01.py"


def main():
    with open(example_file) as f:
        text = f.read()

    print(text)

    tokens, tags = hf.tokenize(text)
    print(tokens)
    print(tags)


if __name__ == "__main__":
    main()
