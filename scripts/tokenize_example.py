import sys

sys.path.append(".")
from src import text_process

examples = [
    "let x = 37 * 9",
    "this_is_a_str ='hej på dig'",
    "let sum <- 0\nfor k in 1:10:\nsum += k",
]

if __name__ == "__main__":
    for text in examples:
        print("\n" + "-" * 20)

        tokens = text_process.process_regex(text)
        print(tokens)
