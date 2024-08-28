import unittest
from src.text_functions import tokenize, tag_individuals


class TestFunctions(unittest.TestCase):
    def test_tokenize(self):
        text = "x = 23\ny = 'abc'"
        tokens, tags = tokenize(text)
        self.assertEqual(
            tokens,
            ["x", " ", "=", " ", "23", "\n", "y", " ", "=", " ", "'abc'"],
            "wrong token list",
        )
        self.assertEqual(len(tags), len(tokens), "inconsistent tags length")
        self.assertEqual(set(tags), {"unk", "num", "str"}, "inconsistent length")

    def test_individuals():
        pass


if __name__ == "__main__":
    unittest.main()
