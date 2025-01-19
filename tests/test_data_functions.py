import unittest

import polars as pl

from src.data_functions import modify_name, randomize_names, data_split


class TestModName(unittest.TestCase):
    def test_var_php(self):
        x = modify_name("$xyz")
        self.assertEqual("$", x[0])
        self.assertTrue(x[1:].islower())

    def test_camel(self):
        x = modify_name("AbraCadabra")
        self.assertTrue(x[0].isupper())
        self.assertTrue(x[1].islower())
        self.assertTrue(x[4].isupper())
        self.assertTrue(x[5].islower())

    def test_snake(self):
        x = modify_name("a_snake")
        self.assertTrue(x.islower())
        self.assertEqual("_", x[1])


class TestRandomizeTokens(unittest.TestCase):
    def test_fncall(self):
        x = randomize_names(
            ["x", "=", "f_un", "(", "x", ")"],
            ["va", "opas", "fnsa", "brop", "va", "brcl"],
        )
        self.assertEqual("=", x[1])
        self.assertEqual("(", x[3])
        self.assertEqual(")", x[5])


class TestSplit(unittest.TestCase):
    def test_4(self):
        df = pl.DataFrame(
            {
                "name": ["a", "b", "c", "d"],
                "group": ["y", "y", "x", "x"],
            },
        )

        splits = [0.5, 0.5]
        for _ in range(5):
            a, b = data_split(df, splits, stratify_col="group", shuffle=True)
            self.assertEqual(2, len(a))
            self.assertEqual(2, len(b))
            self.assertEqual(1, len(a.filter(pl.col("group") == "x")))
            self.assertEqual(1, len(a.filter(pl.col("group") == "y")))
            self.assertEqual(1, len(b.filter(pl.col("group") == "x")))
            self.assertEqual(1, len(b.filter(pl.col("group") == "y")))

    def test_5(self):
        df = pl.DataFrame(
            {
                "name": ["a", "b", "c", "d", "e"],
                "group": ["y", "y", "y", "x", "x"],
            },
        )

        splits = [0.5, 0.5]
        for _ in range(5):
            a, b = data_split(df, splits, stratify_col="group", shuffle=True)
            self.assertEqual({2, 3}, {len(a), len(b)})
            self.assertEqual(1, len(a.filter(pl.col("group") == "x")))
            self.assertEqual(1, len(b.filter(pl.col("group") == "x")))


if __name__ == "__main__":
    unittest.main()
