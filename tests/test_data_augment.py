import unittest

import polars as pl

from src.data_augmentation import modify_name, randomize_names


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


if __name__ == "__main__":
    unittest.main()
