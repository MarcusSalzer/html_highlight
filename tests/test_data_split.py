import unittest

import polars as pl

from src.data_functions import data_split


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
