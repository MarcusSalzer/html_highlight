import polars as pl

from src.data_functions import data_split


class TestSplit:
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
            assert len(a) == 2
            assert len(b) == 2
            assert len(a.filter(pl.col("group") == "x")) == 1
            assert len(a.filter(pl.col("group") == "y")) == 1
            assert len(b.filter(pl.col("group") == "x")) == 1
            assert len(b.filter(pl.col("group") == "y")) == 1

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
            assert {2, 3} == {len(a), len(b)}
            assert len(a.filter(pl.col("group") == "x")) == 1
            assert len(b.filter(pl.col("group") == "x")) == 1
